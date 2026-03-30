"""Audio loading and mel-spectrogram utilities for BirdCLEF+ 2026."""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T

from config import (
    SAMPLE_RATE, CHUNK_SAMPLES,
    N_MELS, N_FFT, HOP_LENGTH, F_MIN, F_MAX,
)

# ── Build transforms once (stateless, thread-safe) ────────────────────────────
_mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_min=F_MIN,
    f_max=F_MAX,
    power=2.0,
    norm="slaney",
    mel_scale="slaney",
    center=True,
)
# PCEN replaces AmplitudeToDB — normalises for recording-device frequency
# response variability (+0.049 soundscape val AUC over AmplitudeToDB).
# torchaudio.functional.pcen is absent in torchaudio 2.x; use manual IIR impl.
_PCEN_GAIN      = 0.98   # alpha  — AGC compression exponent
_PCEN_SMOOTH    = 0.025  # s      — IIR smoother time constant
_PCEN_BIAS      = 2.0    # delta  — stabilising offset
_PCEN_POWER     = 0.5    # r      — root-compression exponent
_PCEN_EPS       = 1e-6


def pcen(mel: torch.Tensor) -> torch.Tensor:
    """Per-Channel Energy Normalization (manual IIR implementation).

    mel  : (1, N_MELS, T)  linear power mel spectrogram (float32)
    Returns an array of the same shape with PCEN applied.

    Formula (Ben-Tzur et al. 2018):
        M[t] = (1 - s) * M[t-1]  +  s * E[t]          (IIR smoother)
        PCEN[t] = (E[t] / (eps + M[t])^gain + bias)^r  - bias^r
    Implemented with a loop over time so gradients are never needed
    (inference utility only — no autograd overhead).
    """
    with torch.no_grad():
        E  = mel.float()              # (1, F, T)
        T  = E.shape[2]
        s  = _PCEN_SMOOTH
        M  = E[:, :, 0].clone()       # (1, F)  initialise smoother at t=0
        out = torch.empty_like(E)
        bias_r = _PCEN_BIAS ** _PCEN_POWER

        for t in range(T):
            M = (1.0 - s) * M + s * E[:, :, t]
            denom = (M + _PCEN_EPS).pow(_PCEN_GAIN)
            out[:, :, t] = (E[:, :, t] / denom + _PCEN_BIAS).pow(_PCEN_POWER) - bias_r

    return out


def load_audio(path: "str | Path", target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file as a mono float32 numpy array.

    Resamples to target_sr if the file's native rate differs.
    Returns an array of arbitrary length.
    """
    waveform, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)   # stereo → mono
    if sr != target_sr:
        wav_t = torch.from_numpy(waveform).unsqueeze(0)  # (1, T)
        wav_t = torchaudio.functional.resample(wav_t, sr, target_sr)
        waveform = wav_t.squeeze(0).numpy()
    return waveform


def pad_or_crop(
    waveform: np.ndarray,
    target_len: int = CHUNK_SAMPLES,
    random_crop: bool = True,
) -> np.ndarray:
    """Return a waveform of exactly target_len samples.

    Short clips are tiled (looped). Long clips are randomly cropped during
    training and front-cropped during validation/inference.
    """
    n = len(waveform)
    if n == target_len:
        return waveform
    if n < target_len:
        repeats = -(-target_len // n)      # ceiling division
        waveform = np.tile(waveform, repeats)
    if len(waveform) > target_len:
        if random_crop:
            start = np.random.randint(0, len(waveform) - target_len)
        else:
            start = 0
        waveform = waveform[start: start + target_len]
    return waveform


def waveform_to_mel(waveform: np.ndarray) -> torch.Tensor:
    """Convert a (CHUNK_SAMPLES,) numpy array to a (3, N_MELS, T) float32 tensor.

    Uses PCEN (Per-Channel Energy Normalization) which normalises for
    recording-device frequency response variability, giving +0.049 soundscape
    val AUC over AmplitudeToDB + min-max norm.  Output is normalised to [0, 1]
    and repeated across 3 channels for ImageNet-pretrained backbones.
    """
    wav_t = torch.from_numpy(waveform).float().unsqueeze(0)   # (1, T)
    mel   = _mel_transform(wav_t)                              # (1, N_MELS, T)

    # PCEN normalization (Per-Channel Energy Normalization)
    out = pcen(mel)                                            # (1, N_MELS, T)

    # Per-sample normalisation to [0, 1]
    out = out - out.min()
    peak = out.max()
    if peak > 0:
        out = out / peak

    return out.repeat(3, 1, 1)   # (3, N_MELS, T)
