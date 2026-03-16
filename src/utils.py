"""Audio loading and mel-spectrogram utilities for BirdCLEF+ 2026."""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T

from config import (
    SAMPLE_RATE, CHUNK_SAMPLES,
    N_MELS, N_FFT, HOP_LENGTH, F_MIN, F_MAX, TOP_DB,
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
_amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=TOP_DB)


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

    The mel-dB spectrogram is normalised per-sample to [0, 1] and repeated
    across 3 channels to match ImageNet-pretrained backbone expectations.
    """
    wav_t  = torch.from_numpy(waveform).float().unsqueeze(0)  # (1, T)
    mel    = _mel_transform(wav_t)                             # (1, N_MELS, T)
    mel_db = _amplitude_to_db(mel)                             # (1, N_MELS, T)

    # Per-sample normalisation to [0, 1]
    mel_db = mel_db - mel_db.min()
    peak = mel_db.max()
    if peak > 0:
        mel_db = mel_db / peak

    return mel_db.repeat(3, 1, 1)   # (3, N_MELS, T)
