"""A3-1: Generate A2 5-fold pseudo-labels on the 10592 unlabeled train_soundscapes.

A2-as-teacher recursive self-training. A2 ensemble broader-pool OOF is
0.8402 (vs A1's 0.7775), so its pseudo-labels should be cleaner than A1's.
Identical pipeline to a2_pseudo_label_a1.py except CKPT_DIR points at the
A2-trained JIT ckpts.

Mirrors cell 41 of the production protossm-postproc kernel:
- mel-spec + PCEN preprocessing (matches src/utils.py:waveform_to_mel)
- 5s windows × 12 per file, tile to 20s for the JIT model
- Forward through 5 A2 JIT folds, sigmoid-mean across folds

Output: data/processed/a2_pseudo_soundscape.npz with:
  filenames    (n_windows,)  str  e.g. 'BC2026_Train_0067_S04_*.ogg'
  start_sec    (n_windows,)  int  in {0,5,10,...,55}
  probs        (n_windows, 234)  float32 sigmoid-mean across 5 folds

Usage:
  python -u src/a3_pseudo_label_a2.py [--device cuda|cpu] [--limit N]
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.transforms as tT

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARENT = ROOT.parent

CKPT_DIR = ROOT / 'kaggle_datasets' / 'a1-effb0-a2-ckpts'
TRAIN_SS_DIR = PARENT / 'data' / 'raw' / 'train_soundscapes'
LABELS_CSV = PARENT / 'data' / 'raw' / 'train_soundscapes_labels.csv'
OUT_DIR = ROOT / 'data' / 'processed'

# A1 preprocessing constants — must match production cell 41
SR = 32_000
A1_DURATION = 20  # JIT was trained on 20s; tile 5s up to 20s
A1_N_MELS = 224
A1_N_FFT = 4096
A1_HOP = 1252  # 512 frames for 20s at 32kHz
A1_F_MIN = 0
A1_F_MAX = 16_000
A1_FOLDS = [0, 1, 2, 3, 4]
N_CLASSES = 234
N_WINDOWS = 12
WINDOW_SAMPLES = SR * 5
FILE_SAMPLES = SR * 60

# PCEN constants (verbatim from BirdCLEF/src/utils.py)
_PCEN_GAIN, _PCEN_SMOOTH, _PCEN_BIAS, _PCEN_POWER, _PCEN_EPS = 0.98, 0.025, 2.0, 0.5, 1e-6


def _build_mel_transform():
    return tT.MelSpectrogram(
        sample_rate=SR, n_fft=A1_N_FFT, hop_length=A1_HOP,
        n_mels=A1_N_MELS, f_min=A1_F_MIN, f_max=A1_F_MAX,
        power=2.0, norm='slaney', mel_scale='slaney', center=True,
    )


def _pcen(mel):
    with torch.no_grad():
        E = mel.float()
        T_ = E.shape[2]
        M = E[:, :, 0].clone()
        out = torch.empty_like(E)
        bias_r = _PCEN_BIAS ** _PCEN_POWER
        for t in range(T_):
            M = (1.0 - _PCEN_SMOOTH) * M + _PCEN_SMOOTH * E[:, :, t]
            denom = (M + _PCEN_EPS).pow(_PCEN_GAIN)
            out[:, :, t] = (E[:, :, t] / denom + _PCEN_BIAS).pow(_PCEN_POWER) - bias_r
    return out


def _waveform_to_mel(wav_1d, mel_transform):
    """5-second slice → (3, N_MELS, T') tensor matching A1 training."""
    chunk_len = SR * A1_DURATION
    reps = -(-chunk_len // len(wav_1d))
    wav = np.tile(wav_1d, reps)[:chunk_len]
    wav_t = torch.from_numpy(wav).float().unsqueeze(0)
    mel = mel_transform(wav_t)
    out = _pcen(mel)
    out = out - out.min()
    peak = out.max()
    if peak > 0:
        out = out / peak
    return out.repeat(3, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    ap.add_argument('--limit', type=int, default=None,
                    help='Process only the first N unlabeled files (for smoke test)')
    ap.add_argument('--out', default=str(OUT_DIR / 'a2_pseudo_soundscape.npz'))
    args = ap.parse_args()

    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if args.device == 'auto' else torch.device(args.device))
    print(f'Device: {device}', flush=True)
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)

    # Discover unlabeled files
    labels_df = pd.read_csv(LABELS_CSV)
    labeled = set(labels_df['filename'].unique())
    all_files = sorted(p.name for p in TRAIN_SS_DIR.glob('*.ogg'))
    unlabeled = sorted(f for f in all_files if f not in labeled)
    print(f'Total: {len(all_files)},  labeled: {len(labeled)},  unlabeled: {len(unlabeled)}',
          flush=True)
    if args.limit:
        unlabeled = unlabeled[:args.limit]
        print(f'Limited to first {len(unlabeled)} files for smoke test', flush=True)

    # Load A1 JIT models
    print('Loading 5 A1 JIT models …', flush=True)
    models = []
    for f in A1_FOLDS:
        p = CKPT_DIR / f'a1_fold{f}.pt'
        m = torch.jit.load(str(p), map_location=device).eval()
        models.append(m)
        print(f'  fold {f}: {p.stat().st_size / 1e6:.1f} MB', flush=True)

    mel_transform = _build_mel_transform()

    # Pre-allocate output
    n_files = len(unlabeled)
    n_rows = n_files * N_WINDOWS
    probs = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
    filenames_out = np.empty(n_rows, dtype=object)
    start_sec_out = np.zeros(n_rows, dtype=np.int32)

    print(f'Processing {n_files} files × {N_WINDOWS} windows × {len(models)} folds …',
          flush=True)
    t_start = time.time()
    row = 0
    for fi, fname in enumerate(unlabeled):
        try:
            y, sr = sf.read(str(TRAIN_SS_DIR / fname), dtype='float32', always_2d=False)
            if y.ndim == 2:
                y = y.mean(axis=1)
            if sr != SR:
                print(f'  [warn] {fname}: sr={sr}, expected {SR} — skipping', flush=True)
                row += N_WINDOWS
                continue
            if len(y) < FILE_SAMPLES:
                y = np.pad(y, (0, FILE_SAMPLES - len(y)))
            elif len(y) > FILE_SAMPLES:
                y = y[:FILE_SAMPLES]

            # Build 12 mels for this file
            file_mels = []
            for w in range(N_WINDOWS):
                seg = y[w * WINDOW_SAMPLES:(w + 1) * WINDOW_SAMPLES]
                file_mels.append(_waveform_to_mel(seg, mel_transform))
            batch = torch.stack(file_mels).to(device)  # (12, 3, 224, 512)

            # All 5 folds → sigmoid-mean (Probe A finding §25.13)
            sig_sum = torch.zeros((N_WINDOWS, N_CLASSES), device=device, dtype=torch.float32)
            with torch.no_grad():
                for m in models:
                    logits = m(batch)
                    sig_sum += torch.sigmoid(logits).float()
            sig_mean = (sig_sum / len(models)).cpu().numpy()  # (12, 234)

            for w in range(N_WINDOWS):
                probs[row + w] = sig_mean[w]
                filenames_out[row + w] = fname
                start_sec_out[row + w] = w * 5
            row += N_WINDOWS

            if (fi + 1) % 50 == 0 or fi == 0:
                el = time.time() - t_start
                eta = el / (fi + 1) * (n_files - fi - 1)
                print(f'  [{fi + 1}/{n_files}] {fname[:50]:<50} '
                      f'cum={el:.1f}s ({el / (fi + 1):.2f}s/file)  ETA {eta / 60:.1f}m',
                      flush=True)
        except Exception as e:
            print(f'  [err] {fname}: {e}', flush=True)
            row += N_WINDOWS  # skip this file's slots, leave zeros

    el = time.time() - t_start
    print(f'\nDone. {n_files} files in {el:.1f}s ({el / n_files:.2f}s/file)', flush=True)

    # Trim if any err rows
    valid = filenames_out != None  # noqa: E711
    print(f'Valid rows: {valid.sum()}/{n_rows}', flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        filenames=filenames_out[valid].astype('U64'),
        start_sec=start_sec_out[valid],
        probs=probs[valid],
    )
    print(f'Saved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)', flush=True)


if __name__ == '__main__':
    main()
