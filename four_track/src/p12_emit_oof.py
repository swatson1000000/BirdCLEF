"""Emit per-fold soundscape val predictions for the v56 4-fold A1 ensemble.

Loads the four restored JIT ckpts in `kaggle_datasets/a1-effb0-ckpts/`,
runs them over every labeled window in `train_soundscapes_labels.csv`,
and saves a single NPZ that downstream P12 isotonic calibration can fit on.

The JIT modules return `clip_logits` directly (see export_a1_jit.A1Wrapper);
we sigmoid those to get class-wise probabilities.

Output: `data/v56_soundscape_oof.npz` with keys
  probs_per_fold : (4, N_windows, 234) float32  — per-fold sigmoid probs
  probs_mean     : (N_windows, 234)    float32  — mean across the 4 folds
  y_true         : (N_windows, 234)    float32  — multi-hot soundscape labels
  fold_ids       : (4,)                int64    — folds covered (0,1,2,4)
  filenames      : (N_windows,)        <U                 — source soundscape file
  start_sec      : (N_windows,)        int32              — window start (s)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# GB10 (compute capability 12.1) exceeds PyTorch's NVRTC max (12.0); the
# JIT-traced module otherwise tries to compile fused kernels at first call
# and dies with "invalid value for --gpu-architecture". Disable every fusion
# path so the traced graph runs in eager mode instead.
torch._C._jit_set_nvfuser_enabled(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(False)

HERE      = Path(__file__).resolve().parent
FT_ROOT   = HERE.parent
ROOT      = FT_ROOT.parent
PARENT_SRC = ROOT / "src"

if str(PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(PARENT_SRC))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config  # noqa: E402
from config import RAW, SAMPLE_RATE, CHUNK_SAMPLES, N_MELS, get_species_index  # noqa: E402
from utils import load_audio, pad_or_crop, waveform_to_mel  # noqa: E402

JIT_DIR  = FT_ROOT / "kaggle_datasets" / "a1-effb0-ckpts"
OUT_PATH = FT_ROOT / "data" / "v56_soundscape_oof.npz"
KEEP_FOLDS = [0, 1, 2, 4]
BATCH = 32


def _parse_time(s: str) -> int:
    h, m, sec = str(s).split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec)


def build_val(sp2idx: dict) -> tuple:
    """Replica of train_a1.build_soundscape_val that also returns
    filenames/start_sec so calibration can be grouped/audited later.
    """
    df = pd.read_csv(RAW / "train_soundscapes_labels.csv")
    n_classes   = config.N_CLASSES
    soundsc_dir = RAW / "train_soundscapes"

    val_mels   = []
    val_labels = np.zeros((len(df), n_classes), dtype=np.float32)
    filenames  = []
    starts     = np.zeros(len(df), dtype=np.int32)

    for i, row in df.iterrows():
        t_start = _parse_time(row["start"])
        t_end   = _parse_time(row["end"])
        path    = soundsc_dir / str(row["filename"])
        try:
            wav = load_audio(path)
            s, e = int(t_start * SAMPLE_RATE), int(t_end * SAMPLE_RATE)
            segment = wav[s:e] if e <= len(wav) else wav[s:]
            segment = pad_or_crop(segment, CHUNK_SAMPLES, random_crop=False)
            mel = waveform_to_mel(segment)
        except Exception as ex:
            print(f"  [warn] skipping {row['filename']} @ {t_start}s: {ex}", flush=True)
            mel = torch.zeros(3, N_MELS, 512)
        val_mels.append(mel)
        filenames.append(str(row["filename"]))
        starts[i] = t_start

        for sp in str(row["primary_label"]).split(";"):
            sp = sp.strip()
            if sp in sp2idx:
                val_labels[i, sp2idx[sp]] = 1.0

    return val_mels, val_labels, np.array(filenames), starts


@torch.no_grad()
def score_fold(jit_path: Path, val_mels: list, device: torch.device) -> np.ndarray:
    model = torch.jit.load(str(jit_path), map_location=device)
    model.eval()
    probs = []
    for i in range(0, len(val_mels), BATCH):
        batch = torch.stack(val_mels[i: i + BATCH]).to(device)
        logits = model(batch)  # FP32; autocast removed to avoid fused-kernel NVRTC compile
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(probs, axis=0).astype(np.float32)


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    sp2idx = get_species_index()
    print(f"sp2idx size = {len(sp2idx)} (expect {config.N_CLASSES})", flush=True)
    assert len(sp2idx) == config.N_CLASSES

    print("building soundscape val mels...", flush=True)
    val_mels, y_true, filenames, starts = build_val(sp2idx)
    n_w = len(val_mels)
    print(f"  N_windows = {n_w}", flush=True)
    print(f"  N_pos per class: min={int(y_true.sum(0).min())} "
          f"median={int(np.median(y_true.sum(0)))} max={int(y_true.sum(0).max())}",
          flush=True)
    n_zero = int((y_true.sum(0) == 0).sum())
    print(f"  classes with ZERO positives in val: {n_zero}/{config.N_CLASSES}",
          flush=True)

    probs_per_fold = np.zeros((len(KEEP_FOLDS), n_w, config.N_CLASSES), dtype=np.float32)
    for j, f in enumerate(KEEP_FOLDS):
        jit_path = JIT_DIR / f"a1_fold{f}.pt"
        assert jit_path.exists(), f"missing JIT ckpt: {jit_path}"
        t1 = time.time()
        probs_per_fold[j] = score_fold(jit_path, val_mels, device)
        present = y_true.sum(0) > 0
        try:
            auc = float(roc_auc_score(
                y_true[:, present], probs_per_fold[j][:, present], average="macro"
            ))
        except ValueError:
            auc = float("nan")
        dt = time.time() - t1
        print(f"  fold {f}: macro AUC = {auc:.4f}  ({dt:.1f}s)", flush=True)

    probs_mean = probs_per_fold.mean(axis=0)
    present = y_true.sum(0) > 0
    auc_mean = float(roc_auc_score(
        y_true[:, present], probs_mean[:, present], average="macro"
    ))
    print(f"  ENSEMBLE mean macro AUC = {auc_mean:.4f} "
          f"(reference v56 baseline ~0.7414)", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        probs_per_fold=probs_per_fold,
        probs_mean=probs_mean,
        y_true=y_true,
        fold_ids=np.array(KEEP_FOLDS, dtype=np.int64),
        filenames=filenames,
        start_sec=starts,
    )
    sz = OUT_PATH.stat().st_size / 1e6
    print(f"\nsaved {OUT_PATH} ({sz:.1f} MB) in {time.time()-t0:.1f}s total",
          flush=True)


if __name__ == "__main__":
    main()
