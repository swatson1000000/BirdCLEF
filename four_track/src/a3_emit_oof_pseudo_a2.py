"""src/a3_emit_oof_pseudo_a2.py — Track A3 step 1: OOF pseudo-label emission with A2 teacher.

A3-recursive: use A2 5-fold ensemble (broader-pool OOF 0.8402, vs A1's 0.7775)
as the teacher for a second self-training round. Same OOF safety rule as
a2_emit_oof_pseudo.py — each soundscape file is assigned a bucket k via a
stable filename hash; predictions for files in bucket k are averaged across
the 4 A2 ckpts trained on folds != k, avoiding the leakage that would occur
if A3 fold k trained on pseudo-labels predicted by A2 fold k on the same file.

Source ckpts: models/a1/a1_..._fold{F}_seed42_asl.pt — the §28 A2 5-fold raw
state_dicts (named with the a1_ prefix because a2_train.py used the default
save_dir; md5-verified equal to DT copies). Broader-pool OOF (sig-mean of
all 5): 0.8402.

Output: data/processed/a3_train_ss_oof_probs.npz with keys
  probs       : (N_chunks, 234) float32  — OOF mean of 4 fold-!=-k probs
  filenames   : (N_chunks,)     <U       — source soundscape file
  start_sec   : (N_chunks,)     int32    — window start in seconds
  oof_bucket  : (N_chunks,)     int8     — bucket assignment 0..4
  fold_set    : (5,)            int32    — which folds were available
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

HERE       = Path(__file__).resolve().parent
FT_ROOT    = HERE.parent
ROOT       = FT_ROOT.parent
PARENT_SRC = ROOT / "src"

if str(PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(PARENT_SRC))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config  # noqa: E402
from config import RAW, SAMPLE_RATE, CHUNK_SAMPLES  # noqa: E402
from utils import load_audio, pad_or_crop, waveform_to_mel  # noqa: E402
from model_a1 import BirdSEDModelA1  # noqa: E402

CKPT_DIR         = FT_ROOT / "models" / "a1"
CKPT_TEMPLATE    = "a1_tf_efficientnet_b0.ns_jft_in1k_fold{f}_seed42_asl.pt"
SS_DIR           = RAW / "train_soundscapes"
OUT_PATH         = FT_ROOT / "data" / "processed" / "a3_train_ss_oof_probs.npz"

FOLDS            = [0, 1, 2, 3, 4]
WINDOW_SEC       = 5
WINDOWS_PER_FILE = 12       # 60s file / 5s window
NUM_WORKERS      = 8
BATCH_FILES      = 4        # 4 files * 12 windows = 48 mels per fold pass


def file_bucket(filename: str, n_buckets: int = 5) -> int:
    """Stable filename -> bucket id via md5."""
    h = hashlib.md5(filename.encode("utf-8")).digest()
    return h[0] % n_buckets


class SSEmitDataset(Dataset):
    """One item = one full 60s soundscape file decoded into 12 x 5s mels.

    Each 5s segment is pad/cropped to CHUNK_SAMPLES (20s) — matches the
    p12_emit_oof.py convention so the trained A1 ckpts see the same
    20s context window they were trained/validated on.
    """

    def __init__(self, filenames: list[str]):
        self.filenames = filenames

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fname = self.filenames[idx]
        path  = SS_DIR / fname
        try:
            wav = load_audio(path)
        except Exception as ex:
            print(f"  [warn] failed to load {fname}: {ex}", flush=True)
            wav = np.zeros(SAMPLE_RATE * 60, dtype=np.float32)

        mels = []
        for w in range(WINDOWS_PER_FILE):
            s = w * WINDOW_SEC * SAMPLE_RATE
            e = s + WINDOW_SEC * SAMPLE_RATE
            seg = wav[s:e] if e <= len(wav) else wav[s:]
            seg = pad_or_crop(seg, CHUNK_SAMPLES, random_crop=False)
            mels.append(waveform_to_mel(seg))
        mels = torch.stack(mels)        # (12, 3, N_MELS, T)
        return mels, fname, file_bucket(fname)


def collate(items):
    mels      = torch.stack([it[0] for it in items])                    # (B, 12, 3, N_MELS, T)
    filenames = [it[1] for it in items]
    buckets   = torch.tensor([it[2] for it in items], dtype=torch.int8)
    return mels, filenames, buckets


def load_models(device: torch.device) -> dict[int, torch.nn.Module]:
    models = {}
    for f in FOLDS:
        path = CKPT_DIR / CKPT_TEMPLATE.format(f=f)
        assert path.exists(), f"missing ckpt: {path}"
        m = BirdSEDModelA1()
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = m.load_state_dict(state, strict=False)
        if missing:
            print(f"  fold {f}: {len(missing)} missing keys (first 3): "
                  f"{sorted(missing)[:3]}", flush=True)
        if unexpected:
            print(f"  fold {f}: {len(unexpected)} unexpected keys (first 3): "
                  f"{sorted(unexpected)[:3]}", flush=True)
        m = m.to(device).eval()
        models[f] = m
    return models


@torch.no_grad()
def main() -> None:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  num_workers={NUM_WORKERS}  batch_files={BATCH_FILES}",
          flush=True)

    filenames = sorted(f for f in os.listdir(SS_DIR) if f.endswith(".ogg"))
    n_files = len(filenames)
    n_chunks = n_files * WINDOWS_PER_FILE
    print(f"files: {n_files}  total windows: {n_chunks}", flush=True)

    bucket_counts = np.zeros(5, dtype=np.int64)
    for fn in filenames:
        bucket_counts[file_bucket(fn)] += 1
    print(f"bucket distribution: {bucket_counts.tolist()}", flush=True)

    print("loading 5 fold ckpts...", flush=True)
    models = load_models(device)
    print(f"  loaded folds {sorted(models.keys())}", flush=True)

    ds = SSEmitDataset(filenames)
    # multiprocessing_context='spawn' prevents the CUDA-fork deadlock: the
    # parent has CUDA initialized (we just loaded 5 models on GPU above), and
    # default fork-mode workers inherit a half-initialized CUDA context that
    # then deadlocks on internal mutexes when they try to do anything torch.
    # Spawn workers start clean and stay CPU-only.
    dl = DataLoader(ds, batch_size=BATCH_FILES, shuffle=False,
                    num_workers=NUM_WORKERS, collate_fn=collate,
                    persistent_workers=True, pin_memory=True,
                    multiprocessing_context="spawn")

    out_probs   = np.zeros((n_chunks, config.N_CLASSES), dtype=np.float32)
    out_files   = np.empty(n_chunks, dtype=object)
    out_starts  = np.zeros(n_chunks, dtype=np.int32)
    out_buckets = np.zeros(n_chunks, dtype=np.int8)

    cursor   = 0
    last_log = time.time()
    log_every = 30.0

    for batch_idx, (mels, fnames, buckets) in enumerate(dl):
        B = mels.shape[0]
        flat = mels.view(B * WINDOWS_PER_FILE, *mels.shape[2:]).to(device, non_blocking=True)

        per_fold_probs = {}
        for f, m in models.items():
            out = m(flat)
            logits = out["clip_logits"]
            per_fold_probs[f] = torch.sigmoid(logits).float().cpu().numpy()

        for i in range(B):
            k = int(buckets[i].item())
            keep = [f for f in FOLDS if f != k]
            stacked = np.stack(
                [per_fold_probs[f][i*WINDOWS_PER_FILE:(i+1)*WINDOWS_PER_FILE]
                 for f in keep], axis=0
            )                                            # (4, 12, 234)
            oof = stacked.mean(axis=0)                   # (12, 234)

            out_probs  [cursor:cursor+WINDOWS_PER_FILE] = oof
            out_files  [cursor:cursor+WINDOWS_PER_FILE] = fnames[i]
            out_starts [cursor:cursor+WINDOWS_PER_FILE] = (
                np.arange(WINDOWS_PER_FILE, dtype=np.int32) * WINDOW_SEC
            )
            out_buckets[cursor:cursor+WINDOWS_PER_FILE] = k
            cursor += WINDOWS_PER_FILE

        if time.time() - last_log > log_every:
            done = (batch_idx + 1) * BATCH_FILES
            rate = done / (time.time() - t0)
            eta  = (n_files - done) / max(rate, 0.01)
            print(f"  [{done}/{n_files}] {rate:.1f} files/s  ETA {eta/60:.1f} min",
                  flush=True)
            last_log = time.time()

    assert cursor == n_chunks, f"cursor mismatch: {cursor} vs {n_chunks}"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        probs=out_probs,
        filenames=out_files.astype(str),
        start_sec=out_starts,
        oof_bucket=out_buckets,
        fold_set=np.array(FOLDS, dtype=np.int32),
    )
    sz = OUT_PATH.stat().st_size / 1e6
    elapsed = time.time() - t0
    print(f"\nsaved {OUT_PATH} ({sz:.1f} MB) in {elapsed:.1f}s "
          f"({elapsed/60:.1f} min) total", flush=True)


if __name__ == "__main__":
    main()
