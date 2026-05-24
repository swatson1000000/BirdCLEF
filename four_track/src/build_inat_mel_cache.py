"""Pre-compute fp16 mel spectrograms for the iNat train manifest.

Eliminates the per-iteration mp3-decode + resample + mel-compute work that
dominates wall-clock during pretrain. On HDD-backed raw audio, training runs
at ~110 min/epoch. With cached mels on NVMe, training is GPU-bound at
~3-5 min/epoch.

Cache layout: one fp16 `.npy` per clip, named by manifest row index. The
INatSoundsDataset class learns the cache directory via `mel_cache_dir`; if
the file exists for that row, it skips the audio path and loads the mel
directly.

Args:
  --inat-root: directory containing inat_manifest.csv + train/ + val/
  --out-dir: where to write cached mels. Defaults to
             {FT_ROOT}/data/processed/inat_mels/
  --split: 'train' (default), 'val', or 'all'
  --num-workers: parallel workers (default 8)
  --batch-log-every: progress print every N samples (default 1000)

Per-sample cost: ~30-100 ms wall (decode + resample + mel). At 8 workers
parallel, ~70-200 samples/sec → 137K train clips in ~12-30 min.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

FT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FT_ROOT.parent / "src"))
sys.path.insert(0, str(FT_ROOT / "src"))

from config import CHUNK_SAMPLES  # noqa: E402
from utils import load_audio, pad_or_crop, waveform_to_mel  # noqa: E402


def _compute_and_save_one(args: tuple) -> tuple:
    """Worker function: read audio, compute mel, save fp16 .npy.

    Returns (idx, ok, error_msg_or_None).
    """
    idx, abs_path, out_path = args
    try:
        if out_path.exists():
            return (idx, True, "exists")
        wav = load_audio(Path(abs_path))
        if wav.size == 0:
            return (idx, False, "zero-length wav")
        wav = pad_or_crop(wav, CHUNK_SAMPLES, random_crop=False)
        mel = waveform_to_mel(wav)
        # mel is a torch.Tensor (3, N_MELS, T)
        if not isinstance(mel, torch.Tensor):
            mel = torch.as_tensor(mel)
        mel_fp16 = mel.to(dtype=torch.float16).contiguous().numpy()
        # write to temp then rename for atomicity. np.save auto-appends ".npy"
        # to string paths but NOT to file objects — so pass a file handle to
        # avoid suffix surprise (the previous version produced .npy.tmp.npy
        # files because of this).
        tmp_path = out_path.with_name(out_path.name + ".tmp")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "wb") as fh:
            np.save(fh, mel_fp16, allow_pickle=False)
        tmp_path.rename(out_path)
        return (idx, True, None)
    except Exception as exc:
        return (idx, False, f"{type(exc).__name__}: {exc}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inat-root", type=Path, required=True,
                   help="Dir containing inat_manifest.csv + train/ + val/")
    p.add_argument("--out-dir", type=Path,
                   default=FT_ROOT / "data" / "processed" / "inat_mels",
                   help="Cache directory (default: four_track/data/processed/inat_mels/)")
    p.add_argument("--split", choices=["train", "val", "all"], default="train")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--batch-log-every", type=int, default=1000)
    args = p.parse_args()

    manifest_path = args.inat_root / "inat_manifest.csv"
    print(f"[load] {manifest_path}", flush=True)
    df = pd.read_csv(manifest_path, dtype=str)
    df["abs_path"] = df["file_path"].apply(lambda p: str(args.inat_root / p))

    if args.split == "train":
        df = df[df["split"] == "train"].reset_index(drop=True)
    elif args.split == "val":
        df = df[df["split"] == "val"].reset_index(drop=True)
    print(f"[load] {len(df)} {args.split} rows", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out] {args.out_dir}", flush=True)

    # Build job list
    jobs = []
    for i, row in df.iterrows():
        out_path = args.out_dir / f"{args.split}_{i:07d}.npy"
        jobs.append((i, row["abs_path"], out_path))

    n_jobs = len(jobs)
    t_start = time.time()
    n_ok = 0
    n_skip_exists = 0
    n_bad = 0
    bad_examples = []

    print(f"[run] {n_jobs} samples, {args.num_workers} workers", flush=True)

    # Use spawn context, NOT fork. PyTorch + concurrent.futures fork deadlocks
    # because torch was imported in main and inherits internal threads/locks.
    # Confirmed empirically 2026-05-10 — first try with default fork left all
    # 8 workers at 0% CPU forever.
    spawn_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.num_workers,
                              mp_context=spawn_ctx) as ex:
        futures = {ex.submit(_compute_and_save_one, j): j[0] for j in jobs}
        done_count = 0
        for fut in as_completed(futures):
            idx, ok, err = fut.result()
            done_count += 1
            if ok:
                if err == "exists":
                    n_skip_exists += 1
                else:
                    n_ok += 1
            else:
                n_bad += 1
                if len(bad_examples) < 30:
                    bad_examples.append((idx, err))

            if done_count % args.batch_log_every == 0:
                elapsed = time.time() - t_start
                rate = done_count / elapsed
                eta_s = (n_jobs - done_count) / rate if rate > 0 else 0
                print(f"  [{done_count}/{n_jobs}] "
                      f"ok={n_ok} skip_exists={n_skip_exists} bad={n_bad}  "
                      f"rate={rate:.1f}/s  eta={eta_s/60:.1f}m",
                      flush=True)

    elapsed = time.time() - t_start
    print(f"\n[done] {n_jobs} samples in {elapsed/60:.1f} min", flush=True)
    print(f"  ok          : {n_ok}", flush=True)
    print(f"  skip_exists : {n_skip_exists}", flush=True)
    print(f"  bad         : {n_bad}", flush=True)
    if bad_examples:
        print(f"  first {len(bad_examples)} bad rows:", flush=True)
        for idx, err in bad_examples[:30]:
            print(f"    idx={idx}: {err}", flush=True)

    # Write a tiny manifest of cached indices for INatSoundsDataset to scan
    cache_manifest = args.out_dir / f"{args.split}_cache_manifest.txt"
    with open(cache_manifest, "w") as f:
        for j in jobs:
            i, _, p = j
            if p.exists():
                f.write(f"{i}\t{p.name}\n")
    print(f"[manifest] {cache_manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
