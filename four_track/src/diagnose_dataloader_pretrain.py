"""Reproduce the worker-pool hang from the 2026-05-10 iNat re-pretrain v2.

The pretrain script's --smoke-test flag forces num_workers=0, so it never
exercises the actual worker code path that hung tonight. This script does
exactly that: builds the dataset, wraps it in a DataLoader with N workers,
and times the first few batches. If MixUp causes the hang, we'll see it
within a few minutes instead of waiting 2+ hours.

Three modes:
  default (balanced sampler, no MixUp) — should match prior working pretrain
  --natural-sampling                   — isolate this flag
  --mixup-prob 0.5                     — isolate this flag
  (both flags)                         — reproduce the hang

Logs per-batch wall time. If a batch takes > 30s, prints a warning.

CLI mirrors pretrain_inat_sounds.py for the relevant args.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

FT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FT_ROOT.parent / "src"))
sys.path.insert(0, str(FT_ROOT / "src"))

import config  # noqa: E402
from pretrain_inat_sounds import (  # noqa: E402
    INatSoundsDataset, make_balanced_sampler,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inat-root", type=Path, required=True,
                   help="Directory containing inat_manifest.csv (matches "
                        "the pretrain --inat-root semantics — adds abs_path "
                        "from file_path + inat_root)")
    p.add_argument("--n-clips", type=int, default=5000,
                   help="Truncate manifest to first N clips for fast iteration")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--n-batches", type=int, default=20,
                   help="How many batches to time before stopping")
    p.add_argument("--natural-sampling", action="store_true")
    p.add_argument("--mixup-prob", type=float, default=0.0)
    p.add_argument("--persistent-workers", action="store_true", default=True)
    p.add_argument("--no-persistent-workers", action="store_false",
                   dest="persistent_workers")
    p.add_argument("--mp-context", type=str, default="spawn",
                   choices=["spawn", "fork", "forkserver"])
    args = p.parse_args()

    print(f"[config] batch_size={args.batch_size}  num_workers={args.num_workers}  "
          f"n_batches={args.n_batches}", flush=True)
    print(f"[config] natural_sampling={args.natural_sampling}  "
          f"mixup_prob={args.mixup_prob}", flush=True)
    print(f"[config] persistent_workers={args.persistent_workers}  "
          f"mp_context={args.mp_context}", flush=True)

    # ── Build dataset ────────────────────────────────────────────────────────
    manifest_path = args.inat_root / "inat_manifest.csv"
    print(f"[load] {manifest_path}", flush=True)
    import pandas as pd
    df = pd.read_csv(manifest_path, dtype=str)
    df["abs_path"] = df["file_path"].apply(lambda p: str(args.inat_root / p))
    print(f"[load] full manifest: {len(df)} rows", flush=True)
    if args.n_clips < len(df):
        df = df.head(args.n_clips).reset_index(drop=True)
        print(f"[load] truncated to {len(df)} rows", flush=True)

    # Species index
    species = sorted(df["scientific_name"].astype(str).unique().tolist())
    sp2idx = {s: i for i, s in enumerate(species)}
    print(f"[load] {len(sp2idx)} unique species in this slice", flush=True)

    ds = INatSoundsDataset(df, sp2idx, augment=True, mixup_prob=args.mixup_prob)

    # ── Build sampler + DataLoader ───────────────────────────────────────────
    if args.natural_sampling:
        sampler = None
        shuffle = True
        sampler_desc = "natural shuffle"
    else:
        sampler = make_balanced_sampler(df, sp2idx)
        shuffle = False
        sampler_desc = "WeightedRandomSampler (1/n)"
    print(f"[loader] sampler: {sampler_desc}", flush=True)

    dl = DataLoader(
        ds, batch_size=args.batch_size,
        sampler=sampler, shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.persistent_workers,
        multiprocessing_context=args.mp_context,
    )
    print(f"[loader] DataLoader built: ~{len(dl)} batches/epoch on this slice",
          flush=True)

    # ── Time batches ─────────────────────────────────────────────────────────
    print(f"[iter] timing first {args.n_batches} batches …", flush=True)
    t_start = time.time()
    t_prev = t_start
    slow_threshold_s = 30.0

    for i, (mels, labels, mask) in enumerate(dl):
        t_now = time.time()
        dt = t_now - t_prev
        total = t_now - t_start
        flag = " ⚠️ SLOW" if dt > slow_threshold_s else ""
        print(f"  batch {i:3d}: shape={tuple(mels.shape)}  dt={dt:.2f}s  "
              f"total={total:.1f}s{flag}", flush=True)
        t_prev = t_now
        if i + 1 >= args.n_batches:
            break

    t_end = time.time()
    print(f"\n[done] {args.n_batches} batches in {t_end - t_start:.1f}s  "
          f"({(t_end - t_start)/args.n_batches:.2f}s/batch avg)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
