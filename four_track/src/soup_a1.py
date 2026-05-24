"""Checkpoint soup builder for Track A1 — §14.11.10.8 T2.3.

Averages per-epoch state_dicts saved by `train_a1.py --save-all-epochs` and
re-validates the averaged weights on the soundscape val set. Prints a
side-by-side table of per-epoch val_roc_auc vs the soup's val_roc_auc so the
caller can decide whether to export the soup or the single-BEST epoch.

Usage:
    python -u src/soup_a1.py --fold 0 --loss asl --start-epoch 8 --end-epoch 25

Output:
    models/a1/a1_<backbone>_fold<F>_seed<S>_<loss>_soup.pt
        (raw state_dict — same key schema as a single-epoch ckpt, so
        export_a1_jit.py can pick it up without change via --loss-suffix)

Design:
- Loads each ckpt as {"epoch", "val_roc_auc", "state_dict"} (dict wrapper).
- Averages parameter tensors only — buffers (running_mean/var in BN, etc.)
  are taken from the last ckpt in the range, since averaging BN running
  stats is not meaningful across epochs trained on different data orders.
  EffNet-B0 NS uses BN; the last-epoch buffers best reflect the true
  distribution at that optimizer state.
- Re-validates the soup weights and compares vs max single-epoch val.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# ── Path wiring (mirror train_a1.py) ──────────────────────────────────────────
HERE       = Path(__file__).resolve().parent
FT_ROOT    = HERE.parent
ROOT       = FT_ROOT.parent
PARENT_SRC = ROOT / "src"

if str(PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(PARENT_SRC))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config  # noqa: E402
from config import get_species_index  # noqa: E402
from model_a1 import BirdSEDModelA1  # noqa: E402
from train_a1 import build_soundscape_val, validate  # noqa: E402

A1_MODELS_DIR = FT_ROOT / "models" / "a1"


def _average_state_dicts(ckpts: list[dict]) -> dict:
    """Average parameter tensors; take buffers from the last ckpt."""
    first_sd = ckpts[0]["state_dict"]
    keys = list(first_sd.keys())
    avg: dict[str, torch.Tensor] = {}

    for k in keys:
        ref = first_sd[k]
        # Buffers (non-float, or BN running stats) — take last ckpt's values.
        is_float = ref.is_floating_point()
        is_bn_stat = k.endswith("running_mean") or k.endswith("running_var") \
            or k.endswith("num_batches_tracked")
        if (not is_float) or is_bn_stat:
            avg[k] = ckpts[-1]["state_dict"][k].clone()
            continue
        stacked = torch.stack([c["state_dict"][k].float() for c in ckpts], dim=0)
        avg[k] = stacked.mean(dim=0).to(ref.dtype)

    return avg


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fold",        type=int, default=0)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--loss",        type=str, default="asl",
                   choices=["asl", "hybrid", "bce", "ce"])
    p.add_argument("--backbone",    type=str, default=config.BACKBONE)
    p.add_argument("--start-epoch", type=int, default=8,
                   help="First epoch to include in the soup (inclusive).")
    p.add_argument("--end-epoch",   type=int, default=25,
                   help="Last epoch to include in the soup (inclusive).")
    p.add_argument("--min-val-auc", type=float, default=None,
                   help="Optional: drop epochs with val_roc_auc below this.")
    p.add_argument("--output-suffix", type=str, default="soup",
                   help="Suffix appended to the ckpt filename before .pt.")
    p.add_argument("--dry-run",     action="store_true",
                   help="Print the plan and exit without building/validating.")
    args = p.parse_args()

    soup_dir = A1_MODELS_DIR / "_soup" / f"fold{args.fold}_{args.loss}_seed{args.seed}"
    if not soup_dir.is_dir():
        print(f"[error] soup dir not found: {soup_dir}", flush=True)
        return 1

    all_paths = sorted(soup_dir.glob("epoch*.pt"))
    if not all_paths:
        print(f"[error] no epoch ckpts in {soup_dir}", flush=True)
        return 1

    print(f"[soup] found {len(all_paths)} per-epoch ckpts in {soup_dir}", flush=True)
    print(f"[soup] range: epochs {args.start_epoch}..{args.end_epoch} "
          f"(min_val_auc={args.min_val_auc})", flush=True)

    ckpts: list[dict] = []
    rows: list[tuple[int, float, bool]] = []  # (epoch, val_auc, included)
    for path in all_paths:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ep = int(ck["epoch"])
        vauc = float(ck["val_roc_auc"])
        keep = (args.start_epoch <= ep <= args.end_epoch)
        if keep and args.min_val_auc is not None and vauc < args.min_val_auc:
            keep = False
        if keep:
            ckpts.append(ck)
        rows.append((ep, vauc, keep))

    print("\n  epoch | val_roc_auc | in-soup", flush=True)
    print("  ------+-------------+--------", flush=True)
    for ep, vauc, keep in rows:
        mark = "  YES " if keep else "  no  "
        print(f"   {ep:3d}  |   {vauc:.4f}    | {mark}", flush=True)

    if len(ckpts) < 2:
        print(f"\n[error] need >=2 ckpts to average, got {len(ckpts)}", flush=True)
        return 1

    max_single = max(v for _, v, k in rows if k)
    print(f"\n[soup] averaging {len(ckpts)} ckpts  "
          f"(max single-epoch val among included = {max_single:.4f})", flush=True)

    if args.dry_run:
        print("[soup] --dry-run: stopping before averaging.", flush=True)
        return 0

    avg_sd = _average_state_dicts(ckpts)

    # Build model and load the averaged state_dict.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cudnn.benchmark = True

    model = BirdSEDModelA1(backbone_name=args.backbone, mixstyle_p=0.0).to(device)
    missing, unexpected = model.load_state_dict(avg_sd, strict=True)
    if missing or unexpected:
        print(f"[warn] missing={missing}  unexpected={unexpected}", flush=True)

    # Validate.
    sp2idx = get_species_index()
    print("[soup] building soundscape val set …", flush=True)
    val_mels, val_labels = build_soundscape_val(sp2idx)
    n_present = int((val_labels.sum(axis=0) > 0).sum())
    print(f"[soup]   {len(val_mels)} val segments, {n_present} species present",
          flush=True)

    soup_auc = validate(model, val_mels, val_labels, device)
    print(f"\n[soup] SOUP val_roc_auc = {soup_auc:.4f}", flush=True)
    print(f"[soup] max single-epoch    = {max_single:.4f}", flush=True)
    delta = soup_auc - max_single
    sign = "+" if delta >= 0 else ""
    print(f"[soup] Δ (soup − best)     = {sign}{delta:+.4f}", flush=True)

    # Save.
    save_path = (A1_MODELS_DIR /
                 f"a1_{args.backbone}_fold{args.fold}_seed{args.seed}_"
                 f"{args.loss}_{args.output_suffix}.pt")
    torch.save(avg_sd, save_path)
    print(f"\n[soup] saved → {save_path}", flush=True)

    # Also drop a tiny metadata sidecar so the morning-after reviewer doesn't
    # have to re-read this script to know what went in.
    meta_path = save_path.with_suffix(".meta.npz")
    np.savez(
        meta_path,
        epochs_included=np.array([ep for ep, _, k in rows if k], dtype=np.int32),
        val_aucs_included=np.array([v for _, v, k in rows if k], dtype=np.float32),
        soup_val_auc=np.float32(soup_auc),
        max_single_val_auc=np.float32(max_single),
        delta=np.float32(delta),
    )
    print(f"[soup] sidecar → {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
