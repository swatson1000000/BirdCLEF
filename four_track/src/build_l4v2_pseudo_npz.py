"""Build L4-v2 pseudo NPZ: rank-mean fusion of L3-prec sigmean and ProtoSSM probs.

Rank-mean is the only fusion mode that gained on the teacher smoke gate
(+0.0122 at w=0.60 vs L3-prec subset 0.8709). Sig-mean regressed.

Per-class ranks are computed across the 127104-window unlabeled pool (10592
train_soundscapes files × 12 windows). The fused-rank value at percentile
τ_rank is then used as the threshold for downstream manifest building.

Threshold calibration (positives/window matching):
  L3-prec @0.7 produced ~2.44 positives/window (per plan §36.1 sidebar).
  We search for τ_rank on the rank-fused outputs that yields ~2.44 too.

Output: data/processed/l4v2_pseudo_soundscape.npz with the same schema as
the L3-prec NPZ:
  filenames  (n_windows,) <U
  start_sec  (n_windows,) int32
  probs      (n_windows, 234) float32  — fused rank values in [0,1]

Plus a calibration report in data/processed/l4v2_pseudo_calibration.json
with τ_rank and positives-per-window stats.

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle-arch
    python -u src/build_l4v2_pseudo_npz.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent

L3_NPZ = PROJECT / "data" / "processed" / "l3prec_pseudo_soundscape.npz"
PROTO_NPZ = PROJECT / "data" / "processed" / "protossm_pseudo_soundscape.npz"
OUT_NPZ = PROJECT / "data" / "processed" / "l4v2_pseudo_soundscape.npz"
OUT_REPORT = PROJECT / "data" / "processed" / "l4v2_pseudo_calibration.json"

# Best fusion config from teacher smoke gate (eval_l3prec_x_protossm_broader_oof.py)
RANK_MEAN_W_L3 = 0.60   # 60% L3-prec, 40% ProtoSSM
# Target positives/window — L3-prec @0.7 produced 2.44 (per plan §36.1)
TARGET_POS_PER_WINDOW = 2.44


def _per_class_ranks(probs: np.ndarray) -> np.ndarray:
    """Per-class rank-normalize to (0, 1). Faster than rankdata in a loop."""
    n, c = probs.shape
    out = np.empty_like(probs, dtype=np.float32)
    for ci in range(c):
        # argsort positions → ranks. Average ties via argsort-twice trick approx;
        # but rankdata("average") is cleanest. Use scipy for correctness.
        from scipy.stats import rankdata
        out[:, ci] = (rankdata(probs[:, ci], method="average") - 0.5) / n
    return out


def main() -> int:
    print(f"[load] {L3_NPZ}", flush=True)
    l3 = np.load(L3_NPZ, allow_pickle=True)
    l3_probs = l3["probs"].astype(np.float32)            # (127104, 234)
    l3_filenames = np.array([str(f) for f in l3["filenames"]])
    l3_start = l3["start_sec"].astype(np.int32)
    print(f"  probs={l3_probs.shape}  files={len(set(l3_filenames))}", flush=True)

    print(f"[load] {PROTO_NPZ}", flush=True)
    pr = np.load(PROTO_NPZ, allow_pickle=True)
    pr_probs = pr["probs"].astype(np.float32)            # (127896, 234)
    pr_filenames = np.array([str(f) for f in pr["filenames"]])
    pr_start = pr["start_sec"].astype(np.int32)
    print(f"  probs={pr_probs.shape}  files={len(set(pr_filenames))}", flush=True)

    # Index ProtoSSM rows by (filename, start_sec) for quick lookup
    pr_key = np.array([f"{f}|{s}" for f, s in zip(pr_filenames, pr_start)])
    pr_index = {k: i for i, k in enumerate(pr_key)}

    # Subset ProtoSSM to L3-prec's row order. L3 covers 10592 unlabeled files;
    # ProtoSSM covers 10658 (incl. 66 labeled).
    print("[align] mapping L3-prec rows → ProtoSSM rows…", flush=True)
    pr_idx = np.empty(len(l3_filenames), dtype=np.int64)
    missing = 0
    for i, (f, s) in enumerate(zip(l3_filenames, l3_start)):
        key = f"{f}|{s}"
        if key not in pr_index:
            missing += 1
            pr_idx[i] = -1
        else:
            pr_idx[i] = pr_index[key]
    if missing:
        print(f"  [err] {missing} L3-prec rows not found in ProtoSSM output", file=sys.stderr)
        return 1
    pr_aligned = pr_probs[pr_idx]
    print(f"  aligned: pr_aligned={pr_aligned.shape}", flush=True)

    # Compute per-class ranks for both
    print("[rank] per-class rank-normalize…", flush=True)
    l3_rank = _per_class_ranks(l3_probs)
    pr_rank = _per_class_ranks(pr_aligned)
    print(f"  l3_rank range: [{l3_rank.min():.4f}, {l3_rank.max():.4f}]", flush=True)
    print(f"  pr_rank range: [{pr_rank.min():.4f}, {pr_rank.max():.4f}]", flush=True)

    # Fuse
    fused = RANK_MEAN_W_L3 * l3_rank + (1.0 - RANK_MEAN_W_L3) * pr_rank
    fused = fused.astype(np.float32)
    print(f"[fuse] rank-mean w_L3={RANK_MEAN_W_L3}  fused={fused.shape}  "
          f"range=[{fused.min():.4f}, {fused.max():.4f}]", flush=True)

    # Calibrate threshold to target positives/window
    n_windows = fused.shape[0]
    print(f"[calibrate] searching τ for positives/window ≈ {TARGET_POS_PER_WINDOW}", flush=True)
    print(f"  {'τ':>8}  {'pos/window':>12}  {'#positive':>11}", flush=True)
    # Bisection over τ in [0.50, 1.00]
    lo, hi = 0.50, 1.00
    for _ in range(20):
        mid = (lo + hi) / 2
        pos_per_win = float((fused > mid).sum() / n_windows)
        if pos_per_win > TARGET_POS_PER_WINDOW:
            lo = mid
        else:
            hi = mid
    tau = (lo + hi) / 2
    achieved = float((fused > tau).sum() / n_windows)
    print(f"  τ = {tau:.4f}  pos/window = {achieved:.3f}", flush=True)

    # Show distribution at chosen tau
    n_pos = (fused > tau).sum(axis=1)  # (n_windows,)
    print(f"  pos/window stats: mean={n_pos.mean():.2f}  median={np.median(n_pos):.0f}  "
          f"p95={np.percentile(n_pos, 95):.0f}  max={n_pos.max()}", flush=True)
    # Per-class positives — show top/bottom classes by positive count
    n_pos_per_class = (fused > tau).sum(axis=0)
    sorted_cls = np.argsort(-n_pos_per_class)
    print(f"  most-positive classes (idx, count): "
          f"{[(int(c), int(n_pos_per_class[c])) for c in sorted_cls[:5]]}", flush=True)
    print(f"  least-positive classes (idx, count): "
          f"{[(int(c), int(n_pos_per_class[c])) for c in sorted_cls[-5:]]}", flush=True)

    # Cross-check: how many positives come from ProtoSSM-trained 71 classes vs 163 untrained
    cache_meta = pd.read_parquet(PROJECT / "data" / "kaggle_perch_cache" / "full_perch_meta.parquet")
    # We need to know which class indices are "ProtoSSM-trained" (saw a positive in training).
    # The training cache labels come from train_soundscapes_labels.csv;
    # let's recompute by loading the training labels.
    sys.path.insert(0, str(PROJECT / "src"))
    sys.path.insert(0, str(PROJECT.parent / "src"))
    from config import RAW, get_species_index
    sp2idx = get_species_index()
    labels_csv = pd.read_csv(RAW / "train_soundscapes_labels.csv")
    labeled_classes = set()
    for lab_str in labels_csv["primary_label"].dropna():
        for lbl in str(lab_str).split(";"):
            lbl = lbl.strip()
            if lbl in sp2idx:
                labeled_classes.add(sp2idx[lbl])
    trained_mask = np.zeros(234, dtype=bool)
    for c in labeled_classes:
        trained_mask[c] = True
    print(f"\n  ProtoSSM-trained class count: {int(trained_mask.sum())} of 234", flush=True)
    pos_trained = (fused[:, trained_mask] > tau).sum()
    pos_untrained = (fused[:, ~trained_mask] > tau).sum()
    pos_total = pos_trained + pos_untrained
    print(f"  positives from trained-71 classes:  {pos_trained}  "
          f"({100*pos_trained/pos_total:.1f}%)", flush=True)
    print(f"  positives from untrained-163 classes: {pos_untrained}  "
          f"({100*pos_untrained/pos_total:.1f}%)", flush=True)
    # Expected baseline if uniform: 163/234 = 69.7%
    expected_untrained_pct = 100 * (~trained_mask).sum() / 234
    print(f"  uniform-baseline expected untrained%: {expected_untrained_pct:.1f}%", flush=True)

    # Save
    print(f"\n[save] {OUT_NPZ}", flush=True)
    np.savez_compressed(
        OUT_NPZ,
        probs=fused,
        filenames=l3_filenames.astype("U64"),
        start_sec=l3_start,
    )
    print(f"  {OUT_NPZ.stat().st_size/1e6:.1f} MB", flush=True)

    report = {
        "fusion_mode": "rank-mean",
        "w_l3": float(RANK_MEAN_W_L3),
        "w_proto": float(1.0 - RANK_MEAN_W_L3),
        "target_pos_per_window": float(TARGET_POS_PER_WINDOW),
        "achieved_pos_per_window": float(achieved),
        "tau": float(tau),
        "n_windows": int(n_windows),
        "n_total_positives": int(pos_total),
        "n_positives_trained71": int(pos_trained),
        "n_positives_untrained163": int(pos_untrained),
        "pct_positives_untrained": float(100 * pos_untrained / max(1, pos_total)),
        "pct_uniform_baseline_untrained": float(expected_untrained_pct),
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2))
    print(f"[save] {OUT_REPORT}", flush=True)

    print("\nNext step: build CSV manifest with this threshold")
    print(f"  python -u src/a2_build_pseudo_manifest.py "
          f"--probs {OUT_NPZ} --threshold {tau:.4f} --max-classes 5 --min-classes 1 "
          f"--out data/processed/l4v2_pseudo_manifest.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
