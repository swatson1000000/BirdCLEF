"""Compute A2 ⊕ L3-precursor 2-recipe bag under sig-mean and rank-mean.

Reports:
  - Standalone A2 (5-fold sig-mean)
  - Standalone L3-precursor (5-fold sig-mean)
  - Bag under sig-mean (10 ckpts averaged in probability space)
  - Bag under rank-mean (10 ckpts averaged in rank space, per-class)
  - Δ vs A2 anchor and Δ vs L3-prec standalone (the *real* diversity test)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

FT_ROOT = Path(__file__).resolve().parents[1]
A2_PATH = FT_ROOT / "data" / "a2_a1_5fold_broader_oof.npz"
L3_PATH = FT_ROOT / "data" / "l3_precursor_broader_oof.npz"

A2_ANCHOR = 0.8402  # plan §663 gate anchor


def macro_auc_present(probs: np.ndarray, y_true: np.ndarray) -> float:
    present = y_true.sum(axis=0) > 0
    return float(
        roc_auc_score(y_true[:, present], probs[:, present], average="macro")
    )


def rank_normalize(probs: np.ndarray) -> np.ndarray:
    """Per-class rank-normalize across the window axis. Returns values in [0, 1)."""
    n_seg = probs.shape[0]
    ranks = np.zeros_like(probs, dtype=np.float32)
    for c in range(probs.shape[1]):
        ranks[:, c] = rankdata(probs[:, c], method="average") / n_seg
    return ranks


def main() -> int:
    a2 = np.load(A2_PATH, allow_pickle=True)
    l3 = np.load(L3_PATH, allow_pickle=True)

    a2_ppf = a2["probs_per_fold"]  # (5, 1478, 234)
    l3_ppf = l3["probs_per_fold"]
    y_true = a2["y_true"]
    assert np.array_equal(y_true, l3["y_true"]), "label mismatch — bug"

    n_a2, n_seg, n_cls = a2_ppf.shape
    n_l3 = l3_ppf.shape[0]
    assert a2_ppf.shape == l3_ppf.shape, "shape mismatch"

    a2_sig = a2_ppf.mean(axis=0)
    l3_sig = l3_ppf.mean(axis=0)
    a2_auc = macro_auc_present(a2_sig, y_true)
    l3_auc = macro_auc_present(l3_sig, y_true)

    bag_ppf = np.concatenate([a2_ppf, l3_ppf], axis=0)
    bag_sig = bag_ppf.mean(axis=0)
    bag_sig_auc = macro_auc_present(bag_sig, y_true)

    a2_ranks = np.stack([rank_normalize(a2_ppf[i]) for i in range(n_a2)], axis=0)
    l3_ranks = np.stack([rank_normalize(l3_ppf[i]) for i in range(n_l3)], axis=0)
    bag_ranks = np.concatenate([a2_ranks, l3_ranks], axis=0).mean(axis=0)
    bag_rank_auc = macro_auc_present(bag_ranks, y_true)

    a2_rank_mean = a2_ranks.mean(axis=0)
    l3_rank_mean = l3_ranks.mean(axis=0)
    rank_mean_a2_only = macro_auc_present(a2_rank_mean, y_true)
    rank_mean_l3_only = macro_auc_present(l3_rank_mean, y_true)

    stronger_standalone = max(a2_auc, l3_auc)

    print("=" * 70, flush=True)
    print("A2 ⊕ L3-precursor 2-recipe bag — broader-pool OOF (1478 segs, 75 cls)", flush=True)
    print("=" * 70, flush=True)
    print(f"  A2 standalone (sig-mean):   {a2_auc:.4f}", flush=True)
    print(f"  L3-prec standalone (sig-mean): {l3_auc:.4f}", flush=True)
    print(f"  A2 standalone (rank-mean):  {rank_mean_a2_only:.4f}", flush=True)
    print(f"  L3-prec standalone (rank-mean): {rank_mean_l3_only:.4f}", flush=True)
    print("", flush=True)
    print(f"  bag sig-mean (10 ckpts):    {bag_sig_auc:.4f}", flush=True)
    print(f"  bag rank-mean (10 ckpts):   {bag_rank_auc:.4f}", flush=True)
    print("", flush=True)
    print(f"  plan gate anchor (A2):      {A2_ANCHOR:.4f}", flush=True)
    print(f"  Δ bag-sig vs A2 anchor:     {bag_sig_auc - A2_ANCHOR:+.4f}", flush=True)
    print(f"  Δ bag-rank vs A2 anchor:    {bag_rank_auc - A2_ANCHOR:+.4f}", flush=True)
    print("", flush=True)
    print(f"  stronger standalone:        {stronger_standalone:.4f}", flush=True)
    print(f"  Δ bag-sig vs stronger:      {bag_sig_auc - stronger_standalone:+.4f}", flush=True)
    print(f"  Δ bag-rank vs stronger:     {bag_rank_auc - stronger_standalone:+.4f}", flush=True)
    print("=" * 70, flush=True)
    print("  NOTE: 'Δ vs A2 anchor' is the plan §663 gate;", flush=True)
    print("        'Δ vs stronger standalone' is the actual recipe-diversity signal.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
