"""Sweep rank-mean fusion of an a5_xarch fold-0 + A2 5-fold ensemble.

Baseline = A2 5-fold sig-mean ensemble (0.8402 broader-pool, the v75 production
anchor). Sweep arch weight 0.05 -> 0.95 in steps of 0.05; report best fusion
AUC, delta vs anchor, and gate check (anchor + 0.05 = 0.8902 per
feedback_min_oof_delta_to_burn_slot).

The §29 AST x A2 result (rank-mean w=0.40 = 0.8630, +0.0227 vs anchor) is the
reference data point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

FT_ROOT = Path(__file__).resolve().parents[1]

A2_OOF_PATH = FT_ROOT / "data" / "a2_a1_5fold_broader_oof.npz"
A2_ANCHOR_AUC = 0.8402
GATE_AUC = 0.8902  # anchor + 0.05 per feedback_min_oof_delta_to_burn_slot


def rank01_per_col(mat: np.ndarray) -> np.ndarray:
    """Per-column rank normalized to [0,1]. Matches probe_b_weight_sweep_v2."""
    n = mat.shape[0]
    order = np.argsort(mat, axis=0, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    rows = np.arange(n, dtype=np.float32)
    for c in range(mat.shape[1]):
        ranks[order[:, c], c] = rows
    if n > 1:
        ranks /= (n - 1)
    return ranks


def macro_auc_present(probs: np.ndarray, y_true: np.ndarray) -> float:
    present = y_true.sum(axis=0) > 0
    return float(
        roc_auc_score(y_true[:, present], probs[:, present], average="macro")
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--arch-oof",
        type=Path,
        required=True,
        help="Path to a5_<short>_fold0_broader_oof.npz from eval_a5_broader_oof.py",
    )
    p.add_argument(
        "--also-sigmean",
        action="store_true",
        help="Also report sigmoid-mean fusion (§29 showed this regresses cross-arch).",
    )
    args = p.parse_args()

    print(f"[arch-oof] {args.arch_oof}", flush=True)
    print(f"[a2-oof]   {A2_OOF_PATH}", flush=True)
    arch = np.load(args.arch_oof, allow_pickle=True)
    a2 = np.load(A2_OOF_PATH, allow_pickle=True)

    arch_probs = arch["probs"]        # (1478, 234)
    arch_y = arch["y_true"]           # (1478, 234)
    a2_probs_mean = a2["probs_mean"]  # (1478, 234) — sig-mean of 5 folds
    a2_y = a2["y_true"]               # (1478, 234)
    arch_short = (
        str(arch["backbone"]) if "backbone" in arch.files else "<unknown>"
    ).split(".")[0]

    assert arch_probs.shape == a2_probs_mean.shape, (
        f"shape mismatch: arch {arch_probs.shape} vs a2 {a2_probs_mean.shape}"
    )
    if not np.allclose(arch_y, a2_y):
        n_diff = int(np.abs(arch_y - a2_y).sum())
        print(
            f"  [WARN] y_true differs by {n_diff} entries (should be 0)",
            flush=True,
        )

    # Re-verify baselines.
    a2_auc = macro_auc_present(a2_probs_mean, a2_y)
    arch_auc = macro_auc_present(arch_probs, arch_y)
    print("", flush=True)
    print(f"  A2 5-fold sig-mean (anchor): {a2_auc:.4f} "
          f"(expected ~{A2_ANCHOR_AUC:.4f})", flush=True)
    print(f"  {arch_short} fold-0 (standalone): {arch_auc:.4f}", flush=True)
    if abs(a2_auc - A2_ANCHOR_AUC) > 0.001:
        print(
            f"  [WARN] A2 anchor recompute differs by "
            f"{abs(a2_auc - A2_ANCHOR_AUC):.4f}",
            flush=True,
        )

    # Rank-normalize.
    a2_rank = rank01_per_col(a2_probs_mean)
    arch_rank = rank01_per_col(arch_probs)

    # Sweep arch weights 0.05 .. 0.95 step 0.05.
    weights = np.round(np.arange(0.05, 1.00, 0.05), 2)
    rows = []
    for w in weights:
        fused = w * arch_rank + (1.0 - w) * a2_rank
        auc = macro_auc_present(fused, a2_y)
        rows.append((float(w), auc))

    print("", flush=True)
    print(f"  {'arch_w':>6}  {'fusion_auc':>11}  {'Δ vs anchor':>14}", flush=True)
    for w, auc in rows:
        marker = ""
        if auc >= GATE_AUC:
            marker = "  ★ GATE PASS"
        print(
            f"  {w:>6.2f}  {auc:>11.4f}  {auc - a2_auc:>+14.4f}{marker}",
            flush=True,
        )

    best_idx = int(np.argmax([auc for _, auc in rows]))
    best_w, best_auc = rows[best_idx]
    delta = best_auc - a2_auc

    print("", flush=True)
    print("=" * 60, flush=True)
    print(f"  best rank-mean fusion: w={best_w:.2f}  "
          f"AUC={best_auc:.4f}  Δ={delta:+.4f}", flush=True)
    print(f"  gate (anchor + 0.05):  {GATE_AUC:.4f}", flush=True)
    if best_auc >= GATE_AUC:
        verdict = (
            f"GATE PASS  ({best_auc:.4f} >= {GATE_AUC:.4f})  → "
            f"justified v77 LB push at w={best_w:.2f}"
        )
    else:
        gap = GATE_AUC - best_auc
        verdict = (
            f"GATE FAIL  ({best_auc:.4f} < {GATE_AUC:.4f}, gap {gap:.4f})  → "
            f"don't burn LB slot per feedback_min_oof_delta_to_burn_slot"
        )
    print(f"  verdict: {verdict}", flush=True)
    print("=" * 60, flush=True)

    if args.also_sigmean:
        print("", flush=True)
        print("Sigmoid-mean fusion (for completeness — §29 shows this regresses):",
              flush=True)
        print(f"  {'arch_w':>6}  {'fusion_auc':>11}  {'Δ vs anchor':>14}", flush=True)
        for w in [0.05, 0.20, 0.40, 0.50]:
            fused = w * arch_probs + (1.0 - w) * a2_probs_mean
            auc = macro_auc_present(fused, a2_y)
            print(
                f"  {w:>6.2f}  {auc:>11.4f}  {auc - a2_auc:>+14.4f}",
                flush=True,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
