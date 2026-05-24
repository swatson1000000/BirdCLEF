"""D1-b — per-fold temperature scaling on existing A1 5-fold OOF.

Per `new_plan.md` line 744: A1's stored 5-fold soft-vote (0.7017) is *worse*
than 4 of its 5 individual folds (0.7414, 0.7232, 0.6970, 0.7250, 0.6636).
The averaging across folds with different per-fold calibration destroys
signal — gap is +0.04 AUC of recoverable lift if we re-scale each fold's
logits with a learned scalar before averaging.

This script fits one scalar temperature T_f per fold to minimize binary
cross-entropy on the OOF (held-out) val set, then re-computes the soft-vote
AUC under the calibrated logits.

Inputs:
  data/v56_soundscape_oof.npz
    probs_per_fold : (F=4, N=1478, C=234)
    probs_mean     : (N, C)             — uncalibrated soft-vote
    y_true         : (N, C)             — binary multi-hot

Outputs (stdout):
  Per-fold uncalibrated AUC + fitted T_f
  Calibrated per-fold AUC
  Uncalibrated vs calibrated soft-vote AUC
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

FT_ROOT = Path(__file__).resolve().parents[1]
NPZ_PATH = FT_ROOT / "data" / "v56_soundscape_oof.npz"

EPS = 1e-7


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _bce(probs: np.ndarray, targets: np.ndarray) -> float:
    """Mean per-element binary cross-entropy."""
    p = np.clip(probs, EPS, 1.0 - EPS)
    return float(-(targets * np.log(p) + (1.0 - targets) * np.log(1.0 - p)).mean())


def _macro_auc_present(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Macro ROC-AUC over classes that have at least one positive label.

    Matches the val_v2 macro-AUC convention used throughout the project.
    """
    present = y_true.sum(axis=0) > 0
    return float(
        roc_auc_score(y_true[:, present], probs[:, present], average="macro")
    )


def _fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Fit a single scalar T > 0 minimizing BCE of sigmoid(logits/T)."""

    def obj(T: float) -> float:
        return _bce(_sigmoid(logits / T), y_true)

    # Bounded search over a wide but sane range.
    res = minimize_scalar(obj, bounds=(0.1, 20.0), method="bounded",
                          options={"xatol": 1e-4})
    return float(res.x), float(res.fun)


def main() -> None:
    if not NPZ_PATH.exists():
        sys.exit(f"missing OOF file: {NPZ_PATH}")

    d = np.load(NPZ_PATH, allow_pickle=True)
    probs_per_fold = d["probs_per_fold"]  # (F, N, C)
    probs_mean = d["probs_mean"]          # (N, C)
    y_true = d["y_true"]                  # (N, C)
    fold_ids = d["fold_ids"]              # (F,)

    F, N, C = probs_per_fold.shape
    print(f"Loaded OOF: F={F}  N={N}  C={C}  fold_ids={list(fold_ids)}",
          flush=True)

    # ── 1. Uncalibrated per-fold + soft-vote AUC ─────────────────────────────
    print("\n[uncalibrated]", flush=True)
    per_fold_aucs_uncal = []
    for i, f in enumerate(fold_ids):
        auc = _macro_auc_present(probs_per_fold[i], y_true)
        per_fold_aucs_uncal.append(auc)
        print(f"  fold {f}: AUC = {auc:.4f}", flush=True)
    softvote_uncal_auc = _macro_auc_present(probs_mean, y_true)
    print(f"  soft-vote (mean of probs): AUC = {softvote_uncal_auc:.4f}",
          flush=True)

    # ── 2. Fit T_f per fold on the OOF ───────────────────────────────────────
    print("\n[fit temperatures]", flush=True)
    Ts = []
    probs_calibrated = np.empty_like(probs_per_fold)
    for i, f in enumerate(fold_ids):
        logits = _logit(probs_per_fold[i])
        T_f, bce_at_T = _fit_temperature(logits, y_true)
        bce_at_1 = _bce(_sigmoid(logits), y_true)
        probs_calibrated[i] = _sigmoid(logits / T_f)
        Ts.append(T_f)
        print(f"  fold {f}: T = {T_f:.4f}   "
              f"BCE: {bce_at_1:.4f} → {bce_at_T:.4f}",
              flush=True)

    # ── 3. Calibrated per-fold + soft-vote AUC ───────────────────────────────
    print("\n[calibrated]", flush=True)
    per_fold_aucs_cal = []
    for i, f in enumerate(fold_ids):
        auc = _macro_auc_present(probs_calibrated[i], y_true)
        per_fold_aucs_cal.append(auc)
        delta = auc - per_fold_aucs_uncal[i]
        print(f"  fold {f}: AUC = {auc:.4f}   (Δ {delta:+.4f})", flush=True)

    softvote_cal_probs = probs_calibrated.mean(axis=0)
    softvote_cal_auc = _macro_auc_present(softvote_cal_probs, y_true)
    softvote_delta = softvote_cal_auc - softvote_uncal_auc
    print(f"  soft-vote (calibrated): AUC = {softvote_cal_auc:.4f}   "
          f"(Δ {softvote_delta:+.4f})", flush=True)

    # ── 4. Summary ───────────────────────────────────────────────────────────
    print("\n[summary]", flush=True)
    print(f"  per-fold AUC mean (uncal): {np.mean(per_fold_aucs_uncal):.4f}",
          flush=True)
    print(f"  per-fold AUC mean (cal):   {np.mean(per_fold_aucs_cal):.4f}",
          flush=True)
    print(f"  soft-vote AUC uncal:       {softvote_uncal_auc:.4f}", flush=True)
    print(f"  soft-vote AUC cal:         {softvote_cal_auc:.4f}", flush=True)
    print(f"  recovered gap:             {softvote_delta:+.4f}", flush=True)
    print(f"  fitted Ts:                 {[f'{t:.3f}' for t in Ts]}",
          flush=True)


if __name__ == "__main__":
    main()
