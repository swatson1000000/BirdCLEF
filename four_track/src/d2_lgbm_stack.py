"""D2-β Phase 2: local stacker over {A1_rank, B1, ProtoSSM} OOF signals.

Reads `four_track/data/d2_beta_oofs.npz` (produced by D2-β Phase 1 — see
`four_track/src/d2_beta_oof_cell.py` and `new_plan.md` §D2-β) and fits a
GroupKFold stacker on the 708-row × 234-class OOF substrate.

**Design rationale:**

The production rank-fusion submit path (`a1_notebook_cell.py` + B1 rank cell)
combines A1, B1, and ProtoSSM by rank-averaging per class with frozen weights
{A1: 0.20, B1: 0.10, Proto: 0.70}. D2-β asks: can a learned per-class blender
beat that frozen-weight baseline on held-out data?

Two key design constraints from the kernel dump:

1. Only 8 unique site-level groups (S03/S08/S13/S15/S18/S19/S22/S23), not 59
   per-file groups. GroupKFold maxes out at 8 splits. We use 5-fold to keep
   each fold's validation site coverage representative.

2. b1_oof and proto_oof are logits in the dump; a1_ranks is already in [0,1]
   rank space. We rank-transform b1 and proto per class to match a1's scale.
   The Kaggle submit path does the same (per-fold rank then average), so the
   stacker's feature space mirrors what the notebook will produce at inference.

**Baselines to beat:**

  BP: `prod_fused` — the actual LB-production rank-fusion output reconstructed
      on the 708-row OOF substrate by the D2-β Phase 1.5 cell. This IS what
      the LB 0.933 notebook produces on this substrate and is the only
      trustworthy local reference (see new_plan.md §D2-β Phase 1.5 and the
      project_b1_weight_sweep memory). **Primary gate baseline.**
  B0: A1-only (a1_ranks alone) — secondary diagnostic, not the gate.
  B1: A1+Proto equal rank-mean — diagnostic only.
  B2: A1+B1+Proto equal rank-mean — diagnostic only.
  B3: Frozen weights {0.20, 0.10, 0.70} in rank space without inverse-CDF —
      diagnostic only (subtle difference from BP: BP preserves ProtoSSM
      marginals per class, B3 doesn't).

Stackers:

  S1: Per-class logistic regression on [a1_rank, b1_rank, proto_rank].
  S2: Single global LightGBM classifier with class_id as categorical.

**Gate (from new_plan.md):** median Δ ≥ +0.001 macro ROC-AUC over 5 seeds
AND sign-stable across all 5 seeds (no seed regression) vs the best baseline.

Usage (from four_track/):
    python -u src/d2_lgbm_stack.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

HERE = Path(__file__).resolve().parent
FT_ROOT = HERE.parent
NPZ_PATH = FT_ROOT / "data" / "d2_beta_oofs.npz"


# --------------------------- utilities --------------------------------------

def rank01_per_col(mat: np.ndarray) -> np.ndarray:
    """Per-column rank transform to [0,1]."""
    n = mat.shape[0]
    out = np.empty_like(mat, dtype=np.float32)
    order = np.argsort(mat, axis=0, kind="mergesort")
    rows = np.arange(n, dtype=np.float32)
    for c in range(mat.shape[1]):
        out[order[:, c], c] = rows
    if n > 1:
        out /= (n - 1)
    return out


def macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Macro ROC-AUC on present classes only (skip degenerate columns)."""
    present = y.sum(axis=0) > 0
    # Also skip classes where y is all 1 — undefined AUC.
    nondeg = present & (y.sum(axis=0) < y.shape[0])
    if nondeg.sum() == 0:
        return float("nan")
    return float(roc_auc_score(y[:, nondeg], p[:, nondeg], average="macro"))


def groupkfold_indices(groups: np.ndarray, n_splits: int, seed: int):
    """Deterministic GroupKFold with seed-controlled group ordering.

    sklearn's GroupKFold is NOT seed-aware — so we permute the unique groups
    by seed and then assign groups to folds round-robin. Gives us genuine
    seed diversity instead of 5 identical runs.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    perm = rng.permutation(unique)
    group_to_fold = {g: i % n_splits for i, g in enumerate(perm)}
    fold_of_row = np.array([group_to_fold[g] for g in groups], dtype=np.int64)
    for k in range(n_splits):
        val_idx = np.where(fold_of_row == k)[0]
        tr_idx = np.where(fold_of_row != k)[0]
        if len(val_idx) == 0 or len(tr_idx) == 0:
            continue
        yield tr_idx, val_idx


# --------------------------- baselines --------------------------------------

def baseline_B0(a1r, b1r, pr):  # A1 only
    return a1r

def baseline_B1(a1r, b1r, pr):  # A1+Proto equal
    return (a1r + pr) / 2.0

def baseline_B2(a1r, b1r, pr):  # A1+B1+Proto equal
    return (a1r + b1r + pr) / 3.0

def baseline_B3(a1r, b1r, pr):  # frozen production weights
    return 0.20 * a1r + 0.10 * b1r + 0.70 * pr


BASELINES = {
    "B0 A1-only":        baseline_B0,
    "B1 A1+Proto":       baseline_B1,
    "B2 A1+B1+Proto":    baseline_B2,
    "B3 frozen-weights": baseline_B3,
}


# --------------------------- stackers ---------------------------------------

def stacker_S1_per_class_logreg(a1r, b1r, pr, y, fold_iter):
    """Per-class logistic on [a1, b1, proto].

    For each GroupKFold split, for each class with ≥2 positives in train,
    fit a logistic and predict on val. Classes with insufficient positives
    fall back to the A1-only prediction.

    Returns out-of-fold predictions of shape (n_rows, n_classes) — zeros
    where a row wasn't in the val set for any fold (shouldn't happen if
    GroupKFold covers all rows).
    """
    n_rows, n_classes = y.shape
    oof = np.full_like(a1r, fill_value=np.nan, dtype=np.float32)

    for tr_idx, val_idx in fold_iter:
        X_tr = np.stack([a1r[tr_idx], b1r[tr_idx], pr[tr_idx]], axis=-1)    # (n_tr, K, 3)
        X_va = np.stack([a1r[val_idx], b1r[val_idx], pr[val_idx]], axis=-1) # (n_va, K, 3)
        y_tr = y[tr_idx]

        for c in range(n_classes):
            y_c = y_tr[:, c]
            n_pos = int(y_c.sum())
            n_neg = int((1 - y_c).sum())
            if n_pos < 2 or n_neg < 2:
                # Fallback: A1 only
                oof[val_idx, c] = a1r[val_idx, c]
                continue
            X_c_tr = X_tr[:, c, :]
            X_c_va = X_va[:, c, :]
            try:
                model = LogisticRegression(
                    C=1.0, solver="lbfgs", max_iter=500, class_weight="balanced"
                )
                model.fit(X_c_tr, y_c)
                p = model.predict_proba(X_c_va)[:, 1]
                oof[val_idx, c] = p.astype(np.float32)
            except Exception:
                oof[val_idx, c] = a1r[val_idx, c]

    # Fill any remaining NaNs with A1-only as a safety net.
    nan_mask = np.isnan(oof)
    if nan_mask.any():
        oof[nan_mask] = a1r[nan_mask]
    return oof


# --------------------------- main sweep -------------------------------------

def run_seed(a1r, b1r, pr, y, groups, prod_fused, seed: int, n_splits: int = 5):
    """Run one seed: all baselines + S1 stacker. Returns dict of AUCs."""
    # Baselines: they don't depend on folds (they are input-only transforms),
    # but we evaluate macro AUC on the whole row space for consistency.
    result = {}
    result["BP prod-fused"] = macro_auc(y, prod_fused)   # primary gate baseline
    for name, fn in BASELINES.items():
        pred = fn(a1r, b1r, pr)
        result[name] = macro_auc(y, pred)

    # Stacker: re-generate GroupKFold splits each time (consumed by generator).
    def _fold_iter():
        return groupkfold_indices(groups, n_splits=n_splits, seed=seed)

    t0 = time.time()
    s1_oof = stacker_S1_per_class_logreg(a1r, b1r, pr, y, _fold_iter())
    result["S1 per-class logreg"] = macro_auc(y, s1_oof)
    result["_S1_wall_s"] = time.time() - t0
    return result


def fit_final_s1(a1r, b1r, pr, y):
    """Fit per-class logistic on ALL 708 OOF rows (no holdout).

    Returns:
      coefs        — (n_classes, 3) float32, weights for [a1_rank, b1_rank, proto_rank]
      intercepts   — (n_classes,)   float32
      fallback     — (n_classes,)   bool, True if class falls back to A1-only at inference
                     (insufficient positives or training failure)
      n_pos        — (n_classes,)   int32, train-time positive count per class

    For fallback classes, coefs/intercepts are filled with [1,0,0] / 0 so that
    the apply step produces a1_rank as the score (i.e. the logistic is the
    identity over a1_rank, which preserves rankings).
    """
    n_rows, n_classes = y.shape
    coefs      = np.zeros((n_classes, 3), dtype=np.float32)
    intercepts = np.zeros((n_classes,),    dtype=np.float32)
    fallback   = np.ones((n_classes,),     dtype=bool)
    n_pos_arr  = y.sum(axis=0).astype(np.int32)

    fitted = 0
    for c in range(n_classes):
        n_pos = int(n_pos_arr[c])
        n_neg = n_rows - n_pos
        if n_pos < 2 or n_neg < 2:
            # Fallback: produce A1-only score (linear identity over a1_rank)
            coefs[c, :] = [1.0, 0.0, 0.0]
            intercepts[c] = 0.0
            fallback[c] = True
            continue
        X = np.stack([a1r[:, c], b1r[:, c], pr[:, c]], axis=-1)
        try:
            model = LogisticRegression(
                C=1.0, solver="lbfgs", max_iter=500, class_weight="balanced"
            )
            model.fit(X, y[:, c])
            coefs[c, :]   = model.coef_[0].astype(np.float32)
            intercepts[c] = float(model.intercept_[0])
            fallback[c]   = False
            fitted += 1
        except Exception:
            coefs[c, :]   = [1.0, 0.0, 0.0]
            intercepts[c] = 0.0
            fallback[c]   = True

    return coefs, intercepts, fallback, n_pos_arr, fitted


def apply_s1(a1r, b1r, pr, coefs, intercepts):
    """Apply per-class S1 logistic at inference time.

    Inputs:
      a1r, b1r, pr — (n_rows, n_classes) float32 in [0,1] rank space.
      coefs        — (n_classes, 3) float32
      intercepts   — (n_classes,)   float32

    Returns:
      logits       — (n_rows, n_classes) float32. Sigmoid is monotonic in
                     logits so for ROC AUC / rank fusion we can pass either.
                     We return logits (no sigmoid) — easier to feed into the
                     downstream inverse-CDF rank-fusion preserve step.
    """
    # logits[i, c] = a*a1r[i,c] + b*b1r[i,c] + c*pr[i,c] + intercept[c]
    # Vectorize across (rows, classes):
    logits = (
        a1r * coefs[:, 0]    # broadcast (n_classes,) over rows
      + b1r * coefs[:, 1]
      + pr  * coefs[:, 2]
      + intercepts
    ).astype(np.float32)
    return logits


def main():
    if not NPZ_PATH.exists():
        sys.exit(f"Missing OOF dump: {NPZ_PATH}")

    print(f"Loading {NPZ_PATH}")
    z = np.load(NPZ_PATH, allow_pickle=False)
    a1_ranks  = z["a1_ranks"].astype(np.float32)
    b1_oof    = z["b1_oof"].astype(np.float32)
    proto_oof = z["proto_oof"].astype(np.float32)
    y_true    = z["y_true"].astype(np.float32)
    fold_ids  = z["fold_ids"].astype(np.int64)  # (708,) per-row file/site group
    n_windows = int(z["n_windows"])
    if "prod_fused" not in z.files:
        sys.exit("Missing `prod_fused` in the npz. Re-run the D2-β Phase 1.5 "
                 "kernel (d2_beta_oof_cell.py must be the post-Phase-1.5 "
                 "version that dumps prod_fused).")
    prod_fused = z["prod_fused"].astype(np.float32)

    n_rows, n_classes = y_true.shape
    print(f"  shape: {n_rows} rows × {n_classes} classes")
    print(f"  unique groups (sites): {len(np.unique(fold_ids))}")
    print(f"  positives: {int(y_true.sum())}  present classes: {int((y_true.sum(0) > 0).sum())}")

    # --- Rank-transform B1 and ProtoSSM to match A1's rank-space scale.
    print("Rank-transforming b1_oof and proto_oof per column …")
    b1_ranks    = rank01_per_col(b1_oof)
    proto_ranks = rank01_per_col(proto_oof)
    print(f"  a1_ranks     min/mean/max = {a1_ranks.min():.4f}/{a1_ranks.mean():.4f}/{a1_ranks.max():.4f}")
    print(f"  b1_ranks     min/mean/max = {b1_ranks.min():.4f}/{b1_ranks.mean():.4f}/{b1_ranks.max():.4f}")
    print(f"  proto_ranks  min/mean/max = {proto_ranks.min():.4f}/{proto_ranks.mean():.4f}/{proto_ranks.max():.4f}")

    # --- Signal-signal correlations (diagnostic only).
    def mean_pearson(A, B):
        from scipy.stats import pearsonr
        rs = []
        for c in range(n_classes):
            if y_true[:, c].sum() < 2:
                continue
            r, _ = pearsonr(A[:, c], B[:, c])
            if not np.isnan(r):
                rs.append(r)
        return float(np.mean(rs)) if rs else float("nan")
    print("\nPairwise mean Pearson (over present classes):")
    print(f"  A1 vs B1     = {mean_pearson(a1_ranks, b1_ranks):.4f}")
    print(f"  A1 vs Proto  = {mean_pearson(a1_ranks, proto_ranks):.4f}")
    print(f"  B1 vs Proto  = {mean_pearson(b1_ranks, proto_ranks):.4f}")

    # --- Run 5 seeds.
    N_SEEDS = 5
    seeds = list(range(N_SEEDS))
    print(f"\n=== Running {N_SEEDS} seeds × 5-fold GroupKFold ===")
    all_results = []
    for seed in seeds:
        print(f"\n--- seed {seed} ---")
        r = run_seed(a1_ranks, b1_ranks, proto_ranks, y_true, fold_ids,
                     prod_fused=prod_fused, seed=seed)
        for k, v in r.items():
            if k.startswith("_"):
                continue
            print(f"  {k:30s} macro AUC = {v:.4f}")
        print(f"  (S1 wall = {r['_S1_wall_s']:.1f}s)")
        all_results.append(r)

    # --- Aggregate across seeds.
    print("\n=== Summary across seeds (median [min, max]) ===")
    keys = [k for k in all_results[0].keys() if not k.startswith("_")]
    aggregates = {}
    for k in keys:
        vals = np.array([r[k] for r in all_results])
        med = float(np.median(vals))
        lo = float(vals.min())
        hi = float(vals.max())
        aggregates[k] = {"median": med, "min": lo, "max": hi, "all": vals.tolist()}
        print(f"  {k:30s} median={med:.4f}  [{lo:.4f}, {hi:.4f}]")

    # --- Gate decision. PRIMARY baseline is BP prod-fused (the LB-production
    # rank-fusion reconstructed on this 708-row substrate). Diagnostics show
    # the other baselines but only BP counts for the gate.
    print("\n=== Gate decision ===")
    gate_baseline_name = "BP prod-fused"
    gate_baseline_med  = aggregates[gate_baseline_name]["median"]
    s1_med = aggregates["S1 per-class logreg"]["median"]
    delta  = s1_med - gate_baseline_med
    print(f"  gate baseline:    {gate_baseline_name}  (median AUC {gate_baseline_med:.4f})")
    print(f"  S1 stacker:       median AUC {s1_med:.4f}")
    print(f"  Δ vs baseline:    {delta:+.4f}")

    # Per-seed sign stability
    per_seed_delta = np.array(aggregates["S1 per-class logreg"]["all"]) - \
                     np.array(aggregates[gate_baseline_name]["all"])
    sign_stable = bool(np.all(per_seed_delta > 0))
    print(f"  per-seed Δ:       {per_seed_delta.round(4).tolist()}")
    print(f"  sign-stable (all >0)? {sign_stable}")
    gate_pass = (delta >= 0.001) and sign_stable
    print(f"  GATE (Δ≥+0.001 AND sign-stable): {'PASS' if gate_pass else 'FAIL'}")

    if gate_pass:
        print("\n→ S1 logistic clears gate. Escalate to S2 LightGBM? Not yet — "
              "first submit an LB probe with S1 to see if local Δ → LB.")
    else:
        print("\n→ S1 logistic does NOT clear gate. Do NOT proceed to LB.")
        print("  Next considerations:")
        print("  - Try S2 (global LightGBM with class_id categorical) if any "
              "seed showed positive lift.")
        print("  - Or conclude D2-β as a null result and file it in new_plan.md.")

    # Write a summary JSON next to the npz for posterity.
    out_json = FT_ROOT / "data" / "d2_beta_phase2_results.json"
    out_json.write_text(json.dumps({
        "npz_path":       str(NPZ_PATH),
        "n_seeds":        N_SEEDS,
        "aggregates":     aggregates,
        "gate_baseline":  gate_baseline_name,
        "delta_vs_gate":  delta,
        "sign_stable":    sign_stable,
        "gate_pass":      gate_pass,
    }, indent=2))
    print(f"\nWrote {out_json}")

    # ---- Fit final S1 on ALL 708 rows for inference shipping. ----
    print("\n=== Fitting final S1 on all 708 rows (no holdout) ===")
    coefs, intercepts, fallback, n_pos, n_fitted = fit_final_s1(
        a1_ranks, b1_ranks, proto_ranks, y_true
    )
    print(f"  fit per-class logistic on {n_fitted} / {n_classes} classes "
          f"(rest fall back to A1-only)")

    # In-sample sanity check: apply S1 to the training data and compute AUC.
    in_sample_logits = apply_s1(a1_ranks, b1_ranks, proto_ranks, coefs, intercepts)
    in_sample_auc = macro_auc(y_true, in_sample_logits)
    print(f"  in-sample macro AUC = {in_sample_auc:.4f}")
    print(f"  (compare: OOF S1 = {s1_med:.4f}, BP prod-fused = {gate_baseline_med:.4f})")

    coefs_path = FT_ROOT / "data" / "d2_beta_s1_coefs.npz"
    np.savez(
        coefs_path,
        coefs       = coefs,
        intercepts  = intercepts,
        fallback    = fallback,
        n_pos       = n_pos,
        n_classes   = np.int64(n_classes),
        n_train_rows= np.int64(n_rows),
    )
    print(f"Wrote coefficients → {coefs_path}  ({coefs_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
