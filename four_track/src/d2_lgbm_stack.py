"""D2-β Phase 2 / D2-α: local stackers over {A1_rank, B1, ProtoSSM} OOF signals.

Reads `four_track/data/d2_beta_oofs.npz` (produced by D2-β Phase 1 — see
`four_track/src/d2_beta_oof_cell.py` and `new_plan.md` §D2-β) and fits
GroupKFold stackers on the 708-row × 234-class OOF substrate.

**Design rationale:**

The production rank-fusion submit path (`a1_notebook_cell.py` + B1 rank cell)
combines A1, B1, and ProtoSSM by rank-averaging per class with frozen weights
{A1: 0.20, B1: 0.10, Proto: 0.70}. D2-β/D2-α ask: can a learned blender beat
that frozen-weight baseline on held-out data?

Two key design constraints from the kernel dump:

1. Only 8 unique site-level groups (S03/S08/S13/S15/S18/S19/S22/S23), not 59
   per-file groups. GroupKFold maxes out at 8 splits. We use 5-fold to keep
   each fold's validation site coverage representative.

2. b1_oof and proto_oof are logits in the dump; a1_ranks is already in [0,1]
   rank space. S1 rank-transformed b1 and proto per class to match a1's scale.
   S2 (D2-α) sigmoids b1 and proto instead — sigmoid features are absolute
   (not batch-size-dependent) and sidestep the rank-distribution-mismatch
   failure mode that killed S1 on LB (v46, LB 0.775; see new_plan.md §9).

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
      Killed on LB 2026-04-15 (v46 LB 0.775, −0.156 vs baseline). Retained
      here for reproducibility / regression checks.
  S2: Single global LightGBM over [a1_rank, sigmoid(b1_oof),
      sigmoid(proto_oof), class_id_categorical]. **D2-α candidate
      (2026-04-16)** — replaces S1's per-class logistic with a tree model
      and rank features with sigmoid features.

**Gate (from new_plan.md):** median Δ ≥ +0.001 macro ROC-AUC over 5 seeds
AND sign-stable across all 5 seeds (no seed regression) vs the best baseline
(BP prod-fused). NOTE the local gate is structurally untrustworthy as an LB
proxy (Phase 1.5 finding: local BP=0.6699 vs A1-alone=0.7359). Local PASS
is necessary but not sufficient for an LB lift.

Usage (from four_track/):
    python -u src/d2_lgbm_stack.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
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
# S2 (with class_id) was killed locally on first sweep — kept as
# `_with_class_id` artifact for forensic comparison only. S2A (no class_id,
# D2-α Fork 1) is the active D2-α candidate; its artifact is what gets
# uploaded to Kaggle.
LGBM_OUT_PATH_S2 = FT_ROOT / "data" / "d2_alpha_lgbm_with_class_id.txt"
LGBM_META_PATH_S2 = FT_ROOT / "data" / "d2_alpha_lgbm_with_class_id_meta.npz"
LGBM_OUT_PATH = FT_ROOT / "data" / "d2_alpha_lgbm.txt"          # S2A (active)
LGBM_META_PATH = FT_ROOT / "data" / "d2_alpha_lgbm_meta.npz"    # S2A (active)

# S2 / D2-α LightGBM hyperparameters. Conservative, regularized.
# Feature order must match D2-α inference:
#   col 0: a1_rank      (float, [0,1])
#   col 1: b1_sigmoid   (float, [0,1])
#   col 2: proto_sigmoid(float, [0,1])
#   col 3: class_id     (int, categorical, [0, N_CLASSES))
LGBM_FEATURE_NAMES = ["a1_rank", "b1_sigmoid", "proto_sigmoid", "class_id"]
LGBM_CAT_FEATURES = [3]  # class_id is categorical
LGBM_PARAMS = {
    "objective":         "binary",
    "metric":            "auc",
    "learning_rate":     0.05,
    "num_leaves":        63,
    "min_data_in_leaf":  50,
    "feature_fraction":  1.0,    # only 4 features, don't sub-sample
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "lambda_l2":         1.0,
    "verbosity":         -1,
}
LGBM_NUM_BOOST_ROUND = 200       # fixed (no early stopping on outer val to keep
                                  # comparison vs BP fair)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable element-wise sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    ).astype(np.float32)


def build_flat_features(a1_rank, b1_sig, proto_sig, *, use_class_id: bool = True):
    """Return (n_rows*n_classes, F) feature matrix in row-major layout.

    F = 4 when `use_class_id=True` (cols a1_rank, b1_sigmoid, proto_sigmoid,
    class_id), F = 3 otherwise (drop class_id). Row i, class c lives at flat
    index `i * n_classes + c`. This matches `arr.reshape(-1)` on a
    (n_rows, n_classes) C-order numpy array, so any label/prediction array
    follows the same indexing.
    """
    n_rows, n_classes = a1_rank.shape
    flat_a1   = a1_rank.reshape(-1).astype(np.float32)
    flat_b1   = b1_sig.reshape(-1).astype(np.float32)
    flat_pr   = proto_sig.reshape(-1).astype(np.float32)
    if use_class_id:
        flat_cls = np.tile(np.arange(n_classes, dtype=np.int32), n_rows)
        return np.column_stack([flat_a1, flat_b1, flat_pr, flat_cls.astype(np.float32)])
    return np.column_stack([flat_a1, flat_b1, flat_pr])


def expand_row_idx_to_flat(row_idx: np.ndarray, n_classes: int) -> np.ndarray:
    """Convert a (k,) array of original-row indices to a (k*n_classes,) array
    of flat indices spanning all classes for those rows."""
    return (row_idx[:, None] * n_classes + np.arange(n_classes)[None, :]).reshape(-1)


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


def stacker_S2_global_lgbm(a1_rank, b1_sig, proto_sig, y, fold_iter, *,
                            use_class_id: bool = True,
                            params=None, num_boost_round=None):
    """Single global LightGBM over flattened (row, class) pairs.

    Features (per row × class):
      use_class_id=True  → [a1_rank, sigmoid(b1_oof), sigmoid(proto_oof),
                            class_id_categorical]   (S2, original D2-α)
      use_class_id=False → [a1_rank, sigmoid(b1_oof), sigmoid(proto_oof)]
                            (S2-A, D2-α-A — no class identity, model is
                            class-agnostic and applied to all 234 classes)

    For each GroupKFold split, train one LGBM on tr_rows × all classes, predict
    on val_rows × all classes, reshape back to (n_val_rows, n_classes). No
    early stopping on outer val (would cherry-pick rounds for OOF AUC and bias
    the gate). Uses fixed `num_boost_round`.

    Returns (n_rows, n_classes) float32 OOF predictions.
    """
    if params is None:
        params = LGBM_PARAMS
    if num_boost_round is None:
        num_boost_round = LGBM_NUM_BOOST_ROUND

    n_rows, n_classes = y.shape
    X_flat = build_flat_features(a1_rank, b1_sig, proto_sig,
                                  use_class_id=use_class_id)
    y_flat = y.reshape(-1).astype(np.float32)
    oof = np.full_like(a1_rank, fill_value=np.nan, dtype=np.float32)

    if use_class_id:
        feat_names = LGBM_FEATURE_NAMES
        cat_feats  = LGBM_CAT_FEATURES
    else:
        feat_names = LGBM_FEATURE_NAMES[:3]
        cat_feats  = []

    for tr_idx, val_idx in fold_iter:
        tr_flat = expand_row_idx_to_flat(tr_idx, n_classes)
        va_flat = expand_row_idx_to_flat(val_idx, n_classes)

        train_set = lgb.Dataset(
            X_flat[tr_flat], label=y_flat[tr_flat],
            feature_name=feat_names,
            categorical_feature=cat_feats,
            free_raw_data=False,
        )
        model = lgb.train(
            params, train_set, num_boost_round=num_boost_round,
        )
        preds = model.predict(X_flat[va_flat], num_iteration=num_boost_round)
        oof[val_idx] = preds.reshape(len(val_idx), n_classes).astype(np.float32)

    nan_mask = np.isnan(oof)
    if nan_mask.any():
        oof[nan_mask] = a1_rank[nan_mask]
    return oof


# --------------------------- main sweep -------------------------------------

def run_seed(a1r, b1r, pr, y, groups, prod_fused, seed: int, n_splits: int = 5,
             *, b1_sig=None, proto_sig=None, run_s1: bool = True,
             run_s2: bool = True):
    """Run one seed: all baselines + (optional) S1 + (optional) S2 stackers.

    `b1_sig`, `proto_sig` are required iff `run_s2` is True. They are
    sigmoid-space features for the S2 LightGBM stacker, distinct from the
    rank-space `b1r`, `pr` arrays used by S1.
    """
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

    if run_s1:
        t0 = time.time()
        s1_oof = stacker_S1_per_class_logreg(a1r, b1r, pr, y, _fold_iter())
        result["S1 per-class logreg"] = macro_auc(y, s1_oof)
        result["_S1_wall_s"] = time.time() - t0

    if run_s2:
        if b1_sig is None or proto_sig is None:
            raise ValueError("run_s2=True requires b1_sig and proto_sig.")
        t0 = time.time()
        s2_oof = stacker_S2_global_lgbm(a1r, b1_sig, proto_sig, y, _fold_iter(),
                                         use_class_id=True)
        result["S2  global LGBM (class_id)"] = macro_auc(y, s2_oof)
        result["_S2_wall_s"] = time.time() - t0

        # D2-α-A (Fork 1, 2026-04-16): drop class_id. Same model architecture
        # as S2 but `use_class_id=False` makes the LGBM class-agnostic.
        t0 = time.time()
        s2a_oof = stacker_S2_global_lgbm(a1r, b1_sig, proto_sig, y, _fold_iter(),
                                          use_class_id=False)
        result["S2A global LGBM (no class_id)"] = macro_auc(y, s2a_oof)
        result["_S2A_wall_s"] = time.time() - t0

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


def fit_final_lgbm(a1_rank, b1_sig, proto_sig, y, *,
                    use_class_id: bool = True,
                    params=None, num_boost_round=None):
    """Fit the D2-α LightGBM on ALL 708 OOF rows × 234 classes (no holdout).

    Returns (booster, meta) where meta is a dict with shape/feature/param info
    suitable for sanity checking at apply time.
    """
    if params is None:
        params = LGBM_PARAMS
    if num_boost_round is None:
        num_boost_round = LGBM_NUM_BOOST_ROUND

    n_rows, n_classes = y.shape
    X_flat = build_flat_features(a1_rank, b1_sig, proto_sig,
                                  use_class_id=use_class_id)
    y_flat = y.reshape(-1).astype(np.float32)

    feat_names = LGBM_FEATURE_NAMES if use_class_id else LGBM_FEATURE_NAMES[:3]
    cat_feats  = LGBM_CAT_FEATURES if use_class_id else []

    train_set = lgb.Dataset(
        X_flat, label=y_flat,
        feature_name=feat_names,
        categorical_feature=cat_feats,
        free_raw_data=False,
    )
    model = lgb.train(params, train_set, num_boost_round=num_boost_round)

    meta = {
        "n_train_rows":    int(n_rows),
        "n_classes":       int(n_classes),
        "feature_names":   feat_names,
        "cat_features":    cat_feats,
        "use_class_id":    bool(use_class_id),
        "num_boost_round": int(num_boost_round),
        "params":          dict(params),
        "best_iteration":  int(num_boost_round),
    }
    return model, meta


def apply_lgbm(model, a1_rank, b1_sig, proto_sig, n_classes: int, *,
                use_class_id: bool = True):
    """Apply a fitted D2-α LightGBM to (n_test_rows, n_classes) inputs.

    Returns (n_test_rows, n_classes) float32 sigmoid-space predictions.
    Caller is responsible for any downstream rank-space remapping. The
    `use_class_id` flag must match how the model was fit.
    """
    n_test, _ = a1_rank.shape
    assert b1_sig.shape == a1_rank.shape == proto_sig.shape, \
        f"shape mismatch: a1={a1_rank.shape}, b1_sig={b1_sig.shape}, proto_sig={proto_sig.shape}"
    X_flat = build_flat_features(a1_rank, b1_sig, proto_sig,
                                  use_class_id=use_class_id)
    preds = model.predict(X_flat).astype(np.float32)
    return preds.reshape(n_test, n_classes)


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

    # --- S1 features: rank-transform B1 and ProtoSSM to match A1's rank-space.
    print("Rank-transforming b1_oof and proto_oof per column (for S1) …")
    b1_ranks    = rank01_per_col(b1_oof)
    proto_ranks = rank01_per_col(proto_oof)
    print(f"  a1_ranks     min/mean/max = {a1_ranks.min():.4f}/{a1_ranks.mean():.4f}/{a1_ranks.max():.4f}")
    print(f"  b1_ranks     min/mean/max = {b1_ranks.min():.4f}/{b1_ranks.mean():.4f}/{b1_ranks.max():.4f}")
    print(f"  proto_ranks  min/mean/max = {proto_ranks.min():.4f}/{proto_ranks.mean():.4f}/{proto_ranks.max():.4f}")

    # --- S2 / D2-α features: sigmoid B1 and ProtoSSM logits.
    print("Sigmoiding b1_oof and proto_oof per cell (for S2 / D2-α) …")
    b1_sig    = sigmoid(b1_oof)
    proto_sig = sigmoid(proto_oof)
    print(f"  b1_sig       min/mean/max = {b1_sig.min():.4f}/{b1_sig.mean():.4f}/{b1_sig.max():.4f}")
    print(f"  proto_sig    min/mean/max = {proto_sig.min():.4f}/{proto_sig.mean():.4f}/{proto_sig.max():.4f}")

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
                     prod_fused=prod_fused, seed=seed,
                     b1_sig=b1_sig, proto_sig=proto_sig,
                     run_s1=True, run_s2=True)
        for k, v in r.items():
            if k.startswith("_"):
                continue
            print(f"  {k:30s} macro AUC = {v:.4f}")
        if "_S1_wall_s" in r:
            print(f"  (S1 wall = {r['_S1_wall_s']:.1f}s)")
        if "_S2_wall_s" in r:
            print(f"  (S2 wall = {r['_S2_wall_s']:.1f}s)")
        if "_S2A_wall_s" in r:
            print(f"  (S2A wall = {r['_S2A_wall_s']:.1f}s)")
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
    # the other baselines but only BP counts for the gate. PRIMARY stacker
    # is S2 (D2-α LightGBM); S1 is shown for comparison.
    print("\n=== Gate decision ===")
    gate_baseline_name = "BP prod-fused"
    gate_baseline_med  = aggregates[gate_baseline_name]["median"]

    def _per_stacker_gate(stacker_name: str):
        if stacker_name not in aggregates:
            return None
        med = aggregates[stacker_name]["median"]
        delta = med - gate_baseline_med
        per_seed_delta = (
            np.array(aggregates[stacker_name]["all"])
            - np.array(aggregates[gate_baseline_name]["all"])
        )
        sign_stable = bool(np.all(per_seed_delta > 0))
        gate_pass = (delta >= 0.001) and sign_stable
        return {
            "median":        med,
            "delta":         float(delta),
            "per_seed_delta":per_seed_delta.round(4).tolist(),
            "sign_stable":   sign_stable,
            "gate_pass":     gate_pass,
        }

    gates = {}
    for s in ("S1 per-class logreg",
              "S2  global LGBM (class_id)",
              "S2A global LGBM (no class_id)"):
        g = _per_stacker_gate(s)
        if g is None:
            continue
        gates[s] = g
        print(f"  {s}:")
        print(f"    median AUC      = {g['median']:.4f}")
        print(f"    Δ vs BP         = {g['delta']:+.4f}")
        print(f"    per-seed Δ      = {g['per_seed_delta']}")
        print(f"    sign-stable     = {g['sign_stable']}")
        print(f"    GATE (Δ≥+0.001 AND sign-stable): {'PASS' if g['gate_pass'] else 'FAIL'}")
    print(f"  gate baseline (BP prod-fused) median AUC = {gate_baseline_med:.4f}")

    # D2-α primary stacker is now S2A (no-class-id, Fork 1). S2 with class_id
    # was killed locally on the first sweep (median AUC 0.34, worse than
    # random) and is retained in the sweep purely for diagnostic comparison.
    s2a_gate = gates.get("S2A global LGBM (no class_id)", {"gate_pass": False})
    print(f"\n→ D2-α-A (Fork 1, S2-no-class-id) GATE: "
          f"{'PASS — proceed to fit final + LB probe' if s2a_gate['gate_pass'] else 'FAIL — do NOT push LB probe'}")

    # Write a summary JSON next to the npz for posterity.
    out_json = FT_ROOT / "data" / "d2_alpha_phase2_results.json"
    out_json.write_text(json.dumps({
        "npz_path":       str(NPZ_PATH),
        "n_seeds":        N_SEEDS,
        "aggregates":     aggregates,
        "gate_baseline":  gate_baseline_name,
        "gates":          gates,
        "lgbm_params":    LGBM_PARAMS,
        "lgbm_num_boost_round": LGBM_NUM_BOOST_ROUND,
    }, indent=2))
    print(f"\nWrote {out_json}")

    # ---- Fit final S1 on ALL 708 rows (kept for reproducibility — S1 was
    # killed on LB v46 0.775 but the artifact is the diff baseline for any
    # future S1 retry attempt). ----
    print("\n=== Fitting final S1 on all 708 rows (S1 retained for repro only) ===")
    coefs, intercepts, fallback, n_pos, n_fitted = fit_final_s1(
        a1_ranks, b1_ranks, proto_ranks, y_true
    )
    print(f"  fit per-class logistic on {n_fitted} / {n_classes} classes "
          f"(rest fall back to A1-only)")

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
    print(f"Wrote S1 coefficients → {coefs_path}  ({coefs_path.stat().st_size} bytes)")

    # ---- Fit final S2 (with class_id) on ALL 708 rows. Retained for
    # forensic comparison only — this variant was killed locally on the
    # first sweep (OOF 0.34, worse than random). ----
    print("\n=== Fitting final S2 (with class_id) on all 708 rows (FORENSIC ONLY) ===")
    booster_s2, meta_s2 = fit_final_lgbm(a1_ranks, b1_sig, proto_sig, y_true,
                                          use_class_id=True)
    in_sample_preds_s2 = apply_lgbm(booster_s2, a1_ranks, b1_sig, proto_sig,
                                     n_classes, use_class_id=True)
    in_sample_auc_s2 = macro_auc(y_true, in_sample_preds_s2)
    s2_oof_med = aggregates.get("S2  global LGBM (class_id)", {}).get("median", float("nan"))
    print(f"  in-sample macro AUC = {in_sample_auc_s2:.4f}")
    print(f"  (compare: OOF S2 = {s2_oof_med:.4f}, BP prod-fused = {gate_baseline_med:.4f})")
    booster_s2.save_model(str(LGBM_OUT_PATH_S2))
    np.savez(
        LGBM_META_PATH_S2,
        n_train_rows = np.int64(meta_s2["n_train_rows"]),
        n_classes    = np.int64(meta_s2["n_classes"]),
        feature_names= np.array(meta_s2["feature_names"]),
        cat_features = np.array(meta_s2["cat_features"], dtype=np.int64),
        use_class_id = np.bool_(meta_s2["use_class_id"]),
        num_boost_round = np.int64(meta_s2["num_boost_round"]),
    )
    size_mb_s2 = LGBM_OUT_PATH_S2.stat().st_size / 1e6
    print(f"Wrote forensic LGBM  → {LGBM_OUT_PATH_S2}  ({size_mb_s2:.2f} MB)")

    # ---- Fit final S2A (no class_id) on ALL 708 rows. ACTIVE D2-α
    # artifact — this is what gets uploaded to Kaggle if the gate passes. ----
    print("\n=== Fitting final S2A (no class_id, D2-α-A) on all 708 rows (ACTIVE) ===")
    booster_s2a, meta_s2a = fit_final_lgbm(a1_ranks, b1_sig, proto_sig, y_true,
                                             use_class_id=False)
    in_sample_preds_s2a = apply_lgbm(booster_s2a, a1_ranks, b1_sig, proto_sig,
                                      n_classes, use_class_id=False)
    in_sample_auc_s2a = macro_auc(y_true, in_sample_preds_s2a)
    s2a_oof_med = aggregates.get("S2A global LGBM (no class_id)", {}).get("median", float("nan"))
    print(f"  in-sample macro AUC = {in_sample_auc_s2a:.4f}")
    print(f"  (compare: OOF S2A = {s2a_oof_med:.4f}, BP prod-fused = {gate_baseline_med:.4f})")
    print(f"  in-sample/OOF gap = {in_sample_auc_s2a - s2a_oof_med:.4f}  "
          f"(small gap = less overfit; compare S2 gap = {in_sample_auc_s2 - s2_oof_med:.4f})")
    booster_s2a.save_model(str(LGBM_OUT_PATH))
    np.savez(
        LGBM_META_PATH,
        n_train_rows = np.int64(meta_s2a["n_train_rows"]),
        n_classes    = np.int64(meta_s2a["n_classes"]),
        feature_names= np.array(meta_s2a["feature_names"]),
        cat_features = np.array(meta_s2a["cat_features"], dtype=np.int64),
        use_class_id = np.bool_(meta_s2a["use_class_id"]),
        num_boost_round = np.int64(meta_s2a["num_boost_round"]),
    )
    size_mb_s2a = LGBM_OUT_PATH.stat().st_size / 1e6
    print(f"Wrote active LGBM    → {LGBM_OUT_PATH}  ({size_mb_s2a:.2f} MB)")
    print(f"Wrote active LGBM meta → {LGBM_META_PATH}")


if __name__ == "__main__":
    main()
