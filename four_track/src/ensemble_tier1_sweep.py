"""§34 Tier 1 ensembling sweep — Chapter 10 techniques on existing OOF probs.

Loads broader-pool OOF probs we already have on disk and runs every
ensembling technique from The Kaggle Book Ch 10 that operates on
existing model outputs (no retraining):

  1. Alternative averaging operators: arithmetic, geometric, harmonic,
     mean-of-powers (n=3), logarithmic, rank-mean
  2. Inverse-correlation weighted averaging (Ch 10 pp 374)
  3. Caruana ensemble selection with replacement (Ch 10 pp 380-383),
     with file-based holdout for honest test AUC
  4. Logistic regression blender (Ch 10 pp 378-380), L1 + positive-only
     coefficients, with file-based holdout

Reports each technique's broader-pool macro AUC vs A2 sig-mean anchor
(0.8402) and the +0.05 slot-burn gate (0.8902). Saves results to
data/ensemble_tier1_results.json.

Honest expectation per §34.5: best of Tier 1 likely lands 0.85-0.87;
gate-fail. Closes the "we never tried" empirical record gap.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

FT_ROOT = Path(__file__).resolve().parents[1]
DATA = FT_ROOT / "data"

A2_OOF = DATA / "a2_a1_5fold_broader_oof.npz"
AST_OOF = DATA / "a4_ast_fold0_broader_oof.npz"
CONVNEXT_OOF = DATA / "a5_convnext_pico_fold0_broader_oof.npz"
MOBILEVIT_OOF = DATA / "a5_mobilevit_s_fold0_broader_oof.npz"
CERRADO_OOF = DATA / "xc_cerrado_5fold_broader_oof.npz"

OUT_PATH = DATA / "ensemble_tier1_results.json"

A2_ANCHOR_AUC = 0.8402
GATE_AUC = 0.8902  # anchor + 0.05 per feedback_min_oof_delta_to_burn_slot

# ── Utilities ─────────────────────────────────────────────────────────────────

def macro_auc(probs: np.ndarray, y_true: np.ndarray) -> float:
    present = y_true.sum(axis=0) > 0
    if not present.any():
        return float("nan")
    return float(
        roc_auc_score(y_true[:, present], probs[:, present], average="macro")
    )


def rank01_per_col(mat: np.ndarray) -> np.ndarray:
    """Per-column rank normalized to [0,1] (matches probe_b_weight_sweep_v2)."""
    n = mat.shape[0]
    order = np.argsort(mat, axis=0, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    rows = np.arange(n, dtype=np.float32)
    for c in range(mat.shape[1]):
        ranks[order[:, c], c] = rows
    if n > 1:
        ranks /= (n - 1)
    return ranks


# ── Operators (parameter-free, operate on probs in [0,1]) ────────────────────

def op_arithmetic(probs_list: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(probs_list, axis=0), axis=0)


def op_geometric(probs_list: list[np.ndarray]) -> np.ndarray:
    eps = 1e-7
    log_mean = np.mean(np.log(np.clip(np.stack(probs_list, 0), eps, 1.0)), axis=0)
    return np.exp(log_mean)


def op_harmonic(probs_list: list[np.ndarray]) -> np.ndarray:
    eps = 1e-7
    return 1.0 / np.mean(1.0 / (np.stack(probs_list, 0) + eps), axis=0)


def op_mean_of_powers(probs_list: list[np.ndarray], n: int = 3) -> np.ndarray:
    return np.mean(np.stack(probs_list, 0) ** n, axis=0) ** (1.0 / n)


def op_logarithmic(probs_list: list[np.ndarray]) -> np.ndarray:
    # log1p + expm1 variant — avoids log(0) without an explicit clip
    return np.expm1(np.mean(np.log1p(np.stack(probs_list, 0)), axis=0))


def op_rank_mean(probs_list: list[np.ndarray]) -> np.ndarray:
    ranks = [rank01_per_col(p) for p in probs_list]
    return np.mean(np.stack(ranks, 0), axis=0)


OPERATORS = {
    "arithmetic": op_arithmetic,
    "geometric": op_geometric,
    "harmonic": op_harmonic,
    "mean_of_powers_n3": op_mean_of_powers,
    "logarithmic": op_logarithmic,
    "rank_mean": op_rank_mean,
}


def inv_corr_weighted(probs_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Ch 10 pp 374 — weight inversely proportional to correlation.

    Builds correlation matrix over the per-window flattened predictions of
    each model, zeros the diagonal, row-averages, takes the reciprocal,
    normalizes to sum=1, applies as a per-model weight to a weighted sum.
    """
    stacked = np.stack([p.ravel() for p in probs_list], axis=0)
    cormat = np.corrcoef(stacked)
    np.fill_diagonal(cormat, 0.0)
    avg_corr = np.mean(cormat, axis=1)
    # avoid div-by-zero when a single model has zero avg corr (single model case)
    W = 1.0 / np.where(np.abs(avg_corr) > 1e-9, avg_corr, 1.0)
    W = W / W.sum()
    out = np.zeros_like(probs_list[0])
    for w, p in zip(W, probs_list):
        out += w * p
    return out, W


# ── Caruana ensemble selection (Ch 10 pp 380-383) ────────────────────────────

def caruana_select(
    component_probs: list[np.ndarray],
    component_names: list[str],
    y_holdout: np.ndarray,
    max_iters: int = 100,
) -> tuple[dict[str, float], float, list[int]]:
    """Hill-climbing forward selection with replacement.

    Returns (weights dict, best_holdout_auc, selection_sequence).
    """
    baseline = 0.5
    models: list[int] = []
    for _ in range(max_iters):
        challengers = []
        for j in range(len(component_probs)):
            candidate_idx = models + [j]
            candidate_probs = np.mean(
                np.stack([component_probs[i] for i in candidate_idx], axis=0),
                axis=0,
            )
            score = macro_auc(candidate_probs, y_holdout)
            challengers.append((score, j))
        challengers.sort(key=lambda x: x[0], reverse=True)
        best_score, best_idx = challengers[0]
        if best_score > baseline:
            models.append(best_idx)
            baseline = best_score
        else:
            break
    freqs = Counter(models)
    weights = {component_names[k]: f / len(models) for k, f in freqs.items()}
    return weights, baseline, models


# ── Logistic-regression blender (Ch 10 pp 378-380) ───────────────────────────

def logreg_blend(
    component_probs_train: list[np.ndarray],   # each (N_train, K)
    y_train: np.ndarray,                       # (N_train, K)
    component_probs_test: list[np.ndarray],    # each (N_test, K)
    positive_only: bool = True,
) -> np.ndarray:
    """Fit one logistic regression per class (multi-label OvR setup).

    Each per-class blender takes the K stacked-model probabilities for
    that class and learns a positive-weighted combination. Returns the
    fused (N_test, K) probabilities.
    """
    n_models = len(component_probs_train)
    n_test, n_classes = component_probs_test[0].shape
    out = np.zeros((n_test, n_classes), dtype=np.float32)
    for c in range(n_classes):
        # Build the per-class feature matrices
        Xtr = np.stack([p[:, c] for p in component_probs_train], axis=1)
        Xte = np.stack([p[:, c] for p in component_probs_test], axis=1)
        yc = y_train[:, c]
        if yc.sum() < 2 or yc.sum() == len(yc):
            # All-zero or all-one column — fall back to arithmetic mean
            out[:, c] = Xte.mean(axis=1)
            continue
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)
        try:
            blender = LogisticRegression(
                solver="liblinear", penalty="l1", C=1.0,
                positive=positive_only, fit_intercept=False, max_iter=200,
            )
            blender.fit(Xtr_s, yc)
            out[:, c] = blender.predict_proba(Xte_s)[:, 1]
        except Exception:
            out[:, c] = Xte.mean(axis=1)
    return out


# ── Load all available OOFs ──────────────────────────────────────────────────

def load_components() -> tuple[list[tuple[str, np.ndarray]], np.ndarray, np.ndarray, np.ndarray]:
    """Load all available OOF probs into a list of (name, (N,K) probs).

    Returns (components, y_true, filenames, start_secs) where the latter
    three come from the A2 file (authoritative for the eval pool).
    """
    if not A2_OOF.exists():
        sys.exit(f"missing A2 OOF: {A2_OOF}")
    a2 = np.load(A2_OOF, allow_pickle=True)
    y_true = a2["y_true"].astype(np.float32)
    filenames = a2["filenames"]
    start_sec = a2["start_sec"]
    a2_probs_mean = a2["probs_mean"].astype(np.float32)
    n_segments = y_true.shape[0]

    components: list[tuple[str, np.ndarray]] = []
    components.append(("a2_ensemble", a2_probs_mean))
    # Also expose individual A2 folds for Caruana to pick from
    for f in range(5):
        components.append((f"a2_fold{f}", a2["probs_per_fold"][f].astype(np.float32)))

    def _maybe_load(path: Path, name: str) -> None:
        if not path.exists():
            print(f"  [skip] {name} (file missing: {path.name})", flush=True)
            return
        arr = np.load(path, allow_pickle=True)
        probs = (arr["probs"] if "probs" in arr.files else arr["probs_mean"]).astype(np.float32)
        assert probs.shape == (n_segments, y_true.shape[1]), (
            f"{name} shape {probs.shape} != A2 {(n_segments, y_true.shape[1])}"
        )
        components.append((name, probs))
        # If the cross-arch file has individual folds, expose those too
        if "probs_per_fold" in arr.files:
            for fi in range(arr["probs_per_fold"].shape[0]):
                components.append(
                    (f"{name}_fold{fi}",
                     arr["probs_per_fold"][fi].astype(np.float32))
                )

    _maybe_load(AST_OOF, "ast_fold0")
    _maybe_load(CONVNEXT_OOF, "convnext_pico_fold0")
    _maybe_load(MOBILEVIT_OOF, "mobilevit_s_fold0")
    _maybe_load(CERRADO_OOF, "cerrado_ensemble")

    return components, y_true, filenames, start_sec


# ── File-based holdout split (66 files → 50/50 random) ───────────────────────

def file_split_indices(
    filenames: np.ndarray, seed: int, test_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique_files = np.array(sorted(set(filenames.tolist())))
    rng.shuffle(unique_files)
    n_test = int(round(len(unique_files) * test_frac))
    test_files = set(unique_files[:n_test])
    is_test = np.array([f in test_files for f in filenames])
    return np.where(~is_test)[0], np.where(is_test)[0]


# ── Sweeps ────────────────────────────────────────────────────────────────────

def sweep_pairwise_operators(
    a2_probs_mean: np.ndarray,
    y_true: np.ndarray,
    cross_arch_components: list[tuple[str, np.ndarray]],
) -> list[dict]:
    """For each cross-arch component, sweep operators × A2-weight grid."""
    a2_auc = macro_auc(a2_probs_mean, y_true)
    rows: list[dict] = []
    weights = np.round(np.arange(0.05, 0.96, 0.05), 2)
    for name, probs in cross_arch_components:
        for op_name, op_fn in OPERATORS.items():
            for w in weights:
                if op_name in {"arithmetic", "rank_mean"}:
                    # Properly weighted variants for these two ops
                    if op_name == "arithmetic":
                        fused = float(w) * probs + (1.0 - float(w)) * a2_probs_mean
                    else:  # rank_mean
                        fused = (
                            float(w) * rank01_per_col(probs)
                            + (1.0 - float(w)) * rank01_per_col(a2_probs_mean)
                        )
                else:
                    # For the other ops, equal-weight 2-way mix is the canonical form
                    fused = op_fn([probs, a2_probs_mean])
                auc = macro_auc(fused, y_true)
                rows.append({
                    "component": name,
                    "operator": op_name,
                    "weight_on_component": float(w) if op_name in {"arithmetic", "rank_mean"} else 0.50,
                    "auc": auc,
                    "delta_vs_a2": auc - a2_auc,
                })
                # Non-weighted ops only run once
                if op_name not in {"arithmetic", "rank_mean"}:
                    break
    return rows


def sweep_pool_inv_corr(
    components: list[tuple[str, np.ndarray]],
    y_true: np.ndarray,
    pool_specs: list[tuple[str, list[str]]],
) -> list[dict]:
    """Inverse-correlation weighted averaging over named pools."""
    name_to_probs = dict(components)
    rows: list[dict] = []
    a2_auc = macro_auc(name_to_probs["a2_ensemble"], y_true)
    for pool_name, member_names in pool_specs:
        missing = [m for m in member_names if m not in name_to_probs]
        if missing:
            continue
        member_probs = [name_to_probs[m] for m in member_names]
        fused, W = inv_corr_weighted(member_probs)
        auc = macro_auc(fused, y_true)
        rows.append({
            "pool": pool_name,
            "members": member_names,
            "weights": [float(w) for w in W],
            "auc": auc,
            "delta_vs_a2": auc - a2_auc,
        })
    return rows


def sweep_caruana(
    components: list[tuple[str, np.ndarray]],
    y_true: np.ndarray,
    filenames: np.ndarray,
    n_splits: int = 5,
) -> dict:
    """Caruana ensemble selection with file-based holdout, averaged over splits."""
    names = [n for n, _ in components]
    probs_list = [p for _, p in components]
    test_aucs = []
    selections = []
    for seed in range(n_splits):
        train_idx, test_idx = file_split_indices(filenames, seed=seed)
        train_probs = [p[train_idx] for p in probs_list]
        train_y = y_true[train_idx]
        weights, _holdout_auc, _seq = caruana_select(train_probs, names, train_y)
        # Compute fused on TEST using the same weight pattern
        test_probs = [p[test_idx] for p in probs_list]
        fused = np.zeros_like(test_probs[0])
        for nm, w in weights.items():
            idx = names.index(nm)
            fused += w * test_probs[idx]
        test_auc = macro_auc(fused, y_true[test_idx])
        test_aucs.append(test_auc)
        selections.append(weights)
    return {
        "mean_test_auc": float(np.mean(test_aucs)),
        "std_test_auc": float(np.std(test_aucs)),
        "per_split_test_auc": [float(a) for a in test_aucs],
        "per_split_weights": selections,
    }


def sweep_logreg_blender(
    components: list[tuple[str, np.ndarray]],
    y_true: np.ndarray,
    filenames: np.ndarray,
    n_splits: int = 5,
) -> dict:
    """Logistic-regression per-class blender, file-based holdout, averaged."""
    # Use only the "top-level" components (not per-fold expansions) to keep
    # the blender feature count tractable
    keep = [(n, p) for n, p in components if "_fold" not in n]
    names = [n for n, _ in keep]
    probs_list = [p for _, p in keep]
    test_aucs = []
    for seed in range(n_splits):
        train_idx, test_idx = file_split_indices(filenames, seed=seed)
        train_probs = [p[train_idx] for p in probs_list]
        test_probs = [p[test_idx] for p in probs_list]
        fused = logreg_blend(train_probs, y_true[train_idx], test_probs)
        test_auc = macro_auc(fused, y_true[test_idx])
        test_aucs.append(test_auc)
    return {
        "mean_test_auc": float(np.mean(test_aucs)),
        "std_test_auc": float(np.std(test_aucs)),
        "per_split_test_auc": [float(a) for a in test_aucs],
        "members": names,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60, flush=True)
    print("§34 Tier 1 ensembling sweep", flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()
    components, y_true, filenames, _start = load_components()
    n_segments, n_classes = y_true.shape
    print(f"[load] {len(components)} components  "
          f"y_true={y_true.shape}  ({time.time() - t0:.1f}s)", flush=True)
    for nm, p in components:
        print(f"  - {nm:<30} {p.shape}", flush=True)
    a2_mean_probs = next(p for n, p in components if n == "a2_ensemble")
    a2_auc = macro_auc(a2_mean_probs, y_true)
    print("", flush=True)
    print(f"  A2 anchor (sig-mean):  {a2_auc:.4f}  (expected ~{A2_ANCHOR_AUC:.4f})",
          flush=True)
    print(f"  Slot-burn gate:        {GATE_AUC:.4f}  (anchor + 0.05)", flush=True)
    print("", flush=True)

    # --- Pairwise operator sweep ---
    print("=" * 60, flush=True)
    print("(1) Pairwise operator sweep: cross-arch × A2", flush=True)
    print("=" * 60, flush=True)
    cross_arch = [(n, p) for n, p in components
                  if n in {"ast_fold0", "convnext_pico_fold0",
                           "mobilevit_s_fold0", "cerrado_ensemble"}]
    pair_rows = sweep_pairwise_operators(a2_mean_probs, y_true, cross_arch)
    # Top 10 by AUC
    pair_sorted = sorted(pair_rows, key=lambda r: r["auc"], reverse=True)
    print(f"  {'component':<24} {'operator':<20} {'w':>5}  {'auc':>7}  {'Δ':>7}",
          flush=True)
    for r in pair_sorted[:15]:
        print(f"  {r['component']:<24} {r['operator']:<20} "
              f"{r['weight_on_component']:>5.2f}  {r['auc']:>7.4f}  "
              f"{r['delta_vs_a2']:>+7.4f}", flush=True)

    # --- Inverse-correlation weighted averaging ---
    print("", flush=True)
    print("=" * 60, flush=True)
    print("(2) Inverse-correlation weighted averaging (Ch 10 pp 374)", flush=True)
    print("=" * 60, flush=True)
    available = {n for n, _ in components}
    pool_specs: list[tuple[str, list[str]]] = [
        ("a2+ast", ["a2_ensemble", "ast_fold0"]),
        ("a2+convnext", ["a2_ensemble", "convnext_pico_fold0"]),
        ("a2+mobilevit", ["a2_ensemble", "mobilevit_s_fold0"]),
        ("a2+cerrado", ["a2_ensemble", "cerrado_ensemble"]),
        ("a2+ast+convnext+mobilevit", [
            "a2_ensemble", "ast_fold0", "convnext_pico_fold0", "mobilevit_s_fold0",
        ]),
        ("all_arch", [
            "a2_ensemble", "ast_fold0", "convnext_pico_fold0",
            "mobilevit_s_fold0", "cerrado_ensemble",
        ]),
    ]
    # Drop pool specs that reference missing components
    pool_specs = [(n, m) for (n, m) in pool_specs if set(m).issubset(available)]
    inv_corr_rows = sweep_pool_inv_corr(components, y_true, pool_specs)
    for r in inv_corr_rows:
        print(f"  {r['pool']:<35} auc={r['auc']:.4f}  Δ={r['delta_vs_a2']:+.4f}",
              flush=True)
        weight_str = ", ".join(f"{m}={w:.3f}" for m, w in
                                zip(r["members"], r["weights"]))
        print(f"    weights: {weight_str}", flush=True)

    # --- Caruana ensemble selection ---
    print("", flush=True)
    print("=" * 60, flush=True)
    print("(3) Caruana ensemble selection (Ch 10 pp 380-383)", flush=True)
    print("    File-based 50/50 holdout × 5 splits, hill-climb on train, AUC on test", flush=True)
    print("=" * 60, flush=True)
    caruana_result = sweep_caruana(components, y_true, filenames)
    print(f"  mean test AUC across 5 splits: {caruana_result['mean_test_auc']:.4f} "
          f"(± {caruana_result['std_test_auc']:.4f})  "
          f"Δ vs anchor: {caruana_result['mean_test_auc'] - a2_auc:+.4f}",
          flush=True)
    print("  selected weights (first split):", flush=True)
    for nm, w in sorted(caruana_result["per_split_weights"][0].items(),
                        key=lambda x: -x[1]):
        print(f"    {nm:<30} {w:.3f}", flush=True)

    # --- Logistic-regression blender ---
    print("", flush=True)
    print("=" * 60, flush=True)
    print("(4) Logistic regression blender per-class, L1, positive-only", flush=True)
    print("    File-based 50/50 holdout × 5 splits", flush=True)
    print("=" * 60, flush=True)
    logreg_result = sweep_logreg_blender(components, y_true, filenames)
    print(f"  mean test AUC across 5 splits: {logreg_result['mean_test_auc']:.4f} "
          f"(± {logreg_result['std_test_auc']:.4f})  "
          f"Δ vs anchor: {logreg_result['mean_test_auc'] - a2_auc:+.4f}",
          flush=True)
    print(f"  members: {logreg_result['members']}", flush=True)

    # --- Verdict ---
    print("", flush=True)
    print("=" * 60, flush=True)
    print("Tier 1 verdict", flush=True)
    print("=" * 60, flush=True)
    best_pairwise = pair_sorted[0]
    best_inv_corr = max(inv_corr_rows, key=lambda r: r["auc"]) if inv_corr_rows else None
    candidates = [
        ("best pairwise operator",
         best_pairwise["auc"],
         f"{best_pairwise['component']} {best_pairwise['operator']} "
         f"w={best_pairwise['weight_on_component']:.2f}"),
        ("best inv-corr pool",
         best_inv_corr["auc"] if best_inv_corr else float("nan"),
         best_inv_corr["pool"] if best_inv_corr else "(none)"),
        ("caruana (mean test)",
         caruana_result["mean_test_auc"],
         "5-split avg"),
        ("logreg blender (mean test)",
         logreg_result["mean_test_auc"],
         "5-split avg"),
    ]
    for name, auc, detail in candidates:
        marker = "  ★ GATE PASS" if auc >= GATE_AUC else ""
        print(f"  {name:<30} auc={auc:.4f}  Δ={auc - a2_auc:+.4f}  "
              f"({detail}){marker}", flush=True)
    best_overall = max(candidates, key=lambda x: x[1])
    print("", flush=True)
    print(f"  best Tier 1: {best_overall[0]} = {best_overall[1]:.4f}  "
          f"(gate {GATE_AUC:.4f})", flush=True)
    if best_overall[1] >= GATE_AUC:
        verdict = "GATE PASS — Tier 1 alone justifies v77"
    elif best_overall[1] >= 0.85:
        verdict = ("GATE FAIL but best ≥ 0.85 — Tier 2 (multi-seed bagging) "
                   "may be worth dispatching")
    else:
        verdict = ("GATE FAIL and < 0.85 — Tier 1 exhausts the ensembling lever; "
                   "Tier 2 priors degrade significantly")
    print(f"  verdict: {verdict}", flush=True)
    print("=" * 60, flush=True)

    # Save
    results = {
        "a2_anchor_auc": a2_auc,
        "gate_auc": GATE_AUC,
        "components": [(n, list(p.shape)) for n, p in components],
        "pairwise_top15": pair_sorted[:15],
        "inv_corr": inv_corr_rows,
        "caruana": caruana_result,
        "logreg_blender": logreg_result,
        "best_overall": {"name": best_overall[0], "auc": best_overall[1],
                          "detail": best_overall[2]},
        "verdict": verdict,
    }
    OUT_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"[save] {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
