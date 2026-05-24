"""Three-gate rank-blend rescue: local OOF sweep against matched A1+ProtoSSM OOF.

Mirrors mtoshidesu 0.947-reproducer cell 34 logic (fake_only + proto_cont +
sed_only gates). Sweeps the most sensitive gate constants and reports
broader-pool macro-AUC delta vs baseline (no gates).

Substrate options:
  --oof-npz data/d2_beta_oofs.npz   (stale April 15 models, fast sanity check)
  --oof-npz data/d2_beta_oofs_fresh.npz   (if regenerated against current models)

Usage:
  python src/postproc_gate_sweep.py --oof-npz data/d2_beta_oofs.npz
  python src/postproc_gate_sweep.py --oof-npz data/d2_beta_oofs.npz --sweep
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


EPS = 1e-5


def rank01(arr: np.ndarray) -> np.ndarray:
    """Per-column percentile rank in [0,1] — same as mtoshidesu (pandas rank pct=True)."""
    return pd.DataFrame(arr).rank(axis=0, pct=True).to_numpy(np.float32)


def macro_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Macro-averaged ROC-AUC over classes that have ≥1 positive."""
    aucs = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() > 0:
            try:
                aucs.append(roc_auc_score(y_true[:, c], scores[:, c]))
            except ValueError:
                pass
    return float(np.mean(aucs)) if aucs else float("nan")


def baseline_blend(p_proto: np.ndarray, p_sed: np.ndarray,
                   w_sed: float = 0.22) -> np.ndarray:
    """Production-style rank blend (no gates).

    Matches mtoshidesu cell 34 baseline: clip both inputs to [EPS, 1-EPS]
    BEFORE ranking, then rank-blend at (1-w_sed, w_sed). Without the clip,
    values at exactly 0 or 1 produce slightly different rank distributions
    than apply_gates() expects, biasing the gate vs baseline comparison.
    """
    p_proto = np.clip(p_proto, EPS, 1.0 - EPS).astype(np.float32)
    p_sed   = np.clip(p_sed,   EPS, 1.0 - EPS).astype(np.float32)
    rp = rank01(p_proto)
    rs = rank01(p_sed)
    return (1.0 - w_sed) * rp + w_sed * rs


def apply_gates(
    p_proto: np.ndarray,
    p_sed: np.ndarray,
    file_ids: np.ndarray,
    *,
    w_sed: float = 0.22,
    # Gate 1: fake_only (noise suppression)
    fake_p_proto_thr: float = 0.55,
    fake_p_sed_thr: float = 0.08,
    fake_blend: float = 0.05,
    # Gate 2: proto_cont (temporal continuity)
    cont_xctx_thr: float = 0.90,
    cont_rp_thr: float = 0.75,
    cont_p_sed_thr: float = 0.12,
    cont_blend: float = 0.12,
    cont_kernel_scale: float = 1.10,
    cont_kernel_power: float = -1.5,
    cont_kernel_halfwidth: int = 3,
    # Gate 3: sed_only (preserve SED spikes)
    sed_rs_thr: float = 0.94,
    sed_rp_thr: float = 0.78,
    sed_blend: float = 0.08,
) -> np.ndarray:
    """Apply mtoshidesu cell 34's three-gate rescue on top of the baseline blend."""
    p_proto = np.clip(p_proto, EPS, 1.0 - EPS).astype(np.float32)
    p_sed   = np.clip(p_sed,   EPS, 1.0 - EPS).astype(np.float32)

    rank_proto = rank01(p_proto)
    rank_sed   = rank01(p_sed)

    pred = (1.0 - w_sed) * rank_proto + w_sed * rank_sed

    # Gate 1: fake_only
    fake_only = (p_proto > fake_p_proto_thr) & (p_sed < fake_p_sed_thr)
    pred = np.where(
        fake_only,
        (1.0 - fake_blend) * pred + fake_blend * rank_proto,
        pred,
    )

    # Gate 2: proto_cont — fat-tail continuity kernel across windows for each file
    hw = cont_kernel_halfwidth
    offs = np.arange(-hw, hw + 1, dtype=np.float32)
    proto_kernel = (1.0 + (offs / cont_kernel_scale) ** 2 / 2.0) ** cont_kernel_power
    proto_kernel = (proto_kernel / proto_kernel.sum()).astype(np.float32)

    pa_ctx = p_proto.copy()
    for fid in pd.unique(file_ids):
        m = file_ids == fid
        x = p_proto[m]
        if len(x) > 1:
            xp = np.pad(x, ((hw, hw), (0, 0)), mode="edge")
            pa_ctx[m] = sum(proto_kernel[i] * xp[i:i + len(x)]
                            for i in range(2 * hw + 1))

    xctx = rank01(pa_ctx)

    proto_cont = (
        (xctx > cont_xctx_thr)
        & (rank_proto > cont_rp_thr)
        & (p_sed < cont_p_sed_thr)
        & (~fake_only)
    )
    pred = np.where(
        proto_cont,
        (1.0 - cont_blend) * pred + cont_blend * np.maximum(rank_proto, xctx),
        pred,
    )

    # Gate 3: sed_only
    sed_only = (
        (rank_sed > sed_rs_thr)
        & (rank_proto < sed_rp_thr)
        & (~fake_only)
        & (~proto_cont)
    )
    pred = np.where(
        sed_only,
        (1.0 - sed_blend) * pred + sed_blend * rank_sed,
        pred,
    )

    return pred.astype(np.float32)


def load_oof(npz_path: Path):
    """Load matched A1+ProtoSSM OOF arrays.

    Recognises two layouts:
      (a) d2_beta_oofs.npz   — keys: a1_ranks, proto_oof, y_true, file_groups (one group id per file), n_windows
      (b) custom merged npz  — keys: p_sed, p_proto, y_true, file_ids
    """
    d = np.load(npz_path, allow_pickle=True)
    keys = set(d.files)

    if {"a1_ranks", "proto_oof", "y_true"}.issubset(keys):
        # d2_beta layout: a1_ranks already in rank space; convert back to "p_sed-like"
        # by treating them as pseudo-probabilities (rank01-of-rank01 ≈ rank01).
        # For the gate logic we need p_sed (sigmoid-like); pass a1_ranks directly —
        # the gate code re-ranks it anyway. p_sed thresholds (0.08, 0.12) target
        # sigmoid scale though, so this is approximate.
        a1_ranks = d["a1_ranks"].astype(np.float32)
        p_proto  = d["proto_oof"].astype(np.float32)
        y_true   = d["y_true"].astype(np.float32)
        # Reconstruct file_ids: file_groups gives one group label per file
        # (length = n_files); rows are flattened (file × window). n_windows is scalar.
        n_windows = int(d["n_windows"])
        file_groups = d["file_groups"]
        n_files = len(file_groups)
        file_ids = np.repeat(np.arange(n_files), n_windows).astype(np.int64)
        return {
            "p_sed":    a1_ranks,   # already ranked; pass-through (caveat noted)
            "p_proto":  p_proto,
            "y_true":   y_true,
            "file_ids": file_ids,
            "n_files":  n_files,
            "n_windows": n_windows,
            "layout":   "d2_beta_a1_ranks",
        }

    if {"p_sed", "p_proto", "y_true", "file_ids"}.issubset(keys):
        return {
            "p_sed":    d["p_sed"].astype(np.float32),
            "p_proto":  d["p_proto"].astype(np.float32),
            "y_true":   d["y_true"].astype(np.float32),
            "file_ids": d["file_ids"].astype(np.int64),
            "n_files":  len(np.unique(d["file_ids"])),
            "n_windows": d["p_sed"].shape[0] // len(np.unique(d["file_ids"])),
            "layout":   "merged",
        }

    raise SystemExit(f"unknown OOF layout: keys={sorted(keys)}")


def run_one(oof, params: dict, label: str) -> dict:
    base = baseline_blend(oof["p_proto"], oof["p_sed"], w_sed=params.get("w_sed", 0.22))
    base_auc = macro_auc(oof["y_true"], base)
    gated = apply_gates(oof["p_proto"], oof["p_sed"], oof["file_ids"], **params)
    gated_auc = macro_auc(oof["y_true"], gated)
    delta = gated_auc - base_auc
    return {
        "label":     label,
        "baseline":  base_auc,
        "gated":     gated_auc,
        "delta":     delta,
        "params":    params,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-npz", required=True, type=Path)
    parser.add_argument("--sweep", action="store_true",
                        help="do a small constant grid sweep (default: just default constants)")
    args = parser.parse_args()

    oof = load_oof(args.oof_npz)
    print(f"OOF substrate: layout={oof['layout']}  "
          f"rows={oof['y_true'].shape[0]}  classes={oof['y_true'].shape[1]}  "
          f"n_files={oof['n_files']}  n_windows={oof['n_windows']}", flush=True)

    if oof["layout"] == "d2_beta_a1_ranks":
        print("  CAVEAT: d2_beta layout passes a1_ranks (already rank-space) as p_sed. "
              "Gate thresholds 0.08/0.12 are calibrated to SIGMOID scale; sed_thr "
              "behaviour will be off until fresh matched A1 sigmoids are regenerated.",
              flush=True)

    # Defaults from mtoshidesu cell 34
    DEFAULTS = dict(
        w_sed=0.22,
        fake_p_proto_thr=0.55, fake_p_sed_thr=0.08, fake_blend=0.05,
        cont_xctx_thr=0.90, cont_rp_thr=0.75, cont_p_sed_thr=0.12, cont_blend=0.12,
        sed_rs_thr=0.94, sed_rp_thr=0.78, sed_blend=0.08,
    )

    print("\n=== Single run with mtoshidesu defaults ===", flush=True)
    r = run_one(oof, DEFAULTS, "mtoshidesu-defaults")
    print(f"  baseline   AUC = {r['baseline']:.4f}", flush=True)
    print(f"  gated      AUC = {r['gated']:.4f}", flush=True)
    print(f"  delta          = {r['delta']:+.4f}", flush=True)

    if not args.sweep:
        return

    print("\n=== Small constant sweep ===", flush=True)
    results = []

    # Sweep over fake_blend, cont_blend, sed_blend (low risk, high sensitivity)
    blend_grid = [0.00, 0.04, 0.08, 0.12, 0.16]
    for fb in blend_grid:
        for cb in blend_grid:
            for sb in blend_grid:
                p = {**DEFAULTS,
                     "fake_blend": fb, "cont_blend": cb, "sed_blend": sb}
                tag = f"f={fb:.2f} c={cb:.2f} s={sb:.2f}"
                results.append(run_one(oof, p, tag))

    # Extended sed-only sweep with fake/cont disabled (the only gate that matters here)
    print("\n=== Extended SED-only sweep (fake=0, cont=0; vary sed_blend + sed_rs_thr) ===",
          flush=True)
    sed_only_results = []
    for sb in [0.00, 0.04, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        for rs_thr in [0.86, 0.90, 0.92, 0.94, 0.96, 0.98]:
            for rp_thr in [0.60, 0.70, 0.78, 0.85]:
                p = {**DEFAULTS,
                     "fake_blend": 0.0, "cont_blend": 0.0,
                     "sed_blend": sb, "sed_rs_thr": rs_thr, "sed_rp_thr": rp_thr}
                tag = f"sb={sb:.2f} rs>{rs_thr:.2f} rp<{rp_thr:.2f}"
                sed_only_results.append(run_one(oof, p, tag))

    results.sort(key=lambda r: r["delta"], reverse=True)
    print(f"\n{'tag':32s} {'baseline':>10s} {'gated':>10s} {'delta':>10s}", flush=True)
    for r in results[:15]:
        print(f"{r['label']:32s} {r['baseline']:10.4f} {r['gated']:10.4f} {r['delta']:+10.4f}",
              flush=True)
    print(f"\nbottom 3 (regressors):", flush=True)
    for r in results[-3:]:
        print(f"{r['label']:32s} {r['baseline']:10.4f} {r['gated']:10.4f} {r['delta']:+10.4f}",
              flush=True)

    sed_only_results.sort(key=lambda r: r["delta"], reverse=True)
    print(f"\n{'tag':40s} {'baseline':>10s} {'gated':>10s} {'delta':>10s}", flush=True)
    for r in sed_only_results[:20]:
        print(f"{r['label']:40s} {r['baseline']:10.4f} {r['gated']:10.4f} {r['delta']:+10.4f}",
              flush=True)
    print(f"\nworst 5 in sed-only sweep:", flush=True)
    for r in sed_only_results[-5:]:
        print(f"{r['label']:40s} {r['baseline']:10.4f} {r['gated']:10.4f} {r['delta']:+10.4f}",
              flush=True)


if __name__ == "__main__":
    main()
