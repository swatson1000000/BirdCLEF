"""Cheap PP probe — apply iter934 V17 post-processing techniques to cached A2 OOF.

Tests rank-aware scaling, delta-shift smoothing, and file-level top-K
confidence scaling. All three are pure-numpy transforms that re-rank
across windows / files, so they can affect macro ROC-AUC.

Baseline = A2 5-fold sig-mean ensemble (broader-pool 0.8402 anchor).

This probe operates on already-cached probs — no model inference, no
retraining. If any variant moves broader-pool AUC >= +0.005 vs anchor,
integrate into the production kernel and submit an LB probe.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FT_ROOT = Path(__file__).resolve().parents[1]
OOF_PATH = FT_ROOT / "data" / "a2_a1_5fold_broader_oof.npz"
OUT_JSON = FT_ROOT / "data" / "probe_postproc_iter934_results.json"

A2_ANCHOR = 0.8402


def macro_auc_present(probs: np.ndarray, y_true: np.ndarray) -> float:
    present = y_true.sum(axis=0) > 0
    return float(
        roc_auc_score(y_true[:, present], probs[:, present], average="macro")
    )


def rank_aware_scaling(scores, file_ids, power=0.5):
    """V17 / 2025 Rank 3. scores *= file_max^power, per-file per-class."""
    out = np.zeros_like(scores)
    for fid in np.unique(file_ids):
        mask = file_ids == fid
        s = scores[mask]
        file_max = s.max(axis=0, keepdims=True)
        out[mask] = s * np.power(file_max, power)
    return out


def file_level_confidence_scale(scores, file_ids, top_k=2):
    """V17 / 2025 Rank 1-2. scores *= mean(top_k per-class scores in file)."""
    out = np.zeros_like(scores)
    for fid in np.unique(file_ids):
        mask = file_ids == fid
        s = scores[mask]
        n_rows = s.shape[0]
        k = min(top_k, n_rows)
        sorted_s = np.sort(s, axis=0)
        topk_mean = sorted_s[-k:, :].mean(axis=0, keepdims=True)
        out[mask] = s * topk_mean
    return out


def delta_shift_smooth(scores, file_ids, starts, alpha=0.15):
    """V17 / 2025 Rank 1. new[t] = (1-alpha)*old[t] + 0.5*alpha*(old[t-1]+old[t+1])
    Temporal moving average within each file, using start_sec for ordering.
    """
    out = scores.copy()
    for fid in np.unique(file_ids):
        mask = np.where(file_ids == fid)[0]
        # Order by start_sec within file
        order = np.argsort(starts[mask])
        idx_ordered = mask[order]
        s = scores[idx_ordered]
        n = s.shape[0]
        if n < 3:
            continue
        # Construct prev/next with edge-replication
        prev_s = np.vstack([s[:1], s[:-1]])
        next_s = np.vstack([s[1:], s[-1:]])
        smoothed = (1 - alpha) * s + 0.5 * alpha * (prev_s + next_s)
        out[idx_ordered] = smoothed
    return out


def main():
    print(f"[load] {OOF_PATH}", flush=True)
    d = np.load(OOF_PATH, allow_pickle=True)
    probs_mean = d["probs_mean"].astype(np.float32)  # (1478, 234)
    y_true = d["y_true"].astype(np.float32)           # (1478, 234)
    filenames = d["filenames"]
    starts = d["start_sec"]
    print(f"  shape: probs {probs_mean.shape} y {y_true.shape}", flush=True)

    # Build integer file_id per row
    fname_to_id = {f: i for i, f in enumerate(np.unique(filenames))}
    file_ids = np.array([fname_to_id[f] for f in filenames])
    n_files = len(fname_to_id)
    windows_per_file = pd.Series(file_ids).value_counts()
    print(f"  files: {n_files}  windows/file mean={windows_per_file.mean():.1f} "
          f"min={windows_per_file.min()} max={windows_per_file.max()}", flush=True)

    # Baseline
    base_auc = macro_auc_present(probs_mean, y_true)
    print(f"\n[baseline] A2 5-fold sig-mean broader-pool: {base_auc:.4f}  "
          f"(anchor expected {A2_ANCHOR})", flush=True)

    results = {"baseline_auc": base_auc, "variants": []}

    # --- T1' probes ---

    # Rank-aware scaling
    print("\n=== rank-aware scaling: scores *= file_max^power ===", flush=True)
    for power in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
        scaled = rank_aware_scaling(probs_mean, file_ids, power=power)
        auc = macro_auc_present(scaled, y_true)
        delta = auc - base_auc
        print(f"  power={power:.2f}: AUC={auc:.4f}  Δ={delta:+.4f}", flush=True)
        results["variants"].append(
            {"variant": "rank_aware", "power": power,
             "auc": auc, "delta": delta}
        )

    # File-level top-K
    print("\n=== file-level top-K confidence scale: scores *= mean(top_k) ===",
          flush=True)
    for k in [1, 2, 3, 4]:
        scaled = file_level_confidence_scale(probs_mean, file_ids, top_k=k)
        auc = macro_auc_present(scaled, y_true)
        delta = auc - base_auc
        print(f"  top_k={k}: AUC={auc:.4f}  Δ={delta:+.4f}", flush=True)
        results["variants"].append(
            {"variant": "file_topk", "top_k": k,
             "auc": auc, "delta": delta}
        )

    # Delta shift smoothing
    print("\n=== delta-shift smoothing: temporal moving average ===", flush=True)
    for alpha in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
        scaled = delta_shift_smooth(probs_mean, file_ids, starts, alpha=alpha)
        auc = macro_auc_present(scaled, y_true)
        delta = auc - base_auc
        print(f"  alpha={alpha:.2f}: AUC={auc:.4f}  Δ={delta:+.4f}", flush=True)
        results["variants"].append(
            {"variant": "delta_smooth", "alpha": alpha,
             "auc": auc, "delta": delta}
        )

    # Combined: rank-aware + delta-smooth (the iter934 pipeline order)
    print("\n=== combined: rank-aware → delta-smooth ===", flush=True)
    for power in [0.3, 0.4, 0.5]:
        for alpha in [0.10, 0.15, 0.20]:
            s1 = rank_aware_scaling(probs_mean, file_ids, power=power)
            s2 = delta_shift_smooth(s1, file_ids, starts, alpha=alpha)
            auc = macro_auc_present(s2, y_true)
            delta = auc - base_auc
            print(f"  power={power:.2f} alpha={alpha:.2f}: AUC={auc:.4f}  "
                  f"Δ={delta:+.4f}", flush=True)
            results["variants"].append(
                {"variant": "rank+delta", "power": power, "alpha": alpha,
                 "auc": auc, "delta": delta}
            )

    # Find overall best
    best = max(results["variants"], key=lambda x: x["auc"])
    print("\n" + "=" * 60, flush=True)
    print(f"BEST: {best}", flush=True)
    delta_best = best["auc"] - base_auc
    if delta_best >= 0.005:
        verdict = (
            f"STRONG signal Δ={delta_best:+.4f} >= +0.005 — integrate into "
            f"kernel + LB probe immediately"
        )
    elif delta_best >= 0.001:
        verdict = (
            f"WEAK positive Δ={delta_best:+.4f} — consider integration but "
            f"may not transfer to LB above ±0.005 SE"
        )
    elif delta_best >= -0.001:
        verdict = (
            f"NEUTRAL Δ={delta_best:+.4f} — public 0.934 gain came from "
            f"elsewhere (likely v5_pseudo multi-ckpt SED bag, ResidualSSM, "
            f"or larger ProtoSSM)"
        )
    else:
        verdict = f"REGRESSION Δ={delta_best:+.4f} — don't ship"
    print(f"VERDICT: {verdict}", flush=True)
    print("=" * 60, flush=True)
    results["best"] = best
    results["verdict"] = verdict

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[save] {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
