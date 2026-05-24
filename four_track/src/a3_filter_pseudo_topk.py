"""A3 step 3 (rescue): top-K filter for A2-derived pseudo-labels.

A2 as a teacher produces diffuse probability mass (median 176 active classes
per chunk at the standard Sydorskyi ZERO_THRESH=0.1), vs A1 which produced
median 4. Diffuse pseudo-labels teach the student to be diffuse — defeats
the point of using a stronger teacher.

This script preserves A2's RANKING signal (which is good — broader-pool
AUC 0.8357) but discards its calibration drift by keeping only the top-K
classes per chunk by A2 probability. K=5 matches A2-as-student sparsity
(median 4, mean 4.6).

Stage A: chunk keep if max(prob) >= KEEP_THRESH (0.6, calibrated 2026-05-16)
Stage B (TOP-K REPLACEMENT): keep top K classes per chunk by prob; zero rest

Inputs:
  data/processed/a3_train_ss_oof_probs.npz

Outputs:
  data/processed/a3_pseudo_soft_topk.npz   same schema as a2_pseudo_soft.npz
  data/processed/a3_pseudo_audit_topk.csv  per-chunk audit
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FT_ROOT = HERE.parent
ROOT = FT_ROOT.parent
PARENT_SRC = ROOT / "src"
if str(PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(PARENT_SRC))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config  # noqa: E402
from config import RAW, get_species_index  # noqa: E402

NPZ_IN = FT_ROOT / "data" / "processed" / "a3_train_ss_oof_probs.npz"
NPZ_OUT = FT_ROOT / "data" / "processed" / "a3_pseudo_soft_topk.npz"
CSV_AUDIT = FT_ROOT / "data" / "processed" / "a3_pseudo_audit_topk.csv"

KEEP_THRESH = 0.60   # chunk-level keep gate (per §30 calibration)
TOP_K = 5            # per-chunk class budget (matches A2-as-student mean 4.6)


def main() -> None:
    print(f"loading NPZ: {NPZ_IN}", flush=True)
    assert NPZ_IN.exists(), f"missing: {NPZ_IN}"
    z = np.load(NPZ_IN, allow_pickle=False)
    probs = z["probs"]
    filenames = z["filenames"].astype(str)
    starts = z["start_sec"].astype(np.int32)
    buckets = z["oof_bucket"].astype(np.int8)
    n_total, n_classes = probs.shape
    print(f"  N_total={n_total}  N_classes={n_classes}", flush=True)
    assert n_classes == config.N_CLASSES

    sp2idx = get_species_index()
    idx2sp = {v: k for k, v in sp2idx.items()}

    # Stage A — chunk keep
    max_per = probs.max(axis=1)
    kept = max_per >= KEEP_THRESH
    n_kept = int(kept.sum())
    print(f"\nstage A — keep @ max>={KEEP_THRESH}: {n_kept}/{n_total} "
          f"({100*n_kept/max(n_total,1):.1f}%)", flush=True)

    probs_k = probs[kept].copy()                              # (N_kept, 234)
    files_k = filenames[kept]
    starts_k = starts[kept]
    buckets_k = buckets[kept]

    # Stage B — top-K per chunk
    # argpartition is O(N*K) — faster than full argsort
    topk_idx = np.argpartition(-probs_k, kth=TOP_K, axis=1)[:, :TOP_K]  # (N_kept, K)
    soft = np.zeros_like(probs_k)
    rows = np.arange(n_kept)[:, None]
    soft[rows, topk_idx] = probs_k[rows, topk_idx]

    n_active = (soft > 0).sum(axis=1).astype(np.int16)
    print(f"\nstage B — TOP-K @ K={TOP_K}:", flush=True)
    print(f"  active per chunk: min={int(n_active.min())} "
          f"median={int(np.median(n_active))} "
          f"max={int(n_active.max())} "
          f"mean={n_active.mean():.2f}", flush=True)
    print(f"  total label rows: {(soft > 0).sum()}", flush=True)
    print(f"  mean active prob (over non-zero entries): "
          f"{soft[soft > 0].mean():.4f}", flush=True)

    # Per-class coverage
    cls_count = (soft > 0).sum(axis=0)
    print(f"\nclass coverage:", flush=True)
    print(f"  classes with >=1 active pseudo row: {int((cls_count > 0).sum())}/{n_classes}",
          flush=True)
    print(f"  per-class active count: min={int(cls_count.min())} "
          f"median={int(np.median(cls_count))} "
          f"max={int(cls_count.max())} "
          f"sum={int(cls_count.sum())}", flush=True)

    print(f"\ntop-10 species by pseudo-row count:", flush=True)
    top_idx = np.argsort(-cls_count)[:10]
    for c in top_idx:
        m = soft[:, c]
        mean_p = float(m[m > 0].mean()) if cls_count[c] > 0 else 0.0
        print(f"  {idx2sp.get(int(c), f'cls{c}'):>15}  "
              f"{int(cls_count[c]):>6}  mean_prob={mean_p:.3f}", flush=True)

    # Save
    NPZ_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        NPZ_OUT,
        filenames=files_k.astype("<U64"),
        start_sec=starts_k,
        oof_bucket=buckets_k,
        soft_labels=soft.astype(np.float32),
        classes_active_count=n_active,
    )
    sz = NPZ_OUT.stat().st_size / 1e6
    print(f"\nsaved → {NPZ_OUT} ({sz:.2f} MB)", flush=True)

    # Audit CSV — top-K classes per chunk for spot-check
    audit_rows = []
    for i in range(min(n_kept, 5000)):  # cap at 5k for sanity
        sorted_classes = np.argsort(-soft[i])[:TOP_K]
        species_list = [idx2sp.get(int(c), f"cls{c}") for c in sorted_classes]
        probs_list = [float(soft[i, c]) for c in sorted_classes]
        audit_rows.append({
            "filename": files_k[i],
            "start_sec": int(starts_k[i]),
            "oof_bucket": int(buckets_k[i]),
            "topk_species": ";".join(species_list),
            "topk_probs": ";".join(f"{p:.3f}" for p in probs_list),
        })
    pd.DataFrame(audit_rows).to_csv(CSV_AUDIT, index=False)
    print(f"audit (first 5k) → {CSV_AUDIT}", flush=True)


if __name__ == "__main__":
    main()
