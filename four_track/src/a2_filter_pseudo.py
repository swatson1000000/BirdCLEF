"""src/a2_filter_pseudo.py — Track A2 step 3: Sydorskyi pseudo-label filter.

Applies the BC2025 2nd-place filter to the full OOF NPZ from step 1:

  1. Keep chunk if max(prob) >= KEEP_THRESH (default 0.5)
  2. Within retained chunks, zero per-class prob < ZERO_THRESH (default 0.1)
  3. Use as SOFT labels, not hard.

The kept rows become Track A2's pseudo substrate. They retain their
`oof_bucket` so the trainer can implement Sydorskyi's fold-safe sampling:
when training fold k, only mix in pseudo rows where bucket == k (those
were predicted by models from folds != k, so fold k's trainer is not
learning from its own predictions).

Inputs:
  data/processed/a2_train_ss_oof_probs.npz  — from a2_emit_oof_pseudo.py

Outputs:
  data/processed/a2_pseudo_soft.npz   — training-side substrate
    filenames           (N,)        <U
    start_sec           (N,)        int32
    oof_bucket          (N,)        int8
    soft_labels         (N, 234)    float32   — zeroed below ZERO_THRESH
    classes_active_count(N,)        int16     — # non-zero per row
  data/processed/a2_pseudo_audit.csv  — per-chunk human-readable preview
    cols: filename, start_sec, oof_bucket, max_prob, n_active,
          top1_species, top1_p, top3_species, top3_probs

Run via:
  python -u src/a2_filter_pseudo.py 2>&1 | tee log/a2_filter_$(date +%Y%m%d_%H%M%S).log
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE       = Path(__file__).resolve().parent
FT_ROOT    = HERE.parent
ROOT       = FT_ROOT.parent
PARENT_SRC = ROOT / "src"

if str(PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(PARENT_SRC))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config  # noqa: E402
from config import RAW, get_species_index  # noqa: E402

NPZ_IN     = FT_ROOT / "data" / "processed" / "a2_train_ss_oof_probs.npz"
NPZ_OUT    = FT_ROOT / "data" / "processed" / "a2_pseudo_soft.npz"
CSV_AUDIT  = FT_ROOT / "data" / "processed" / "a2_pseudo_audit.csv"

KEEP_THRESH = 0.50      # Sydorskyi: chunk kept iff max(prob) >= this
ZERO_THRESH = 0.10      # Sydorskyi: per-class probs below this -> 0


def main() -> None:
    print(f"loading NPZ: {NPZ_IN}", flush=True)
    assert NPZ_IN.exists(), f"NPZ missing: {NPZ_IN} — run a2_emit_oof_pseudo.py first"
    z = np.load(NPZ_IN, allow_pickle=False)
    probs       = z["probs"]                       # (N, 234) float32
    filenames   = z["filenames"].astype(str)       # (N,)
    starts      = z["start_sec"].astype(np.int32)  # (N,)
    buckets     = z["oof_bucket"].astype(np.int8)  # (N,)
    n_classes   = probs.shape[1]
    n_total     = len(probs)
    print(f"  N_total chunks={n_total}  N_classes={n_classes}", flush=True)
    assert n_classes == config.N_CLASSES

    sp2idx = get_species_index()
    idx2sp = {v: k for k, v in sp2idx.items()}

    # ── Stage A: chunk keep rule ─────────────────────────────────────────────
    max_per_chunk = probs.max(axis=1)
    kept_mask = max_per_chunk >= KEEP_THRESH
    n_kept = int(kept_mask.sum())
    print(f"\nstage A — chunk keep @ max>={KEEP_THRESH}:", flush=True)
    print(f"  kept {n_kept}/{n_total} chunks ({n_kept/max(n_total,1)*100:.1f}%)",
          flush=True)
    print(f"  per-bucket kept distribution:", flush=True)
    for b in range(5):
        in_bucket = (buckets == b)
        kept_in_bucket = int((kept_mask & in_bucket).sum())
        total_in_bucket = int(in_bucket.sum())
        pct = kept_in_bucket / max(total_in_bucket, 1) * 100
        print(f"    bucket {b}: {kept_in_bucket}/{total_in_bucket} ({pct:.1f}%)",
              flush=True)

    if n_kept == 0:
        print("\n[FATAL] zero chunks survived stage A — recipe broken or threshold too strict",
              flush=True)
        sys.exit(1)

    # Subset to kept rows
    probs_k    = probs[kept_mask]
    files_k    = filenames[kept_mask]
    starts_k   = starts[kept_mask]
    buckets_k  = buckets[kept_mask]

    # ── Stage B: per-class zero-out ──────────────────────────────────────────
    soft_labels = probs_k.copy()
    soft_labels[soft_labels < ZERO_THRESH] = 0.0
    classes_active = (soft_labels > 0)
    n_active_per_row = classes_active.sum(axis=1).astype(np.int16)

    print(f"\nstage B — per-class zero @ <{ZERO_THRESH}:", flush=True)
    print(f"  active classes per kept chunk: "
          f"min={int(n_active_per_row.min())} "
          f"median={int(np.median(n_active_per_row))} "
          f"max={int(n_active_per_row.max())} "
          f"mean={n_active_per_row.mean():.2f}",
          flush=True)

    # ── Per-class active count + class coverage ──────────────────────────────
    class_active_count = classes_active.sum(axis=0)   # (234,)
    n_classes_with_any = int((class_active_count > 0).sum())
    print(f"\nclass coverage:", flush=True)
    print(f"  classes with >=1 active pseudo row: {n_classes_with_any}/{n_classes}",
          flush=True)
    print(f"  per-class active count: "
          f"min={int(class_active_count.min())} "
          f"median={int(np.median(class_active_count))} "
          f"max={int(class_active_count.max())} "
          f"sum={int(class_active_count.sum())}",
          flush=True)

    # Top-10 over-represented species
    print(f"\ntop-10 species by pseudo-row count:", flush=True)
    top_idx = np.argsort(-class_active_count)[:10]
    for c in top_idx:
        print(f"  {idx2sp.get(int(c), f'cls{c}'):>15}  {int(class_active_count[c]):>6}  "
              f"mean_prob={float(soft_labels[:, c][soft_labels[:, c] > 0].mean() if class_active_count[c] > 0 else 0):.3f}",
              flush=True)

    # Train.csv comparison
    train_csv = RAW / "train.csv"
    if train_csv.exists():
        train_df = pd.read_csv(train_csv)
        train_per_class = train_df["primary_label"].value_counts().to_dict()
        print(f"\nclass-balance comparison (top-5 most-augmented by pseudo):", flush=True)
        # Augmentation factor = pseudo_count / train_count
        aug_factor = []
        for c in range(n_classes):
            sp = idx2sp.get(c, f"cls{c}")
            tc = train_per_class.get(sp, 0)
            pc = int(class_active_count[c])
            if tc > 0 and pc > 0:
                aug_factor.append((sp, pc / tc, pc, tc))
        aug_factor.sort(key=lambda x: -x[1])
        print(f"  {'species':>15}  {'aug_factor':>10}  {'pseudo':>7}  {'train':>7}",
              flush=True)
        for sp, af, pc, tc in aug_factor[:5]:
            print(f"  {sp:>15}  {af:>10.2f}  {pc:>7d}  {tc:>7d}", flush=True)

    # ── Save NPZ ─────────────────────────────────────────────────────────────
    NPZ_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        NPZ_OUT,
        filenames=files_k,
        start_sec=starts_k,
        oof_bucket=buckets_k,
        soft_labels=soft_labels,
        classes_active_count=n_active_per_row,
    )
    sz = NPZ_OUT.stat().st_size / 1e6
    print(f"\nsaved NPZ: {NPZ_OUT} ({sz:.1f} MB)", flush=True)

    # ── Audit CSV ────────────────────────────────────────────────────────────
    rows = []
    for i in range(n_kept):
        row_probs = soft_labels[i]
        # top-3 active classes
        top3 = np.argsort(-row_probs)[:3]
        top3 = [int(c) for c in top3 if row_probs[c] > 0]
        rows.append({
            "filename":      files_k[i],
            "start_sec":     int(starts_k[i]),
            "oof_bucket":    int(buckets_k[i]),
            "max_prob":      float(probs_k[i].max()),
            "n_active":      int(n_active_per_row[i]),
            "top1_species":  idx2sp.get(top3[0], "") if top3 else "",
            "top1_p":        float(row_probs[top3[0]]) if top3 else 0.0,
            "top3_species":  ";".join(idx2sp.get(c, str(c)) for c in top3),
            "top3_probs":    ";".join(f"{row_probs[c]:.3f}" for c in top3),
        })
    audit_df = pd.DataFrame(rows)
    CSV_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(CSV_AUDIT, index=False)
    print(f"saved audit CSV: {CSV_AUDIT} ({len(audit_df)} rows)", flush=True)

    print("\nDONE — next step: src/a2_train.py (overnight retrain)", flush=True)


if __name__ == "__main__":
    main()
