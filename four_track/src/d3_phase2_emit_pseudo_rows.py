"""D3 Phase 2: threshold pseudo-labels and emit train_folds rows.

Reads `data/processed/pseudo_bc25ss_probs.npz` (written by Phase 1), picks a
threshold τ (auto-adapted to a sane retention band), and for each retained
(file, window) emits one train_folds row:

  primary_label        = argmax class  (species code)
  secondary_labels     = str(list of other classes > τ)
  secondary_labels_parsed = same
  filename             = BC2025 soundscape filename
  pseudo_window_start  = window index × 5s (in seconds)
  collection           = "BC2025_SS_PSEUDO"
  fold                 = -1   (always in train; never in val)

Side-effects:
  1. Snapshot current `data/processed/train_folds.csv` →
     `data/processed/train_folds_pre_d3.csv`
  2. Write new `data/processed/train_folds.csv` with pseudo rows appended and
     a new `pseudo_window_start` column (NaN for non-pseudo rows).

Diagnostic gate (§14.11.6.6.1):
  Compute the fraction of emitted rows whose primary_label is a val-present
  species (appears in train_soundscapes_labels.csv). If <30% (i.e. >70% of
  the new rows target val-absent species), exit 1 — the T2.6 failure mode.

CLI:
  python -u src/d3_phase2_emit_pseudo_rows.py [--tau 0.5]
      [--target-min-retention 0.03] [--target-max-retention 0.25]
      [--no-swap]   (emit but do not overwrite train_folds.csv)
"""

import argparse
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

from config import PROC, RAW, get_species_index, N_CLASSES  # noqa: E402

PROBS_NPZ        = FT_ROOT / "data" / "processed" / "pseudo_bc25ss_probs.npz"
TRAIN_FOLDS_CSV  = PROC / "train_folds.csv"
SNAPSHOT_CSV     = PROC / "train_folds_pre_d3.csv"
DIAG_TXT         = FT_ROOT / "data" / "d3_phase2_diagnostic.txt"

WINDOW_SEC       = 5
N_WINDOWS        = 12
VAL_ABSENT_KILL  = 0.70   # if >70% pseudo rows target val-absent species, kill


def _load_val_species() -> set:
    val_df = pd.read_csv(RAW / "train_soundscapes_labels.csv")
    sp2idx = get_species_index()
    out = set()
    for pl in val_df["primary_label"]:
        for sp in str(pl).split(";"):
            sp = sp.strip()
            if sp in sp2idx:
                out.add(sp)
    return out


def _pick_tau(max_prob: np.ndarray, tau_init: float,
              lo: float, hi: float) -> float:
    """Adapt τ so retention ∈ [lo, hi]. Clamp τ to [0.3, 0.8]."""
    n = max_prob.size
    tau = float(tau_init)
    kept = float((max_prob > tau).sum()) / n
    if lo <= kept <= hi:
        return tau
    step = 0.05
    # If too loose (too many kept), raise τ
    while kept > hi and tau < 0.8:
        tau = min(0.8, tau + step)
        kept = float((max_prob > tau).sum()) / n
    # If too strict (too few kept), lower τ
    while kept < lo and tau > 0.3:
        tau = max(0.3, tau - step)
        kept = float((max_prob > tau).sum()) / n
    return round(tau, 2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--target-min-retention", type=float, default=0.03)
    p.add_argument("--target-max-retention", type=float, default=0.25)
    p.add_argument("--no-swap", action="store_true",
                   help="Emit rows but do not overwrite train_folds.csv")
    args = p.parse_args()

    assert PROBS_NPZ.exists(), f"missing {PROBS_NPZ} — Phase 1 did not complete"

    print(f"[d3-p2] loading {PROBS_NPZ.name}", flush=True)
    npz = np.load(PROBS_NPZ)
    probs     = npz["probs"]       # (N_FILES, 12, N_CLASSES) float32
    filenames = npz["filenames"]   # (N_FILES,) <U64
    print(f"[d3-p2]   probs.shape={probs.shape}  filenames={len(filenames)}", flush=True)
    assert probs.ndim == 3 and probs.shape[2] == N_CLASSES, probs.shape
    assert probs.shape[1] == N_WINDOWS

    # Print retention stats across several τ (for visibility)
    max_prob = probs.max(axis=2)    # (N_FILES, 12)
    print("[d3-p2] retention stats:", flush=True)
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        k = int((max_prob > t).sum())
        print(f"  tau={t:.2f}  kept {k}/{max_prob.size}  ({100*k/max_prob.size:.2f}%)",
              flush=True)

    tau = _pick_tau(
        max_prob,
        tau_init=args.tau,
        lo=args.target_min_retention,
        hi=args.target_max_retention,
    )
    n_kept_total = int((max_prob > tau).sum())
    retention = n_kept_total / max_prob.size
    print(f"[d3-p2] chosen tau={tau:.2f}  retention={retention:.4f}  "
          f"kept_windows={n_kept_total}", flush=True)

    # Build idx→species mapping
    sp2idx = get_species_index()
    idx2sp = {v: k for k, v in sp2idx.items()}
    val_species = _load_val_species()
    print(f"[d3-p2] val-present species: {len(val_species)}", flush=True)

    # Emit rows
    rows = []
    for fi, fname in enumerate(filenames):
        for wi in range(N_WINDOWS):
            window_probs = probs[fi, wi]
            if window_probs.max() <= tau:
                continue
            argmax = int(window_probs.argmax())
            primary = idx2sp[argmax]
            others = np.where(window_probs > tau)[0]
            sec = [idx2sp[int(j)] for j in others if int(j) != argmax]
            sec_str = str(sec)
            rows.append({
                "primary_label":           primary,
                "secondary_labels":        sec_str,
                "secondary_labels_parsed": sec_str,
                "type":                    "[]",
                "latitude":                np.nan,
                "longitude":               np.nan,
                "scientific_name":         "",
                "common_name":             "",
                "class_name":              "",
                "inat_taxon_id":           np.nan,
                "author":                  "pseudo_bc25ss",
                "license":                 "cc0",
                "rating":                  0.0,
                "url":                     "",
                "filename":                str(fname),
                "collection":              "BC2025_SS_PSEUDO",
                "fold":                    -1,
                "pseudo_window_start":     wi * WINDOW_SEC,
            })

    pseudo_df = pd.DataFrame(rows)
    print(f"[d3-p2] emitted pseudo rows: {len(pseudo_df)}", flush=True)

    # Per-species distribution
    primary_counts = pseudo_df["primary_label"].value_counts()
    val_absent_rows = int(pseudo_df[~pseudo_df["primary_label"].isin(val_species)]
                          .shape[0])
    val_present_rows = int(pseudo_df[pseudo_df["primary_label"].isin(val_species)]
                           .shape[0])
    val_absent_frac = val_absent_rows / max(len(pseudo_df), 1)
    print(f"[d3-p2] primary-label val coverage:", flush=True)
    print(f"  val-present species rows: {val_present_rows}  "
          f"({100*val_present_rows/max(len(pseudo_df),1):.1f}%)", flush=True)
    print(f"  val-absent  species rows: {val_absent_rows}  "
          f"({100*val_absent_frac:.1f}%)", flush=True)
    print(f"[d3-p2] top 15 primary species:", flush=True)
    for sp, n in primary_counts.head(15).items():
        tag = "✓val" if sp in val_species else "✗"
        print(f"  {sp:>10s}  {n:>6d}  [{tag}]", flush=True)
    print(f"[d3-p2] unique primary species: {pseudo_df['primary_label'].nunique()}",
          flush=True)

    # Diagnostic summary to disk
    DIAG_TXT.parent.mkdir(parents=True, exist_ok=True)
    with DIAG_TXT.open("w") as f:
        f.write(f"D3 Phase 2 diagnostic\n")
        f.write(f"=====================\n")
        f.write(f"tau                    : {tau}\n")
        f.write(f"kept_windows           : {n_kept_total}\n")
        f.write(f"retention              : {retention:.4f}\n")
        f.write(f"emitted_pseudo_rows    : {len(pseudo_df)}\n")
        f.write(f"val_present_species_rows : {val_present_rows}\n")
        f.write(f"val_absent_species_rows  : {val_absent_rows}\n")
        f.write(f"val_absent_frac          : {val_absent_frac:.4f}\n")
        f.write(f"unique_primary_species : {pseudo_df['primary_label'].nunique()}\n")
        f.write(f"\nTop 30 primary species:\n")
        for sp, n in primary_counts.head(30).items():
            tag = "val" if sp in val_species else "abs"
            f.write(f"  {sp:>10s}  {n:>6d}  [{tag}]\n")
    print(f"[d3-p2] wrote diagnostic → {DIAG_TXT}", flush=True)

    # Gate check
    if val_absent_frac > VAL_ABSENT_KILL:
        print(f"[d3-p2] GATE FAILED: val-absent fraction "
              f"{val_absent_frac:.2%} > {VAL_ABSENT_KILL:.0%}.",
              flush=True)
        print("[d3-p2] (§14.11.6.6.1 kill rule — D3 not worth retraining)",
              flush=True)
        sys.exit(1)
    print(f"[d3-p2] gate OK: val-absent frac {val_absent_frac:.2%} ≤ "
          f"{VAL_ABSENT_KILL:.0%}", flush=True)

    if args.no_swap:
        print("[d3-p2] --no-swap: leaving train_folds.csv untouched", flush=True)
        return

    # Snapshot + swap
    base_df = pd.read_csv(TRAIN_FOLDS_CSV)
    print(f"[d3-p2] current train_folds.csv: {len(base_df)} rows "
          f"{list(base_df.columns)}", flush=True)

    if "pseudo_window_start" not in base_df.columns:
        base_df["pseudo_window_start"] = np.nan

    # Make sure the pseudo_df has exactly the same columns in the same order
    pseudo_df = pseudo_df.reindex(columns=base_df.columns)

    # Snapshot only if one does not exist already
    if not SNAPSHOT_CSV.exists():
        base_df.to_csv(SNAPSHOT_CSV, index=False)
        print(f"[d3-p2] snapshot → {SNAPSHOT_CSV.name}", flush=True)
    else:
        print(f"[d3-p2] snapshot {SNAPSHOT_CSV.name} already exists — not overwriting",
              flush=True)

    merged = pd.concat([base_df, pseudo_df], ignore_index=True)
    merged.to_csv(TRAIN_FOLDS_CSV, index=False)
    print(f"[d3-p2] wrote {TRAIN_FOLDS_CSV.name}: {len(merged)} rows "
          f"(+{len(pseudo_df)} pseudo)", flush=True)
    print(f"[d3-p2] collection distribution after merge:", flush=True)
    print(merged["collection"].value_counts(dropna=False).to_string(), flush=True)
    print(f"[d3-p2] fold distribution after merge:", flush=True)
    print(merged["fold"].value_counts().sort_index().to_string(), flush=True)


if __name__ == "__main__":
    main()
