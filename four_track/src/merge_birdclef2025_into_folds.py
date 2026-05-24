"""Merge BirdCLEF 2025 training rows for 2026-overlapping species into train_folds.csv.

Scope (§14.11.6.4 T2.6):
  - Source: data/raw/birdclef_2025/train.csv + train_audio/
  - Filter to species that exist in the 2026 taxonomy (41 of 206 in 2025).
  - Dedupe against existing train_folds.csv by filename basename
    (iNat and XC IDs are globally unique).
  - Assign fold: for authors who already appear in 2026 rows, use their
    existing fold (prevents val leakage where the same author's clips
    would split across train/val). For new authors, GroupKFold(5) by
    author with seed 42.
  - Tag `collection = "BC2025"` so the dataset path router loads from
    data/raw/birdclef_2025/train_audio/ instead of the 2026 tree.

Idempotent: refuses to re-add if BC2025 rows already present.
Backs up train_folds.csv → train_folds_pre_bc2025.csv on first run.

Usage:
    python -u src/merge_birdclef2025_into_folds.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

HERE       = Path(__file__).resolve().parent
FT_ROOT    = HERE.parent
ROOT       = FT_ROOT.parent
PARENT_SRC = ROOT / "src"

if str(PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(PARENT_SRC))

from config import PROC, RAW                                    # noqa: E402

FOLDS         = PROC / "train_folds.csv"
BC2025_ROOT   = RAW / "birdclef_2025"
BC2025_TRAIN  = BC2025_ROOT / "train.csv"
BC2025_TAX    = BC2025_ROOT / "taxonomy.csv"
BC2026_TAX    = RAW / "taxonomy.csv"
BC2025_AUDIO  = BC2025_ROOT / "train_audio"
BACKUP        = PROC / "train_folds_pre_bc2025.csv"

N_FOLDS = 5
SEED    = 42


def _basename(filename: str) -> str:
    return Path(str(filename)).name


def main() -> None:
    assert FOLDS.exists(), f"missing {FOLDS}"
    assert BC2025_TRAIN.exists(), f"missing {BC2025_TRAIN}"
    assert BC2025_TAX.exists(), f"missing {BC2025_TAX}"
    assert BC2026_TAX.exists(), f"missing {BC2026_TAX}"

    df = pd.read_csv(FOLDS)
    print(f"  current train_folds.csv: {len(df)} rows", flush=True)
    print(f"  current fold counts:     {df['fold'].value_counts().sort_index().to_dict()}", flush=True)
    print(f"  current collections:     {df['collection'].value_counts().to_dict()}", flush=True)

    if "BC2025" in df["collection"].astype(str).values:
        print("  [abort] BC2025 rows already in train_folds.csv — nothing to do", flush=True)
        return

    tax_2026 = pd.read_csv(BC2026_TAX)
    bc25     = pd.read_csv(BC2025_TRAIN)
    print(f"  BC2025 train.csv: {len(bc25)} rows, {bc25['primary_label'].nunique()} species", flush=True)

    # --- Species filter: only keep rows whose primary_label is in 2026 ---
    sp_2026 = set(tax_2026["primary_label"].astype(str))
    bc25["primary_label"] = bc25["primary_label"].astype(str)
    bc25 = bc25[bc25["primary_label"].isin(sp_2026)].reset_index(drop=True)
    print(f"  after species filter (2026 overlap): {len(bc25)} rows, "
          f"{bc25['primary_label'].nunique()} species", flush=True)

    # --- Dedupe against existing train_folds by filename basename ---
    existing_basenames = set(df["filename"].astype(str).map(_basename))
    bc25["_basename"] = bc25["filename"].astype(str).map(_basename)
    before_dedupe = len(bc25)
    bc25 = bc25[~bc25["_basename"].isin(existing_basenames)].reset_index(drop=True)
    dropped = before_dedupe - len(bc25)
    print(f"  dedupe by filename basename: dropped {dropped}, kept {len(bc25)}", flush=True)

    # --- Verify audio files exist on disk ---
    missing_audio = []
    for fn in bc25["filename"].astype(str).tolist():
        if not (BC2025_AUDIO / fn).exists():
            missing_audio.append(fn)
    if missing_audio:
        print(f"  [warn] {len(missing_audio)} rows have missing audio files — dropping them",
              flush=True)
        bc25 = bc25[~bc25["filename"].astype(str).isin(missing_audio)].reset_index(drop=True)
    print(f"  after audio existence check: {len(bc25)} rows", flush=True)

    # --- Fill schema columns to match 2026 train_folds ---
    tax_idx = tax_2026.set_index("primary_label")
    tax_idx.index = tax_idx.index.astype(str)
    bc25["inat_taxon_id"] = bc25["primary_label"].map(tax_idx["inat_taxon_id"])
    bc25["class_name"]    = bc25["primary_label"].map(tax_idx["class_name"])
    bc25["secondary_labels_parsed"] = "[]"

    # collection: overwrite with "BC2025" so dataset_a1.py routes to the 2025 tree.
    # (Preserve original collection in `collection_origin` for audit.)
    bc25["collection_origin"] = bc25["collection"].astype(str)
    bc25["collection"]        = "BC2025"

    # --- Fold assignment: match existing author folds; new authors → GroupKFold ---
    df["author"]   = df["author"].astype(str)
    bc25["author"] = bc25["author"].astype(str)

    # Build author → fold map from 2026 rows with fold in {0..4}
    existing = df[df["fold"].isin(range(N_FOLDS))]
    # Most-common fold per author handles the rare case where same author has
    # multiple folds (shouldn't happen with GroupKFold but defensive).
    author_fold_map = (
        existing.groupby("author")["fold"]
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )
    print(f"  2026 author→fold map: {len(author_fold_map)} authors", flush=True)

    bc25["fold"] = bc25["author"].map(author_fold_map)
    known_mask   = bc25["fold"].notna()
    print(f"  BC2025 rows with known author: {int(known_mask.sum())} / {len(bc25)}",
          flush=True)

    # GroupKFold on the unknown-author subset, stable assignment
    unknown = bc25[~known_mask].reset_index(drop=True)
    if len(unknown) > 0:
        n_groups = unknown["author"].nunique()
        n_splits = min(N_FOLDS, max(2, n_groups))
        gkf = GroupKFold(n_splits=n_splits)
        unknown_folds = np.full(len(unknown), -1, dtype=int)
        for fi, (_, val_idx) in enumerate(
            gkf.split(unknown, groups=unknown["author"].values)
        ):
            unknown_folds[val_idx] = fi
        # Map n_splits → N_FOLDS if mismatched (wrap around; rare)
        if n_splits < N_FOLDS:
            unknown_folds = unknown_folds % N_FOLDS
        bc25.loc[~known_mask, "fold"] = unknown_folds
    bc25["fold"] = bc25["fold"].astype(int)

    print(f"  BC2025 new fold distribution: {bc25['fold'].value_counts().sort_index().to_dict()}",
          flush=True)

    # --- Align columns to df schema ---
    bc25 = bc25.drop(columns=["_basename"], errors="ignore")
    missing_cols = [c for c in df.columns if c not in bc25.columns]
    extra_cols   = [c for c in bc25.columns if c not in df.columns]
    if missing_cols:
        for c in missing_cols:
            bc25[c] = pd.NA
    if extra_cols:
        print(f"  [info] dropping extra BC2025 columns: {extra_cols}", flush=True)
        bc25 = bc25.drop(columns=extra_cols)
    bc25 = bc25[df.columns.tolist()]

    # --- Backup + write ---
    if not BACKUP.exists():
        df.to_csv(BACKUP, index=False)
        print(f"  backup → {BACKUP.name} ({BACKUP.stat().st_size/1e6:.1f} MB)", flush=True)
    else:
        print(f"  backup already exists at {BACKUP.name}, leaving it", flush=True)

    out = pd.concat([df, bc25], ignore_index=True)
    out.to_csv(FOLDS, index=False)

    print(f"\n  wrote {len(out)} rows → {FOLDS.name}  (+{len(bc25)})", flush=True)
    print(f"  new fold counts:        {out['fold'].value_counts().sort_index().to_dict()}",
          flush=True)
    print(f"  new collection counts:  {out['collection'].value_counts().to_dict()}",
          flush=True)


if __name__ == "__main__":
    main()
