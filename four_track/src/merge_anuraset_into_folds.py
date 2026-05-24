"""Concat AnuraSet supplement into train_folds.csv with fold=-1.

fold=-1 means "always train, never val" — train_a1.py loads
`df[df["fold"] != fold]`, so -1 is included in every fold's training set
but never selected as the held-out val fold (which only iterates 0..4).

The AnuraSet rows do NOT participate in soundscape val (that comes from
train_soundscapes_labels.csv via build_soundscape_val), so fold=-1 keeps
them out of any val computation cleanly.

Idempotent: refuses to re-add if AnuraSet rows already present.
Backs up the original train_folds.csv to train_folds_pre_anuraset.csv
on the first run.

Usage:
    python -u src/merge_anuraset_into_folds.py
"""

import sys
from pathlib import Path

import pandas as pd

HERE       = Path(__file__).resolve().parent
FT_ROOT    = HERE.parent
ROOT       = FT_ROOT.parent
PARENT_SRC = ROOT / "src"

if str(PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(PARENT_SRC))

from config import PROC                                       # noqa: E402

FOLDS  = PROC / "train_folds.csv"
SUPPL  = FT_ROOT / "data" / "processed" / "anuraset_supplement.csv"
BACKUP = PROC / "train_folds_pre_anuraset.csv"


def main() -> None:
    assert FOLDS.exists(), f"missing {FOLDS}"
    assert SUPPL.exists(), f"missing {SUPPL} — run fetch_anuraset.py --phase csv first"

    df = pd.read_csv(FOLDS)
    print(f"  current train_folds.csv: {len(df)} rows", flush=True)
    print(f"  current fold counts:     {df['fold'].value_counts().sort_index().to_dict()}", flush=True)

    if "AnuraSet" in df.get("collection", pd.Series(dtype=object)).astype(str).values:
        print("  [abort] AnuraSet rows already in train_folds.csv — nothing to do", flush=True)
        return

    sup = pd.read_csv(SUPPL)
    print(f"  supplement: {len(sup)} AnuraSet rows", flush=True)

    sup["secondary_labels_parsed"] = "[]"
    sup["fold"] = -1

    missing = [c for c in df.columns if c not in sup.columns]
    extra   = [c for c in sup.columns if c not in df.columns]
    assert not missing, f"supplement missing columns: {missing}"
    assert not extra,   f"supplement has unexpected columns: {extra}"
    sup = sup[df.columns.tolist()]

    if not BACKUP.exists():
        df.to_csv(BACKUP, index=False)
        print(f"  backup → {BACKUP.name} ({BACKUP.stat().st_size/1e6:.1f} MB)", flush=True)
    else:
        print(f"  backup already exists at {BACKUP.name}, leaving it", flush=True)

    out = pd.concat([df, sup], ignore_index=True)
    out.to_csv(FOLDS, index=False)
    print(f"  wrote {len(out)} rows → {FOLDS.name}", flush=True)
    print(f"  new fold counts:        {out['fold'].value_counts().sort_index().to_dict()}", flush=True)
    print(f"  collection counts:      {out['collection'].value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    main()
