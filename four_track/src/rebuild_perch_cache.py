"""Rebuild data/kaggle_perch_cache/ from the 2026-05-21 Kaggle Perch extraction.

The legacy kaggle_perch_cache (built 2026-04-15) covered only 59 of the 66
labeled train_soundscapes. The new merge at
kaggle_datasets/train-soundscapes-perch/ covers all 10658 train_soundscapes
files (66 labeled + 10592 unlabeled).

This rewrites the labeled-only subset (66 files * 12 windows = 792 rows) in
the schema train_protossm_local.py expects:
    emb_full         (N, 1536) float32   — keyed "emb" in merged source
    scores_full_raw  (N, 234)  float32   — keyed "scores" in merged source
    meta             [row_id, filename, site, hour_utc]
        row_id = "<basename_without_ogg>_<end_sec>"
        end_sec = (window_idx + 1) * 5

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle-arch
    python -u src/rebuild_perch_cache.py
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent           # four_track/
BIRDCLEF = PROJECT.parent                                  # BirdCLEF/

MERGED_DIR = PROJECT / "kaggle_datasets" / "train-soundscapes-perch"
RAW_DIR = BIRDCLEF / "data" / "raw"  # config.RAW points here on this machine
CACHE_DIR = PROJECT / "data" / "kaggle_perch_cache"

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")


def main() -> int:
    sys.path.insert(0, str(BIRDCLEF / "src"))
    sys.path.insert(0, str(PROJECT / "src"))
    from config import RAW  # noqa: E402

    labels_csv = Path(RAW) / "train_soundscapes_labels.csv"
    labels_df = pd.read_csv(labels_csv)
    all_labeled_files = sorted(labels_df["filename"].astype(str).unique())
    # Filter to files with FULL 12-window label coverage. train_protossm_local.py
    # requires every meta_full row_id to be present in soundscape_labels; 7 of
    # the 66 labeled files (added post 2026-04) have only partial labels
    # (2-9 windows) and would break the row_id alignment.
    windows_per_file = (
        labels_df.groupby("filename").apply(
            lambda s: len(s[["start", "end"]].drop_duplicates())
        )
    )
    full_labeled = sorted(windows_per_file[windows_per_file == 12].index.tolist())
    partial = sorted(set(all_labeled_files) - set(full_labeled))
    print(f"[labels] {labels_csv.name}: {len(all_labeled_files)} files total, "
          f"{len(full_labeled)} with full 12-window labels, "
          f"{len(partial)} partial (excluded)")
    labeled_files = full_labeled

    emb_full = np.load(MERGED_DIR / "full_train_soundscapes_perch.npz")
    emb = emb_full["emb"]              # (127896, 1536)
    scores = emb_full["scores"]        # (127896, 234)
    meta = pd.read_parquet(MERGED_DIR / "full_train_soundscapes_meta.parquet")
    assert len(meta) == emb.shape[0] == scores.shape[0]
    print(f"[merged] emb={emb.shape} scores={scores.shape} rows={len(meta)} files={meta['filename'].nunique()}")

    keep_mask = meta["filename"].isin(labeled_files).to_numpy()
    n_keep = int(keep_mask.sum())
    n_files_kept = int(meta.loc[keep_mask, "filename"].nunique())
    print(f"[subset] kept rows={n_keep}  files={n_files_kept}")

    if n_files_kept != len(labeled_files):
        miss = set(labeled_files) - set(meta.loc[keep_mask, "filename"].unique())
        print(f"  ERR: missing {len(miss)} labeled file(s) from merged extraction", file=sys.stderr)
        for f in sorted(miss)[:5]:
            print(f"    {f}", file=sys.stderr)
        return 1

    emb_sub = emb[keep_mask].astype(np.float32, copy=False)
    scores_sub = scores[keep_mask].astype(np.float32, copy=False)
    meta_sub = meta.loc[keep_mask, ["filename", "window_idx"]].reset_index(drop=True)

    rows = []
    for i in range(len(meta_sub)):
        fname = str(meta_sub.iloc[i]["filename"])
        widx = int(meta_sub.iloc[i]["window_idx"])
        end_sec = (widx + 1) * 5
        basename = fname[:-4] if fname.endswith(".ogg") else fname
        row_id = f"{basename}_{end_sec}"
        m = FNAME_RE.match(fname)
        site = m.group(2) if m else ""
        hour_utc = int(m.group(4)[:2]) if m else -1
        rows.append((row_id, fname, site, hour_utc))
    out_meta = pd.DataFrame(rows, columns=["row_id", "filename", "site", "hour_utc"])

    rows_per_file = out_meta.groupby("filename").size()
    print(f"[subset] rows-per-file dist: min={rows_per_file.min()} max={rows_per_file.max()} mean={rows_per_file.mean():.1f}")
    if rows_per_file.min() != 12 or rows_per_file.max() != 12:
        print(f"  ERR: not exactly 12 windows per file across subset", file=sys.stderr)
        return 2

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_npz = CACHE_DIR / "full_perch_arrays.npz"
    out_pq = CACHE_DIR / "full_perch_meta.parquet"
    print(f"[write] {out_npz}")
    np.savez_compressed(out_npz, emb_full=emb_sub, scores_full_raw=scores_sub)
    print(f"[write] {out_pq}")
    out_meta.to_parquet(out_pq, index=False)

    print(f"\n[done] cache rebuilt at {CACHE_DIR}")
    print(f"  emb_full       {emb_sub.shape} {emb_sub.dtype}  {out_npz.stat().st_size/1e6:.1f} MB NPZ")
    print(f"  scores_full_raw {scores_sub.shape} {scores_sub.dtype}")
    print(f"  meta            {out_meta.shape}  cols={list(out_meta.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
