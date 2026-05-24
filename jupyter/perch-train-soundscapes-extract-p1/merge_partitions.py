"""Merge 4 Perch-extraction partitions into a single dataset staging dir.

Reads:  /tmp/perch_ss_extract_v{0,1,2,3}/full_train_soundscapes_{perch,meta}_p{0,1,2,3}of4.{npz,parquet}
Writes: kaggle_datasets/train-soundscapes-perch/{full_train_soundscapes_perch.npz,
                                                  full_train_soundscapes_meta.parquet,
                                                  dataset-metadata.json}

Verifies shape, dtype, row-count alignment, and unique filename count ≥10,500.
Exits non-zero on any mismatch so the autopilot bails before upload.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PARTITION_COUNT = 4
PART_DIRS = [Path(f"/tmp/perch_ss_extract_v{i}") for i in range(PARTITION_COUNT)]
STAGE = Path("/home/swatson/work/MachineLearning/kaggle/BirdCLEF/kaggle_datasets/train-soundscapes-perch")


def main() -> int:
    STAGE.mkdir(parents=True, exist_ok=True)

    embs, scores, metas = [], [], []
    for part_id, d in enumerate(PART_DIRS):
        npz_path = d / f"full_train_soundscapes_perch_p{part_id}of{PARTITION_COUNT}.npz"
        meta_path = d / f"full_train_soundscapes_meta_p{part_id}of{PARTITION_COUNT}.parquet"
        if not npz_path.exists() or not meta_path.exists():
            print(f"MISSING partition {part_id}: {npz_path.exists()=} {meta_path.exists()=}", file=sys.stderr)
            return 1
        print(f"loading {npz_path.name}")
        arr = np.load(npz_path)
        emb = arr["emb"]
        sc = arr["scores"]
        meta = pd.read_parquet(meta_path)
        print(f"  emb={emb.shape} scores={sc.shape} meta={len(meta)}")
        if not (emb.shape[0] == sc.shape[0] == len(meta)):
            print(f"ROW MISMATCH partition {part_id}", file=sys.stderr)
            return 2
        embs.append(emb)
        scores.append(sc)
        metas.append(meta)

    emb_full = np.concatenate(embs, axis=0)
    scores_full = np.concatenate(scores, axis=0)
    meta_full = pd.concat(metas, ignore_index=True)
    print(f"\nMerged: emb={emb_full.shape} scores={scores_full.shape} meta={len(meta_full)}")

    if not (emb_full.shape[0] == scores_full.shape[0] == len(meta_full)):
        print("MERGED ROW MISMATCH", file=sys.stderr)
        return 3
    if emb_full.dtype != np.float32 or emb_full.shape[1] != 1536:
        print(f"EMB SCHEMA FAIL: dtype={emb_full.dtype} shape={emb_full.shape}", file=sys.stderr)
        return 4
    if scores_full.dtype != np.float32 or scores_full.shape[1] != 234:
        print(f"SCORES SCHEMA FAIL: dtype={scores_full.dtype} shape={scores_full.shape}", file=sys.stderr)
        return 5

    n_files = meta_full["filename"].nunique()
    print(f"unique filenames: {n_files}")
    if n_files < 10500:
        print(f"FILE COUNT BELOW GATE: {n_files} < 10500", file=sys.stderr)
        return 6

    out_npz = STAGE / "full_train_soundscapes_perch.npz"
    out_meta = STAGE / "full_train_soundscapes_meta.parquet"
    print(f"\nwriting {out_npz}")
    np.savez_compressed(out_npz, emb=emb_full, scores=scores_full)
    print(f"writing {out_meta}")
    meta_full.to_parquet(out_meta, index=False)

    dataset_meta = {
        "title": "BirdCLEF 2026 train_soundscapes Perch features",
        "id": "stevewatson999/birdclef-2026-train-soundscapes-perch",
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(STAGE / "dataset-metadata.json", "w") as f:
        json.dump(dataset_meta, f, indent=2)

    print(f"\ndone. stage dir: {STAGE}")
    print(f"  {out_npz.stat().st_size / 1e6:.1f} MB NPZ")
    print(f"  {out_meta.stat().st_size / 1e6:.2f} MB parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
