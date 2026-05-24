"""Build matched A2 SED + ProtoSSM OOF on the 59-file substrate (708 rows).

The A2 broader-pool eval has 24 rows per file (each (filename, start_sec)
duplicated). The ProtoSSM OOF has 12 windows per file. Both substrates use
5s windows at offsets 0, 5, ..., 55 over the first 60s of the file.

This script:
  1. Loads A2 5-fold broader-pool OOF (data/a2_a1_5fold_broader_oof.npz)
  2. Loads ProtoSSM 3-seed mean OOF (data/protossm_oof.npz)
  3. Filters A2 to the 59 ProtoSSM-substrate files and dedups start_sec
     (taking first of each duplicate pair)
  4. Reorders both to a canonical (file_id, window) index
  5. Saves data/matched_oof_v75.npz with keys:
       p_sed     (708, 234)    — A2 sig-mean across 5 folds
       p_proto   (708, 234)    — ProtoSSM 3-seed mean
       y_true    (708, 234)
       file_ids  (708,)        — integer per source file (0..58)
       filenames (59,)         — for reference
"""

from pathlib import Path
import numpy as np

FT_ROOT = Path(__file__).resolve().parents[1]
A2_PATH    = FT_ROOT / "data" / "a2_a1_5fold_broader_oof.npz"
PROTO_PATH = FT_ROOT / "data" / "protossm_oof.npz"
OUT_PATH   = FT_ROOT / "data" / "matched_oof_v75.npz"


def main() -> None:
    a2    = np.load(A2_PATH, allow_pickle=True)
    proto = np.load(PROTO_PATH, allow_pickle=True)

    a2_files  = np.array([str(f) for f in a2["filenames"]])
    a2_starts = a2["start_sec"].astype(np.int32)
    a2_sed    = a2["probs_mean"].astype(np.float32)          # (1478, 234) sig-mean
    a2_y      = a2["y_true"].astype(np.float32)

    proto_files = [str(f) for f in proto["file_list"]]
    proto_logit = proto["oof_mean"].astype(np.float32)        # (59, 12, 234) — LOGIT space
    proto_mean  = 1.0 / (1.0 + np.exp(-proto_logit))          # → sigmoid, matches mtoshidesu p_proto scale
    proto_y     = proto["labels"].astype(np.float32)          # (59, 12, 234)

    n_files   = len(proto_files)
    n_windows = proto_mean.shape[1]
    assert n_windows == 12, f"expected 12 windows per file, got {n_windows}"

    n_classes = a2_sed.shape[1]
    assert proto_mean.shape[2] == n_classes, "class count mismatch"

    p_sed_out   = np.zeros((n_files * n_windows, n_classes), dtype=np.float32)
    p_proto_out = np.zeros_like(p_sed_out)
    y_out       = np.zeros_like(p_sed_out)
    file_ids    = np.zeros(n_files * n_windows, dtype=np.int64)

    label_mismatches = 0

    for fi, fname in enumerate(proto_files):
        # Find A2 rows for this file
        a2_mask = a2_files == fname
        if a2_mask.sum() == 0:
            raise SystemExit(f"file {fname} missing from A2 broader-pool OOF")

        starts_here = a2_starts[a2_mask]
        sed_here    = a2_sed[a2_mask]
        y_here      = a2_y[a2_mask]

        # Dedup by start_sec — keep first occurrence
        _, first_idx = np.unique(starts_here, return_index=True)
        if len(first_idx) != n_windows:
            raise SystemExit(
                f"file {fname}: expected {n_windows} unique starts, got {len(first_idx)}"
            )
        order = np.argsort(starts_here[first_idx])
        first_idx = first_idx[order]

        sed_w  = sed_here[first_idx]                       # (12, 234)
        y_w_a2 = y_here[first_idx]                         # (12, 234)

        for wi in range(n_windows):
            row = fi * n_windows + wi
            p_sed_out[row]   = sed_w[wi]
            p_proto_out[row] = proto_mean[fi, wi]
            y_out[row]       = y_w_a2[wi]
            file_ids[row]    = fi

            # Cross-check label consistency
            if not np.array_equal(y_w_a2[wi], proto_y[fi, wi]):
                label_mismatches += 1

    print(f"matched OOF: n_files={n_files}  n_windows={n_windows}  "
          f"rows={p_sed_out.shape[0]}  classes={n_classes}", flush=True)
    print(f"label mismatches (A2 vs ProtoSSM y_true): {label_mismatches}", flush=True)
    print(f"p_sed range:   [{p_sed_out.min():.4f}, {p_sed_out.max():.4f}]  "
          f"mean={p_sed_out.mean():.4f}", flush=True)
    print(f"p_proto range: [{p_proto_out.min():.4f}, {p_proto_out.max():.4f}]  "
          f"mean={p_proto_out.mean():.4f}", flush=True)
    print(f"y_true positives total: {int(y_out.sum())}", flush=True)

    np.savez_compressed(
        OUT_PATH,
        p_sed=p_sed_out,
        p_proto=p_proto_out,
        y_true=y_out,
        file_ids=file_ids,
        filenames=np.array(proto_files, dtype=object),
    )
    print(f"saved → {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
