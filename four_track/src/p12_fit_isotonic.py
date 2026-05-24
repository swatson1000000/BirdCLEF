"""Fit per-class isotonic calibrators on the v56 soundscape OOF predictions.

Reads `data/v56_soundscape_oof.npz` (emitted by `src/p12_emit_oof.py`),
fits `sklearn.isotonic.IsotonicRegression(out_of_bounds='clip', y_min=0.0,
y_max=1.0)` per class where `y_true[:, c].sum() >= 5`, and pickles a
`dict[int -> IsotonicRegression]` to `data/p12_isotonic_calibrators.pkl`.
Classes below the threshold get `None` (identity passthrough downstream).

Sanity metrics reported on the same val (fit == eval, so this is a
consistency check not a generalization test):
  - Calibrated-class macro AUC before vs after (isotonic is monotone,
    so per-class AUC is rank-invariant — expect ~0 Δ; drift flags a bug).
  - Untouched-class macro AUC before vs after (must be byte-identical).
  - Global "present classes" macro AUC before vs after (matches the
    §14.14.10 baseline 0.7414 on load).

The downstream effect on LB comes from the production notebook's Cell 18
cross-class post-proc (P1 taxon T, P3 rank-power, P11 adaptive smoothing,
P13 top-k mean), which consume per-class magnitudes; the sanity block
here only verifies the calibrator itself is well-formed.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

HERE      = Path(__file__).resolve().parent
FT_ROOT   = HERE.parent
PARENT_SRC = FT_ROOT.parent / "src"
if str(PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(PARENT_SRC))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config  # noqa: E402

OOF_PATH = FT_ROOT / "data" / "v56_soundscape_oof.npz"
OUT_PATH = FT_ROOT / "data" / "p12_isotonic_calibrators.pkl"
MIN_POS  = 5


def macro_auc(y_true: np.ndarray, probs: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    return float(roc_auc_score(y_true[:, mask], probs[:, mask], average="macro"))


def main() -> None:
    t0 = time.time()
    print(f"loading {OOF_PATH}", flush=True)
    d = np.load(OOF_PATH)
    probs_mean = d["probs_mean"].astype(np.float64)  # (N, C)
    y_true     = d["y_true"].astype(np.float64)       # (N, C)
    n_w, n_classes = probs_mean.shape
    assert n_classes == config.N_CLASSES
    print(f"  N_windows={n_w}  N_classes={n_classes}", flush=True)

    pos_per_class = y_true.sum(axis=0).astype(int)
    present_mask  = pos_per_class > 0
    calib_mask    = pos_per_class >= MIN_POS
    untouched_mask = present_mask & ~calib_mask
    print(f"  present classes            : {int(present_mask.sum())}/{n_classes}",
          flush=True)
    print(f"  calibratable (>= {MIN_POS} pos) : {int(calib_mask.sum())}/{n_classes}",
          flush=True)
    print(f"  present but uncalibratable : {int(untouched_mask.sum())}/{n_classes}",
          flush=True)
    print(f"  zero-positive (untouched)  : {int((~present_mask).sum())}/{n_classes}",
          flush=True)

    calibrators: dict[int, IsotonicRegression | None] = {}
    probs_cal = probs_mean.copy()

    t1 = time.time()
    for c in range(n_classes):
        if not calib_mask[c]:
            calibrators[c] = None
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(probs_mean[:, c], y_true[:, c])
        calibrators[c] = iso
        probs_cal[:, c] = iso.predict(probs_mean[:, c])
    print(f"  fit {int(calib_mask.sum())} calibrators in {time.time()-t1:.2f}s",
          flush=True)

    auc_calib_before = macro_auc(y_true, probs_mean, calib_mask)
    auc_calib_after  = macro_auc(y_true, probs_cal,  calib_mask)
    auc_untouched_before = macro_auc(y_true, probs_mean, untouched_mask)
    auc_untouched_after  = macro_auc(y_true, probs_cal,  untouched_mask)
    auc_global_before = macro_auc(y_true, probs_mean, present_mask)
    auc_global_after  = macro_auc(y_true, probs_cal,  present_mask)

    print("\nsanity AUCs (expect ~0 Δ; monotone calib is rank-invariant):", flush=True)
    print(f"  calibrated-class macro AUC : {auc_calib_before:.6f} -> "
          f"{auc_calib_after:.6f}   (Δ={auc_calib_after - auc_calib_before:+.6f})",
          flush=True)
    print(f"  untouched-class macro AUC  : {auc_untouched_before:.6f} -> "
          f"{auc_untouched_after:.6f}   (Δ={auc_untouched_after - auc_untouched_before:+.6f})",
          flush=True)
    print(f"  GLOBAL    macro AUC        : {auc_global_before:.6f} -> "
          f"{auc_global_after:.6f}   (Δ={auc_global_after - auc_global_before:+.6f})",
          flush=True)
    print(f"  (§14.14.10 baseline = 0.7414; untouched slice must be byte-identical)",
          flush=True)

    untouched_drift = abs(auc_untouched_after - auc_untouched_before)
    if untouched_drift > 1e-9:
        print(f"\n  [WARN] untouched-slice AUC drifted by {untouched_drift:.2e} — "
              "calibrator leaked outside calib_mask", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(
            {
                "calibrators":    calibrators,
                "calib_mask":     calib_mask,
                "pos_per_class":  pos_per_class,
                "min_pos":        MIN_POS,
                "n_classes":      n_classes,
                "source":         str(OOF_PATH.name),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    sz = OUT_PATH.stat().st_size / 1e3
    print(f"\nsaved {OUT_PATH} ({sz:.1f} KB) in {time.time()-t0:.1f}s total",
          flush=True)


if __name__ == "__main__":
    main()
