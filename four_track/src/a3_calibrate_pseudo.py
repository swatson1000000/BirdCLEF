"""Thin wrapper: run a2_calibrate_pseudo with A3 paths.

A3-recursive (A2-as-teacher) — see new_plan.md §30. Rebinds NPZ_PATH and
OUT_CSV before calling the original main(); zero edits to the A2 script.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import a2_calibrate_pseudo as cal  # noqa: E402

FT_ROOT = HERE.parent
cal.NPZ_PATH = FT_ROOT / "data" / "processed" / "a3_train_ss_oof_probs.npz"
cal.OUT_CSV = FT_ROOT / "data" / "processed" / "a3_calibration_report.csv"

if __name__ == "__main__":
    cal.main()
