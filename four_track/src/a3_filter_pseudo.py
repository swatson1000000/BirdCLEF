"""Thin wrapper: run a2_filter_pseudo with A3 paths + tightened threshold.

A3-recursive (A2-as-teacher) — see new_plan.md §30. A2-derived pseudo-labels
gate-failed P@0.5 (0.633 < 0.70 required) but cleared P@0.6 (0.822). This
wrapper bumps KEEP_THRESH 0.5 → 0.6 and rebinds I/O paths; zero edits to
the A2 script.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import a2_filter_pseudo as flt  # noqa: E402

FT_ROOT = HERE.parent
flt.NPZ_IN = FT_ROOT / "data" / "processed" / "a3_train_ss_oof_probs.npz"
flt.NPZ_OUT = FT_ROOT / "data" / "processed" / "a3_pseudo_soft.npz"
flt.CSV_AUDIT = FT_ROOT / "data" / "processed" / "a3_pseudo_audit.csv"
flt.KEEP_THRESH = 0.60  # tightened from BC2025-default 0.50; calibration §30

if __name__ == "__main__":
    flt.main()
