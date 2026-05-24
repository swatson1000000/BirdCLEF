"""Thin wrapper: run eval_a2_broader_oof with A3 ckpts.

A3-recursive (A2-as-teacher, top-K-rescued pseudos) — see new_plan.md §30.
Rebinds CKPT_DIR → models/a3/ and OUT_PATH before calling main(); zero
edits to the A2 eval script. Same fold layout, same filename pattern
(a2_train.py kept the 'a2_' prefix in saved ckpts).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import eval_a2_broader_oof as ev  # noqa: E402

FT_ROOT = HERE.parent
ev.CKPT_DIR = FT_ROOT / "models" / "a3"
ev.CKPT_NAME_FMT = "a2_tf_efficientnet_b0.ns_jft_in1k_fold{f}_seed42_asl.pt"
ev.OUT_PATH = FT_ROOT / "data" / "a3_5fold_broader_oof.npz"
# A2's anchor (0.8402) — the gate that A3 must clear to justify a slot.
ev.V4_ANCHOR_AUC = 0.8402
ev.GATE_DELTA = 0.05
ev.GATE_AUC = ev.V4_ANCHOR_AUC + ev.GATE_DELTA  # 0.8902

if __name__ == "__main__":
    raise SystemExit(ev.main())
