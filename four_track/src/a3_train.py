"""Thin wrapper: run a2_train with A3 paths.

A3-recursive (A2-as-teacher, top-K-rescued pseudos) — see new_plan.md §30.
Rebinds A2_MODELS_DIR → models/a3/ and DEFAULT_PSEUDO_NPZ → A3 path before
calling a2_train.main(); zero edits to the A2 script.

Usage (same args as a2_train.py):
  python -u src/a3_train.py --folds 0,1,2,3,4 --epochs 25 --pseudo-ratio 0.4

Ckpts land at models/a3/a2_{backbone}_fold{f}_seed{seed}_{loss}.pt
(filename retains 'a2_' prefix from the underlying script; only directory differs).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import a2_train as t  # noqa: E402

FT_ROOT = HERE.parent
t.A2_MODELS_DIR = FT_ROOT / "models" / "a3"
t.DEFAULT_PSEUDO_NPZ = FT_ROOT / "data" / "processed" / "a3_pseudo_soft_topk.npz"

if __name__ == "__main__":
    t.main()
