"""Thin wrapper: run a2_train with cross-arch backbones, save to models/a5_xarch/.

A5 cross-arch probe (see new_plan.md §31): test if a backbone from a
different architecture family — MobileViT-S, ConvNeXt-Pico, etc. —
gates broader-pool ≥ A1's 0.7775 anchor at single-fold. If yes, rank-fuse
with A2 ensemble in production (per the §29 cross-arch rank-mean
finding: +0.0227 broader-pool gain on AST × A2; AST couldn't ship due
to CPU cost, but a smaller cross-arch backbone should).

Reuses the A2 trainer (focal + A1-teacher pseudos at pseudo_ratio=0.4)
unchanged except for save dir and default --backbone. Pass --backbone
explicitly per invocation.

Usage:
  python -u src/a5_train_xarch.py --fold 0 --epochs 25 --loss asl \\
      --backbone mobilevit_s.cvnets_in1k
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import a2_train as t  # noqa: E402

FT_ROOT = HERE.parent
t.A2_MODELS_DIR = FT_ROOT / "models" / "a5_xarch"
# Default to canonical A2 pseudo-labels (A1 teacher); user may override
# with --pseudo-npz to test recipe variants.
t.DEFAULT_PSEUDO_NPZ = FT_ROOT / "data" / "processed" / "a2_pseudo_soft.npz"

if __name__ == "__main__":
    t.main()
