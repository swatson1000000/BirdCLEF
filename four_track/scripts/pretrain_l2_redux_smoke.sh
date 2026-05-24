#!/usr/bin/env bash
# L2-redux Phase 2b — 5-epoch smoke pretrain on BC-historic focal Aves.
# Per new_plan.md §14.17.15.7.2.
#
# Sources: BC2023 + BC2024 + BC2025 train_audio (~70K clips / ~636 species).
# Goal: validate the L2-redux pipeline end-to-end before committing to
#   Phase 2c (XC bulk download, 3-7 days). Subsequent encoder probe on
#   BC2026 val_v2 gates Phase 2c launch.
#
# Launch (per four_track/CLAUDE.md):
#   cd /home/swatson/work/kaggle/BirdCLEF/four_track
#   rm -f log/*.log
#   nohup bash scripts/pretrain_l2_redux_smoke.sh \
#       > log/pretrain_l2_redux_smoke_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate kaggle

python -u src/pretrain_l2_redux.py \
    --epochs 5 \
    --lr 2.5e-4 \
    --lr-min 1e-6 \
    --batch-size 64 \
    --val-frac 0.05 \
    --seed 42 \
    --mixstyle-p 0.5 \
    --focal-gamma 2.0
