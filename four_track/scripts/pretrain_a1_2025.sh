#!/usr/bin/env bash
# L2 — A1 EffNet-B0 pretrain on BirdCLEF-2025 focal audio (10 epochs).
# Per new_plan.md §14.10. Smoke-test first via:
#   python -u src/pretrain_a1_2025.py --epochs 1 --smoke-test
set -euo pipefail

cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate kaggle

python -u src/pretrain_a1_2025.py \
    --epochs 10 \
    --lr 1e-3 \
    --batch-size 64 \
    --val-frac 0.05 \
    --seed 42 \
    --mixstyle-p 0.5
