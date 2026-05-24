#!/usr/bin/env bash
# Track D — Multi-seed ensemble (new_plan.md §14.12 D step).
#
# Trains fold-0 with two new seeds (1337, 2026) using the v56 hybrid recipe.
# Seed 42 is reused from the existing local checkpoint
# `models/a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_hybrid.pt`
# to save ~1.5h GPU.
#
# Usage:
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle
#   cd /home/swatson/work/kaggle/BirdCLEF/four_track
#   rm -f log/*.log
#   nohup bash scripts/d_multi_seed_train.sh \
#     > log/d_multi_seed_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -euo pipefail

cd /home/swatson/work/kaggle/BirdCLEF/four_track

for SEED in 1337 2026; do
    echo ""
    echo "=========================================="
    echo "=== Seed $SEED  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    python -u src/train_a1.py \
        --folds 0 \
        --epochs 25 \
        --loss hybrid \
        --mixstyle-p 0.5 \
        --seed "$SEED"
done

echo ""
echo "=========================================="
echo "=== All seeds complete  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
