#!/usr/bin/env bash
# §34 Tier 2 — Multi-seed A2 bagging, DT chunk (8 of 10 runs).
#
# Reproduces the production A2 recipe at seeds 43 and 44, for folds 0-3.
# Skynet runs fold 4 in parallel (scripts/tier2_skynet.sh).
#
# Recipe matches production A2 (train_a1.py with --pseudo-manifest, --loss asl,
# --mixstyle-p 0.5, 25 epochs). Per `feedback_dispatch_5fold_split_4_plus_1`
# and the CLAUDE.md ratio table, 10 runs → 8 DT + 2 skynet.
#
# Usage (via runon, env auto-activated):
#   runon deepthought bash scripts/tier2_dt.sh
set -euo pipefail

cd /home/swatson/work/MachineLearning/_runon/BirdCLEF/four_track

START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "=== Tier 2 DT dispatch started at $START ==="

for SEED in 43 44; do
    for FOLD in 0 1 2 3; do
        echo ""
        echo "=========================================="
        echo "=== Seed $SEED  Fold $FOLD  $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="
        python -u src/train_a1.py \
            --fold "$FOLD" \
            --epochs 25 \
            --loss asl \
            --mixstyle-p 0.5 \
            --pseudo-manifest data/processed/a2_pseudo_manifest.csv \
            --seed "$SEED"
    done
done

echo ""
echo "=== Tier 2 DT dispatch complete  start=$START  end=$(date '+%Y-%m-%d %H:%M:%S') ==="
