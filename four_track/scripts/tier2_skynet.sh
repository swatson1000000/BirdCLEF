#!/usr/bin/env bash
# §34 Tier 2 — Multi-seed A2 bagging, skynet chunk (2 of 10 runs).
#
# Trains fold 4 only, at seeds 43 and 44. DT handles seeds 43, 44 × folds 0-3
# in parallel (scripts/tier2_dt.sh).
#
# Recipe matches production A2 (train_a1.py with --pseudo-manifest, --loss asl,
# --mixstyle-p 0.5, 25 epochs).
#
# Usage:
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle-arch
#   cd /home/swatson/work/kaggle/BirdCLEF/four_track
#   nohup bash scripts/tier2_skynet.sh \
#     > log/tier2_skynet_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -euo pipefail

cd /home/swatson/work/kaggle/BirdCLEF/four_track

START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "=== Tier 2 skynet dispatch started at $START ==="

for SEED in 43 44; do
    echo ""
    echo "=========================================="
    echo "=== Seed $SEED  Fold 4  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    python -u src/train_a1.py \
        --fold 4 \
        --epochs 25 \
        --loss asl \
        --mixstyle-p 0.5 \
        --pseudo-manifest data/processed/a2_pseudo_manifest.csv \
        --seed "$SEED"
done

echo ""
echo "=== Tier 2 skynet dispatch complete  start=$START  end=$(date '+%Y-%m-%d %H:%M:%S') ==="
