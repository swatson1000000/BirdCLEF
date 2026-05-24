#!/usr/bin/env bash
# §36 L3-precursor — Multi-recipe SED bag (1-recipe validation).
#
# Trains 4 folds (0-3) of a CE-loss + seed-123 + no-mixstyle recipe.
# Skynet runs fold 4 in parallel (scripts/l3_precursor_skynet.sh).
# Per `feedback_dispatch_5fold_split_4_plus_1`, N=5 dispatches as 4+1.
#
# Recipe contrasts maximally with production A2 (ASL + seed42 + mixstyle=0.5)
# to test whether recipe diversity (vs fold diversity) breaks the within-arch
# fusion ceiling (§32 finding: same-recipe folds cap at +0.001 broader-pool
# rank-mean gain; cross-recipe should gain meaningfully more if the
# diversity-via-loss hypothesis is real).
#
# Ckpt naming: a1_<backbone>_fold{f}_seed123_ce.pt
#
# Usage (via runon, env auto-activated):
#   runon deepthought bash scripts/l3_precursor_dt.sh
set -euo pipefail

cd /home/swatson/work/MachineLearning/_runon/BirdCLEF/four_track

START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "=== L3-precursor DT dispatch started at $START ==="

for FOLD in 0 1 2 3; do
    echo ""
    echo "=========================================="
    echo "=== L3-precursor Fold $FOLD CE seed123  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    python -u src/train_a1.py \
        --fold "$FOLD" \
        --epochs 25 \
        --loss ce \
        --mixstyle-p 0 \
        --pseudo-manifest data/processed/a2_pseudo_manifest.csv \
        --seed 123
done

echo ""
echo "=== L3-precursor DT dispatch complete  start=$START  end=$(date '+%Y-%m-%d %H:%M:%S') ==="
