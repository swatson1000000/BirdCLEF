#!/usr/bin/env bash
# §36 L4-v2 MAIN RUN — folds 1, 2, 3 sequential on DT.
#
# Recipe: same as L4-v2 fold-0 (seed 123, CE, no mixstyle, l4v2 pseudo manifest).
# Fold 0 ckpt (broader-pool 0.9253) already exists at:
#   models/a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed123_ce_l4v2.pt
#
# Ckpts produced: a1_<bb>_fold{1,2,3}_seed123_ce_l4v2.pt
#
# Per-fold time on DT (RTX 4080) measured 2026-05-22 fold-0 seed-123: 7h 41m.
# Three folds sequential ETA: ~23h 0m.
#
# Usage (via runon):
#   runon deepthought bash scripts/l4v2_folds123_dt.sh
set -euo pipefail

cd /home/swatson/work/MachineLearning/_runon/BirdCLEF/four_track

START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "=== L4-v2 MAIN RUN folds 1,2,3 DT dispatch started at $START ==="

python -u src/train_a1.py \
    --folds 1,2,3 \
    --epochs 25 \
    --loss ce \
    --mixstyle-p 0 \
    --pseudo-manifest data/processed/l4v2_pseudo_manifest.csv \
    --seed 123 \
    --ckpt-suffix _l4v2

echo ""
echo "=== L4-v2 MAIN RUN folds 1,2,3 DT dispatch complete  start=$START  end=$(date '+%Y-%m-%d %H:%M:%S') ==="
