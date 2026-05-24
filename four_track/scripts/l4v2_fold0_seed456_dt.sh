#!/usr/bin/env bash
# §36 L4-v2 fold-0 SANITY CHECK — same recipe as l4v2_fold0_smoke_dt.sh
# but with seed 456 instead of 123. Confirms the 0.9253 broader-pool
# fold-0 isn't a freak seed result before committing to 5-fold dispatch.
#
# Ckpt naming: a1_<backbone>_fold0_seed456_ce_l4v2.pt
#
# Pass criterion: broader-pool fold-0 within ±0.010 of seed123's 0.9253
# (i.e. 0.9153 — 0.9353 inclusive).
#
# Usage (via runon, env auto-activated):
#   runon deepthought bash scripts/l4v2_fold0_seed456_dt.sh
set -euo pipefail

cd /home/swatson/work/MachineLearning/_runon/BirdCLEF/four_track

START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "=== L4-v2 fold-0 SEED-456 sanity DT dispatch started at $START ==="

python -u src/train_a1.py \
    --fold 0 \
    --epochs 25 \
    --loss ce \
    --mixstyle-p 0 \
    --pseudo-manifest data/processed/l4v2_pseudo_manifest.csv \
    --seed 456 \
    --ckpt-suffix _l4v2

echo ""
echo "=== L4-v2 fold-0 SEED-456 sanity DT dispatch complete  start=$START  end=$(date '+%Y-%m-%d %H:%M:%S') ==="
