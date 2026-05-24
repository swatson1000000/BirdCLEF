#!/usr/bin/env bash
# §36 L4-v2 MAIN RUN — fold 4 on skynet (parallel to DT's folds 1,2,3).
#
# Recipe: same as L4-v2 fold-0 (seed 123, CE, no mixstyle, l4v2 pseudo manifest).
# B0-SED on GB10 runs ~2.8× slower than RTX 4080 (per
# reference_b0_sed_skynet_dt_ratio). Fold-0 on DT took 7h 41m; expect
# this skynet fold to land at ~21h 30m.
#
# Ckpt produced: a1_<bb>_fold4_seed123_ce_l4v2.pt
#
# Usage (skynet-local, with kaggle-arch env active):
#   nohup bash scripts/l4v2_fold4_skynet.sh > log/l4v2_fold4_skynet_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -euo pipefail

cd /home/swatson/work/kaggle/BirdCLEF/four_track

START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "=== L4-v2 MAIN RUN fold 4 skynet dispatch started at $START ==="

python -u src/train_a1.py \
    --fold 4 \
    --epochs 25 \
    --loss ce \
    --mixstyle-p 0 \
    --pseudo-manifest data/processed/l4v2_pseudo_manifest.csv \
    --seed 123 \
    --ckpt-suffix _l4v2

echo ""
echo "=== L4-v2 MAIN RUN fold 4 skynet dispatch complete  start=$START  end=$(date '+%Y-%m-%d %H:%M:%S') ==="
