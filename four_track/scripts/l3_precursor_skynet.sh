#!/usr/bin/env bash
# §36 L3-precursor — Multi-recipe SED bag, skynet chunk (fold 4 only).
#
# CE loss + seed 123 + no mixstyle, otherwise identical to A2 production recipe.
#
# Ckpt naming: a1_<backbone>_fold4_seed123_ce.pt
set -euo pipefail

cd /home/swatson/work/kaggle/BirdCLEF/four_track

START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "=== L3-precursor skynet dispatch started at $START ==="

echo "=========================================="
echo "=== L3-precursor Fold 4 CE seed123  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
python -u src/train_a1.py \
    --fold 4 \
    --epochs 25 \
    --loss ce \
    --mixstyle-p 0 \
    --pseudo-manifest data/processed/a2_pseudo_manifest.csv \
    --seed 123

echo ""
echo "=== L3-precursor skynet dispatch complete  start=$START  end=$(date '+%Y-%m-%d %H:%M:%S') ==="
