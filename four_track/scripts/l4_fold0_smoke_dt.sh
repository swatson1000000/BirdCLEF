#!/usr/bin/env bash
# §36 L4 fold-0 smoke — pseudo round-2 with L3-prec as teacher.
#
# Recipe: identical to L3-prec (CE+seed123+no-mixstyle), only the pseudo
# manifest changes from a2_pseudo_manifest.csv (A1 teacher) to
# l4_pseudo_manifest_t070.csv (L3-prec teacher @ threshold 0.7,
# precision-matched to A2's A1@0.5 — see §35).
#
# Ckpt naming: a1_<backbone>_fold0_seed123_ce_l4.pt (via --ckpt-suffix _l4).
#
# Gate: broader-pool fold-0 ≥ +0.010 over L3-prec fold-0 (0.8596) → 0.8696.
# Below gate → abort L4. Above gate → commit to full 5-fold.
#
# Usage (via runon, env auto-activated):
#   runon deepthought bash scripts/l4_fold0_smoke_dt.sh
set -euo pipefail

cd /home/swatson/work/MachineLearning/_runon/BirdCLEF/four_track

START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "=== L4 fold-0 smoke DT dispatch started at $START ==="

python -u src/train_a1.py \
    --fold 0 \
    --epochs 25 \
    --loss ce \
    --mixstyle-p 0 \
    --pseudo-manifest data/processed/l4_pseudo_manifest_t070.csv \
    --seed 123 \
    --ckpt-suffix _l4

echo ""
echo "=== L4 fold-0 smoke DT dispatch complete  start=$START  end=$(date '+%Y-%m-%d %H:%M:%S') ==="
