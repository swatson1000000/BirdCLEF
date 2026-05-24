#!/usr/bin/env bash
# §36 L4-v2 fold-0 smoke — pseudo round-2 with COMBINED teacher (DT dispatch).
#
# Recipe: identical to L4-v1 (CE+seed123+no-mixstyle), but the pseudo
# manifest swaps from l4_pseudo_manifest_t070.csv (L3-prec teacher alone)
# to l4v2_pseudo_manifest.csv (rank-mean fusion of L3-prec + ProtoSSM,
# w_L3=0.60, threshold τ=0.9667 → 2.44 positives/window).
#
# Ckpt naming: a1_<backbone>_fold0_seed123_ce_l4v2.pt (via --ckpt-suffix _l4v2).
#
# Gate: broader-pool fold-0 ≥ 0.8696 (L3-prec fold-0 0.8596 + 0.010).
#
# Caveat: 60.8% of L4-v2 pseudo positives come from 159 classes ProtoSSM
# never saw in training. The +0.0122 teacher OOF gain was measured on 71
# in-val classes only; realized student gain may be smaller.
#
# Usage (via runon, env auto-activated, only after L4-v1 finishes on DT):
#   runon deepthought bash scripts/l4v2_fold0_smoke_dt.sh
set -euo pipefail

cd /home/swatson/work/MachineLearning/_runon/BirdCLEF/four_track

START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "=== L4-v2 fold-0 smoke DT dispatch started at $START ==="

python -u src/train_a1.py \
    --fold 0 \
    --epochs 25 \
    --loss ce \
    --mixstyle-p 0 \
    --pseudo-manifest data/processed/l4v2_pseudo_manifest.csv \
    --seed 123 \
    --ckpt-suffix _l4v2

echo ""
echo "=== L4-v2 fold-0 smoke DT dispatch complete  start=$START  end=$(date '+%Y-%m-%d %H:%M:%S') ==="
