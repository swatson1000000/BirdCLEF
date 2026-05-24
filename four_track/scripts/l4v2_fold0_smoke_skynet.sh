#!/usr/bin/env bash
# §36 L4-v2 fold-0 smoke — pseudo round-2 with COMBINED teacher.
#
# Recipe: identical to L4-v1 (CE+seed123+no-mixstyle), but the pseudo
# manifest swaps from l4_pseudo_manifest_t070.csv (L3-prec teacher alone)
# to l4v2_pseudo_manifest.csv (rank-mean fusion of L3-prec + ProtoSSM,
# w_L3=0.60, threshold τ=0.9667 → 2.44 positives/window matching L3-prec
# @0.7).
#
# Ckpt naming: a1_<backbone>_fold0_seed123_ce_l4v2.pt (via --ckpt-suffix _l4v2).
#
# Gate: broader-pool fold-0 ≥ +0.010 over L3-prec fold-0 (0.8596) → 0.8696.
# Below gate → abort L4-v2. Above gate → commit to full 5-fold.
#
# Caveat (worth checking against the result): 60.8% of L4-v2 pseudo positives
# come from 159 classes ProtoSSM never saw in training (rank-mean dilutes with
# noise on those). The +0.0122 teacher OOF gain was measured on 71 in-val
# classes; full-pseudo training spans all 234, so the realized gain may be
# smaller than the gate suggested.
#
# Skynet dispatch (DT is occupied with L4-v1 fold-0). Skynet:DT speed
# ratio is ~3× for B0-SED — ETA ~13-15h.
set -euo pipefail

cd /home/swatson/work/kaggle/BirdCLEF/four_track

START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "=== L4-v2 fold-0 smoke skynet dispatch started at $START ==="

python -u src/train_a1.py \
    --fold 0 \
    --epochs 25 \
    --loss ce \
    --mixstyle-p 0 \
    --pseudo-manifest data/processed/l4v2_pseudo_manifest.csv \
    --seed 123 \
    --ckpt-suffix _l4v2

echo ""
echo "=== L4-v2 fold-0 smoke skynet dispatch complete  start=$START  end=$(date '+%Y-%m-%d %H:%M:%S') ==="
