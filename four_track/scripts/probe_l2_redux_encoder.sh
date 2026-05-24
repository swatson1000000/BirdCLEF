#!/usr/bin/env bash
# L2-redux Phase 2b gate — encoder linear probe (paired runs).
# Per new_plan.md §14.17.15.7.2.
#
# Runs two head-only fine-tunes back-to-back:
#   1. ImageNet baseline (no --init-from)
#   2. L2-redux init     (--init-from models/l2_redux/l2_redux_best.pt)
# Both train the BirdSEDModelA1 head (cls_conv + att_conv) for 5 epochs
# on BC2026 fold-0 train data with frozen backbone, eval val_v2 macro-AUC.
#
# Decision rule: l2redux − imagenet ≥ 0.01 → green-light Phase 2c XC bulk.
# At end, the script prints the Δ for fast scanning.
#
# Launch (per four_track/CLAUDE.md):
#   cd /home/swatson/work/kaggle/BirdCLEF/four_track
#   rm -f log/*.log
#   nohup bash scripts/probe_l2_redux_encoder.sh \
#       > log/probe_l2_redux_encoder_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate kaggle

L2_CKPT=models/l2_redux/l2_redux_best.pt
if [ ! -f "$L2_CKPT" ]; then
    echo "ERROR: $L2_CKPT not found — run scripts/pretrain_l2_redux_smoke.sh first."
    exit 1
fi

echo "==== Probe 1/2: ImageNet baseline ===="
python -u src/probe_l2_redux_encoder.py \
    --tag imagenet \
    --epochs 5 \
    --lr 1e-3 \
    --batch-size 64 \
    --seed 42

echo
echo "==== Probe 2/2: L2-redux init ===="
python -u src/probe_l2_redux_encoder.py \
    --tag l2redux \
    --init-from "$L2_CKPT" \
    --epochs 5 \
    --lr 1e-3 \
    --batch-size 64 \
    --seed 42

echo
echo "==== Phase 2b gate verdict ===="
python - <<'PY'
import json
from pathlib import Path
ROOT = Path("models/l2_redux")
imagenet = json.loads((ROOT / "probe_imagenet_log.json").read_text())
l2redux  = json.loads((ROOT / "probe_l2redux_log.json").read_text())
delta = l2redux["best_val_v2"] - imagenet["best_val_v2"]
print(f"  ImageNet  best val_v2 AUC: {imagenet['best_val_v2']:.4f}")
print(f"  L2-redux  best val_v2 AUC: {l2redux['best_val_v2']:.4f}")
print(f"  Δ (l2redux − imagenet)  : {delta:+.4f}")
gate = 0.01
if delta >= gate:
    print(f"  ✅ PASS (Δ ≥ {gate}) — green-light Phase 2c XC bulk download")
else:
    print(f"  ❌ FAIL (Δ < {gate}) — encoder transfer below noise; reconsider Phase 2c")
PY
