#!/usr/bin/env bash
# inat_scan_and_resume.sh — scan iNat manifest, then resume pretrain to e25.
#
# Step 1: scan_inat_manifest.py — drops zero-frame / unreadable .wav rows.
# Step 2: pretrain_inat_sounds.py --resume <e13 best> --epochs 25
#         → loop runs e14..e25 with cosine T_max=25 (steeper LR descent
#         than the original 50-epoch cosine; reaches lr_min by e25).
#
# Dispatched via:
#   runon deepthought bash four_track/scripts/inat_scan_and_resume.sh
#
# Assumes cwd on remote is the runon project root (.../BirdCLEF), so
# `four_track/` is reachable as a relative subdir.

set -euo pipefail

INAT_ROOT=/home/swatson/work/MachineLearning/kaggle/BirdCLEF/four_track/data/external/inat_sounds_2024
RESUME_CKPT=/mnt/mytoshiba/MachineLearning/_runon/BirdCLEF/four_track/models/pretrain_inat/inat_best_tf_efficientnet_b0_ns_jft_in1k.pt

echo "==========================================================="
echo "=== Step 1/2: scan iNat manifest ($(date)) ==="
echo "==========================================================="
python -u four_track/src/scan_inat_manifest.py \
    --inat-root "$INAT_ROOT" \
    --workers 16

echo
echo "==========================================================="
echo "=== Step 2/2: resume iNat pretrain to e25 ($(date)) ==="
echo "==========================================================="
[[ -f "$RESUME_CKPT" ]] || {
    echo "ERROR: resume ckpt not found: $RESUME_CKPT" >&2
    exit 2
}
python -u four_track/src/pretrain_inat_sounds.py \
    --inat-root "$INAT_ROOT" \
    --epochs 25 \
    --resume "$RESUME_CKPT" \
    --start-epoch 13

echo
echo "==========================================================="
echo "=== done $(date) ==="
echo "==========================================================="
