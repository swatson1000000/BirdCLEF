#!/bin/bash
# Stage 1 training: 5 folds × 15 epochs, EfficientNet-B0, seed 42
# Run with: nohup bash scripts/train_stage1.sh > log/train_stage1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -e

ROOT=/home/swatson/work/MachineLearning/kaggle/BirdCLEF
PYTHON=/home/swatson/miniconda3/envs/kaggle/bin/python

cd "$ROOT"

# Clean previous training logs
rm -f "$ROOT/log/train_fold"*.log

echo "========================================="
echo "BirdCLEF+ 2026 — Stage 1 Training"
echo "Backbone : tf_efficientnet_b0.ns_jft_in1k"
echo "Epochs   : 15   Seed: 42"
echo "Started  : $(date)"
echo "========================================="

for FOLD in 0 1 2 3 4; do
    echo ""
    echo "----- Fold $FOLD / 4 -----"
    "$PYTHON" -u "$ROOT/src/train.py" \
        --fold    "$FOLD" \
        --epochs  15 \
        --seed    42 \
        2>&1 | tee "$ROOT/log/train_fold${FOLD}_$(date +%Y%m%d_%H%M%S).log"
done

echo ""
echo "========================================="
echo "Stage 1 complete: $(date)"
echo "========================================="
