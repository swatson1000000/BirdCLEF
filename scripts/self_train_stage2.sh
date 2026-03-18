#!/bin/bash
# Self-training Stage 2: 5 folds × 30 epochs, EfficientNet-B0, seed 42
# Warm-starts from Stage 1 checkpoints, mixes focal clips with pseudo-labels v1.
#
# Run with:
#   nohup bash /home/swatson/work/MachineLearning/kaggle/BirdCLEF/scripts/self_train_stage2.sh \
#     > /home/swatson/work/MachineLearning/kaggle/BirdCLEF/log/self_train_stage2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -e

ROOT=/home/swatson/work/MachineLearning/kaggle/BirdCLEF
PYTHON=/home/swatson/miniconda3/envs/kaggle/bin/python
BACKBONE="tf_efficientnet_b0.ns_jft_in1k"
EPOCHS=30
SEED=42
VERSION=1
PSEUDO_CSV="$ROOT/data/processed/pseudo_labels_v1.csv"

cd "$ROOT"

rm -f "$ROOT/log/self_train_fold"*.log

echo "========================================="
echo "BirdCLEF+ 2026 — Self-Training Stage 2"
echo "Backbone : $BACKBONE"
echo "Epochs   : $EPOCHS   Seed: $SEED"
echo "Pseudo   : $PSEUDO_CSV"
echo "Started  : $(date)"
echo "========================================="

for FOLD in 0 1 2 3 4; do
    INIT_CKPT="$ROOT/models/stage1/sed_${BACKBONE}_fold${FOLD}_seed${SEED}.pt"
    echo ""
    echo "----- Fold $FOLD / 4 -----"
    "$PYTHON" -u "$ROOT/src/self_train.py" \
        --fold         "$FOLD" \
        --backbone     "$BACKBONE" \
        --epochs       "$EPOCHS" \
        --seed         "$SEED" \
        --pseudo-csv   "$PSEUDO_CSV" \
        --pseudo-power 1.0 \
        --init-ckpt    "$INIT_CKPT" \
        --version      "$VERSION" \
        2>&1 | tee "$ROOT/log/self_train_fold${FOLD}_$(date +%Y%m%d_%H%M%S).log"
done

echo ""
echo "========================================="
echo "Self-training Stage 2 complete: $(date)"
echo "========================================="
