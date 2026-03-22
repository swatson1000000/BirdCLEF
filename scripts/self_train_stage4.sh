#!/bin/bash
# Self-training Stage 4: two backbones, 5 folds × 20 epochs each, seed 42
#
#   Pass 1 — EfficientNet-B0  (warm-start from v3 checkpoints, version=4)
#   Pass 2 — RegNetY-016      (warm-start from v1 checkpoints, version=2)  ← Option A
#
# Both use pseudo_labels_v3.csv (power=1.5).
# 20 epochs (down from 30): Iter 3 best checkpoints peaked at ep8–ep17, no gains after.
#
# Run with:
#   nohup bash /home/swatson/work/MachineLearning/kaggle/BirdCLEF/scripts/self_train_stage4.sh \
#     > /home/swatson/work/MachineLearning/kaggle/BirdCLEF/log/self_train_stage4_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -e

ROOT=/home/swatson/work/MachineLearning/kaggle/BirdCLEF
PYTHON=/home/swatson/miniconda3/envs/kaggle/bin/python
EPOCHS=20
SEED=42
PSEUDO_CSV="$ROOT/data/processed/pseudo_labels_v3.csv"
PSEUDO_POWER=1.5

cd "$ROOT"

rm -f "$ROOT/log/self_train_fold"*.log

# ─────────────────────────────────────────────────────────────
# Pass 1: EfficientNet-B0  (warm-start from v3, output v4)
# ─────────────────────────────────────────────────────────────
BACKBONE_B0="tf_efficientnet_b0.ns_jft_in1k"
VERSION_B0=4

echo "========================================="
echo "BirdCLEF+ 2026 — Self-Training Stage 4"
echo "Pass 1 Backbone : $BACKBONE_B0  (version $VERSION_B0)"
echo "Epochs          : $EPOCHS   Seed: $SEED"
echo "Pseudo          : $PSEUDO_CSV  (power=$PSEUDO_POWER)"
echo "Started         : $(date)"
echo "========================================="

for FOLD in 0 1 2 3 4; do
    INIT_CKPT="$ROOT/models/sed_${BACKBONE_B0}_fold${FOLD}_seed${SEED}_v3.pt"
    echo ""
    echo "----- B0 Fold $FOLD / 4 -----"
    "$PYTHON" -u "$ROOT/src/self_train.py" \
        --fold         "$FOLD" \
        --backbone     "$BACKBONE_B0" \
        --epochs       "$EPOCHS" \
        --seed         "$SEED" \
        --pseudo-csv   "$PSEUDO_CSV" \
        --pseudo-power "$PSEUDO_POWER" \
        --init-ckpt    "$INIT_CKPT" \
        --version      "$VERSION_B0" \
        2>&1 | tee "$ROOT/log/self_train_fold${FOLD}_$(date +%Y%m%d_%H%M%S).log"
done

echo ""
echo "========================================="
echo "Pass 1 (EffB0 v4) complete: $(date)"
echo "========================================="

# ─────────────────────────────────────────────────────────────
# Pass 2: RegNetY-016  (warm-start from v1, output v2)  ← Option A
# ─────────────────────────────────────────────────────────────
BACKBONE_RY="regnety_016.tv2_in1k"
VERSION_RY=2

echo ""
echo "========================================="
echo "Pass 2 Backbone : $BACKBONE_RY  (version $VERSION_RY, warm-start from v1)"
echo "Epochs          : $EPOCHS   Seed: $SEED"
echo "Pseudo          : $PSEUDO_CSV  (power=$PSEUDO_POWER)"
echo "Started         : $(date)"
echo "========================================="

for FOLD in 0 1 2 3 4; do
    INIT_CKPT="$ROOT/models/sed_${BACKBONE_RY}_fold${FOLD}_seed${SEED}_v1.pt"
    echo ""
    echo "----- RegNetY Fold $FOLD / 4 -----"
    "$PYTHON" -u "$ROOT/src/self_train.py" \
        --fold         "$FOLD" \
        --backbone     "$BACKBONE_RY" \
        --epochs       "$EPOCHS" \
        --seed         "$SEED" \
        --pseudo-csv   "$PSEUDO_CSV" \
        --pseudo-power "$PSEUDO_POWER" \
        --init-ckpt    "$INIT_CKPT" \
        --version      "$VERSION_RY" \
        2>&1 | tee "$ROOT/log/self_train_fold${FOLD}_$(date +%Y%m%d_%H%M%S).log"
done

echo ""
echo "========================================="
echo "Self-training Stage 4 complete: $(date)"
echo "========================================="
