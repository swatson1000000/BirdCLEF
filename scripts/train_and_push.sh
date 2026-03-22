#!/bin/bash
# Full pipeline: self-training Stage 3 → evaluate → push models → push notebook
#
#   Stage 3 trains two backbones on pseudo_labels_v2.csv:
#     Pass 1 — EfficientNet-B0 v3  (warm-start from v2, 5 folds × 30 epochs)
#     Pass 2 — RegNetY-016 v1      (fresh ImageNet init, 5 folds × 30 epochs)
#
# Usage:
#   nohup bash /home/swatson/work/MachineLearning/kaggle/BirdCLEF/scripts/train_and_push.sh \
#     > /home/swatson/work/MachineLearning/kaggle/BirdCLEF/log/train_and_push_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -e

ROOT=/home/swatson/work/MachineLearning/kaggle/BirdCLEF
PYTHON=/home/swatson/miniconda3/envs/kaggle/bin/python
KAGGLE=/home/swatson/miniconda3/envs/kaggle/bin/kaggle

DATASET_SLUG="stevewatson999/birdclef2026-sed-models"
NOTEBOOK_SLUG="stevewatson999/birdclef-2026-sed-inference"
DATASET_URL="https://www.kaggle.com/datasets/${DATASET_SLUG}"
NOTEBOOK_URL="https://www.kaggle.com/code/${NOTEBOOK_SLUG}"

cd "$ROOT"

echo "========================================="
echo "BirdCLEF+ 2026 — Full Pipeline (Stage 3)"
echo "Started : $(date)"
echo "========================================="

# ── 1. Clean training logs (not this log file) ────────────────────────────────
echo ""
echo "[1/5] Cleaning training logs..."
rm -f "$ROOT/log/self_train_fold"*.log "$ROOT/log/train_fold"*.log "$ROOT/log/self_train_stage3_"*.log

# ── 2. Self-training Stage 3 ──────────────────────────────────────────────────
echo ""
echo "[2/5] Starting self-training Stage 3 (EffB0-v3 + RegNetY016-v1, ~26h)..."
bash "$ROOT/scripts/self_train_stage3.sh"
echo "Self-training Stage 3 complete: $(date)"

# ── 3. Evaluate EffB0-v3 ensemble ─────────────────────────────────────────────
echo ""
echo "[3/5] Evaluating EffB0-v3 ensemble on validation soundscapes..."
"$PYTHON" -u "$ROOT/src/evaluate.py" \
    --backbone tf_efficientnet_b0.ns_jft_in1k \
    --version 3
echo "Evaluation complete: $(date)"

# ── 4. Push models dataset ─────────────────────────────────────────────────────
echo ""
echo "[4/5] Pushing models dataset to Kaggle..."
"$KAGGLE" datasets version \
    -p "$ROOT/models" \
    -m "Stage 3: EffB0-v3 + RegNetY016-v1, pseudo_labels_v2, power=1.5" \
    --dir-mode zip
echo "Dataset push complete: $(date)"
echo "  → $DATASET_URL"

# ── 5. Push inference notebook ─────────────────────────────────────────────────
echo ""
echo "[5/5] Pushing inference notebook to Kaggle..."
"$KAGGLE" kernels push -p "$ROOT/jupyter/sed"
echo "Notebook push complete: $(date)"
echo "  → $NOTEBOOK_URL"

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "========================================="
echo "Pipeline complete: $(date)"
echo ""
echo "Kaggle URLs:"
echo "  Dataset  : $DATASET_URL"
echo "  Notebook : $NOTEBOOK_URL"
echo "========================================="
