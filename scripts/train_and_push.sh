#!/bin/bash
# Full pipeline: self-training → evaluate → push models → push notebook
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
echo "BirdCLEF+ 2026 — Full Pipeline"
echo "Started : $(date)"
echo "========================================="

# ── 1. Clean training logs (not this log file) ────────────────────────────────
echo ""
echo "[1/4] Cleaning training logs..."
rm -f "$ROOT/log/self_train_fold"*.log "$ROOT/log/train_fold"*.log "$ROOT/log/self_train_stage2_"*.log

# ── 2. Self-training ───────────────────────────────────────────────────────────
echo ""
echo "[2/4] Starting self-training (5 folds × 30 epochs)..."
bash "$ROOT/scripts/self_train_stage2.sh"
echo "Self-training complete: $(date)"

# ── 3. Evaluate ensemble ───────────────────────────────────────────────────────
echo ""
echo "[3/4] Evaluating ensemble on validation soundscapes..."
"$PYTHON" -u "$ROOT/src/evaluate.py" --version 1
echo "Evaluation complete: $(date)"

# ── 4. Push models dataset ─────────────────────────────────────────────────────
echo ""
echo "[4/5] Pushing models dataset to Kaggle..."
"$KAGGLE" datasets version \
    -p "$ROOT/models" \
    -m "Self-training v2 (leakage-fixed, warm-started from Stage 1)" \
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
