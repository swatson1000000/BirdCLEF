#!/bin/bash
# Full pipeline: pseudo-labels v3 → self-training Stage 4 → evaluate → push
#
#   pseudo_labels_v3.csv is a 10-model ensemble average:
#     5× EffB0-v3  +  5× RegNetY016-v1
#
#   Stage 4 trains two backbones on pseudo_labels_v3.csv (20 epochs each):
#     Pass 1 — EfficientNet-B0 v4     (warm-start from v3, 5 folds × 20 epochs)
#     Pass 2 — RegNetY-016 v2         (warm-start from v1, 5 folds × 20 epochs)  ← Option A
#
# Usage:
#   nohup bash /home/swatson/work/MachineLearning/kaggle/BirdCLEF/scripts/train_and_push_stage4.sh \
#     > /home/swatson/work/MachineLearning/kaggle/BirdCLEF/log/train_and_push_stage4_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -e

ROOT=/home/swatson/work/MachineLearning/kaggle/BirdCLEF
PYTHON=/home/swatson/miniconda3/envs/kaggle/bin/python
KAGGLE=/home/swatson/miniconda3/envs/kaggle/bin/kaggle

DATASET_SLUG="stevewatson999/birdclef2026-sed-models"
NOTEBOOK_SLUG="stevewatson999/birdclef-2026-sed-inference"
DATASET_URL="https://www.kaggle.com/datasets/${DATASET_SLUG}"
NOTEBOOK_URL="https://www.kaggle.com/code/${NOTEBOOK_SLUG}"

PSEUDO_B0="$ROOT/data/processed/pseudo_labels_v3_b0.csv"
PSEUDO_RY="$ROOT/data/processed/pseudo_labels_v3_rg.csv"
PSEUDO_V3="$ROOT/data/processed/pseudo_labels_v3.csv"

cd "$ROOT"

echo "========================================="
echo "BirdCLEF+ 2026 — Full Pipeline (Stage 4)"
echo "Started : $(date)"
echo "========================================="

# ── 1. Clean old training fold logs ──────────────────────────────────────────
echo ""
echo "[1/6] Cleaning training logs..."
rm -f "$ROOT/log/self_train_fold"*.log "$ROOT/log/train_fold"*.log "$ROOT/log/self_train_stage4_"*.log

# ── 2. Generate pseudo-labels v3: EffB0-v3 (5 folds) ─────────────────────────
echo ""
echo "[2/6] Generating pseudo-labels from EffB0-v3 (5 folds)..."
"$PYTHON" -u "$ROOT/src/pseudo_label.py" \
    --backbone    tf_efficientnet_b0.ns_jft_in1k \
    --ckpt-version 3 \
    --version     3 \
    --output      "$PSEUDO_B0"
echo "EffB0-v3 pseudo-labels written to: $PSEUDO_B0"

# ── 3. Generate pseudo-labels v3: RegNetY016-v1 (5 folds) ────────────────────
echo ""
echo "[3/6] Generating pseudo-labels from RegNetY016-v1 (5 folds)..."
"$PYTHON" -u "$ROOT/src/pseudo_label.py" \
    --backbone    regnety_016.tv2_in1k \
    --ckpt-version 1 \
    --version     3 \
    --output      "$PSEUDO_RY"
echo "RegNetY-v1 pseudo-labels written to: $PSEUDO_RY"

# ── 4. Average the two pseudo-label CSVs → pseudo_labels_v3.csv ──────────────
echo ""
echo "[4/6] Averaging 10-model ensemble pseudo-labels → $PSEUDO_V3 ..."
"$PYTHON" -u - <<'EOF'
import sys
import pandas as pd

root = "/home/swatson/work/MachineLearning/kaggle/BirdCLEF"
path_b0 = f"{root}/data/processed/pseudo_labels_v3_b0.csv"
path_rg = f"{root}/data/processed/pseudo_labels_v3_rg.csv"
out     = f"{root}/data/processed/pseudo_labels_v3.csv"

df1 = pd.read_csv(path_b0)
df2 = pd.read_csv(path_rg)

key_cols     = ["filename", "start_time", "end_time"]
species_cols = [c for c in df1.columns if c not in key_cols]

assert list(df1.columns) == list(df2.columns), "Column mismatch between CSVs"
assert len(df1) == len(df2), f"Row count mismatch: {len(df1)} vs {len(df2)}"

merged = df1.copy()
merged[species_cols] = (df1[species_cols].values + df2[species_cols].values) / 2.0
merged.to_csv(out, index=False)
print(f"Saved {len(merged)} rows, {len(species_cols)} species → {out}")
EOF
echo "pseudo_labels_v3.csv generated: $(date)"

# ── 5. Self-training Stage 4 ──────────────────────────────────────────────────
echo ""
echo "[5/6] Starting self-training Stage 4 (EffB0-v4 + RegNetY016-v2, ~17h)..."
bash "$ROOT/scripts/self_train_stage4.sh"
echo "Self-training Stage 4 complete: $(date)"

# ── 6. Evaluate EffB0-v4 ensemble ─────────────────────────────────────────────
echo ""
echo "[6/6a] Evaluating EffB0-v4 ensemble on validation soundscapes..."
"$PYTHON" -u "$ROOT/src/evaluate.py" \
    --backbone tf_efficientnet_b0.ns_jft_in1k \
    --version 4
echo "EffB0-v4 evaluation complete: $(date)"

echo ""
echo "[6/6b] Evaluating RegNetY016-v2 ensemble on validation soundscapes..."
"$PYTHON" -u "$ROOT/src/evaluate.py" \
    --backbone regnety_016.tv2_in1k \
    --version 2
echo "RegNetY-v2 evaluation complete: $(date)"

# ── 7. Push models dataset ─────────────────────────────────────────────────────
echo ""
echo "[7/7] Pushing models dataset to Kaggle..."
"$KAGGLE" datasets version \
    -p "$ROOT/models" \
    -m "Stage 4: EffB0-v4 + RegNetY016-v2 (warm-start), pseudo_labels_v3, power=1.5, 20 epochs" \
    --dir-mode zip
echo "Dataset push complete: $(date)"
echo "  → $DATASET_URL"

# ── 8. Push inference notebook ─────────────────────────────────────────────────
echo ""
echo "[8/8] Pushing inference notebook to Kaggle..."
"$KAGGLE" kernels push -p "$ROOT/jupyter/sed"
echo "Notebook push complete: $(date)"
echo "  → $NOTEBOOK_URL"

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "========================================="
echo "Stage 4 pipeline complete: $(date)"
echo ""
echo "Kaggle URLs:"
echo "  Dataset  : $DATASET_URL"
echo "  Notebook : $NOTEBOOK_URL"
echo "========================================="
