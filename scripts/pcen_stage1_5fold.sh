#!/bin/bash
# PCEN Stage 1 (5-fold) + Pseudo-label generation + 5-fold self-train (#15i)
#
# PURPOSE: Complete the v13 PCEN Stage 1 run for folds 1-4 (fold-0 already
#          exists as v13), then generate a 5-fold ensemble pseudo-label CSV
#          (pseudo_labels_pcen_v2.csv), then self-train all 5 folds as v15.
#
# WHY:
#   v14 self-train used pseudo-labels from fold-0-only v13 → too noisy → 0.749 LB
#   (regression vs v13 single-fold 0.762 LB). A 5-fold ensemble generates
#   substantially better pseudo-labels (each fold predicts on held-out data),
#   breaking the self-training regression.
#
# STEPS:
#   1. Train folds 1-4 with same hyperparams as v13 (PCEN, BCE, from scratch,
#      20 epochs, focal-only). Fold-0 v13 already exists — skip it.
#   2. Generate pseudo_labels_pcen_v2.csv from all 5 folds v13 ensemble.
#   3. Self-train all 5 folds as v15 (warm-start each fold from its own v13,
#      PCEN, BCE, single-cosine LR, pcen_v2 pseudo-labels).
#   4. Push all 5 v15 checkpoints + submit for LB.
#
# GATE (from #15i):
#   val_roc_auc ≥ 0.76 at v15 fold-0 epoch 1 (warm-start quality check)
#   Final target: 5-fold v15 ensemble LB ≥ 0.780
#
# Usage:
#   conda activate kaggle
#   cd /home/swatson/work/MachineLearning/kaggle/BirdCLEF
#   rm -f log/pcen_*.log
#   nohup bash scripts/pcen_stage1_5fold.sh \
#     > log/pcen_stage1_5fold_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   tail -f log/pcen_stage1_5fold_*.log

set -e

ROOT=/home/swatson/work/MachineLearning/kaggle/BirdCLEF
PYTHON=/home/swatson/miniconda3/envs/kaggle/bin/python
KAGGLE=/home/swatson/miniconda3/envs/kaggle/bin/kaggle

BACKBONE="tf_efficientnet_b0.ns_jft_in1k"
STAGE1_VERSION=13    # v13 = PCEN Stage 1 focal-only (fold-0 already exists)
SELF_TRAIN_VERSION=15 # v15 = 5-fold PCEN self-train with pcen_v2 pseudo-labels
STAGE1_EPOCHS=25
SELF_TRAIN_EPOCHS=30
SEED=42
PSEUDO_CSV="$ROOT/data/processed/pseudo_labels_pcen_v2.csv"

DATASET_SLUG="stevewatson999/birdclef2026-sed-models"

cd "$ROOT"

echo "============================================================"
echo "BirdCLEF+ 2026 — PCEN Stage 1 (5-fold) + Self-Train (#15i)"
echo "Backbone      : $BACKBONE"
echo "Stage 1 v     : v$STAGE1_VERSION (folds 1-4; fold-0 already exists)"
echo "Self-train v  : v$SELF_TRAIN_VERSION (all 5 folds)"
echo "Stage 1 epochs: $STAGE1_EPOCHS"
echo "Self-train ep : $SELF_TRAIN_EPOCHS"
echo "Seed          : $SEED"
echo "Pseudo CSV    : $PSEUDO_CSV"
echo "Started       : $(date)"
echo "============================================================"

# ── Step 1: Train all 5 folds (focal-only, PCEN, from scratch) ───────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "STEP 1: PCEN Stage 1 — training folds 0-4 (v$STAGE1_VERSION)"
echo "══════════════════════════════════════════════════════"

for FOLD in 0 1 2 3 4; do
    CKPT="$ROOT/models/sed_${BACKBONE}_fold${FOLD}_seed${SEED}_v${STAGE1_VERSION}.pt"
    if [ -f "$CKPT" ]; then
        echo ""
        echo "----- Fold $FOLD already trained (v$STAGE1_VERSION) — skipping -----"
        ls -lh "$CKPT"
        continue
    fi

    echo ""
    echo "----- Stage 1 Fold $FOLD / 4 -----"
    echo "  Started: $(date)"

    "$PYTHON" -u "$ROOT/src/train.py" \
        --fold      "$FOLD" \
        --backbone  "$BACKBONE" \
        --epochs    "$STAGE1_EPOCHS" \
        --seed      "$SEED" \
        --version   "$STAGE1_VERSION"

    echo "  Finished: $(date)"
    echo "  Checkpoint:"
    ls -lh "$CKPT" 2>/dev/null || echo "  WARNING: checkpoint not found"
done

echo ""
echo "All Stage 1 folds complete. Checkpoints:"
for FOLD in 0 1 2 3 4; do
    ls -lh "$ROOT/models/sed_${BACKBONE}_fold${FOLD}_seed${SEED}_v${STAGE1_VERSION}.pt" 2>/dev/null \
        || echo "  MISSING: fold $FOLD"
done

# ── Step 2: Generate 5-fold ensemble pseudo-labels ─────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "STEP 2: Generate PCEN pseudo-labels (5-fold ensemble)"
echo "  Output: $PSEUDO_CSV"
echo "══════════════════════════════════════════════════════"

"$PYTHON" -u "$ROOT/src/pseudo_label.py" \
    --backbone     "$BACKBONE" \
    --seed         "$SEED" \
    --folds        "0,1,2,3,4" \
    --ckpt-version "$STAGE1_VERSION" \
    --version      2 \
    --output       "$PSEUDO_CSV"

echo ""
echo "Pseudo-labels generated:"
wc -l "$PSEUDO_CSV"

# ── Step 3: 5-fold PCEN self-train (warm-start each fold from its own v13) ──
echo ""
echo "══════════════════════════════════════════════════════"
echo "STEP 3: 5-fold PCEN self-train (v$SELF_TRAIN_VERSION)"
echo "  Warm-start: each fold from its own v$STAGE1_VERSION checkpoint"
echo "  Pseudo CSV: $PSEUDO_CSV"
echo "══════════════════════════════════════════════════════"

for FOLD in 0 1 2 3 4; do
    INIT_CKPT="$ROOT/models/sed_${BACKBONE}_fold${FOLD}_seed${SEED}_v${STAGE1_VERSION}.pt"
    OUT_CKPT="$ROOT/models/sed_${BACKBONE}_fold${FOLD}_seed${SEED}_v${SELF_TRAIN_VERSION}.pt"

    if [ -f "$OUT_CKPT" ]; then
        echo ""
        echo "----- Self-train Fold $FOLD already done (v$SELF_TRAIN_VERSION) — skipping -----"
        ls -lh "$OUT_CKPT"
        continue
    fi

    echo ""
    echo "----- Self-Train Fold $FOLD / 4  (v$SELF_TRAIN_VERSION) -----"
    echo "  Warm-start: $INIT_CKPT"
    echo "  Started   : $(date)"

    "$PYTHON" -u "$ROOT/src/self_train.py" \
        --fold          "$FOLD" \
        --backbone      "$BACKBONE" \
        --epochs        "$SELF_TRAIN_EPOCHS" \
        --seed          "$SEED" \
        --pseudo-csv    "$PSEUDO_CSV" \
        --pseudo-power  1.5 \
        --init-ckpt     "$INIT_CKPT" \
        --version       "$SELF_TRAIN_VERSION" \
        --use-bce \
        --no-freq-mixstyle \
        --no-dual-loss \
        --lr-schedule   cosine \
        --soup-start-ep 0

    echo "  Finished: $(date)"
done

echo ""
echo "============================================================"
echo "All 5-fold self-train complete: $(date)"
echo ""
echo "Checkpoints (v$SELF_TRAIN_VERSION):"
for FOLD in 0 1 2 3 4; do
    ls -lh "$ROOT/models/sed_${BACKBONE}_fold${FOLD}_seed${SEED}_v${SELF_TRAIN_VERSION}.pt" 2>/dev/null \
        || echo "  MISSING: fold $FOLD"
done
echo "============================================================"

# ── Push all v15 checkpoints to Kaggle dataset ──────────────────────────────
TMP_DS="$(mktemp -d)"
cp "$ROOT/models/dataset-metadata.json" "$TMP_DS/"
for FOLD in 0 1 2 3 4; do
    cp "$ROOT/models/sed_${BACKBONE}_fold${FOLD}_seed${SEED}_v${SELF_TRAIN_VERSION}.pt" "$TMP_DS/" \
        || echo "WARNING: fold $FOLD v$SELF_TRAIN_VERSION checkpoint missing, skipping"
done
echo ""
echo "Pushing checkpoints to Kaggle dataset: $DATASET_SLUG"
echo "  Files:"
ls -lh "$TMP_DS/"
"$KAGGLE" datasets version \
    -p "$TMP_DS" \
    -m "PCEN 5-fold self-train: EffB0-v${SELF_TRAIN_VERSION} (${SELF_TRAIN_EPOCHS}ep, PCEN, BCE, warm-start v${STAGE1_VERSION}, 5-fold ensemble pseudo-labels)" \
    --dir-mode zip
rm -rf "$TMP_DS"

echo ""
echo "Waiting 30s for dataset version to propagate..."
sleep 30

# ── Push inference notebook ──────────────────────────────────────────────────
echo ""
echo "Pushing inference notebook: stevewatson999/birdclef-2026-sed-inference"
"$KAGGLE" kernels push -p "$ROOT/jupyter/sed"

echo ""
echo "============================================================"
echo "PCEN 5-fold pipeline complete: $(date)"
echo ""
echo "NEXT STEPS:"
echo "  1. Wait for notebook to finish running on Kaggle (~30 min)"
echo "  2. Submit output to competition — tag: PCEN_5fold_v${SELF_TRAIN_VERSION}"
echo "  3. Record LB score in plan.md"
echo "  4. Gate: If LB >= 0.780 → proceed to add EffB3 backbone (Stage 7)"
echo "============================================================"
