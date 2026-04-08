#!/usr/bin/env bash
# Track A1 — PCEN + ASL + FreqMixStyle EffNet-B0 SED, 5-fold × 25 epochs.
#
# Run from the four_track workspace with the kaggle conda env active:
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle
#   cd /home/swatson/work/MachineLearning/kaggle/BirdCLEF/four_track
#   nohup bash scripts/train_a1_5fold.sh \
#     > log/train_a1_5fold_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -euo pipefail

cd /home/swatson/work/MachineLearning/kaggle/BirdCLEF/four_track

python -u src/train_a1.py \
    --folds 0,1,2,3,4 \
    --epochs 25 \
    --loss hybrid \
    --mixstyle-p 0.5
