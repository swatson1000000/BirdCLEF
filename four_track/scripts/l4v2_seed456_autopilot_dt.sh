#!/usr/bin/env bash
# Autopilot for the L4-v2 fold-0 SEED-456 sanity check.
#
# Steps:
#   1. Dispatch `runon deepthought bash scripts/l4v2_fold0_seed456_dt.sh`
#      (runon blocks until the remote script finishes)
#   2. rsync the seed-456 ckpt back to skynet using EXPLICIT path (avoids
#      the previous autopilot's path bug — it used _runon/BirdCLEF/models/
#      instead of _runon/BirdCLEF/four_track/models/)
#   3. Run broader-pool eval with --seed 456 --ckpt-suffix _l4v2
#   4. Report verdict against seed-123's 0.9253 (±0.010 window)
#
# Usage (run with nohup from skynet):
#   cd /home/swatson/work/kaggle/BirdCLEF/four_track
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle-arch
#   nohup bash scripts/l4v2_seed456_autopilot_dt.sh > log/l4v2_seed456_autopilot_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -euo pipefail

SKYNET_ROOT="/home/swatson/work/kaggle/BirdCLEF/four_track"
DT_FT_ROOT="/home/swatson/work/MachineLearning/_runon/BirdCLEF/four_track"
CKPT_NAME="a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed456_ce_l4v2.pt"
SEED123_AUC="0.9253"
PASS_WINDOW="0.010"

cd "$SKYNET_ROOT"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP 1: dispatch seed-456 training on DT (~9h)"
echo "    runon deepthought bash scripts/l4v2_fold0_seed456_dt.sh"
runon deepthought bash scripts/l4v2_fold0_seed456_dt.sh

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP 1 OK: DT training complete."

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP 2: rsync seed-456 ckpt back to skynet"
rsync -av "deepthought:${DT_FT_ROOT}/models/a1/${CKPT_NAME}" "${SKYNET_ROOT}/models/a1/"

if [[ ! -f "${SKYNET_ROOT}/models/a1/${CKPT_NAME}" ]]; then
    echo "[err] ckpt not present locally after rsync: ${SKYNET_ROOT}/models/a1/${CKPT_NAME}"
    exit 1
fi

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP 3: broader-pool eval (seed 456)"
python -u src/eval_l4_fold0_broader_oof.py \
    --ckpt-suffix _l4v2 \
    --seed 456 \
    2>&1 | tee log/eval_l4v2_seed456_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "================================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] L4-v2 SEED-456 sanity complete."
echo ""
echo "  seed-123 baseline broader-pool: ${SEED123_AUC}"
echo "  PASS window: 0.9153 — 0.9353 (±${PASS_WINDOW})"
echo ""
echo "  Inspect the eval log above for the seed-456 number and compare."
echo "  If within window → commit to 5-fold dispatch."
echo "  If outside window → investigate before committing."
echo "================================================================"
