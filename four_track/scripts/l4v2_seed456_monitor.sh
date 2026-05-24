#!/usr/bin/env bash
# Monitor + eval for the L4-v2 seed-456 sanity check.
#
# The training was ALREADY dispatched (PID 2526229 on DT).
# This monitor:
#   1. Polls the DT log for completion (5-min interval)
#   2. rsyncs the seed-456 ckpt back to skynet using EXPLICIT path
#   3. Runs broader-pool eval with --seed 456 --ckpt-suffix _l4v2
#
# Usage (run with nohup from skynet, kaggle-arch env):
#   cd /home/swatson/work/kaggle/BirdCLEF/four_track
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle-arch
#   nohup bash scripts/l4v2_seed456_monitor.sh > log/l4v2_seed456_monitor_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -uo pipefail

SKYNET_ROOT="/home/swatson/work/kaggle/BirdCLEF/four_track"
DT_FT_ROOT="/home/swatson/work/MachineLearning/_runon/BirdCLEF/four_track"
DT_LOG="/home/swatson/work/MachineLearning/_runon/BirdCLEF/log/runon_deepthought_20260522_145624.log"
CKPT_NAME="a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed456_ce_l4v2.pt"
DONE_MARKER='L4-v2 fold-0 SEED-456 sanity DT dispatch complete'
SEED123_AUC="0.9253"

cd "$SKYNET_ROOT"

echo "================================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] L4-v2 seed-456 monitor starting"
echo "  DT log: $DT_LOG"
echo "  Looking for marker: '${DONE_MARKER}'"
echo "================================================================"

# ---------------------------------------------------------------- step 1: poll
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP 1: polling DT for completion (5-min interval)"
until ssh deepthought "tail -50 $DT_LOG 2>/dev/null" \
        | grep -qE "${DONE_MARKER}|Traceback|Error: "; do
    sleep 300
done

if ssh deepthought "tail -50 $DT_LOG 2>/dev/null" | grep -qE "Traceback|Error: "; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP 1 ABORTED: training hit Traceback/Error on DT."
    echo "  Last 30 lines of DT log:"
    ssh deepthought "tail -30 $DT_LOG"
    exit 1
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP 1 OK: training complete."

# ---------------------------------------------------------------- step 2: rsync
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP 2: rsync seed-456 ckpt back to skynet"
rsync -av "deepthought:${DT_FT_ROOT}/models/a1/${CKPT_NAME}" "${SKYNET_ROOT}/models/a1/"

if [[ ! -f "${SKYNET_ROOT}/models/a1/${CKPT_NAME}" ]]; then
    echo "[err] ckpt not present locally after rsync: ${SKYNET_ROOT}/models/a1/${CKPT_NAME}"
    exit 1
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP 2 OK."

# ---------------------------------------------------------------- step 3: eval
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
echo "  PASS window: 0.9153 — 0.9353 (±0.010)"
echo ""
echo "  Inspect the eval log above for the seed-456 number and compare."
echo "  If within window → seed-123 result is robust; commit to 5-fold dispatch."
echo "  If outside window → investigate before committing."
echo "================================================================"
