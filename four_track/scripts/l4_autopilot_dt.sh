#!/usr/bin/env bash
# L4 autopilot — wait for L4-v1 fold-0 on DT, eval, then dispatch L4-v2 fold-0 on DT.
#
# Launched at 2026-05-21 ~23:15 EDT, while L4-v1 fold-0 is at epoch 14/25 on DT.
# L4-v1 ETA ~02:30 EDT 2026-05-22.
# L4-v2 ETA ~12:00 EDT 2026-05-22.
#
# Run as:
#   nohup bash scripts/l4_autopilot_dt.sh > log/l4_autopilot_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -uo pipefail

PROJECT=/home/swatson/work/kaggle/BirdCLEF/four_track
cd "$PROJECT"

# Pin to the existing L4-v1 dispatch log on DT
L4V1_DT_LOG=/home/swatson/work/MachineLearning/_runon/BirdCLEF/log/runon_deepthought_20260521_172151.log
L4V1_DONE_MARKER='L4 fold-0 smoke DT dispatch complete'
L4V2_DONE_MARKER='L4-v2 fold-0 smoke DT dispatch complete'

echo "================================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] L4 autopilot starting"
echo "  L4-v1 DT log: $L4V1_DT_LOG"
echo "  cwd: $PROJECT"
echo "================================================================"

# ---------------------------------------------------------------- step 1
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP 1: polling DT for L4-v1 fold-0 completion (5-min interval)"
until ssh deepthought "tail -50 $L4V1_DT_LOG 2>/dev/null" \
        | grep -qE "${L4V1_DONE_MARKER}|Traceback|Error: "; do
    sleep 300
done

if ssh deepthought "tail -50 $L4V1_DT_LOG 2>/dev/null" | grep -qE "Traceback|Error: "; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP 1 ABORTED: L4-v1 hit Traceback/Error on DT."
    echo "  Last 20 lines of DT log:"
    ssh deepthought "tail -20 $L4V1_DT_LOG"
    exit 1
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP 1 OK: L4-v1 fold-0 complete on DT."

# ---------------------------------------------------------------- step 2
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP 2: syncback DT ckpts + eval L4-v1 fold-0 broader-pool"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kaggle-arch

syncback deepthought 2>&1 | tail -20

python -u src/eval_l4_fold0_broader_oof.py --ckpt-suffix _l4 2>&1 \
    | tee log/eval_l4_fold0_autopilot.log

# ---------------------------------------------------------------- step 3
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP 3: dispatching L4-v2 fold-0 on DT (runon, expected to block ~9h)"

# Capture the runon log path so step 4 can poll it
RUNON_TS=$(date '+%Y%m%d_%H%M%S')
runon deepthought bash scripts/l4v2_fold0_smoke_dt.sh 2>&1 \
    | tee "log/l4v2_runon_dispatch_${RUNON_TS}.log" &
RUNON_PID=$!
echo "[$(date '+%Y-%m-%d %H:%M:%S')] runon PID=$RUNON_PID, sleeping 60s for log to appear on DT"
sleep 60

# Find the latest runon log on DT (the dispatch creates one)
L4V2_DT_LOG=$(ssh deepthought \
    "ls -t /home/swatson/work/MachineLearning/_runon/BirdCLEF/log/runon_deepthought_*.log | head -1")
echo "[$(date '+%Y-%m-%d %H:%M:%S')] L4-v2 DT log: $L4V2_DT_LOG"

# ---------------------------------------------------------------- step 4
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP 4: polling DT for L4-v2 fold-0 completion (5-min interval)"
until ssh deepthought "tail -50 $L4V2_DT_LOG 2>/dev/null" \
        | grep -qE "${L4V2_DONE_MARKER}|Traceback|Error: "; do
    sleep 300
done

if ssh deepthought "tail -50 $L4V2_DT_LOG 2>/dev/null" | grep -qE "Traceback|Error: "; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP 4 ABORTED: L4-v2 hit Traceback/Error on DT."
    echo "  Last 20 lines of DT log:"
    ssh deepthought "tail -20 $L4V2_DT_LOG"
    exit 1
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP 4 OK: L4-v2 fold-0 complete on DT."

# Make sure the background runon process has cleaned up
wait $RUNON_PID 2>/dev/null || true

# ---------------------------------------------------------------- step 5
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP 5: syncback DT ckpts + eval L4-v2 fold-0 broader-pool"
syncback deepthought 2>&1 | tail -20

python -u src/eval_l4_fold0_broader_oof.py --ckpt-suffix _l4v2 2>&1 \
    | tee log/eval_l4v2_fold0_autopilot.log

echo ""
echo "================================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] L4 autopilot COMPLETE."
echo "  L4-v1 fold-0 eval: log/eval_l4_fold0_autopilot.log"
echo "  L4-v2 fold-0 eval: log/eval_l4v2_fold0_autopilot.log"
echo "  see new_plan.md §36.D for decision tree."
echo "================================================================"
