#!/usr/bin/env bash
# Monitor for the L4-v2 MAIN RUN (folds 1,2,3 on DT + fold 4 on skynet).
#
# Steps:
#   1. Poll DT log for `=== L4-v2 MAIN RUN folds 1,2,3 DT dispatch complete ===`
#   2. Poll skynet log for `=== L4-v2 MAIN RUN fold 4 skynet dispatch complete ===`
#   3. When both done: explicit rsync of fold-1/2/3 ckpts from DT (skynet
#      fold-4 ckpt is already local)
#   4. Run 5-fold broader-pool eval (eval_l4v2_5fold_broader_oof.py)
#
# This monitor MUST be launched AFTER both dispatches are running (it
# captures the DT log path via `ls -t` on the remote log dir).
#
# Usage (from skynet, kaggle-arch env):
#   cd /home/swatson/work/kaggle/BirdCLEF/four_track
#   nohup bash scripts/l4v2_main_run_monitor.sh > log/l4v2_main_run_monitor_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -uo pipefail

SKYNET_ROOT="/home/swatson/work/kaggle/BirdCLEF/four_track"
DT_FT_ROOT="/home/swatson/work/MachineLearning/_runon/BirdCLEF/four_track"

DT_DONE_MARKER='L4-v2 MAIN RUN folds 1,2,3 DT dispatch complete'
SKYNET_DONE_MARKER='L4-v2 MAIN RUN fold 4 skynet dispatch complete'

cd "$SKYNET_ROOT"

# Find the latest DT runon log (the one for our folds-123 dispatch)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] L4-v2 MAIN RUN monitor starting"

# Allow a moment for runon to create its log file
sleep 30

DT_LOG=$(ssh deepthought \
    "ls -t /home/swatson/work/MachineLearning/_runon/BirdCLEF/log/runon_deepthought_*.log | head -1")
SKYNET_LOG=$(ls -t "${SKYNET_ROOT}/log/l4v2_fold4_skynet_"*.log 2>/dev/null | head -1)

echo "  DT log:     $DT_LOG"
echo "  skynet log: $SKYNET_LOG"

if [[ -z "$DT_LOG" || -z "$SKYNET_LOG" ]]; then
    echo "[err] could not locate DT or skynet log; abort."
    exit 1
fi

# ---------------------------------------------------------------- step 1: poll DT
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP 1: polling DT log for folds 1,2,3 completion (5-min interval)"
until ssh deepthought "tail -100 $DT_LOG 2>/dev/null" \
        | grep -qE "${DT_DONE_MARKER}|Traceback|Error: "; do
    sleep 300
done

if ssh deepthought "tail -100 $DT_LOG 2>/dev/null" | grep -qE "Traceback|Error: "; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP 1 ABORTED: DT training hit Traceback/Error."
    ssh deepthought "tail -30 $DT_LOG"
    exit 1
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP 1 OK: DT folds 1,2,3 complete."

# ---------------------------------------------------------------- step 2: poll skynet
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP 2: polling skynet log for fold 4 completion (5-min interval)"
until tail -100 "$SKYNET_LOG" 2>/dev/null \
        | grep -qE "${SKYNET_DONE_MARKER}|Traceback|Error: "; do
    sleep 300
done

if tail -100 "$SKYNET_LOG" 2>/dev/null | grep -qE "Traceback|Error: "; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP 2 ABORTED: skynet training hit Traceback/Error."
    tail -30 "$SKYNET_LOG"
    exit 1
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP 2 OK: skynet fold 4 complete."

# ---------------------------------------------------------------- step 3: rsync DT ckpts
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP 3: rsync DT fold-1/2/3 ckpts back"
for f in 1 2 3; do
    CKPT_NAME="a1_tf_efficientnet_b0.ns_jft_in1k_fold${f}_seed123_ce_l4v2.pt"
    rsync -av "deepthought:${DT_FT_ROOT}/models/a1/${CKPT_NAME}" "${SKYNET_ROOT}/models/a1/"
    if [[ ! -f "${SKYNET_ROOT}/models/a1/${CKPT_NAME}" ]]; then
        echo "[err] ckpt fold-${f} not present locally after rsync"
        exit 1
    fi
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP 3 OK."

# ---------------------------------------------------------------- step 4: 5-fold eval
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP 4: 5-fold broader-pool eval"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kaggle-arch
python -u src/eval_l4v2_5fold_broader_oof.py 2>&1 \
    | tee "log/eval_l4v2_5fold_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "================================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] L4-v2 MAIN RUN monitor complete."
echo "  References:"
echo "    A2 anchor:       0.8402"
echo "    L3-prec anchor:  0.8700"
echo "    +0.05 slot gate: 0.8902"
echo "    fold-0 standalone (already measured): 0.9253"
echo "================================================================"
