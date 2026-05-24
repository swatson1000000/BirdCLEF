#!/bin/bash
# Check A2 5-fold A1 retrain progress on deepthought and email a summary.
# Designed to be invoked via `at` at ~19:30 EDT 2026-05-15.

set -u
RECIPIENT="swatson1000000@gmail.com"
LOG_REMOTE="/home/swatson/work/MachineLearning/_runon/BirdCLEF/four_track/log/train_a1_a2_dt_folds0123.log"
CKPT_GLOB="/home/swatson/work/MachineLearning/_runon/BirdCLEF/four_track/models/a1/a1_*_seed42_asl.pt"
SKYNET_FOLD4="/home/swatson/work/kaggle/BirdCLEF/four_track/models/a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold4_seed42_asl.pt"
DT_PID=312252

ts=$(date '+%Y-%m-%d %H:%M:%S %Z')
body=$(mktemp)

{
    echo "A2 5-fold check @ ${ts}"
    echo ""
    echo "=== Deepthought PID ${DT_PID} status ==="
    ssh -o ConnectTimeout=15 deepthought "ps -p ${DT_PID} -o pid,etime,stat,pcpu,cmd 2>&1 | tail -2" 2>&1
    echo ""
    echo "=== DT log tail (last 50 lines) ==="
    ssh -o ConnectTimeout=15 deepthought "tail -50 ${LOG_REMOTE}" 2>&1
    echo ""
    echo "=== DT fold ckpts ==="
    ssh -o ConnectTimeout=15 deepthought "ls -la ${CKPT_GLOB} 2>&1"
    echo ""
    echo "=== Per-fold best val_roc_auc (folds 0-3, deepthought) ==="
    ssh -o ConnectTimeout=15 deepthought "grep -E '^Fold [0-3] complete' ${LOG_REMOTE}" 2>&1
    echo ""
    echo "=== Skynet fold 4 ckpt ==="
    ls -la "${SKYNET_FOLD4}" 2>&1
    echo ""
    echo "=== Gate-check readiness ==="
    dt_done=$(ssh -o ConnectTimeout=15 deepthought "ls ${CKPT_GLOB} 2>/dev/null | wc -l")
    skynet_done=0
    [ -f "${SKYNET_FOLD4}" ] && skynet_done=1
    echo "DT fold ckpts present: ${dt_done}/4"
    echo "Skynet fold 4 ckpt present: ${skynet_done}/1"
    if [ "${dt_done}" = "4" ] && [ "${skynet_done}" = "1" ]; then
        echo "STATUS: ALL 5 FOLDS READY. Proceed to rsync DT ckpts and run broader-pool OOF gate-check."
    else
        echo "STATUS: NOT READY. $((4 - dt_done)) DT fold(s) and $((1 - skynet_done)) skynet fold still pending."
    fi
} > "${body}" 2>&1

subject="[BirdCLEF A2] 5-fold check @ ${ts}"

{
    echo "To: ${RECIPIENT}"
    echo "From: swatson1000000@gmail.com"
    echo "Subject: ${subject}"
    echo "Content-Type: text/plain; charset=utf-8"
    echo ""
    cat "${body}"
} | msmtp -t

rm -f "${body}"
