#!/bin/bash
# B2 ConvNeXt 5-fold status check — fired by `at` at 08:30 local on 2026-04-27.
# Reads the training log and writes a status report. Read-only; does NOT
# launch the JIT export (that step requires user confirmation).

set -u

LOG="/home/swatson/work/kaggle/BirdCLEF/four_track/log/train_b2_5fold_resume_20260425_164603.log"
CKPT_DIR="/home/swatson/work/kaggle/BirdCLEF/four_track/models/b2"
REPORT_DIR="/home/swatson/work/kaggle/BirdCLEF/four_track/log"
REPORT="${REPORT_DIR}/b2_5fold_status_$(date +%Y%m%d_%H%M%S).txt"
GATE="0.7414"

{
  echo "B2 5-fold ConvNeXt status report"
  echo "Generated: $(date)"
  echo "Log: $LOG"
  echo "================================================================"
  echo

  if [ ! -f "$LOG" ]; then
    echo "[FATAL] Log file not found."
    exit 1
  fi

  echo "--- Per-fold BEST val_roc_auc ---"
  grep "Best val ROC-AUC" "$LOG" || echo "(no fold completion lines found)"
  echo

  count=$(grep -c "Best val ROC-AUC" "$LOG")
  echo "Folds complete: ${count}/5"
  echo

  echo "--- Process status ---"
  if pgrep -af "train_b2.py" > /dev/null; then
    echo "[RUNNING] train_b2.py still active:"
    pgrep -af "train_b2.py" | head -3
  else
    echo "[NOT RUNNING] no train_b2.py process"
  fi
  echo

  echo "--- Last 40 lines of log ---"
  tail -40 "$LOG"
  echo

  echo "--- Checkpoints in ${CKPT_DIR} ---"
  ls -la "${CKPT_DIR}"/*.pt 2>/dev/null || echo "(none)"
  echo

  echo "--- Verdict ---"
  if [ "$count" -eq 5 ]; then
    mean=$(grep "Best val ROC-AUC" "$LOG" | awk -F': ' '{sum+=$2; n+=1} END {printf "%.4f", sum/n}')
    echo "5-fold mean BEST val_roc_auc: ${mean}"
    pass=$(awk -v m="$mean" -v g="$GATE" 'BEGIN { print (m+0 >= g+0) ? "1" : "0" }')
    if [ "$pass" = "1" ]; then
      echo "[GATE PASS] mean ${mean} >= ${GATE} — proceed to JIT export"
      echo "Next step: review fold-4 details, then run src/export_b2_jit.py (verify it exists)"
    else
      echo "[GATE FAIL] mean ${mean} < ${GATE} — kill, do not LB-probe"
    fi
  elif [ "$count" -lt 5 ]; then
    echo "[INCOMPLETE] only ${count}/5 folds finished — training still running or stalled"
    echo "If train_b2.py is dead and count<5, training crashed mid-fold"
  fi

  echo
  echo "Report path: $REPORT"
} > "$REPORT" 2>&1

# Best-effort desktop notification (no-op if no session bus available)
notify-send "B2 5-fold check complete" "$REPORT" 2>/dev/null || true

exit 0
