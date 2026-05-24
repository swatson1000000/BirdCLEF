#!/usr/bin/env bash
# Snapshot the state of the XC v3 download (PID 764735) and write to
# four_track/log/xc_v3_status_<timestamp>.log. Designed to be run from
# `at` or manually.

set -u

ROOT=/home/swatson/work/kaggle/BirdCLEF
FT=$ROOT/four_track
LOG_GLOB="$FT/log/l2_redux_xc_v3_full_*.log"
BULK_DIR="$ROOT/data/external/xenocanto_bulk"
STATUS_LOG="$FT/log/xc_v3_status_$(date +%Y%m%d_%H%M%S).log"
PID=764735

{
    echo "=== XC v3 download status check $(date) ==="
    echo
    if kill -0 "$PID" 2>/dev/null; then
        echo "PID $PID: ALIVE"
        ps -p "$PID" -o pid,etime,stat,cmd | sed 's/XC_API_KEY=[^ ]*/XC_API_KEY=<redacted>/g'
        echo
        echo "--- last 30 log lines ---"
        tail -30 $LOG_GLOB
    else
        echo "PID $PID: EXITED"
        echo
        echo "--- last 80 log lines (looking for exit reason) ---"
        tail -80 $LOG_GLOB
        echo
        # Pattern-match common abort gates
        echo "--- abort-pattern scan ---"
        grep -E "\[auth\]|\[gate\]|\[setup\]|429|EXCEPTION" $LOG_GLOB | tail -20 || echo "no abort patterns found"
    fi
    echo
    echo "--- disk ---"
    df -h /home/swatson | tail -1
    echo
    echo "--- bulk dir size + species count ---"
    if [ -d "$BULK_DIR" ]; then
        du -sh "$BULK_DIR"
        n=$(find "$BULK_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
        echo "$n species directories"
        echo "ogg file count: $(find "$BULK_DIR" -name "*.ogg" | wc -l)"
    else
        echo "MISSING: $BULK_DIR"
    fi
    echo
    echo "=== status log written to: $STATUS_LOG ==="
} > "$STATUS_LOG" 2>&1
