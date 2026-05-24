#!/usr/bin/env bash
# post_xc_badblocks.sh
#
# Wait for the XC v3 download (PID at launch time) to finish, then run a
# read-only badblocks scan on /dev/sda (Seagate One Touch external).
# Safe to launch now and leave running — it polls cheaply and only kicks
# off badblocks once the download exits.
#
# 2026-05-05: queued during XC v3 download (~3.3d remaining).

set -u

# --- config -----------------------------------------------------------------
XC_PATTERN='src/l2_redux/download_xenocanto.py'
DEVICE='/dev/sda'
LOG_DIR='/home/swatson/work/kaggle/BirdCLEF/four_track/log'
TS=$(date +%Y%m%d_%H%M%S)
PROGRESS_LOG="${LOG_DIR}/post_xc_badblocks_${TS}.log"
BADBLOCKS_OUT="${LOG_DIR}/badblocks_sda_${TS}.txt"
POLL_SECS=300   # 5 min — cheap, fine since the download has days to go

# --- helpers ----------------------------------------------------------------
log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$PROGRESS_LOG"; }

# --- wait for XC v3 to finish ----------------------------------------------
log "queued. waiting for XC download process matching '$XC_PATTERN' to exit."
log "device under test: $DEVICE"
log "badblocks output : $BADBLOCKS_OUT"

while pgrep -f "$XC_PATTERN" > /dev/null; do
    sleep "$POLL_SECS"
done

log "XC download process is gone. sleeping 60s for filesystem flush, then starting badblocks."
sleep 60

# --- pre-flight -------------------------------------------------------------
if mount | grep -q "^${DEVICE}"; then
    log "ABORT: $DEVICE (or a partition) is mounted. Refusing to scan."
    exit 1
fi

if [[ ! -b "$DEVICE" ]]; then
    log "ABORT: $DEVICE is not a block device."
    exit 1
fi

log "starting read-only badblocks scan (-sv -b 4096). expect 12-18 h."

# --- run --------------------------------------------------------------------
# -s show progress, -v verbose, -b 4096 match physical block size,
# -o write bad-block list (will be empty if drive is clean).
# Default mode (no -w / -n) is READ-ONLY.
sudo -n badblocks -sv -b 4096 -o "$BADBLOCKS_OUT" "$DEVICE" \
    >> "$PROGRESS_LOG" 2>&1
RC=$?

log "badblocks exited with rc=$RC."
if [[ -s "$BADBLOCKS_OUT" ]]; then
    log "WARNING: bad blocks found. See $BADBLOCKS_OUT"
else
    log "OK: no bad blocks reported."
fi

exit "$RC"
