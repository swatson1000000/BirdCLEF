#!/usr/bin/env bash
# rsync_to_deepthought_throttled.sh
#
# One-shot throttled rsync of /home/swatson/work/MachineLearning/ to
# deepthought:/mnt/mypassport/backup_ai_skynet. Designed to survive
# the WD MyPassport 260D's USB-3 link instability documented in
# ~/bin/backup line 59-66 (and dmesg sdc1->sdd1 re-enumeration on
# 2026-05-06 22:01:19, where EXT4-fs hit "I/O error while writing
# superblock" mid-rsync and remounted read-only).
#
# Strategy:
#   --bwlimit=50M  caps write rate; empirically the link drops
#                  correlate with the moment the drive's USB FIFO +
#                  cache saturate, which a slow steady stream avoids.
#   --inplace +    on resume, append directly to the existing dest
#   --partial-dir  file after checksum-verifying the prefix matches
#   --append-verify src, instead of re-copying from byte 0.
#   retry loop     transient rsync exit codes (10/12/23/30/35) trigger
#                  a sleep + retry, up to MAX_RETRIES times. Each
#                  retry is a fresh rsync, so the full file-list scan
#                  re-runs and any new src changes get picked up.
#
# This is a workaround, NOT a fix for the USB-link instability.
# Replacing the cable / enclosure is still the correct long-term fix.

set -u

# === Mutex against ~/bin/backup's nightly ai rsync ===
# Both scripts target deepthought:/mnt/mypassport/backup_ai_skynet with
# --delete --inplace, so a concurrent run would thrash files. Acquire
# the shared lock non-blockingly and bail loudly if held — the operator
# can decide whether to wait or kill the in-flight run.
exec 8>/tmp/backup_ai_skynet.lock
if ! flock -n 8; then
    echo "ABORT: /tmp/backup_ai_skynet.lock is held by another rsync to backup_ai_skynet."
    echo "       holder PIDs: $(fuser /tmp/backup_ai_skynet.lock 2>/dev/null | tr -d ' ')"
    echo "       Wait for it to finish, or kill the holder, before re-running."
    exit 75
fi

SRC="/home/swatson/work/MachineLearning/"
DEST_HOST="deepthought"
DEST_PATH="/mnt/mypassport/backup_ai_skynet"
DEST="${DEST_HOST}:${DEST_PATH}"
BWLIMIT="50M"
LOG_DIR="/home/swatson/work/kaggle/BirdCLEF/four_track/log"
LOG="${LOG_DIR}/rsync_to_deepthought_$(date +%Y%m%d_%H%M%S).log"
MAX_RETRIES=30
RETRY_SLEEP=180   # 3 min — long enough for USB reset + ext4 remount

mkdir -p "${LOG_DIR}"

# Rsync options.
#   -a              archive (perms / owner / time / symlinks / dirs)
#   --delete        mirror — remove dest files no longer on src
#   --inplace       write directly to the dest path (no temp file).
#                   With --inplace, an interrupted transfer leaves the
#                   dest file partially written in place, which is the
#                   "partial" we want to resume against. --partial-dir
#                   would conflict (and is mutually exclusive with
#                   --append-verify).
#   --append-verify on resume, only extend an existing dest file if
#                   its prefix checksum matches src — guards against
#                   torn writes from a USB-link drop mid-transfer
#   --bwlimit       cap rate (50M = ~50 MB/s)
#   --info=name1,stats2,flist0 one filename per line + final stats; no
#                   progress-bar carriage returns to trash the log
#   ServerAliveInterval/Count keep ssh up across slow disk stalls
RSYNC_OPTS=(
    -a
    --delete
    --inplace
    --append-verify
    --bwlimit="${BWLIMIT}"
    --info=name1,stats2,flist0
    -e "ssh -T -o LogLevel=QUIET -o ServerAliveInterval=30 -o ServerAliveCountMax=4"
)

# Add EXCLUDES entries here if you want to skip subtrees.
# xenocanto_bulk excluded 2026-05-08: 92 GB across 524,572 small audio
# files would saturate the flaky MyPassport's metadata path on the
# first recovery attempt. Its plan-designated home is the new
# /mnt/MachineLearning cold archive on skynet (see new_plan.md §14.21.7
# step 4e). Re-include here only after the cable/enclosure is
# replaced and a clean baseline rsync has succeeded.
#
# kaggle/ and nomos-1/ excluded 2026-05-09: split the in-flight backup
# so the smaller dirs (~135 GB total) finish in ~4-5 h instead of
# blocking on these two large trees (kaggle ~291 GB, nomos-1 ~244 GB)
# for ~17 additional hours. Run them in a separate later pass by
# commenting out (not deleting) the two lines below.
EXCLUDES=(
    --exclude='kaggle/BirdCLEF/data/external/xenocanto_bulk/'
    --exclude='kaggle/'
    --exclude='nomos-1/'
)

RETRYABLE=(10 12 23 24 30 35)

red()    { printf "\033[31m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }
green()  { printf "\033[32m%s\033[0m\n" "$*"; }

trap 'echo "[$(date -Iseconds)] interrupted by signal; partial dest files kept in place for resume" | tee -a "${LOG}"; exit 130' INT TERM

# === Preflight ===
echo "[$(date -Iseconds)] preflight checks" | tee -a "${LOG}"

if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "${DEST_HOST}" true 2>>"${LOG}"; then
    red "ABORT: ssh ${DEST_HOST} failed (no key auth or host down)"
    exit 1
fi
green "[ok] ssh ${DEST_HOST} reachable" | tee -a "${LOG}"

# Confirm dest mountpoint is actually mounted on deepthought (a USB
# drop can leave the path present but unmounted; rsync would happily
# fill the underlying root fs).
if ! ssh "${DEST_HOST}" "mountpoint -q /mnt/mypassport" 2>>"${LOG}"; then
    red "ABORT: /mnt/mypassport on ${DEST_HOST} is NOT mounted"
    yellow "  fix: ssh ${DEST_HOST} 'sudo mount /mnt/mypassport' (or check dmesg for USB drops)"
    exit 1
fi
green "[ok] /mnt/mypassport mounted on ${DEST_HOST}" | tee -a "${LOG}"

if ! ssh "${DEST_HOST}" "test -d ${DEST_PATH}" 2>>"${LOG}"; then
    red "ABORT: ${DEST_PATH} does not exist on ${DEST_HOST}"
    exit 1
fi
green "[ok] ${DEST_PATH} exists on ${DEST_HOST}" | tee -a "${LOG}"

# Show free space at dest.
echo "[dest free]" | tee -a "${LOG}"
ssh "${DEST_HOST}" "df -h /mnt/mypassport" | tee -a "${LOG}"

echo ""                                                             | tee -a "${LOG}"
echo "[$(date -Iseconds)] starting throttled rsync"                 | tee -a "${LOG}"
echo "  src         : ${SRC}"                                       | tee -a "${LOG}"
echo "  dest        : ${DEST}"                                      | tee -a "${LOG}"
echo "  bwlimit     : ${BWLIMIT}"                                   | tee -a "${LOG}"
echo "  max_retries : ${MAX_RETRIES} (sleep ${RETRY_SLEEP}s each)"  | tee -a "${LOG}"
echo "  excludes    : ${EXCLUDES[*]:-<none>}"                       | tee -a "${LOG}"
echo "  log         : ${LOG}"                                       | tee -a "${LOG}"
echo ""                                                             | tee -a "${LOG}"

# === Retry loop ===
attempt=0
while true; do
    attempt=$((attempt + 1))
    echo "[$(date -Iseconds)] === attempt ${attempt}/${MAX_RETRIES} ===" | tee -a "${LOG}"

    rsync "${RSYNC_OPTS[@]}" "${EXCLUDES[@]}" "${SRC}" "${DEST}" 2>&1 | tee -a "${LOG}"
    rc=${PIPESTATUS[0]}
    echo "[$(date -Iseconds)] attempt ${attempt} exit=${rc}" | tee -a "${LOG}"

    if [[ ${rc} -eq 0 ]]; then
        green "[$(date -Iseconds)] DONE after ${attempt} attempt(s)" | tee -a "${LOG}"
        exit 0
    fi

    is_retryable=0
    for code in "${RETRYABLE[@]}"; do
        [[ ${rc} -eq ${code} ]] && is_retryable=1 && break
    done

    if [[ ${is_retryable} -eq 0 ]]; then
        red "[$(date -Iseconds)] non-retryable exit ${rc}; aborting" | tee -a "${LOG}"
        exit "${rc}"
    fi

    if [[ ${attempt} -ge ${MAX_RETRIES} ]]; then
        red "[$(date -Iseconds)] hit MAX_RETRIES=${MAX_RETRIES}; aborting (last exit=${rc})" | tee -a "${LOG}"
        exit "${rc}"
    fi

    yellow "[$(date -Iseconds)] retryable exit ${rc}; sleeping ${RETRY_SLEEP}s before retry"  | tee -a "${LOG}"

    # If the drive re-enumerated, it may need to be remounted before retry.
    # Check and ask deepthought to remount if needed.
    if ! ssh -o ConnectTimeout=10 "${DEST_HOST}" "mountpoint -q /mnt/mypassport" 2>/dev/null; then
        yellow "[$(date -Iseconds)] /mnt/mypassport on ${DEST_HOST} is not mounted; attempting remount" | tee -a "${LOG}"
        ssh "${DEST_HOST}" "sudo -n mount /mnt/mypassport" 2>&1 | tee -a "${LOG}" || true
    fi

    sleep "${RETRY_SLEEP}"
done
