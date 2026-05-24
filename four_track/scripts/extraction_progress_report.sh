#!/usr/bin/env bash
# Generic long-running extraction progress reporter.
#
# Usage:
#   extraction_progress_report.sh \
#     --label "<short name>" \
#     --log-glob "<absolute glob>" \
#     --proc-pattern "<pgrep -f pattern>" \
#     [--progress-regex "<grep -E regex>"] \
#     [--recipient <email>]
#
# Sends a one-shot progress email. Designed for cron.

set -u
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

LABEL=""
LOG_GLOB=""
PROC_PATTERN=""
PROGRESS_REGEX='\[[0-9]+/[0-9]+\]'
RECIPIENT="swatson1000000@gmail.com"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --label)           LABEL="$2"; shift 2 ;;
    --log-glob)        LOG_GLOB="$2"; shift 2 ;;
    --proc-pattern)    PROC_PATTERN="$2"; shift 2 ;;
    --progress-regex)  PROGRESS_REGEX="$2"; shift 2 ;;
    --recipient)       RECIPIENT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$LABEL" || -z "$LOG_GLOB" ]]; then
  echo "missing required --label or --log-glob" >&2
  exit 2
fi

HOST=$(hostname)
NOW=$(date '+%Y-%m-%d %H:%M:%S %Z')

LOG_FILE=$(ls -t $LOG_GLOB 2>/dev/null | head -1)

if [[ -z "${LOG_FILE:-}" ]]; then
  printf 'No log matching %s on %s as of %s.\n' "$LOG_GLOB" "$HOST" "$NOW" \
    | mail -s "[BirdCLEF $LABEL] no log found on $HOST" "$RECIPIENT"
  exit 0
fi

if [[ -n "$PROC_PATTERN" ]]; then
  PID=$(pgrep -f "$PROC_PATTERN" | head -1 || true)
  if [[ -n "${PID:-}" ]]; then
    ETIME=$(ps -o etime= -p "$PID" | tr -d ' ')
    STATE="RUNNING (pid=$PID, etime=$ETIME)"
    SHORT="RUNNING"
  else
    STATE="NOT RUNNING"
    SHORT="DONE"
  fi
else
  STATE="(process check disabled)"
  SHORT=""
fi

LAST_PROGRESS=$(grep -E "$PROGRESS_REGEX" "$LOG_FILE" 2>/dev/null | tail -1)
if [[ -z "${LAST_PROGRESS:-}" ]]; then
  LAST_PROGRESS=$(tail -1 "$LOG_FILE")
fi
SUBJ_TAIL=$(printf '%s' "$LAST_PROGRESS" | tr -s ' ' | cut -c1-90)

PROGRESS_PCT=""
if [[ "$LAST_PROGRESS" =~ \[([0-9]+)/([0-9]+)\] ]]; then
  N="${BASH_REMATCH[1]}"
  M="${BASH_REMATCH[2]}"
  if (( M > 0 )); then
    PROGRESS_PCT=$(awk -v n="$N" -v m="$M" 'BEGIN { printf "%.2f", (n*100)/m }')
  fi
fi

SUBJECT="[BirdCLEF $LABEL] ${SUBJ_TAIL:-no progress line}${SHORT:+ ($SHORT)}"

LOG_SIZE=$(du -h "$LOG_FILE" | awk '{print $1}')
LOG_MTIME=$(date -r "$LOG_FILE" '+%Y-%m-%d %H:%M:%S %Z')
DISK_FREE=$(df -h "$(dirname "$LOG_FILE")" | awk 'NR==2 {print $4 " free of " $2 " (" $5 " used)"}')

{
  printf '%s progress report\n' "$LABEL"
  printf '===========================\n'
  printf 'host:      %s\n' "$HOST"
  printf 'time:      %s\n' "$NOW"
  printf 'process:   %s\n' "$STATE"
  printf 'log file:  %s\n' "$LOG_FILE"
  printf 'log size:  %s (last write %s)\n' "$LOG_SIZE" "$LOG_MTIME"
  printf 'disk:      %s\n' "$DISK_FREE"
  printf '\n--- last progress line ---\n'
  printf '%s\n' "${LAST_PROGRESS:-(empty log)}"
  if [[ -n "$PROGRESS_PCT" ]]; then
    printf 'species progress: %s/%s (%s%%)\n' "$N" "$M" "$PROGRESS_PCT"
  fi
  printf '\n--- tail (last 15 lines) ---\n'
  tail -15 "$LOG_FILE"
} | mail -s "$SUBJECT" "$RECIPIENT"
