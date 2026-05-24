#!/bin/bash
set -u

LOG_DIR=/home/swatson/work/kaggle/BirdCLEF/four_track/log
BULK_DIR=/home/swatson/work/kaggle/BirdCLEF/data/external/xenocanto_bulk
LATEST_LOG=$(ls -t "$LOG_DIR"/l2_redux_xc_v3_full_*.log 2>/dev/null | head -1)
REPORT=$(mktemp)
RECIPIENT=swatson1000000@gmail.com

{
  echo "XC v3 download check-in — $(date)"
  echo "host: $(hostname)"
  echo "log : ${LATEST_LOG:-NONE FOUND}"
  echo

  if [[ -z "${LATEST_LOG:-}" ]]; then
    echo "ERROR: no log file matched pattern. Aborting report."
    exit 1
  fi

  # Process liveness — match python process running download_xenocanto.py
  PIDS=$(pgrep -f "python.*download_xenocanto.py" || true)
  if [[ -z "$PIDS" ]]; then
    echo "STATUS: NO LIVE PROCESS — download has stopped or crashed."
  else
    echo "STATUS: ALIVE — PID(s): $PIDS"
    ps -o pid,etime,stat,cmd -p $PIDS | tail -n +1
  fi
  echo

  # Latest progress line
  echo "--- last 5 progress lines ---"
  grep -E "^  \[[0-9]+/" "$LATEST_LOG" | tail -5
  echo

  # Throughput + ETA
  echo "--- throughput / ETA ---"
  python3 - <<PYEOF
import re, sys
from pathlib import Path
log = Path("$LATEST_LOG").read_text(errors="replace").splitlines()
prog = [l for l in log if re.match(r'^  \[\d+/\d+\]', l)]
if not prog:
    print("no progress lines found"); sys.exit(0)
m_first = re.search(r't=([\d.]+)h', prog[0])
m_last  = re.search(r't=([\d.]+)h', prog[-1])
m_idx   = re.search(r'\[(\d+)/(\d+)\]', prog[-1])
m_cum   = re.search(r'cum_dl=(\d+)', prog[-1])
if not all([m_first, m_last, m_idx, m_cum]):
    print("could not parse progress fields"); sys.exit(0)
t0 = float(m_first.group(1)); t1 = float(m_last.group(1))
elapsed = max(t1 - t0, 0.001)
idx, total = int(m_idx.group(1)), int(m_idx.group(2))
cum = int(m_cum.group(1))
spc_per_hr = idx / max(t1, 0.001)
dl_per_hr  = cum / max(t1, 0.001)
remaining_spc = total - idx
eta_hr_by_spc = remaining_spc / max(spc_per_hr, 0.001)
# Project total downloads: linear extrap from current cum_dl/idx
proj_total_dl = cum * (total / max(idx, 1))
eta_hr_by_dl  = (proj_total_dl - cum) / max(dl_per_hr, 0.001)
print(f"  elapsed in this run     : {t1:.2f} h")
print(f"  species progress         : {idx}/{total} ({100*idx/total:.2f}%)")
print(f"  cum downloads            : {cum}")
print(f"  rate (species/hr)        : {spc_per_hr:.1f}")
print(f"  rate (downloads/hr)      : {dl_per_hr:.0f}")
print(f"  ETA by species           : {eta_hr_by_spc:.1f} h ({eta_hr_by_spc/24:.2f} d)")
print(f"  proj total downloads     : {proj_total_dl:.0f}")
print(f"  ETA by downloads         : {eta_hr_by_dl:.1f} h ({eta_hr_by_dl/24:.2f} d)")
print()
print(f"  vs original 6.7-day projection (from 4.3 h, idx 178, 2444 dl/hr):")
days_now = max(eta_hr_by_dl, eta_hr_by_spc)/24
print(f"    current ETA            : {days_now:.2f} d")
print(f"    delta vs 6.7d          : {days_now - 6.7:+.2f} d")
if days_now > 7:
    print(f"  *** WARNING: ETA exceeds 7-day mark. Hard auto-abort was demoted to")
    print(f"      warn-only on 2026-04-30 (per user); the live process honors that")
    print(f"      change (was restarted with patched code). No abort risk. ***")
PYEOF
  echo

  # Failure tallies
  echo "--- failure tallies (this run) ---"
  printf "  ffmpeg transcode failures: %s\n" "$(grep -cE '\[transcode\] ffmpeg failed' "$LATEST_LOG")"
  printf "  download failures        : %s\n" "$(grep -cE '\[dl\].*failed:' "$LATEST_LOG")"
  printf "  HTTP 4xx/5xx mentions    : %s\n" "$(grep -cE 'HTTP [45][0-9]{2}' "$LATEST_LOG")"
  echo
  echo "  recent failures (tail 5):"
  grep -E "(\[transcode\] ffmpeg failed|\[dl\].*failed:|HTTP [45][0-9]{2})" "$LATEST_LOG" | tail -5 | sed 's/^/    /'
  echo

  # Disk
  echo "--- disk ---"
  df -h /home/swatson/work | tail -1
  echo "  bulk dir size : $(du -sh "$BULK_DIR" 2>/dev/null | cut -f1)"
  echo "  species dirs  : $(ls -d "$BULK_DIR"/*/ 2>/dev/null | wc -l)"
  echo "  ogg files     : $(find "$BULK_DIR" -name '*.ogg' 2>/dev/null | wc -l)"
} > "$REPORT" 2>&1

# Email it
SUBJECT="XC v3 download check-in $(date '+%Y-%m-%d %H:%M %Z')"
mail -s "$SUBJECT" "$RECIPIENT" < "$REPORT"
MAIL_RC=$?

# Also drop a local copy under log/ for the user to read directly
ARCHIVE_REPORT="$LOG_DIR/xc_v3_checkin_$(date +%Y%m%d_%H%M%S).txt"
cp "$REPORT" "$ARCHIVE_REPORT"
rm -f "$REPORT"

echo "checkin done: mail rc=$MAIL_RC, local copy=$ARCHIVE_REPORT"
exit 0
