#!/usr/bin/env bash
set -uo pipefail

SRC="/home/swatson/work/MachineLearning"
DST="/mnt/MachineLearning"

ts() { date +'%Y-%m-%d %H:%M:%S'; }

echo "=== mv $SRC/* -> $DST/ started $(ts) ==="
echo
df -h "$SRC" "$DST"
echo

shopt -s dotglob nullglob
entries=( "$SRC"/* )
total=${#entries[@]}

echo "--- $total top-level entries discovered ---"
for e in "${entries[@]}"; do
  echo "  $(basename "$e")"
done
echo

i=0; ok=0; fail=0; skip=0
for e in "${entries[@]}"; do
  i=$((i+1))
  name=$(basename "$e")

  if [[ "$name" == "kaggle" ]]; then
    echo "[$i/$total $(ts)] SKIP $name (claude-harness keeps recreating cwd here)"
    skip=$((skip+1))
    continue
  fi

  size=$(du -sh "$e" 2>/dev/null | awk '{print $1}')
  start=$(date +%s)
  echo "[$i/$total $(ts)] mv $name ($size) ..."

  if mv "$e" "$DST/" 2>&1; then
    elapsed=$(( $(date +%s) - start ))
    rate=""
    bytes=$(du -sb "$DST/$name" 2>/dev/null | awk '{print $1}')
    if [[ -n "${bytes:-}" && "$elapsed" -gt 0 ]]; then
      rate=$(awk -v b="$bytes" -v e="$elapsed" 'BEGIN { printf "%.1f MB/s", (b/1048576)/e }')
    fi
    echo "[$i/$total $(ts)] OK   $name  (${elapsed}s, $rate)"
    ok=$((ok+1))
  else
    rc=$?
    echo "[$i/$total $(ts)] FAIL $name (rc=$rc)"
    fail=$((fail+1))
  fi
  sync
  df -h "$DST" | tail -1
done

echo
echo "=== done $(ts) ==="
echo "totals: ok=$ok fail=$fail skip=$skip total=$total"
echo
echo "--- residual in $SRC ---"
ls -la "$SRC"
echo
echo "--- $DST after move ---"
ls -la "$DST"
