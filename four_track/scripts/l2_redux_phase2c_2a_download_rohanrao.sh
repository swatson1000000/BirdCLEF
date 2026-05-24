#!/usr/bin/env bash
# L2-redux Phase 2c.2a — rohanrao XC bulk dataset download.
# Per new_plan.md §14.17.16.7.3.
#
# Pulls two Kaggle datasets:
#   rohanrao/xeno-canto-bird-recordings-extended-a-m  (~18 GB)
#   rohanrao/xeno-canto-bird-recordings-extended-n-z  (~11 GB)
# Layout: <A-M|N-Z>/<ebird_code>/XC<id>.mp3
#
# Stages at data/external/xenocanto_raw/. Idempotent — skips a dataset
# if its top-level subdir already exists. Stream-extracts: deletes each
# zip after extraction to bound peak disk.
#
# Launch (per four_track/CLAUDE.md):
#   cd /home/swatson/work/kaggle/BirdCLEF/four_track
#   rm -f log/*.log
#   nohup bash scripts/l2_redux_phase2c_2a_download_rohanrao.sh \
#       > log/l2_redux_phase2c_2a_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate kaggle

ROOT=/home/swatson/work/kaggle/BirdCLEF
STAGE_DIR="$ROOT/data/external/xenocanto_raw"
mkdir -p "$STAGE_DIR"
cd "$STAGE_DIR"

DISK_CAP_PCT=92   # higher than Phase 2a since this dataset is bigger

check_disk() {
    local use
    use=$(df -P /home/swatson | awk 'NR==2 {gsub("%","",$5); print $5}')
    echo "[$(date +%H:%M:%S)] Disk usage: ${use}%"
    if [ "$use" -gt "$DISK_CAP_PCT" ]; then
        echo "ERROR: disk usage at ${use}% exceeds ${DISK_CAP_PCT}% cap; aborting."
        exit 1
    fi
}

download_dataset() {
    local dataset_ref="$1"   # e.g. rohanrao/xeno-canto-bird-recordings-extended-a-m
    local subdir="$2"        # e.g. A-M
    echo
    echo "==== Phase 2c.2a: $dataset_ref ===="

    if [ -d "$STAGE_DIR/$subdir" ] && [ -n "$(find "$STAGE_DIR/$subdir" -type d -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
        local n
        n=$(find "$STAGE_DIR/$subdir" -mindepth 1 -maxdepth 1 -type d | wc -l)
        echo "[$(date +%H:%M:%S)] $subdir/ already populated ($n species dirs); skipping."
        return 0
    fi

    check_disk

    echo "[$(date +%H:%M:%S)] Downloading $dataset_ref..."
    kaggle datasets download "$dataset_ref" --force --path "$STAGE_DIR"

    local zip_path
    zip_path=$(ls -t "$STAGE_DIR"/*.zip 2>/dev/null | head -1)
    [ -n "$zip_path" ] || { echo "ERROR: no zip found after download"; exit 1; }
    echo "[$(date +%H:%M:%S)] Got $(basename "$zip_path") ($(du -h "$zip_path" | cut -f1))"
    check_disk

    echo "[$(date +%H:%M:%S)] Extracting $(basename "$zip_path") ..."
    # -o = overwrite without prompting (both A-M and N-Z share train_extended.csv at root)
    unzip -o -q "$zip_path" -d "$STAGE_DIR"

    local n_species n_ogg ext_size
    n_species=$(find "$STAGE_DIR/$subdir" -mindepth 1 -maxdepth 1 -type d | wc -l)
    n_ogg=$(find "$STAGE_DIR/$subdir" -type f -name "*.mp3" | wc -l)
    ext_size=$(du -sh "$STAGE_DIR/$subdir" | cut -f1)
    echo "[$(date +%H:%M:%S)] Extracted $n_species species dirs, $n_ogg .mp3 files ($ext_size)"

    rm -f "$zip_path"
    echo "[$(date +%H:%M:%S)] Removed zip $(basename "$zip_path")"
    check_disk
}

echo "==== L2-redux Phase 2c.2a starting at $(date) ===="
check_disk

download_dataset "rohanrao/xeno-canto-bird-recordings-extended-a-m" "A-M"
download_dataset "rohanrao/xeno-canto-bird-recordings-extended-n-z" "N-Z"

echo
echo "==== Phase 2c.2a complete at $(date) ===="
echo "Final stage state:"
for sub in A-M N-Z; do
    if [ -d "$STAGE_DIR/$sub" ]; then
        ns=$(find "$STAGE_DIR/$sub" -mindepth 1 -maxdepth 1 -type d | wc -l)
        nf=$(find "$STAGE_DIR/$sub" -type f -name "*.mp3" | wc -l)
        sz=$(du -sh "$STAGE_DIR/$sub" | cut -f1)
        echo "  $sub: $ns species dirs, $nf .mp3 files, $sz"
    fi
done
df -h /home/swatson | tail -1
