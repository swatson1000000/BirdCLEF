#!/usr/bin/env bash
# L2-redux Phase 2a — BC-historic train_audio download for L2-redux pretrain.
# Per new_plan.md §14.17.15.3 Phase 2a.
#
# Downloads BC2023 + BC2024 train_audio only:
#   - BC2021 has no focal train_audio (XC-only, comes via Phase 2c).
#   - BC2025 already on disk at data/raw/birdclef_2025/.
#
# Stages at data/external/birdclef_history/{birdclef-2023,birdclef-2024}/train_audio/.
# Idempotent: if a competition's train_audio/ is already populated, skips it.
# Stream-extracts: deletes each zip after extraction to bound peak disk.
#
# Launch (per four_track/CLAUDE.md):
#   cd /home/swatson/work/kaggle/BirdCLEF/four_track
#   rm -f log/*.log
#   nohup bash scripts/l2_redux_phase2a_download_bc_historic.sh \
#       > log/l2_redux_phase2a_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate kaggle

ROOT=/home/swatson/work/kaggle/BirdCLEF
STAGE_DIR="$ROOT/data/external/birdclef_history"
mkdir -p "$STAGE_DIR"
cd "$STAGE_DIR"

DISK_CAP_PCT=90

check_disk() {
    local use
    use=$(df -P /home/swatson | awk 'NR==2 {gsub("%","",$5); print $5}')
    echo "[$(date +%H:%M:%S)] Disk usage: ${use}%"
    if [ "$use" -gt "$DISK_CAP_PCT" ]; then
        echo "ERROR: disk usage at ${use}% exceeds ${DISK_CAP_PCT}% cap; aborting."
        exit 1
    fi
}

download_one() {
    local comp="$1"
    local dest="$STAGE_DIR/$comp"
    echo
    echo "==== Phase 2a: $comp ===="

    if [ -d "$dest/train_audio" ] && [ "$(find "$dest/train_audio" -type f -name '*.ogg' 2>/dev/null | head -1)" ]; then
        local n
        n=$(find "$dest/train_audio" -type f -name '*.ogg' | wc -l)
        echo "[$(date +%H:%M:%S)] $comp/train_audio already populated ($n .ogg files); skipping."
        return 0
    fi

    check_disk
    mkdir -p "$dest"

    echo "[$(date +%H:%M:%S)] Downloading $comp competition zip..."
    # --force overwrites any partial zip from a prior interrupted run
    kaggle competitions download -c "$comp" --force --path "$STAGE_DIR"

    local zip_path
    zip_path=$(ls -t "$STAGE_DIR/${comp}"*.zip | head -1)
    echo "[$(date +%H:%M:%S)] Got $(basename "$zip_path") ($(du -h "$zip_path" | cut -f1))"
    check_disk

    echo "[$(date +%H:%M:%S)] Extracting train_audio/ from $(basename "$zip_path")..."
    unzip -q "$zip_path" 'train_audio/*' -d "$dest"

    local n_files
    n_files=$(find "$dest/train_audio" -type f -name '*.ogg' | wc -l)
    local extracted_size
    extracted_size=$(du -sh "$dest/train_audio" | cut -f1)
    echo "[$(date +%H:%M:%S)] Extracted $n_files .ogg files ($extracted_size) into $dest/train_audio/"

    rm -f "$zip_path"
    echo "[$(date +%H:%M:%S)] Removed zip $(basename "$zip_path")"
    check_disk
}

echo "==== L2-redux Phase 2a starting at $(date) ===="
check_disk

for comp in birdclef-2023 birdclef-2024; do
    download_one "$comp"
done

echo
echo "==== Phase 2a complete at $(date) ===="
echo "Final stage state:"
for comp in birdclef-2023 birdclef-2024; do
    if [ -d "$STAGE_DIR/$comp/train_audio" ]; then
        local_n=$(find "$STAGE_DIR/$comp/train_audio" -type f -name '*.ogg' | wc -l)
        local_sz=$(du -sh "$STAGE_DIR/$comp/train_audio" | cut -f1)
        echo "  $comp: $local_n .ogg files, $local_sz"
    fi
done
df -h /home/swatson | tail -1
