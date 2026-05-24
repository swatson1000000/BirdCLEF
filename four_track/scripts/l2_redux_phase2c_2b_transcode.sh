#!/usr/bin/env bash
# L2-redux Phase 2c.2b — filter + transcode rohanrao bulk to ogg vorbis.
# Per new_plan.md §14.17.16.7.3.
#
# Walks our 6,971 target eBird codes, finds matching dirs in
# data/external/xenocanto_raw/{A-M,N-Z}/<code>/, ffmpeg-transcodes each
# .mp3 to 32 kHz mono ogg vorbis q4 with 30s/60s duration cap,
# per-species cap 500 recordings.
#
# Output: data/external/xenocanto_bulk/<target_code>/XC<id>.ogg
#
# Launch:
#   cd /home/swatson/work/kaggle/BirdCLEF/four_track
#   rm -f log/*.log
#   nohup bash scripts/l2_redux_phase2c_2b_transcode.sh \
#       > log/l2_redux_phase2c_2b_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate kaggle

python -u src/l2_redux/transcode_xenocanto.py \
    --workers 4 \
    --per-species-cap 500 \
    --rare-threshold 20 \
    --dur-cap-normal 30 \
    --dur-cap-rare 60
