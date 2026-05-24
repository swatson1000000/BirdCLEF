#!/usr/bin/env bash
# D3 overnight pipeline: pseudo-label BC2025 Colombia soundscapes →
# emit pseudo rows (with val-coverage gate) → retrain fold-0 (ASL) →
# val gate → JIT export → push Kaggle dataset v57.
#
# Safe to run under nohup. The shell-level gates short-circuit on failure
# so nothing downstream touches Kaggle if an earlier phase fails.
#
# Usage:
#   cd /home/swatson/work/kaggle/BirdCLEF/four_track
#   rm -f log/*.log
#   nohup bash scripts/d3_overnight.sh > log/d3_overnight_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -eo pipefail

FT_ROOT="/home/swatson/work/kaggle/BirdCLEF/four_track"
cd "$FT_ROOT"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate kaggle

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$FT_ROOT/log"
BACKUP_DIR="$FT_ROOT/kaggle_datasets/_backups"
mkdir -p "$LOG_DIR" "$BACKUP_DIR"

P1_LOG="$LOG_DIR/d3_phase1_pseudo_label_${TS}.log"
P2_LOG="$LOG_DIR/d3_phase2_emit_${TS}.log"
P4_LOG="$LOG_DIR/d3_phase4_train_${TS}.log"
P5_LOG="$LOG_DIR/d3_phase5_export_push_${TS}.log"

log() { printf '\n==== %s  %s ====\n' "$(date +'%F %T')" "$*" ; }

trap 'log "FAILED at line $LINENO"; exit 1' ERR

log "D3 overnight start"

# ── Phase 1 ───────────────────────────────────────────────────────────────────
log "Phase 1 — pseudo-label 9,726 BC2025 soundscapes → $P1_LOG"
python -u src/pseudo_label_bc25ss.py > "$P1_LOG" 2>&1
tail -25 "$P1_LOG"

NPZ="data/processed/pseudo_bc25ss_probs.npz"
if [ ! -s "$NPZ" ]; then
    log "Phase 1 FAILED: $NPZ missing"
    exit 1
fi

# ── Phase 2 ───────────────────────────────────────────────────────────────────
log "Phase 2 — emit pseudo rows + val-coverage gate → $P2_LOG"
set +e
python -u src/d3_phase2_emit_pseudo_rows.py > "$P2_LOG" 2>&1
P2_RC=$?
set -e
tail -40 "$P2_LOG"
if [ $P2_RC -ne 0 ]; then
    log "Phase 2 GATE FAIL (rc=$P2_RC). Not retraining. train_folds.csv unchanged."
    exit 1
fi

# Sanity: pre-D3 snapshot exists. Phase 2 writes to the parent BirdCLEF
# data/processed/ via config.PROC, NOT four_track/data/processed/.
SNAPSHOT="$FT_ROOT/../data/processed/train_folds_pre_d3.csv"
if [ ! -s "$SNAPSHOT" ]; then
    log "Phase 2 aborted: snapshot missing at $SNAPSHOT"
    exit 1
fi

# ── Phase 3 ───────────────────────────────────────────────────────────────────
log "Phase 3 — verify dataset_a1.py BC2025_SS_PSEUDO branch"
if ! grep -q 'BC2025_SS_PSEUDO' src/dataset_a1.py; then
    log "Phase 3 FAIL: BC2025_SS_PSEUDO branch missing from src/dataset_a1.py"
    exit 1
fi

# ── Phase 4 ───────────────────────────────────────────────────────────────────
log "Phase 4 — retrain fold-0 (ASL, mixstyle-p=0.5, 25 epochs) → $P4_LOG"
python -u src/train_a1.py \
    --fold 0 \
    --epochs 25 \
    --loss asl \
    --mixstyle-p 0.5 \
    > "$P4_LOG" 2>&1
tail -25 "$P4_LOG"

# Parse best val AUC from the log
BEST_AUC="$(grep -oP 'Best val ROC-AUC:\s+\K[0-9.]+' "$P4_LOG" | tail -1 || echo "0")"
BASELINE="0.7414"
log "Phase 4 BEST val_roc_auc = $BEST_AUC (baseline $BASELINE)"

# Gate: BEST >= baseline  (numeric compare via awk, if-form plays nicely with set -e)
if ! awk -v b="$BEST_AUC" -v base="$BASELINE" 'BEGIN{exit !(b+0 >= base+0)}'; then
    log "Phase 4 GATE FAIL: $BEST_AUC < $BASELINE. Not pushing dataset v57."
    log "D3 fold-0 ckpt at models/a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_asl.pt — decide manually."
    exit 1
fi
log "Phase 4 GATE OK: $BEST_AUC ≥ $BASELINE"

# ── Phase 5 ───────────────────────────────────────────────────────────────────
log "Phase 5 — backup current Kaggle fold-0, export JIT, push dataset v57 → $P5_LOG"

BKP_NAME="a1_fold0_v56_preD3_${TS}.pt"
cp "kaggle_datasets/a1-effb0-ckpts/a1_fold0.pt" "$BACKUP_DIR/$BKP_NAME"
log "backed up current fold-0 → $BACKUP_DIR/$BKP_NAME"

python -u src/export_a1_jit.py \
    --folds 0 \
    --loss-suffix asl \
    --skip-metadata \
    > "$P5_LOG" 2>&1
tail -10 "$P5_LOG"

JIT="$FT_ROOT/kaggle_datasets/a1-effb0-ckpts/a1_fold0.pt"
if [ ! -s "$JIT" ]; then
    log "Phase 5 FAIL: JIT export did not produce $JIT"
    exit 1
fi

log "Phase 5 — kaggle datasets version push (v57)"
kaggle datasets version \
    -p kaggle_datasets/a1-effb0-ckpts \
    -m "v57: D3 BC2025_SS_PSEUDO fold-0 retrain (val=${BEST_AUC})" \
    >> "$P5_LOG" 2>&1
tail -15 "$P5_LOG"

log "D3 overnight complete"
log "NEXT: manual kernel push + LB probe in the morning (§14.11.7 Phase 5 gate)"
