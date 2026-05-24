#!/usr/bin/env bash
# dispatch_a1_inat_v2_fold0.sh — fold-0 A1 finetune from iNat re-pretrain v2
#
# Triggered when the DT iNat re-pretrain (natural sampling + MixUp, dispatched
# 2026-05-10 19:27) finishes. The expected best ckpt path is:
#   models/pretrain_inat/inat_best_tf_efficientnet_b0_ns_jft_in1k_natfreq_mixup50.pt
#
# This script:
#   1. Verifies the ckpt exists locally (run `syncback deepthought four_track/models/`
#      first if not)
#   2. Cleans local log/*.log
#   3. Dispatches the fold-0 finetune to DT via runon with --ft-recipe production
#
# Expected wall time: ~85 min on DT (single fold, 25 epochs at ~3:15/epoch
#   after first epoch's val-cache build).
#
# Result interpretation per new_plan.md PICK UP HERE decision tree:
#   ≥0.7414 → recipe was the dominant bug; run folds 1-3 production + re-run
#             fold 4 production (full 5-fold)
#   0.72-0.7414 → partial recovery; hold and discuss before commit
#   ~0.71 → sampler + MixUp didn't help; pivot to XC v3 or off pretrain
#   <0.71 → broke something; investigate

set -euo pipefail

FT_ROOT="/home/swatson/work/kaggle/BirdCLEF/four_track"
CKPT_REL="models/pretrain_inat/inat_best_tf_efficientnet_b0_ns_jft_in1k_natfreq_melcache.pt"
CKPT_LOCAL="${FT_ROOT}/${CKPT_REL}"

cd "${FT_ROOT}"

# 1. Verify ckpt exists
if [[ ! -f "${CKPT_LOCAL}" ]]; then
  echo "[dispatch] ERROR: ckpt missing at ${CKPT_LOCAL}" >&2
  echo "[dispatch] Run 'syncback deepthought four_track/models/' first." >&2
  exit 2
fi
echo "[dispatch] ckpt exists: $(ls -l "${CKPT_LOCAL}")"

# 2. Pre-flight DT idle check
echo "[dispatch] checking deepthought GPU …"
DT_MEM=$(ssh deepthought "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" | tr -d ' ')
if (( DT_MEM > 500 )); then
  echo "[dispatch] WARNING: deepthought has ${DT_MEM} MiB GPU in use." >&2
  echo "[dispatch] Continue anyway? Re-run with FORCE=1 to bypass." >&2
  [[ "${FORCE:-0}" != "1" ]] && exit 3
fi
echo "[dispatch] deepthought GPU memory used: ${DT_MEM} MiB (idle threshold ok)"

# 3. Clean logs
rm -f log/*.log
echo "[dispatch] cleaned log/"

# 4. Dispatch
echo "[dispatch] launching fold-0 A1 finetune from v2 backbone with --ft-recipe production"
runon deepthought python -u src/train_a1.py \
  --fold 0 \
  --epochs 25 \
  --loss hybrid \
  --mixstyle-p 0.5 \
  --init-from "${CKPT_REL}" \
  --ft-recipe production

echo "[dispatch] launched. Monitor with:"
echo "  ssh deepthought \"tail -f /home/swatson/work/MachineLearning/_runon/BirdCLEF/log/runon_deepthought_*.log\""
