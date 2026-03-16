# BirdCLEF+ 2026 — Copilot Instructions

**Task**: Multi-label audio classification of 234 wildlife species from passive acoustic monitoring (PAM) recordings in the Brazilian Pantanal. Metric: macro-averaged ROC-AUC on 5-second windows.

See [plan.md](../plan.md) for full competition plan and [CLAUDE.md](../CLAUDE.md) for execution policy details.

---

## Environment

- **Conda env**: `kaggle` — always activate before running Python
- **Hardware**: NVIDIA GB10 (Blackwell, 128 GB unified memory) — use BF16 for training
- **Project root**: `/home/swatson/work/MachineLearning/kaggle/BirdCLEF`

---

## Script Execution Policy

All Python scripts **must** run with `nohup` in the background with timestamped logs:

```bash
conda activate kaggle
cd /home/swatson/work/MachineLearning/kaggle/BirdCLEF

# Clean old logs before a new training run
rm -f log/train_*.log

nohup python -u src/<script>.py [args] > log/<script>_$(date +%Y%m%d_%H%M%S).log 2>&1 &
tail -f log/<script>_*.log
```

- Use Python scripts (`.py`), **not** Jupyter notebooks, for all core implementation
- Shell scripts in `scripts/` must use **absolute paths** (relative paths cause silent failures when invoked from different working directories)
- Jupyter notebooks in `jupyter/` are only for the final Kaggle inference submission

---

## Architecture

### Model: SED (Sound Event Detection)

```python
# src/model.py — BirdSEDModel
# CNN backbone (timm) → GEM frequency pooling → attention-weighted clip prediction
backbone = timm.create_model(backbone_name, features_only=True, out_indices=(4,))
# SED head: Conv1d attention + classifier over time frames
# clip_prob = weighted_sum(frame_prob * frame_att) / sum(frame_att)
```

**Backbone priority**: `tf_efficientnet_b0.ns_jft_in1k` → `efficientvit_b0.r224_in1k` (2–3× faster ONNX) → `regnety_008` → `tf_efficientnet_b3` → `tf_efficientnet_b4` → `eca_nfnet_l0`

### Audio Pipeline

| Parameter | Value |
|-----------|-------|
| Sample rate | 32 kHz (no resampling needed) |
| Chunk duration | 20 seconds = 640,000 samples |
| Mel bins (`n_mels`) | 224 |
| FFT size (`n_fft`) | 4096 |
| Hop length | 1252 → output width ≈ 512 |
| Frequency range | 0–16,000 Hz |
| Dynamic range | 80 dB (top_db) |
| Input tensor | (B, 3, 224, 512) — 3-channel by repeating mel |

### MixUp (critical details)

```python
# Normalize by absmax before mixing
w1 = w1 / (np.abs(w1).max() + 1e-8)
w2 = w2 / (np.abs(w2).max() + 1e-8)
mixed = 0.5 * w1 + 0.5 * w2          # Fixed 0.5 weight — NOT random beta
labels = np.maximum(label1, label2)   # Element-wise max — NOT average
```

### Augmentations

| Augmentation | Notes |
|---|---|
| Background noise injection | p=0.3–0.5, mix real env noise (freefield1010, warblrb, birdvox) at low gain |
| PitchShift | p=0.3, ±2 semitones via `torch_audiomentations` on GPU |
| TimeShift | p=0.5, ±25% via `torch_audiomentations` on GPU |
| SpecAugment time/freq mask | p=0.3 each |
| Random gain | p=0.5, ±6 dB |

### Secondary Labels: Mask Loss

Set secondary label loss weight to **0** (not 0.5): species location within the clip is unknown — forcing a label at a specific 5s window adds noise. This gave +0.01 LB in BirdCLEF 2024.

```python
# secondary_mask: 0 for secondary label positions, 1 for primary
loss = (bce_loss * secondary_mask).mean()
```

---

## Training

### BF16 Boilerplate (include in every training script)

```python
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cudnn.benchmark = True   # fixed input shapes

device = torch.device("cuda")
model = model.to(device)
model = torch.compile(model)            # fused kernels for GB10

autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
# BF16 does NOT need GradScaler
```

> `torch.compile` is for **training only**. Export un-compiled weights for ONNX. `torch.compile` + `torch.onnx.export` are incompatible in the same session.

### Key Hyperparameters (Stage 1)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW, weight_decay=1e-4 |
| LR | 5e-4 → 1e-6, CosineAnnealingWarmRestarts (T_0=5 epochs) |
| Epochs | 15 (Stage 1), 25–35 (self-training) |
| Batch size | 64 |
| Folds | 5-fold stratified (MultilabelStratifiedKFold) |
| Seeds | 42, 123, 777 (multi-seed ensemble) |
| Precision | BF16 |

### Epoch Logging Format (required in all training scripts)

Every epoch must output this summary line (implement via callback):

```
========================================
Epoch  N/15: train_loss=X.XXXX val_roc_auc=X.XXXX time=Xm XXs ★ BEST
========================================
```

- Time as `Xm XXs` (not raw seconds)
- Append `★ BEST` when best validation ROC-AUC is achieved

---

## Validation

- Validate against **`data/raw/train_soundscapes_labels.csv`** (expert-labeled 5-second segments)
- This matches the competition metric domain (field recordings, not focal clips)
- Per-segment macro ROC-AUC = competition metric

---

## Pseudo-Labeling & Self-Training

- Pseudo-label unlabeled `train_soundscapes/` with Stage 1 ensemble
- Self-train with Noisy Student: mix focal clips + pseudo-labeled soundscape chunks at 0.5/0.5
- Apply power transform after each iteration: `pseudo_prob ** power` (e.g., 1.5 for sharpening)
- See [plan.md](../plan.md) Phase 2–3 for full details

---

## ONNX Export (Kaggle Inference)

- Export each model to ONNX for CPU inference (no quantization)
- `dynamic_axes={"input": {0: "batch_size"}}`
- Inference: sliding window, 20s chunks with 5s stride, average overlapping frame predictions
- Submission constraint: CPU-only notebook, ≤90 min runtime, no internet

---

## Project Layout

```
src/
  config.py         # Central config: paths, mel params, training hyperparams
  dataset.py        # BirdTrainDataset + augmentations
  model.py          # BirdSEDModel (SED head + timm backbone)
  train.py          # Supervised training (Stage 1)
  self_train.py     # Noisy Student self-training
  pseudo_label.py   # Generate pseudo-labels from soundscapes
  evaluate.py       # Local ROC-AUC vs train_soundscapes_labels.csv
  ensemble.py       # Weighted average ensemble
  export_onnx.py    # ONNX export
  utils.py          # Audio loading, mel spectrogram, pad_or_crop
scripts/
  train_stage1.sh               # 5-fold Stage 1 (use absolute paths)
  pseudo_label_soundscapes.sh
  self_train_stage2.sh
  export_ensemble_onnx.sh
data/raw/           # Competition data (gitignored)
data/processed/     # train_folds.csv, pseudo_labels_v*.csv, eda_report.txt
models/             # Saved weights (gitignored)
jupyter/            # birdclef2026-inference.ipynb (Kaggle submission)
log/                # nohup logs (gitignored)
```

---

## Key Pitfalls

- **Insect sonotypes** (e.g., `47158son16`) are valid class labels — treat as unique species
- **Secondary labels**: mask their loss to 0 (not soft weight 0.5) — location within clip is unknown
- **iNat clips** have no quality ratings — do not filter; XC clips: prefer rating ≥ 3.5
- **`torch.compile` + ONNX export** are incompatible — export un-compiled weights
- **BF16** does not need `GradScaler` (unlike FP16)
- **Shell scripts**: always use absolute paths (relative paths fail when invoked from different cwd)
- **Logs**: clean `log/train_*.log` before each new training run to avoid confusion
- **OpenVINO**: ~2× faster than ONNX but causes ~0.01 accuracy drop; `eca_nfnet_l0` fails conversion — use ONNX only
- **Aves/BirdAves model**: +0.01 public LB but hurts private LB — don't include without LB gate
- **Minimum 10 samples per class**: duplicate rare clips before fold splitting
- **Cap external XC data at 500 records/class** to avoid class imbalance
