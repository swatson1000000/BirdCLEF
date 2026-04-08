# BirdCLEF+ 2026 — Project Summary

## What This Project Does

This project builds a machine-learning system for the [BirdCLEF+ 2026 Kaggle competition](https://www.kaggle.com/competitions/birdclef-2026), hosted by the Cornell Lab of Ornithology. The goal is to automatically identify **234 wildlife species** (birds, amphibians, mammals, reptiles, and insects) from passive acoustic monitoring (PAM) recordings collected in the **Brazilian Pantanal** wetlands.

### Competition requirements
- **Input**: 1-minute field soundscape recordings at 32 kHz
- **Output**: Per-species probability score for each 5-second window within the recording
- **Metric**: Macro-averaged ROC-AUC across all 234 species (classes with no true positives in the test set are skipped)
- **Submission constraint**: CPU-only Kaggle notebook, ≤ 90 minutes runtime, no internet access

### Current scores
| Run | Local Val ROC-AUC | Kaggle LB ROC-AUC |
|-----|-------------------|-------------------|
| Perch baseline | — | 0.590 |
| Stage 1 SED EffB0 (5-fold) | 0.7636 | 0.752 |
| SelfTrain v1 (val data leaked) | 0.8395 (inflated) | 0.751 |
| **Self-Train v2 (leakage fixed)** | **0.7858** | **0.762** |

---

## Architecture Overview

The pipeline has four major stages:

```
Raw audio (.ogg)
    ↓
Mel spectrogram (224 × 512, float32)
    ↓
BirdSEDModel (CNN backbone + GEM pool + Conv1d attention)
    ↓
clip_logits → sigmoid → per-species probability
```

Training uses a **noisy student self-training** loop:
1. Train a supervised SED model on labelled focal clips (Stage 1).
2. Use the Stage 1 ensemble to generate soft pseudo-labels for 10,658 unlabelled train soundscapes.
3. Re-train mixing focal clips and pseudo-labelled soundscape segments (Stage 2+).
4. Iterate: regenerate pseudo-labels from the latest ensemble, train the next version.

---

## Directory Structure

```
BirdCLEF/
├── src/            # All Python source code
├── scripts/        # Bash training orchestration scripts
├── data/
│   ├── raw/        # Competition data (audio, CSVs)
│   └── processed/  # Fold splits, pseudo-labels, evaluation outputs
├── models/         # Saved .pt checkpoints and .onnx exports
│   └── stage1/     # Clean Stage 1 weights (warm-start source)
├── jupyter/        # Kaggle submission notebooks
│   └── sed/
└── log/            # Timestamped training logs
```

---

## Python Scripts (`src/`)

### `config.py` — Global Configuration

Central constants and path definitions used by all other scripts.

**Key settings:**
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `SAMPLE_RATE` | 32,000 Hz | Audio sample rate |
| `DURATION` | 20 s | Training chunk length |
| `N_MELS` | 224 | Mel spectrogram height |
| `N_FFT` | 4096 | FFT window size |
| `HOP_LENGTH` | 1252 | → 512 time frames for 20 s |
| `BACKBONE` | `tf_efficientnet_b0.ns_jft_in1k` | Default CNN backbone |
| `N_CLASSES` | 234 | Species count (including insect sonotypes) |
| `BATCH_SIZE` | 64 | Training batch size |
| `EPOCHS` | 15 | Default epoch count (Stage 1) |
| `LR` | 5e-4 | Initial learning rate (AdamW) |
| `T_0` | 5 | CosineAnnealingWarmRestarts period |
| `N_FOLDS` | 5 | Stratified k-fold count |

Also provides `get_species_list()` and `get_species_index()` (LRU-cached) which give a stable alphabetically sorted mapping from species label string to integer index, reading from `data/raw/taxonomy.csv`.

---

### `utils.py` — Audio Utilities

Low-level audio loading and mel spectrogram conversion shared by every other script.

**`load_audio(path)`**
- Reads any `.ogg` or `.wav` file using `soundfile`.
- Converts stereo to mono by averaging channels.
- Resamples to 32 kHz using `torchaudio.functional.resample` if needed.
- Returns a 1-D `float32` NumPy array.

**`pad_or_crop(waveform, target_len, random_crop)`**
- Ensures the waveform is exactly `CHUNK_SAMPLES` (640,000) samples long.
- Short clips are tiled (looped) to fill the target length.
- During training: random crop. During validation/inference: front crop from offset 0.

**`waveform_to_mel(waveform)`**
- Converts a `(640000,)` array to a `(3, 224, 512)` float32 tensor.
- Applies `torchaudio.transforms.MelSpectrogram` then `AmplitudeToDB` (power=2, norm=slaney, top_db=80).
- Per-sample normalises the dB spectrogram to `[0, 1]`.
- Repeats the single channel across 3 channels to match ImageNet-pretrained backbone expectations.

The mel transform object is built once at module import time (stateless and thread-safe), so DataLoader workers can call it concurrently without re-initialising.

---

### `model.py` — Neural Network Architecture

Implements `BirdSEDModel`, a Sound Event Detection model that produces both clip-level and frame-level predictions.

#### `GEMFrequencyPool`
Generalised Mean (GEM) pooling over the frequency (height) dimension of the feature map:
- Input: `(B, C, F, T)` → Output: `(B, C, T)`
- Pooling exponent `p` is a **learnable parameter** initialised to 3.
- Collapses the frequency axis while retaining temporal resolution for per-frame prediction.

#### `BirdSEDModel`
```
Input mel: (B, 3, 224, 512)
    ↓
timm CNN backbone (features_only, last stage only — out_indices=(4,))
    ↓  (B, C, F', T')
GEMFrequencyPool
    ↓  (B, C, T')
            ├── cls_conv  (Conv1d, 1×1)  → frame_logits (B, T', 234)
            └── att_conv  (Conv1d 3×3 → BN → ReLU → Dropout → Conv1d 1×1)
                                         → att_weights  (B, T', 234)  [softmax over T']
clip_logits = (frame_logits × att_weights).sum(dim=T')   → (B, 234)
```

The model returns a dict with `clip_logits`, `frame_logits`, and `att_weights`. Training uses `clip_logits` for BCE loss. Pseudo-labelling uses `frame_logits` to get per-5-second-segment predictions.

**Backbone selection** (in priority order):
1. `tf_efficientnet_b0.ns_jft_in1k` ← current default
2. `regnety_016.tv2_in1k` ← added in Stage 3
3. `efficientnet_b3`, `eca_nfnet_l0` (future)

---

### `dataset.py` — Training Dataset

`BirdTrainDataset` is a PyTorch `Dataset` that handles the supervised focal-recording training data.

**Initialisation:**
- Reads `train_folds.csv` (the fold-split version of `train.csv`).
- **Rare species oversampling**: species with fewer than 10 clips are duplicated by row repetition to reach the minimum floor (before fold-level splitting).
- Optionally scans a background noise directory for `.ogg`/`.wav` files for noise injection.

**`__getitem__` pipeline:**
1. Load waveform from `data/raw/train_audio/`.
2. Build target vector: `labels` (float32, 0/1) and `secondary_mask` (float32, 0/1).
   - Primary label → `labels[idx] = 1.0`, `mask[idx] = 1.0` (included in loss).
   - Secondary labels → `labels[idx] = 1.0`, **`mask[idx] = 0.0`** (excluded from loss — location within clip is unknown).
3. If augmenting:
   - **Random gain** ±6 dB (p=0.5)
   - **Circular time shift** ±25% of clip length (p=0.5)
   - **Background noise injection** at 5–15% gain (p=0.3)
   - **Fixed 0.5/0.5 MixUp**: blend with a randomly selected sample; merge labels with element-wise max; merge masks with element-wise max (union of primary positions).
4. Convert waveform to mel spectrogram via `waveform_to_mel`.
5. Return `(mel, labels, secondary_mask)`.

Mel computation runs inside the DataLoader worker processes, overlapping with GPU forward/backward passes.

---

### `eda.py` — Exploratory Data Analysis & Fold Splitting

One-time script that analyses the training data and generates `data/processed/train_folds.csv`.

**Outputs:**
- `data/processed/eda_report.txt`: Human-readable stats including global counts, collection breakdown (XC vs iNat), taxonomy class breakdown (Aves/Amphibia/Insecta/…), samples-per-species distribution, rare species list, and soundscape segment statistics.
- `data/processed/train_folds.csv`: `train.csv` plus a `fold` column (0–4), created using `MultilabelStratifiedKFold` from `iterstrat` to ensure rare species appear in all folds.

**Key decisions encoded here:**
- Insect sonotypes (e.g. `47158son16`) are treated as unique classes — 234 total.
- Rare species (<10 samples) are flagged; `BirdTrainDataset` handles duplication.

---

### `train.py` — Stage 1 Supervised Training

Trains one fold of the `BirdSEDModel` on the expert-labelled focal recordings.

**Validation set:**
- Built from `data/raw/train_soundscapes_labels.csv` — the expert-annotated 5-second segments from Pantanal field recordings.
- This gives ground-truth evaluation in the same domain as the test set.
- Mel spectrograms are precomputed once at the start of each fold and kept in CPU RAM.

**Training loop (per epoch):**
1. Load mel batch from `BirdTrainDataset` workers (augmented, mixed-up).
2. Apply **SpecAugment** on GPU:
   - `FrequencyMasking(freq_mask_param=27)` (p=0.3)
   - `TimeMasking(time_mask_param=64)` (p=0.3)
3. Optional GPU augmentations via `torch_audiomentations`:
   - `PitchShift` ±2 semitones (p=0.3)
   - `Shift` (time shift, p=0.5)
4. Forward pass with `torch.amp.autocast(dtype=torch.bfloat16)`.
5. `BCEWithLogitsLoss` with `secondary_mask` applied: `loss = (bce_per_element × mask).mean()` — masked positions contribute zero gradient.
6. Gradient clipping at norm 5.0, AdamW step.
7. `CosineAnnealingWarmRestarts` LR schedule (T_0=5, eta_min=1e-6).

**Checkpoint saved** only when a new best validation ROC-AUC is achieved.  
Output: `models/sed_<backbone>_fold<N>_seed<S>.pt` (state dict only for ONNX compatibility).

**CLI:**
```bash
python src/train.py --fold 0 --backbone tf_efficientnet_b0.ns_jft_in1k --epochs 15 --seed 42
```

---

### `pseudo_label.py` — Pseudo-Label Generation

Generates soft pseudo-labels for all ~10,658 train soundscape files using the trained fold ensemble.

**Algorithm:**
1. Load all N-fold checkpoints into a list of `BirdSEDModel` instances (eval mode, BF16).
2. For each soundscape file:
   - Slide a **20-second window** with a **5-second stride** across the full audio.
   - For each window: compute mel → run all N models → average `frame_logits` across folds (ensemble).
   - Divide the T' backbone time frames into 4 groups (one per 5-second sub-segment).
   - Take the **max over frames** within each sub-segment → per-segment prediction vector `(N_CLASSES,)`.
   - Accumulate predictions for each 5-second segment index (multiple windows overlap the same segment).
3. Divide accumulated predictions by overlap count → average probability for each 5-second segment.
4. Write `data/processed/pseudo_labels_v{N}.csv` with columns: `filename`, `start_time`, `end_time`, and one column per species.

**CLI:**
```bash
python src/pseudo_label.py \
    --backbone tf_efficientnet_b0.ns_jft_in1k \
    --folds 0,1,2,3,4 \
    --version 2 \
    --ckpt-version 2   # loads sed_*_v2.pt checkpoints
```

The `--ckpt-version` argument allows selecting versioned checkpoints (e.g. `_v2.pt`) independently of the output `--version` numbering.

---

### `self_train.py` — Noisy Student Self-Training

Re-trains the model mixing supervised focal clips with soft pseudo-labelled soundscape segments. This is the core self-training ("noisy student") loop.

**`PseudoLabelDataset`:**
- Loads `pseudo_labels_v{N}.csv`.
- **Leakage prevention**: excludes all rows whose filename appears in `train_soundscapes_labels.csv` (the expert-labelled validation files). This was the critical fix between v1 and v2.
- Applies a **power transform** to the soft probabilities: `p^power` (default power=1.5 sharpens high-confidence predictions; p<1 would smooth them).
- Lazy-loads soundscape waveforms with an LRU cache (≤200 files at once) to reduce disk I/O.
- Each item: load the 5-second waveform segment, apply random augmentation (gain, time shift, background noise), convert to mel.
- Returns `(mel, soft_labels)` — no secondary mask since these are soft pseudo-labels.

**Mixed batch construction:**
- A `WeightedRandomSampler` ensures a configurable mix ratio (default 50/50) of focal clips and pseudo-label segments within each batch.
- Both datasets feed into a single collated batch.

**Training loop:**
- Same BF16 autocast + SpecAugment + gradient clipping as Stage 1.
- Warm-starts from a Stage 1 checkpoint (`--init-ckpt`) if provided.
- Output checkpoint versioned via `--version N`: `models/sed_<backbone>_fold<N>_seed<S>_v{N}.pt`.

**CLI:**
```bash
python src/self_train.py \
    --fold 0 \
    --backbone tf_efficientnet_b0.ns_jft_in1k \
    --epochs 30 \
    --seed 42 \
    --pseudo-csv data/processed/pseudo_labels_v1.csv \
    --pseudo-power 1.5 \
    --init-ckpt models/stage1/sed_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42.pt \
    --version 2
```

---

### `evaluate.py` — Ensemble Evaluation

Evaluates a trained ensemble on the expert-labelled soundscape validation set and identifies hard species.

**Outputs (written to `data/processed/`):**
- `hard_species_stage1.txt` — 30 worst-scoring species by ROC-AUC.
- `per_species_auc_stage1.csv` — full per-species AUC table (234 rows).
- `eval_stage1_predictions.csv` — raw prediction scores for all segments.

**Algorithm:**
- Loads N-fold checkpoints (supports `--version` suffix).
- For each row in `train_soundscapes_labels.csv`: extracts the labelled 5-second segment, runs ensemble inference, accumulates predictions.
- Computes macro ROC-AUC and per-species ROC-AUC using `sklearn.metrics.roc_auc_score`.
- Skips classes with no positive labels in the validation set (mirroring competition metric).

**CLI:**
```bash
python src/evaluate.py --version 2   # loads _v2.pt checkpoints
```

---

### `export_onnx.py` — ONNX Export

Exports trained `.pt` checkpoints to `.onnx` format for potential CPU inference.

**Process:**
1. Loads a checkpoint into `BirdSEDModel`.
2. Wraps it in `BirdSEDModelONNX` — a thin `nn.Module` that returns only `sigmoid(clip_logits)` as a flat tensor (ONNX cannot return dicts).
3. Exports using `torch.onnx.export` (TorchScript exporter, opset 17) with a dynamic batch axis.
4. Validates the exported model with `onnxruntime` on a dummy input.

**Important note:** ONNX runtime is not available in the Kaggle no-internet environment. The Kaggle submission notebook uses PyTorch CPU inference (`.pt` checkpoints, `map_location="cpu"`) instead. ONNX export is available locally for debugging or potential future use.

**CLI:**
```bash
python src/export_onnx.py --backbone tf_efficientnet_b0.ns_jft_in1k --folds 0,1,2,3,4 --seed 42
```

---

### `eda.py` — EDA & Fold Construction

One-time analysis script. Key outputs:

| Output | Description |
|--------|-------------|
| `eda_report.txt` | Stats: 46K clips, 234 species, sample distribution, XC vs iNat split |
| `train_folds.csv` | `train.csv` + `fold` column (MultilabelStratifiedKFold, 5 folds) |

Species with <10 samples are flagged and later duplicated by `BirdTrainDataset`.

---

### `convert_perch_tflite.py` / `download_perch.py`

Utility scripts for the Google Perch baseline (Phase 0.5):
- `download_perch.py`: Downloads the Perch v4 TFLite model from the Google `bird-vocalization-classifier` hub.
- `convert_perch_tflite.py`: Wraps the TFLite model and runs inference for the initial Perch baseline submission.

These are not part of the main SED training pipeline; they produced the Phase 0.5 LB score of 0.590.

---

## Shell Scripts (`scripts/`)

### `train_stage1.sh` — Supervised Stage 1 Training

Runs 5-fold Stage 1 supervised training sequentially.

```
for FOLD in 0 1 2 3 4:
    python src/train.py --fold $FOLD --epochs 15 --seed 42
```

- Cleans per-fold log files before starting.
- Each fold's stdout is tee'd to a timestamped fold log.
- Saves `models/stage1/sed_<backbone>_fold{N}_seed42.pt`.
- Runtime: ~2.5 hours per fold (~12.5 hours total on GB10).

**Usage:**
```bash
nohup bash scripts/train_stage1.sh > log/train_stage1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

### `self_train_stage2.sh` — Self-Training Stage 2

Runs 5-fold self-training warm-started from Stage 1 checkpoints, using `pseudo_labels_v1.csv` with power=1.5.

```
PSEUDO_CSV=data/processed/pseudo_labels_v1.csv
PSEUDO_POWER=1.5
VERSION=2

for FOLD in 0 1 2 3 4:
    python src/self_train.py \
        --fold $FOLD --epochs 30 --pseudo-power 1.5 --version 2 \
        --init-ckpt models/stage1/sed_..._fold{N}_seed42.pt \
        --pseudo-csv $PSEUDO_CSV
```

Produces `models/sed_<backbone>_fold{N}_seed42_v2.pt`.  
Result: local val 0.7858, LB **0.762**.

**Usage:**
```bash
nohup bash scripts/self_train_stage2.sh > log/self_train_stage2_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

### `self_train_stage3.sh` — Self-Training Stage 3 (Two Backbones)

Runs two sequential passes of 5-fold self-training using `pseudo_labels_v2.csv`.

**Pass 1 — EfficientNet-B0 (warm-start from v2, output v3):**
```
for FOLD in 0 1 2 3 4:
    python src/self_train.py \
        --backbone tf_efficientnet_b0.ns_jft_in1k \
        --init-ckpt models/sed_..._fold{N}_seed42_v2.pt \
        --version 3 --epochs 30 --pseudo-power 1.5
```

**Pass 2 — RegNetY-016 (fresh ImageNet init, output v1):**
```
for FOLD in 0 1 2 3 4:
    python src/self_train.py \
        --backbone regnety_016.tv2_in1k \
        --version 1 --epochs 30 --pseudo-power 1.5
        # no --init-ckpt → fresh init from ImageNet weights
```

Produces:
- `models/sed_tf_efficientnet_b0..._fold{N}_seed42_v3.pt`
- `models/sed_regnety_016.tv2_in1k_fold{N}_seed42_v1.pt`

Ensemble of 10 models (5 B0-v3 + 5 RegNetY-v1) provides backbone diversity.  
Estimated runtime: ~26 hours total.

**Usage:**
```bash
nohup bash scripts/self_train_stage3.sh > log/self_train_stage3_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

### `train_and_push.sh` — Full Pipeline Automation

End-to-end automation script that chains the full pipeline:

1. **Clean** old per-fold training logs.
2. **Self-training** — runs `self_train_stage2.sh`.
3. **Evaluate** — runs `evaluate.py --version 1` for local ROC-AUC.
4. **Push models dataset** — uploads checkpoint directory to `stevewatson999/birdclef2026-sed-models` via `kaggle datasets version`.
5. **Push notebook** — pushes the inference notebook via `kaggle kernels push`.

This script is used to streamline the evaluate → submit cycle.

**Usage:**
```bash
nohup bash scripts/train_and_push.sh > log/train_and_push_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

## End-to-End Pipeline

```
                    ┌──────────────────────────────────────┐
                    │  data/raw/                           │
                    │    train_audio/ (~46K OGG clips)     │
                    │    train_soundscapes/ (10,658 OGG)   │
                    │    train_soundscapes_labels.csv      │
                    │    train.csv / taxonomy.csv          │
                    └───────────────┬──────────────────────┘
                                    │
                             src/eda.py
                                    │
                    ┌───────────────▼──────────────────────┐
                    │  data/processed/train_folds.csv      │
                    │  (5-fold multilabel stratified split) │
                    └───────────────┬──────────────────────┘
                                    │
              scripts/train_stage1.sh  (src/train.py ×5 folds)
                                    │  15 epochs, BF16, AdamW+CosineWarmRestart
                                    │  BCE loss with secondary-label masking
                                    │
                    ┌───────────────▼──────────────────────┐
                    │  models/stage1/                      │
                    │    sed_..._fold{0-4}_seed42.pt        │
                    └───────────────┬──────────────────────┘
                                    │
              src/pseudo_label.py  (sliding 20s window, 5s stride)
                                    │  5-fold ensemble → frame_logits → max-per-5s-segment
                                    │
                    ┌───────────────▼──────────────────────┐
                    │  data/processed/pseudo_labels_v1.csv  │
                    │  (127,896 rows, soft probabilities)   │
                    └───────────────┬──────────────────────┘
                                    │
       scripts/self_train_stage2.sh  (src/self_train.py ×5 folds)
                                    │  30 epochs, warm-start, power=1.5
                                    │  val leakage excluded
                                    │
                    ┌───────────────▼──────────────────────┐
                    │  models/sed_..._fold{0-4}_seed42_v2.pt│
                    │  local val 0.7858 / LB 0.762         │
                    └───────────────┬──────────────────────┘
                                    │
         src/pseudo_label.py  (--ckpt-version 2 --version 2)
                                    │
                    ┌───────────────▼──────────────────────┐
                    │  data/processed/pseudo_labels_v2.csv  │
                    └───────────────┬──────────────────────┘
                                    │
       scripts/self_train_stage3.sh  (Pass 1: EffB0-v3, Pass 2: RegNetY016-v1)
                                    │
              src/evaluate.py  →  local ROC-AUC
                                    │
              scripts/train_and_push.sh  → Kaggle dataset + notebook push
                                    │
              jupyter/sed/birdclef2026-sed-inference.ipynb
                                    │  PyTorch CPU inference, sliding window
                                    │  5-fold ensemble, majority-vote / mean
                                    ▼
                        Kaggle submission CSV
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **SED architecture** (clip + frame predictions) | Frame-level granularity enables per-5-second pseudo-labelling; clip-level logits used for training loss |
| **GEM pooling over frequency** | Learnable pooling exponent adapts to species frequency ranges better than average/max pooling |
| **Secondary label masking to 0** | Secondary labels have unknown temporal location within the clip; including them in the loss degrades calibration |
| **MixUp at 0.5/0.5 fixed weight** | Increases effective training diversity; element-wise max label merge preserves all species |
| **Power transform (p=1.5) on pseudo-labels** | Sharpens high-confidence predictions; reduces noise from near-zero scores |
| **Leakage prevention in self-training** | The 66 expert-labelled soundscape files are excluded from pseudo-label training data — they serve exclusively as validation |
| **BF16 (not FP16)** | NVIDIA GB10 (Blackwell) supports BF16 natively; no `GradScaler` needed unlike FP16 |
| **No `torch.compile`** | GB10 is sm_121, exceeding PyTorch's max sm_120, causing silent hangs during compiled kernel execution |
| **PyTorch CPU inference on Kaggle** | `onnxruntime` is unavailable in the no-internet Kaggle environment; `.pt` checkpoints loaded with `map_location="cpu"` |
| **Mel spectrogram in DataLoader workers** | Overlaps CPU mel computation with GPU forward/backward, hiding latency |

---

## Environment

- **Hardware**: NVIDIA GB10 (Blackwell, 128 GB unified memory)
- **Conda env**: `kaggle` (`/home/swatson/miniconda3/envs/kaggle/`)
- **Key packages**: PyTorch, torchaudio, timm, soundfile, scikit-learn, tqdm, iterstrat
- **Python**: run all scripts with `nohup ... > log/<name>_$(date +%Y%m%d_%H%M%S).log 2>&1 &`
