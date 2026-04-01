# BirdCLEF+ 2026 — Competition Plan

**Competition**: [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)  
**Host**: Cornell Lab of Ornithology  
**Start**: March 11, 2026 | **Entry Deadline**: May 27, 2026 | **Final Submission**: June 3, 2026  
**Metric**: Macro-averaged ROC-AUC (skips classes with no true positive labels)  
**Prizes**: $50,000 total ($15K / $10K / $8K / $7K / $5K)  
**Submission type**: CPU-only notebook, ≤90 min runtime, no internet  

---

## Current State

| Item | Value |
|------|-------|
| Best Kaggle LB score | **0.908** (Perch v2 + LogReg probes + site/hour priors, 2026-03-30) |
| Best local val ROC-AUC | 0.7958 (EffB0-v4 soundscape val) |
| Currently running | **#25A — Dual-probe ensemble** (LogReg + LightGBM rank-average). Target: > 0.908 |
| Architecture | SED — EfficientNet-B0/B3 + GEM pool + Conv1d attention |
| Loss | **BCE** (production, validated); ASL incompatible with BCE warm-start (v10/v11/v12 all failed) |
| Spectrogram | **PCEN** (v8 test) / AmplitudeToDB+min-max (production v17 notebook) |
| Active notebook | `jupyter/sed/birdclef2026-sed-inference.ipynb` (v17, EffB0-v4 + RegNetY-v2) |
| Inference notebook URL | https://www.kaggle.com/code/stevewatson999/birdclef-2026-sed-inference |

### LB Submission History
| Date | Approach | LB score | Notes |
|------|----------|----------|-------|
| Mar 16 | Perch TFLite zero-shot | 0.590 | 158/234 species mapped |
| Mar 16 | SED EffB0 Stage 1 | 0.752 | 5-fold supervised |
| Mar 19 | SED EffB0-v2 self-train | 0.762 | Leakage fixed |
| Mar 20 | SED EffB0-v3 + RegNetY-v1 | 0.762 | No improvement from RegNetY |
| Mar 21 | SED EffB0-v4 + RegNetY-v2 | **0.769** | Best to date (warm-start RegNetY) |
| Mar 24 | Ensemble v1: SED-v6 70% + Perch MLP 30% | 0.667 | SED v6 degraded by Perch KD |
| Mar 24 | Ensemble v2: Perch MLP 70% + SED 30% | 0.573 | Perch MLP fails on soundscapes |
| Mar 24 | SED-only v5 (EffB0-v5 + EffB3-v1, 10 models) | 0.682 | EffB3-v1 fresh init dragged ensemble down |
| Mar 25 | Stage 6 — PCEN+ASL+Freq-MixStyle+soup (EffB0-v7, 5-fold, scratch) | 0.705 ❌ | PCEN trained on v4 log-mel warm-start → mismatch |
| Mar 26 | Restored v17 notebook (EffB0-v4 + RegNetY-v2, log-mel) | **0.769** ✅ | Regression fix |
| Mar 27 | Single-fold PCEN ablation v8 (PCEN-only, BCE, scratch, 20ep) | 0.7214 local val only | Peak ep14; training loss still declining at ep20 — likely under-training, not PCEN bug |
| Mar 27 | PCEN ablation v9 (PCEN+BCE, scratch, 40ep, killed ep22) | 0.7308 local val only (ep7) then plateau | Bell-curve pattern: peak ep7 then decline — overfitting to log-mel pseudo-labels, not PCEN bug |
| Mar 27 | ASL γ−=2 warm-start v4 (v10, 9ep shown) | 0.7748 (ep1) → 0.7221 (ep8) ❌ | Peak ep1 = warm-start value; immediate monotone decline; gate FAILED (needed ≥0.79) |
| Mar 27 | ASL γ−=0.5 warm-start v4 (v11, 6ep shown) | 0.7917 (ep1) → 0.7679 (ep4) ❌ | Same peak-ep1 pattern; softer γ didn't help; ASL+warm-start fundamentally incompatible |
| Mar 27 | ASL focal-only v12 (v12, ep1) | 0.7079 (ep1) ❌ | Much worse — model penalized for detecting species in pseudo audio half; focal-only loss is broken |
| Mar 27 | PCEN Stage 1 focal-only v13 (EffB0, PCEN, BCE, scratch, 20ep) | 0.7188 local val (ep18), **0.762 LB** | Single-fold scratch PCEN — only −0.007 vs v4 10-model ensemble (0.769); validates PCEN strongly |
| Mar 28 | PCEN self-train v14 (EffB0, PCEN, BCE, warm-start v13, 30ep, PCEN pseudo-labels) | 0.7494 local val (ep12), **0.749 LB** ❌ | Self-train with PCEN pseudo-labels degraded LB vs v13 (0.762→0.749); PCEN pseudo-labels adding noise |
| Mar 29 | PCEN 5-fold self-train v15 (EffB0, PCEN, BCE, warm-start v13, 30ep, 5-fold PCEN pseudo-labels) | 0.7259 best fold val, **0.754 LB** ❌ | 5-fold self-train STILL degrades vs single-fold v13 (0.762). Self-train is net negative for PCEN. |
| Mar 29 | PCEN 5-fold Stage 1 v13 (EffB0, PCEN, BCE, scratch, 25ep, focal-only) | **0.765 LB** | 5-fold no-self-train > v15 (0.754) but < best 0.769. Self-train confirmed harmful; PCEN needs ensemble diversity to compete. |
| Mar 29 | PCEN v13 5-fold + temporal smoothing (gaussian sigma=1) | **0.773 LB** ✅ | Smoothing added +0.008 over v13 baseline (0.765). First time beating 0.769. |
| Mar 30 | Perch v2 + LogReg probes + site/hour priors + dual smoothing + temp scaling | **0.908 LB** ✅ | **New best.** Adapted from public 0.908 notebook. No local training — all inference-time sklearn. Massive jump from 0.773. |
| Mar 31 | #22: Perch probe upgrade (PCA 64, wider temporal features, LogReg) | **0.904 LB** ❌ | PCA 64 overfits on ~708 samples → −0.004 regression. LightGBM grid searched but not used in submit. |
| Mar 31 | #22b: Revert PCA to 32 (confirm 0.908 baseline) | **0.908 LB** ✅ | Confirmed: PCA 64 was the cause of regression. Baseline restored. |

### LB Gap Analysis (2026-03-24)
| Approach | LB score | Delta vs ours |
|----------|----------|---------------|
| **Us (best, Iter 4)** | **0.769** | — |
| Perch TFLite zero-shot | 0.825 | +0.056 |
| Perch v2 + MLP head (public notebook) | 0.905–0.910 | +0.14 |
| SED + ASL + PCEN + Perch soft KD | ~0.918 | +0.15 |
| SED + model soup + Freq-MixStyle | ~0.921 | +0.15 |
| Top LB (#1 yuanzhe zhou) | **0.9334** | +0.164 |

**Root causes of the 0.769 → 0.93 gap:**
1. **Missing PCEN** — removing it costs −0.049 soundscape val AUC; we never had it  
2. **BCE instead of ASL** — costs −0.02–0.03  
3. **No Freq-MixStyle** — +1–3pp soundscape val AUC by simulating different ARU frequency responses  
4. **No model soup** — single best checkpoint per fold; checkpoint averaging (SWAD-style) improves OOD generalization  
5. **No circ-shift augmentation** — +0.005–0.01  
6. **No dual loss (clip + frame)** — slight improvement from frame-level supervision  

### ❌ Perch MLP Probe — Dead End (2026-03-24)

The Perch MLP approach was evaluated extensively in `perch_v2/` and **does not work for us**:

| Version | Change | Val AUC | LB AUC |
|---------|--------|---------|--------|
| v1 | BN + averaged emb | 0.739 | 0.690 |
| v2 | BN + averaged emb + L2-norm | 0.777 | 0.723 |
| v3 | LayerNorm + averaged emb | 0.700 | — |
| v4 | No norm + averaged emb | 0.655 | — |
| v5 | No norm + per-window emb | 0.801 | ~0.49 (est. from ensemble 0.573) |

**Why it fails on Kaggle despite 0.90 local val AUC:**
- Trained on per-window embeddings from clean 5s focal clips; test soundscapes are noisy 60s field recordings
- Perch embeddings from noisy soundscapes are muddled — shallow MLP (1536→512→234) cannot handle this
- MLP classifies each 5s window independently with no temporal context (SED uses 20s with attention pooling)
- 31/234 species (insect sonotypes) are not in Perch's vocabulary → always near-zero predictions
- Public notebooks claiming 0.905 with Perch MLP may use different training data, deeper models, or inference tricks we haven't replicated

**Also failed: Perch KD for SED** — SED v6 trained with Perch soft labels (×10 oversample) scored *worse* than v5 (0.737 vs 0.799 val AUC). 10× oversampling caused catastrophic forgetting of focal data. The 31 missing species got zero-signal KD labels, actively harming those classes.

**Conclusion: Abandon Perch as standalone model or direct KD teacher.** The path forward is improving SED itself (PCEN, ASL, Freq-MixStyle, model soup). Perch KD could work with careful tuning (lower oversample ratio, mask missing species) but is not the priority.

*Update this table after each LB submission.*

---

## Table of Contents

1. [Competition Task](#1-competition-task)
2. [Data Overview](#2-data-overview)
3. [Project Directory Structure](#3-project-directory-structure)
4. [Environment & Infrastructure](#4-environment--infrastructure)
5. [Phase 0 — Data Download & EDA](#phase-0--data-download--eda)
6. [Phase 0.5 — Google Perch Quick Baseline](#phase-05--google-perch-quick-baseline)
7. [Phase 1 — Baseline SED Model (Supervised)](#phase-1--baseline-sed-model-supervised)
8. [Phase 2 — Pseudo-Labeling & Noisy Student Self-Training](#phase-2--pseudo-labeling--noisy-student-self-training)
9. [Phase 3 — Multi-Iterative Pseudo-Labeling](#phase-3--multi-iterative-pseudo-labeling)
10. [Phase 4 — Ensemble & PyTorch Export](#phase-4--ensemble--pytorch-export)
11. [Phase 5 — Kaggle Inference Notebook](#phase-5--kaggle-inference-notebook)
12. [Inference Optimization Constraints](#inference-optimization-constraints)
13. [Experiment Tracking](#experiment-tracking)
14. [Key Lessons from BirdCLEF 2025 Top Solutions](#key-lessons-from-birdclef-2025-top-solutions)
15. [Hard Constraints & Known Pitfalls](#hard-constraints--known-pitfalls)
16. [Prioritized Action Plan](#prioritized-action-plan)

---

## 1. Competition Task

Build ML models that identify **234 wildlife species** (birds, amphibians, mammals, reptiles, insects) from **passive acoustic monitoring (PAM)** audio recordings in the **Brazilian Pantanal wetlands**.

- **Input**: 1-minute field soundscape recordings at 32 kHz
- **Output**: Per-species probability for each 5-second window
- **Challenge**: Multi-label, 234 classes, highly class-imbalanced, few samples for rare species, domain shift between curated training clips and field soundscapes loaded with background noise
- **Special difficulty**: Insects and amphibians (particularly insect sonotypes) are drastically under-represented; some species only exist in `train_soundscapes` labels, not in `train_audio`

### Competition-specific aspects (2026 vs 2025)
| | BirdCLEF+ 2025 | BirdCLEF+ 2026 |
|-|-|-|
| Location | Middle Magdalena Valley, Colombia | Brazilian Pantanal |
| Classes | 206 | **234** |
| Data size | ~12 GB | **16.14 GB** |
| Expert-labeled soundscapes | No | **Yes** (train_soundscapes_labels.csv) |

The presence of **expert-labeled train soundscapes** is a major difference in 2026 — use these as high-quality ground truth in addition to the focal recordings.

---

## 2. Data Overview

### Files

| File / Folder | Size / Count | Description |
|--------------|-------------|-------------|
| `train_audio/` | ~46K OGG files | Focal recordings from Xeno-canto (XC) and iNaturalist (iNat), 32 kHz. Variable length. Named `[XC/iNat][file_id].ogg` |
| `test_soundscapes/` | ~600 OGG files, 1-min each | Hidden at submission time; field recordings from Pantanal sites. Named `BC2026_Test_<id>_<site>_<date>_<time>.ogg` |
| `train_soundscapes/` | — | Additional field recordings from same Pantanal sites. Some overlap in sites (NOT in time) with test. |
| `train_soundscapes_labels.csv` | — | Expert-annotated 5-second segments: `filename`, `start`, `end`, `primary_label` (semicolon-separated species) |
| `train.csv` | 1 row/file | `primary_label`, `secondary_labels`, `latitude`, `longitude`, `author`, `filename`, `rating`, `collection` |
| `taxonomy.csv` | 234 rows | Species info: `primary_label`, iNat taxon ID, class (Aves, Amphibia, Mammalia, Insecta, Reptilia) |
| `sample_submission.csv` | — | `row_id`, 234 species columns. One row per 5-second window |
| `recording_location.txt` | 204 B | High-level location description |

### train.csv Key Fields
| Column | Notes |
|--------|-------|
| `primary_label` | eBird code (birds) or iNat taxon ID (non-birds). Maps to submission column names. |
| `secondary_labels` | Other species audible in recording. Incomplete — use as soft labels |
| `rating` | 0–5 (higher = better quality, XC only; iNat has no ratings). Filter on rating ≥ 3.5 for clean training. |
| `latitude` / `longitude` | Geographic origin of clip |
| `collection` | `XC` or `iNat` |
| `filename` | Relative path inside `train_audio/` |

### Submission row_id format
```
BC2026_Test_0001_S05_20250227_010002_20
                                      ↑ end_time in seconds (e.g. 20 = window 00:15-00:20)
```
- 12 windows × ~600 soundscapes ≈ **7,200 rows**
- One row per 5-second window, 234 probability columns each

### Species classes (234 total)
- **Aves** (birds): eBird codes, majority of classes
- **Amphibia**: iNat taxon IDs
- **Mammalia**: iNat taxon IDs (e.g., Jaguar = 41970)
- **Reptilia**: iNat taxon IDs
- **Insecta**: Mix of iNat IDs + sonotypes (e.g., `47158son16` = insect sonotype 16)

**Critical**: Insect sonotypes represent family-level labels for hard-to-identify insects. Some sonotypes are very location-specific and must be treated as unique classes despite not having a true species ID.

---

## 3. Project Directory Structure

```
kaggle/BirdCLEF/
├── CLAUDE.md                      # Execution guidelines
├── plan.md                        # This file
├── data/
│   ├── raw/                       # Competition data (downloaded from Kaggle)
│   │   ├── train_audio/           # ~46K OGG clips
│   │   ├── train_soundscapes/     # Field recordings (labeled subset)
│   │   ├── test_soundscapes/      # Hidden at inference time
│   │   ├── train.csv
│   │   ├── train_soundscapes_labels.csv
│   │   ├── taxonomy.csv
│   │   ├── sample_submission.csv
│   │   └── recording_location.txt
│   └── processed/
│       ├── train_folds.csv        # Train data with fold assignments
│       ├── pseudo_labels_v1.csv   # Pseudo-labels from Stage 1 ensemble
│       ├── pseudo_labels_v2.csv   # Pseudo-labels after 1st self-training iteration
│       ├── pseudo_labels_v3.csv   # etc.
│       └── eda_report.txt
├── src/
│   ├── __init__.py
│   ├── config.py                  # Central config (paths, mel params, training params)
│   ├── dataset.py                 # Dataset + augmentation classes
│   ├── model.py                   # SED model (EfficientNet backbone + SED head)
│   ├── train.py                   # Main supervised training script
│   ├── self_train.py              # Noisy Student self-training script
│   ├── pseudo_label.py            # Generate pseudo-labels from soundscapes
│   ├── ensemble.py                # Weighted average ensemble
│   ├── export_onnx.py             # (unused — ONNX not available on Kaggle)
│   ├── evaluate.py                # Local ROC-AUC evaluation against train_soundscapes_labels.csv
│   └── utils.py                   # Audio loading, mel spectrogram, padding helpers
├── scripts/
│   ├── train_stage1.sh            # Run full 5-fold Stage 1 training
│   ├── pseudo_label_soundscapes.sh
│   └── self_train_stage2.sh
├── models/                        # Saved .pt checkpoints (gitignored)
│   ├── stage1_effb0_fold0/
│   │   ├── best.pth
│   │   └── last.pth
│   └── ...
├── jupyter/
│   └── birdclef2026-inference.ipynb   # Kaggle submission notebook
├── log/                           # nohup training logs (gitignored)
└── requirements.txt
```

---

## 4. Environment & Infrastructure

### Conda Environment
```bash
conda activate kaggle
```

### Key Dependencies
```
torch >= 2.0
torchaudio >= 2.0
timm >= 0.9            # EfficientNet, RegNetY, NFNet, EfficientViT backbones
librosa >= 0.10        
soundfile              # OGG loading
# onnxruntime NOT used — unavailable on Kaggle (no-internet env); use PyTorch .pt checkpoints
pandas, numpy, scikit-learn
torch-audiomentations  # GPU-accelerated PitchShift, TimeShift, AddBackgroundNoise
tensorboard            # optional
```

### Hardware (local training)
- NVIDIA GB10 (Blackwell, 128 GB unified memory, 273 GB/s LPDDR5X)
- Training in **BF16** — BF16 delivers 92 TFLOPS (vs 46 TFLOPS FP32) and avoids NaN issues (unlike FP16). EfficientNet/RegNetY/NFNet are all safe under BF16. See `optimize.md` in the Akkadian project for full reference.
- Inference PyTorch on CPU (ONNX not available on Kaggle — `onnxruntime` absent from no-internet env)

#### GB10 PyTorch Boilerplate (include in every training script)
```python
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cudnn.benchmark = True  # Fixed input shapes → faster

device = torch.device("cuda")
model = model.to(device)
model = torch.compile(model)  # Fused kernels → reduces memory-bandwidth bottleneck

autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
# BF16 does NOT need GradScaler — full exponent range like FP32
for batch in loader:
    with autocast_ctx:
        out = model(batch)
        loss = criterion(out, batch["labels"])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```
> **Note**: Do NOT use ONNX export — `onnxruntime` is unavailable in the Kaggle no-internet environment. Use `.pt` checkpoints with `torch.load(..., map_location="cpu")` for submission inference. `torch.compile` is also not supported (checkpoint portability issues).

### Script Execution Policy (per CLAUDE.md)
All training runs use `nohup` with timestamped logs:
```bash
conda activate kaggle
cd /home/swatson/work/MachineLearning/kaggle/BirdCLEF

# Clean old logs before new training run
rm -f log/train_*.log

nohup python -u src/train.py [args] \
    > log/train_stage1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

tail -f log/train_stage1_*.log
```

---

## Phase 0 — Data Download & EDA

### 0.1 Download Data
```bash
conda activate kaggle
kaggle competitions download -c birdclef-2026 -p data/raw/
cd data/raw && unzip birdclef-2026.zip
```

### 0.2 EDA Tasks (`src/eda.py`)
- **Class distribution**: samples per species, median/mean duration per species class
- **Taxonomy breakdown**: Aves vs Amphibia vs Mammalia vs Insecta vs Reptilia counts
- **Rating distribution**: XC ratings histogram; what fraction have rating ≥ 3.5?
- **Duration distribution**: histogram of training clip lengths (determine padding/truncation strategy)
- **Secondary labels**: frequency of co-occurring species; how to weight them during training
- **Train soundscape labels**: which species appear only in soundscapes? How many 5-sec segments per species?
- **Species with < 5 samples**: flag for special treatment (augmentation, separate model)
- Output: `data/processed/eda_report.txt`

### 0.3 Fold Assignment
- 5-fold stratified split by `primary_label` (ensure all folds have ≥ 1 sample per class)
- Save to `data/processed/train_folds.csv` with added `fold` column
- Use `MultilabelStratifiedKFold` or custom assignment for rare species

---

## Phase 0.5 — Google Perch Quick Baseline

Before building the full custom pipeline, submit a Google Perch-based notebook to establish an early LB score. This costs 1–2 hours and gives a concrete target to beat.

```python
# Google Perch (from Google Research) is a pretrained bioacoustic model
# Available on TensorFlow Hub / HuggingFace
# Install: pip install tensorflow tensorflow-hub

import tensorflow_hub as hub
model = hub.load("https://tfhub.dev/google/bird-vocalization-classifier/1")
```

**Strategy**:
1. Load the Perch model weights from a Kaggle dataset (no internet allowed at submission)
2. Run inference on test soundscapes using 5-second windows (Perch native window)
3. Map Perch's 10K+ global bird labels to the 234 BirdCLEF 2026 species
4. Submit → get LB baseline score

**Expected score**: ~0.70–0.75 ROC-AUC (strong starting point, but below custom SED)

**Tag this submission**: `Perch_baseline`

**Why to do this early**: The Akkadian project showed that early known-good baselines are essential for detecting regressions in later experiments. Fast baseline = fast feedback loop.

---

## Phase 1 — Baseline SED Model (Supervised)

This is the core starting point. Match what 1st-place BirdCLEF 2025 established as the strongest supervised baseline, **adapted for 2026's data** (234 classes, expert-labeled soundscapes available).

### 1.1 Audio Processing Pipeline (`src/utils.py`)

#### Loading
```python
# Load OGG at native 32 kHz (no resampling needed)
audio, sr = torchaudio.load(filepath)  # sr=32000
```

#### Padding / Truncation (for 20-second chunks)
```python
CHUNK_DURATION = 20  # seconds (optimal from 2025 experiments: 5→15→20→30)
CHUNK_SAMPLES = 20 * 32000  # = 640,000 samples

# Random crop if longer, left-pad with zeros if shorter (for MixUp compatibility)
def pad_or_crop(wave, expected_len, mode="random"):
    ...
```

#### Mel Spectrogram Parameters
Based on BirdCLEF 2025 1st-place optimal settings:
```python
mel_params = {
    "sample_rate": 32000,
    "n_fft": 4096,
    "hop_length": 1252,       # 20s → image width = ceil(640000 / 1252) ≈ 512
    "n_mels": 224,             # More mel bins = better Insecta/Amphibia discrimination
    "f_min": 0,
    "f_max": 16000,
    "power": 2.0,
    "top_db": 80.0,            # After converting to dB, clip at 80 dB dynamic range
    "normalized": True,        # 0-1 normalize
}
# Output shape: (224, 512) for 20-second chunk → use as (3, 224, 512) by repeating channels
```

**Alternate params to test** (3rd place configuration):
```python
mel_params_alt = {
    "sample_rate": 32000,
    "n_fft": 2048,
    "hop_length": 512,
    "n_mels": 128,  # or 96
    "f_min": 0,
    "f_max": 16000,
    "norm": "slaney",
    "mel_scale": "htk",
}
```

### 1.2 Dataset (`src/dataset.py`)

#### `BirdTrainDataset`
- Loads from `train.csv` (+ `train_soundscapes_labels.csv` for labeled soundscapes)
- Reads audio → random 20s crop → mel spectrogram → 3-channel image
- Loads from `train.csv` (+ `train_soundscapes_labels.csv` for labeled soundscapes)
- Labels: `primary_label = 1.0`; **secondary labels: mask loss to 0** (species is present *somewhere* in clip but not necessarily in current window — forcing label either way adds noise). See loss masking below.
- `rating`-based sampling: weight high-rated clips (≥ 3.5) more heavily
- Handles variable-length clips with padding
- **Minimum 10 samples per class**: rare classes duplicated (with different augmentation seeds) before fold splitting, so every class has ≥ 10 training samples. Critical for insect sonotypes.

#### Augmentations
| Augmentation | Probability | Notes |
|-------------|------------|-------|
| **MixUp** | p=1.0, weight=0.5 | On raw waveforms (`absmax` normalized). Fixed weight 0.5 (NOT beta-sampled — prevents suppressing meaningful signals). Take `max` of labels. |
| Background noise injection | p=0.3–0.5 | Mix in real environmental noise at low gain (−10 to −20 dB). Sources: BirdCLEF nocall recordings, freefield1010, warblrb, birdvox. Bridges train/test domain gap. |
| PitchShift | p=0.3 | ±2 semitones via `torch_audiomentations` on GPU (fast) |
| TimeShift | p=0.5 | ±25% shift via `torch_audiomentations` on GPU |
| SpecAugment (time mask) | p=0.3 | Randomly zero out time bands |
| SpecAugment (freq mask) | p=0.3 | Randomly zero out frequency bands |
| Random gain | p=0.5 | ±6 dB |

**Self-mixup for background-only clips** (BirdCLEF 2023): for clips where the target species is *only* a secondary label, split the 60s clip into 6×10s segments and sum-mix them into one 10s clip. This ensures the target species is acoustically present in the training window.

```python
from torch_audiomentations import Compose, PitchShift, Shift, AddBackgroundNoise

waveform_augment = Compose([
    AddBackgroundNoise(background_paths="data/noise/", p=0.4, min_snr_db=10, max_snr_db=20),
    PitchShift(min_transpose_semitones=-2, max_transpose_semitones=2, p=0.3, sample_rate=32000),
    Shift(min_shift=-0.25, max_shift=0.25, p=0.5),
], output_type="tensor")
```

**MixUp implementation detail** (critical from 2025 winner):
```python
# Normalize both clips by absmax before mixing
wave1 = wave1 / (np.abs(wave1).max() + 1e-8)
wave2 = wave2 / (np.abs(wave2).max() + 1e-8)
# Fixed 0.5 weight (NOT random Beta) — preserves both signals equally
mixed = 0.5 * wave1 + 0.5 * wave2
# Labels: element-wise max (NOT average)
mixed_label = np.maximum(label1, label2)
```

### 1.3 Model Architecture (`src/model.py`)

#### SED (Sound Event Detection) Model
Based on the 4th-place BirdCLEF 2021 SED head, consistently the best-performing architecture across multiple years:

```python
class BirdSEDModel(nn.Module):
    def __init__(self, backbone_name, num_classes=234, pretrained=True):
        super().__init__()
        # CNN backbone from timm
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained,
            features_only=True, out_indices=(4,)
        )
        
        # SED head
        self.attention = nn.Conv1d(backbone_channels, backbone_channels, 1)  
        self.classifier = nn.Conv1d(backbone_channels, num_classes, 1)
        
        # GEM frequency pooling
        self.gem_pool = GeMFrequencyPooling(p=3.0)
    
    def forward(self, x):
        # x: (B, 3, 224, 512) — 3-channel mel spectrogram
        feat = self.backbone(x)[-1]          # (B, C, H, W)
        feat = self.gem_pool(feat)           # (B, C, W) — pool freq dim with GEM
        
        # Framewise predictions (W = time frames)
        frame_att = torch.sigmoid(self.attention(feat))  # (B, C, W)
        frame_logit = self.classifier(feat)              # (B, num_classes, W)
        frame_prob = torch.sigmoid(frame_logit)          # (B, num_classes, W)
        
        # Clip-level prediction: attention-weighted sum over time
        clip_prob = (frame_prob * frame_att).sum(dim=-1) / (frame_att.sum(dim=-1) + 1e-8)
        
        return {
            "clip_prob": clip_prob,           # (B, num_classes)
            "frame_prob": frame_prob,         # (B, num_classes, T)  
            "frame_logit": frame_logit,       # (B, num_classes, T)
        }
```

#### Backbones (priority order)
| Backbone | Stage | Notes |
|----------|-------|-------|
| `tf_efficientnet_b0.ns_jft_in1k` | Stage 1 | Fast, strong baseline |
| `regnety_008.pycls_in1k` | Stage 1 | Complementary to EfficientNet |
| `efficientvit_b0.r224_in1k` | Stage 1+ | **NEW**: ~2–3× faster ONNX inference than EfficientNet-b0, comparable accuracy. 2024 3rd-place ran 5 folds in 40 min on CPU. Use in inference ensemble to fit more models in 90-min budget. |
| `tf_efficientnet_b3.ns_jft_in1k` | Stage 2+ | Better capacity |
| `tf_efficientnet_b4.ns_jft_in1k` | Stage 3+ | Largest practical |
| `eca_nfnet_l0.ra2_in1k` | Stage 3+ | Diverse architecture for ensemble. **Note**: cannot convert to OpenVINO (stdconv issue). |
| `regnety_016.tv2_in1k` | Stage 2+ | Good diversity |

All from `timm`. Note `.ns_jft_in1k` = trained with Noisy Student on JFT+ImageNet → better transfer for noisy bioacoustic data.

### 1.4 Loss Function

**CrossEntropy (CE)** preferred over BCE/Focal based on 2025 winner's findings:
```python
def compute_loss(clip_prob, frame_logit, labels, secondary_mask):
    # secondary_mask: (B, num_classes) — 0 for secondary label positions, 1 otherwise
    # Masking secondary label loss gives +0.01 LB (2024 3rd place):
    # We know species is present somewhere in clip but NOT which 5s window → don't force
    
    # Clip-level CE loss
    loss_clip = F.cross_entropy(torch.logit(clip_prob), labels)
    
    # Frame-level CE loss (max over time → clip prediction)
    frame_max = frame_logit.max(dim=-1).values   # (B, num_classes)
    loss_frame = F.cross_entropy(frame_max, labels)
    
    return 0.5 * loss_clip + 0.5 * loss_frame
```

**Two-stage loss training** (BirdCLEF 2023, worth testing): start with CE (~10 epochs for fast convergence), then fine-tune with `BCEWithLogitsLoss(reduction='sum')` (~5 epochs for better final score). CE converges 3–5× faster but BCE gives better asymptotic quality.

**Alternative to test**: `BCEWithLogitsLoss` (Focal variant) — 3rd place used Focal BCE.

### 1.5 Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| `epochs` | 15 | Stage 1 supervised |
| `batch_size` | 64 | Per GPU |
| `optimizer` | AdamW | weight_decay=1e-4 |
| `lr` | 5e-4 → 1e-6 | CosineAnnealingWarmRestarts, restart every 5 epochs |
| `scheduler` | CosineAnnealingWarmRestarts | T_0=5 epochs |
| `chunk_duration` | 20 sec | Optimal from 2025 |
| `precision` | **BF16** | GB10 has 92 TFLOPS BF16 vs 46 FP32; EfficientNet/RegNetY safe under BF16. Use `torch.amp.autocast(dtype=torch.bfloat16)` — no GradScaler needed |
| `num_folds` | 5 | Train all 5, ensemble for pseudo-labeling |
| `seed` | 42, 123, 777 | Multi-seed. Ensemble of 3 seeds consistently beats any single seed (validated in Akkadian project and BirdCLEF 2025) |

### 1.6 Validation

- Use the **expert-labeled train soundscapes** (`train_soundscapes_labels.csv`) as validation
- Convert 5-second segment predictions against the multi-label ground truth
- Compute **per-segment ROC-AUC** matching the competition metric exactly
- This is much more reliable than using a held-out subset of `train_audio` (domain is closer to test)

```python
# Evaluation: predict on train_soundscapes, compare to train_soundscapes_labels.csv
python src/evaluate.py \
    --soundscapes-dir data/raw/train_soundscapes/ \
    --labels data/raw/train_soundscapes_labels.csv \
    --model models/stage1_effb0_fold0/best.pth
```

### 1.7 Running Stage 1

```bash
conda activate kaggle
cd /home/swatson/work/MachineLearning/kaggle/BirdCLEF
rm -f log/train_*.log

# Train 5 folds of EfficientNet-B0
nohup bash scripts/train_stage1.sh \
    > log/train_stage1_b0_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

`scripts/train_stage1.sh` — use **absolute paths** to ensure scripts run correctly regardless of cwd:
```bash
#!/bin/bash
PROJECT=/home/swatson/work/MachineLearning/kaggle/BirdCLEF
for FOLD in 0 1 2 3 4; do
    python -u $PROJECT/src/train.py \
        --data $PROJECT/data/processed/train_folds.csv \
        --fold $FOLD \
        --model tf_efficientnet_b0.ns_jft_in1k \
        --output-dir $PROJECT/models/stage1_effb0_fold${FOLD} \
        --epochs 15 \
        --batch-size 64 \
        --lr 5e-4 \
        --chunk-duration 20 \
        --seed 42 \
        --bf16
done
```
> **Lesson from SIFD project**: always use absolute paths in shell scripts — relative paths cause silent failures when scripts are invoked from different working directories.

**Expected Stage 1 baseline ROC-AUC**: ~0.84–0.87 (based on 2025 5-fold EfficientNet-B0 ensemble)

### 1.8 Hard Negative Mining After Stage 1

Inspired by the SIFD project's hard negative mining approach: after Stage 1 validation, identify **species with the worst per-class ROC-AUC** and increase their sampling weight in subsequent training.

```python
# Compute per-species AUC from Stage 1 eval
per_class_auc = roc_auc_score(y_true.T, y_pred.T, average=None)
hard_species = [taxonomy[i]["primary_label"] for i in np.argsort(per_class_auc)[:30]]
# → 30 species with worst detection performance

# In dataset: upsample clips from hard_species by 3–5x weight
# In WeightedRandomSampler: assign weight * 4 to any clip whose primary_label is in hard_species
```

**Also target**: soundscape segments from `train_soundscapes_labels.csv` where Stage 1 assigns low confidence despite a ground-truth label (confidence < 0.2 when label = 1). These are hard positives — add with 3× weight.

**Tag hard negative list**: save `data/processed/hard_species_stage1.txt` with species codes and per-class AUC.

---

## Phase 2 — Pseudo-Labeling & Noisy Student Self-Training

After the supervised Stage 1 ensemble is trained, exploit the **unlabeled portion of `train_soundscapes`** via Noisy Student self-training.

### 2.1 Generate Pseudo-Labels (`src/pseudo_label.py`)

```bash
nohup python -u src/pseudo_label.py \
    --soundscapes-dir data/raw/train_soundscapes/ \
    --models models/stage1_effb0_fold{0..4}/best.pth \
    --output data/processed/pseudo_labels_v1.csv \
    > log/pseudo_label_v1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

For each unlabeled 1-minute soundscape:
1. Split into overlapping 20-second chunks (stride = 5s)
2. Run inference with full SED inference (sliding window averaging — see below)
3. Store **framewise max probabilities** per 5-second segment (4 frames per 5s at hop_size=1252)
4. The max over the 4 framewise predictions = pseudo-label for that 5-second window

**Pseudo-label sampler weights**:
```python
# Weight each soundscape by sum of max probabilities across all classes/segments
# High-sum soundscapes = models are confident = more accurate pseudo-labels
weight[soundscape] = sum(max_prob over segments and classes)
```
Use `WeightedRandomSampler` so high-quality pseudo-labeled soundscapes are sampled more often.

> ⚠️ **Critical: exclude val files from pseudo-label training dataset**
> `train_soundscapes_labels.csv` contains 66 files that are also in `train_soundscapes/`.
> If these are included in the pseudo-label training data, the model trains on pseudo-labels
> for the exact files used for validation → artificially inflated val ROC-AUC (~+0.07)
> with no LB improvement. `src/self_train.py` filters these out automatically.
> Learned from SED_B0_SelfTrain1: local val=0.8395 but LB=0.751 (≈ Stage 1).

### 2.2 Self-Training (`src/self_train.py`)

Self-training loop (Noisy Student style):

| Parameter | Self-Training Value | Notes |
|-----------|-------------------|-------|
| `epochs` | 25–35 | More than Stage 1 |
| `drop_path_rate` | 0.15 | Stochastic Depth — critical for Noisy Student |
| `pseudo_mix_ratio` | 1.0 | Mix every train sample with a pseudo-labeled sample (optimal) |
| `mixup_weight` | 0.5 (fixed) | Equal blend of focal training + pseudo-labeled soundscape |
| Other params | Same as Stage 1 | LR, optimizer, scheduler |

**Self-training MixUp** (different from Stage 1):
```python
# Sample one focal training clip + one pseudo-labeled soundscape chunk
focal_wave, focal_labels = train_dataset[i]
pseudo_wave, pseudo_soft_labels = pseudo_dataset[j]  # soft labels from teacher model

# Same absmax normalization + fixed 0.5 blend
mixed = 0.5 * focal + 0.5 * pseudo
mixed_labels = np.maximum(focal_labels, pseudo_soft_labels)  # element-wise max
```

**Why this works (Noisy Student logic)**:
- Mixing clean focal clips with noisy soundscapes forces the model to learn more robust, generalizable features
- The model must output the same pseudo-label despite the added noise injection
- Simple concatenation (no mixing) does NOT work — model just memorizes noise

**Power transform on pseudo-labels** (critical for multi-iteration):
```python
# After each self-training iteration, apply power > 1 to reduce noise
# before using as pseudo-labels in the next iteration
clean_pseudo = pseudo_prob ** power  # e.g., power=1.5 → sharpens confident, kills noise
```

### 2.3 Expected Gains

| Stage | Expected ROC-AUC | Notes |
|-------|-----------------|-------|
| Stage 1 supervised (5-fold EffB0 ensemble) | ~0.87 | Baseline |
| + Self-training iter 1 | ~0.90–0.91 | Major jump from soundscape data |
| + Power transform + iter 2 | ~0.91–0.92 | Continued gain |
| + Larger models (B3, B4, RegNetY, ECA-NFNet) | ~0.92–0.93 | Architecture diversity |
| + Multi-iteration (3–4 total) | ~0.93+ | Approaches 2025 winner level |

---

## Phase 3 — Multi-Iterative Pseudo-Labeling

Repeat self-training with progressively better teachers:

```
Stage 1 → Pseudo-labels v1 → Self-train iter 2 (EffB0-v2) → Pseudo-labels v2 → Self-train iter 3 (EffB0-v3 + RegNetY016-v1) → ...
```

> **Actual iteration numbering** (our codebase uses `--version` to track checkpoint generations):
> - Stage 1 = supervised EffB0 (stored in `models/stage1/`)
> - Iter 2 = self-train on pseudo_labels_v1, power=1.5 → `_v2.pt` — **done, LB=0.762**
> - Iter 3 = self-train on pseudo_labels_v2, power=1.5 → `_v3.pt` (EffB0) + `_v1.pt` (RegNetY016) — **in progress**

| Iteration | Pseudo CSV | Power | Backbones | Script | Status |
|-----------|-----------|-------|-----------|--------|--------|
| 2 | v1 | 1.5 | EffB0 (×5 folds) | `self_train_stage2.sh` | ✅ Done — LB 0.762 |
| 3 | v2 | 1.5 | EffB0-v3 (×5) + RegNetY016-v1 (×5) | `self_train_stage3.sh` | ✅ Done — LB 0.762 |
| 4 | v3 | 1.5 | EffB0-v4 (×5) + RegNetY016-v2 (×5) | `self_train_stage4.sh` | ✅ Done — LB 0.769 |
| 5 | v4 | 1.5 | EffB0-v5 (×5) + EffB3-v1 (×5) | `self_train_stage5.sh` | 🔄 In progress (Option B, 2026-03-21) |

**Decision gate**: Check LB before iteration 6. If no improvement → stop.

### 3.0 Stage 3 Script (`scripts/self_train_stage3.sh`)

Two-pass script: runs EffB0 v3 first (warm-started from v2), then RegNetY-016 v1 (fresh init). Both use `pseudo_labels_v2.csv` with `power=1.5`.

```bash
# Launch after pseudo_labels_v2.csv is ready:
nohup bash /home/swatson/work/MachineLearning/kaggle/BirdCLEF/scripts/self_train_stage3.sh \
  > /home/swatson/work/MachineLearning/kaggle/BirdCLEF/log/self_train_stage3_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Estimated runtime: ~26 hours (2 backbones × 5 folds × ~2.5 h/fold).

**Note on `pseudo_label.py`**: added `--ckpt-version` flag to load versioned checkpoints (e.g., `--ckpt-version 2` loads `_v2.pt` files). Use `--version N` for the output CSV version.

### 3.1 Dedicated Insecta/Amphibia Model

Train a **separate model** using Xeno-canto data for expanded Insecta/Amphibia species:
- Target classes: all Insecta + Amphibia from taxonomy.csv
- Additional classes: extra species from same families downloaded from Xeno-canto
- Model: EfficientNet-B0 (deeper models didn't help in 2025)
- Training: 40 epochs, BS=128
- At inference: run for all species, insert non-zero predictions only for Insecta/Amphibia columns
- Expected gain: +0.002–0.003 ROC-AUC

---

## Phase 4 — Ensemble & PyTorch Export

### 4.1 Final Ensemble Composition
Based on BirdCLEF 2025 winner strategy (7 models):

| Model | Training stage | Backbone |
|-------|---------------|---------|
| Model 1 | Stage 1 (supervised) | EffB0-NS fold avg |
| Model 2 | Self-train iter 1 | EffB0-NS |
| Model 3 | Self-train iter 2 | EffB3-NS |
| Model 4 | Self-train iter 3 | EffB4-NS |
| Model 5 | Self-train iter 3–4 | RegNetY-016 |
| Model 6 | Self-train iter 3 | ECA-NFNet-L0 |
| Model 7 | Supervised (Insecta/Amphibia specific) | EffB0-NS |

Use **equal weights** (1/7 each). The best private LB in 2025 used equal weights, not tweaked weights.

### 4.2 Model Soup (within-backbone averaging)
Before ensembling across architectures, average checkpoint weights within the same backbone:
```python
# Average epoch-50, epoch-45, epoch-40 checkpoints
# → single checkpoint with better generalization than any individual epoch
```

### 4.3 PyTorch Export (`.pt` checkpoints)

**ONNX is NOT usable on Kaggle** — `onnxruntime` is absent from the no-internet competition environment. Use `.pt` checkpoints loaded with PyTorch directly:

```python
model = BirdSEDModel(backbone=BACKBONE, num_classes=NUM_CLASSES)
checkpoint = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

PyTorch CPU inference fits comfortably within the 90-min budget for a 5-fold EffB0 + 5-fold RegNetY ensemble.

**Note**: Do NOT use `torch.compile` — it breaks checkpoint portability across environments.

---

## Phase 5 — Kaggle Inference Notebook

### 5.1 Inference Algorithm

Frame-level sliding window inference (1st place BirdCLEF 2025 technique):

```python
INFERENCE_CHUNK = 20  # seconds
STRIDE_FRAMES = 5     # stride 5 seconds (overlap = 15s)

# Load 1-minute soundscape (60 seconds = 1,920,000 samples at 32kHz)
# Generate chunks: starts at 0, 5, 10, ..., 40 seconds → 9 overlapping 20-sec chunks

for chunk_start in range(0, 60 - INFERENCE_CHUNK + 1, STRIDE_FRAMES):
    chunk = wave[chunk_start*sr : (chunk_start+INFERENCE_CHUNK)*sr]
    mel = compute_mel(chunk)                # (224, 512)
    out = model(mel.unsqueeze(0))          # framewise predictions
    # out["frame_prob"]: (1, 234, T) where T = time frames
    # Store framewise predictions for this chunk

# Average overlapping framewise predictions across all chunks
# Each 5-second window gets predictions from multiple overlapping chunks
# This is analogous to 2D sliding-window image segmentation TTA
```

**Left/right padding**: Pad the beginning and end of the soundscape by half a chunk so the first and last 5-second windows are centered in a full 20-second context. Remove predictions from padded regions.

### 5.2 Post-processing

```python
# 1. Temporal smoothing over framewise predictions
kernel = [0.1, 0.2, 0.4, 0.2, 0.1]  # 5-tap smoothing
smoothed_probs = convolve1d(raw_probs, kernel, axis=1)

# 2. Delta-shift TTA (from BirdCLEF 2023 2nd place)
# Shift the soundscape by ±2.5 seconds, average with original prediction
# Catches species calling near window boundaries

# 3. Rank-aware power adjustment (from BirdCLEF 2025 3rd place)
# Adjust low-confidence predictions based on class ranking
# Reduces false positives for rare classes
```

### 5.3 Submission Notebook Constraints
- **CPU only** (GPU submissions: 1 minute — unusable)
- **≤90 minutes** total runtime
- Loading ~600 soundscapes: ~5 minutes
- Per-soundscape inference time: (90 - 5) min / 600 ≈ 8.5 sec/soundscape
- With 9 overlapping 20-second chunks × 7 models → need very fast inference
- **Strategy**: Pre-compute spectrograms once, share across models (saves ~50% compute)
- **ONNX**: 2–3× faster than PyTorch → essential for fitting within 90 min

### 5.4 Timing Budget
| Operation | Est. Time |
|----------|----------|
| Load models (7 ONNX) | 3 min |
| Load 600 soundscapes (parallel) | 5 min |
| Inference per soundscape (9 chunks × 7 models) | ~5 sec |
| Total inference (600 × 5 sec) | 50 min |
| Post-processing + CSV write | 2 min |
| **Total** | ~60 min (30 min buffer) |

### 5.5 Model Size Constraints
- Kaggle dataset storage: models uploaded to a Kaggle dataset
- Total ensemble ONNX weight budget: aim for ≤ 2 GB total
- EfficientNet-B0: ~16 MB; B3: ~47 MB; B4: ~73 MB; NFNet-L0: ~72 MB
- 7-model ensemble: ~400 MB ONNX total — well within budget

---

## Inference Optimization Constraints

These constraints are absolute and must never be violated in the submission:

| Constraint | Value | Why |
|-----------|-------|-----|
| CPU only | Yes | GPU disabled on Kaggle for this competition |
| Max runtime | 90 min | Notebook times out |
| No internet | Yes | Must upload all data/models to Kaggle Dataset |
| Submission filename | `submission.csv` | Required exactly |
| Spectrogram reuse | Required | Pre-compute mel once, feed to all 7 models |
| PyTorch (not ONNX) | `.pt` checkpoints via `torch.load` | ONNX export dropped — PyTorch CPU inference fits within 90-min budget |
| No quantization | Yes | Can hurt accuracy — don't use unless fitting 90min otherwise |
| Model upload | Kaggle Dataset | Upload to `stevewatson999/birdclef2026-models` |

---

## Experiment Tracking

Each training run creates:
1. Timestamped log file in `log/`
2. Model checkpoints in `models/<name>/`
3. Entry in this plan.md experiment log table below
4. **Git tag** marking the submission (e.g., `git tag SED_B0_Stage1_0.872` after LB submission)

### Experiment Log

| Tag | Backbone | Stage | Val ROC-AUC | LB ROC-AUC | Notes |
|-----|---------|-------|-------------|-----------|-------|
| `Perch_baseline` | Google Perch | 0.5 | — | 0.590 | Zero-shot, 158/234 species mapped, 2026-03-16 |
| `SED_B0_Stage1` | EfficientNet-B0 | 1 | 0.7408 (mean) | 0.752 | Fold scores: F0=0.7601 F1=0.7500 F2=0.7295 F3=0.7339 F4=0.7305; 15 epochs BF16, 2026-03-16 |
| `SED_B0_SelfTrain1` | EfficientNet-B0 | 2 | 0.8395 (ensemble) | 0.751 | Fold best: F0=0.8252 F1=0.8310 F2=0.8283 F3=0.8263 F4=0.8246; 30 epochs Noisy Student, power=1.0, 2026-03-17. **Val inflated by leakage** — all 66 val files were in pseudo-label training data. Fixed in v2: exclude val files from pseudo dataset. |
| `Bird_0.762` | EfficientNet-B0 | 2 (v2) | 0.7858 (ensemble) | 0.762 | Fold best: F0=0.7635 F1=0.7811 F2=0.7702 F3=0.7731 F4=0.7545; 30 epochs, power=1.5, val leakage fixed (792 rows excluded), 2026-03-19 |
| `Bird_0.769` | EffB0-v4 + RegNetY-v2 | 4 | 0.786 (ensemble) | 0.769 | 10-model ensemble, warm-start both backbones, pseudo_labels_v3, 2026-03-21 |
| `Perch_MLP_v1-v5` | Perch v2 + MLP | — | 0.904 (local) | ~0.49 | 5 iterations, all failed on LB. Domain gap: clean focal → noisy soundscapes, 2026-03-23 |
| `SED_Perch_KD` | EffB0 + Perch KD | 2 (v6) | 0.737 | 0.667 | Perch KD caused catastrophic forgetting. 10× oversample + 31 missing species, 2026-03-23 |
| `SED_EffB0v5_EffB3v1` | EffB0-v5 + EffB3-v1 | 5 | 0.799/0.737 | pending | SED-only 10-model ensemble, v5 notebook pushed 2026-03-24 |

### Git Tag Conventions
```bash
# After each LB submission, tag the code state:
git tag SED_EffB0_Stage1_0.872   # Tag = model_description_LBscore
git push origin --tags
```
This pattern is validated from the Akkadian project where `Byt5_27.5` tags made it easy to recover the exact code state for any given LB score.

### Local Val vs LB Disconnect Warning
- **Local validation** uses `train_soundscapes_labels.csv` which is expert-annotated — this is the best proxy for LB, but not identical
- Expect **5–10% relative differences** between local val ROC-AUC and public LB (test soundscapes are from different recording dates/conditions, even at the same sites)
- The Akkadian project had a ~15-point absolute gap between val and LB — BirdCLEF should be much smaller since validation domain is closer, but calibrate expectations per first LB submission
- **Rule**: if local val improves but LB drops, trust LB. Don't commit to an approach without checking LB.

### Decision Gate: LB Validation
- After each stage, submit a Kaggle notebook to check public LB
- Public LB = ~34% of test data (based on prior years); public ≈ private for BirdCLEF (2025 winner confirmed <0.005 delta)
- Only proceed to next stage if LB improves

---

## Key Lessons from BirdCLEF 2025 Top Solutions

| Finding | Impact | Action |
|---------|--------|--------|
| **20-second chunks beat 5-second** | +3 pts vs 5s, +0.5 vs 15s | Always use 20s chunks |
| **224 mel bins beat 128** | Significant for Insecta/Amphibia | Use 224 n_mels |
| **CE loss ≈ BCE but slightly better** | Minor | Default to CE; test BCE if CE underperforms |
| **Fixed 0.5 MixUp weight beats random Beta** | Essential — random Beta can suppress signal | Use `weight = 0.5` always |
| **MixUp target = element-wise max** | Correct multi-label handling | `np.maximum(label1, label2)` |
| **Mask secondary label loss** | +0.01 LB (2024 3rd place) | Set loss weight = 0 for secondary labels; species location within clip is unknown |
| **Noisy Student > supervised alone** | +4–5 pts | Self-training is mandatory for top scores |
| **Power transform for pseudo-labels** | Prevents noise accumulation in multi-iteration | Apply `prob ** power` before each new iteration |
| **WeightedRandomSampler for pseudo-labels** | Stabilizes training, boosts LB | Weight by sum of max probabilities per soundscape |
| **Stochastic Depth (drop_path=0.15)** | +0.005 LB in self-training only | Enable only during self-training, not Stage 1 |
| **Ensemble from diverse stages** | +1–2 pts vs single-stage ensemble | Include Stage 1 model in final ensemble |
| **ONNX export: DO NOT USE** | `onnxruntime` unavailable on Kaggle (no-internet env) | Use `.pt` checkpoints + PyTorch CPU inference instead |
| **EfficientViT-b0 for inference** | 5 folds in 40 min on CPU (2024 3rd place) | Use `efficientvit_b0.r224_in1k` to fit more models in 90-min budget |
| **OpenVINO: skip** | ~2× faster but −0.01 accuracy drop; eca_nfnet fails | Use ONNX only |
| **Spectrogram reuse across models** | ~50% inference speed gain | Compute mel once, feed to all models |
| **Overlapping framewise averaging** | +0.002–0.003 LB | Use stride=5s inference, average overlapping frames |
| **Temporal smoothing** | Minor | 5-tap kernel [0.1, 0.2, 0.4, 0.2, 0.1] |
| **Background noise injection** | Bridges train/test domain gap | Mix real env recordings (freefield1010, warblrb, birdvox) at low gain |
| **GPU augmentations (torch-audiomentations)** | Fast PitchShift/TimeShift on GB10 | ±2 semitones pitch, ±25% time shift |
| **Cap extra data at 500 records/class** | Adding all external data causes class imbalance | Apply when using XC downloads or prior BirdCLEF data |
| **Minimum 10 samples per class floor** | Avoids gradient starvation for rare insect sonotypes | Duplicate rare clips before fold splitting |
| **Two-stage CE→BCE loss** | CE converges 3–5× faster, BCE better asymptotic | ~10 epochs CE then ~5 epochs BCE fine-tune |
| **Separate Insecta/Amphibia model** | +0.002–0.003 LB | Train dedicated model with more diverse species |
| **BirdCLEF 2025 1st place public/private delta** | 0.933 → 0.930 (very small shake) | Don't over-optimize for public LB |
| **Google Perch model** | Good baseline for 2026 | Use as additional ensemble member or feature extractor |
| **Aves/BirdAves foundation model: risky** | +0.01 public but −0.01 private LB (2024) | If tried, gate strictly by private LB before including in ensemble |

## New 2026 Experiment Findings (validated in competition, March 2026)

| Finding | Gain | Action |
|---------|------|--------|
| **~~Perch v2 embeddings + MLP probe = 0.905+ LB~~** | ❌ Failed: 0.49–0.72 LB | Shallow MLP on frozen embeddings fails to generalize to noisy soundscapes. **Abandoned.** |
| **ASL loss (γ−=4, γ+=0, clip=0.05) > BCE** | +0.02–0.03 soundscape val | Replace BCE in all training scripts — **highest priority** |
| **PCEN normalization is load-bearing** | −0.049 without it | Add PCEN to `utils.py` (replaces AmplitudeToDB) |
| **~~Perch soft labels as KD targets (×10 oversample)~~** | ❌ Failed: −0.06 val AUC | 10× oversample caused catastrophic forgetting; 31 missing spp got zero-signal labels. Needs careful tuning if retried (×2–3, mask missing spp). |
| **Dual loss (clip 50% + frame 50%)** | Slight + | Add frame-level loss branch |
| **Circular shift augmentation (p=0.5)** | +0.005–0.01 | Add to dataset pipeline |
| **Model soup (dense checkpoint avg) ≈ SWAD** | OOD robustness | Save every epoch; average epochs after warmup |
| **Freq-MixStyle** | +1–3pp soundscape val | Mix per-frequency-bin stats between batch samples to simulate ARU device variance |
| **~~SED 70% + Perch TFLite 30% ensemble~~** | ❌ Failed: 0.573 LB | Perch MLP branch (~0.49 LB) destroys SED predictions. Perch MLP not viable as ensemble component. |
| **Soundscape val AUC (not OOF) correlates with LB** | Critical decision metric | Validate on `train_soundscapes_labels.csv` only |
| **Rating ≥ 3 filter HURTS** | −0.047 soundscape val | Never filter by rating |
| **Perch-augmented training hurts** | −0.039 | Use Perch as KD teacher, NOT as training augmenter |
| **V2S (22.4M params) underperforms EffB0 (6.2M)** | −0.057 at equal epochs | V2S needs LR=5e-4 (not 3e-4) and 15+ epochs to converge |
| **Self-training with our own model predictions** | +0.017 LB (0.752→0.769) | Works but gains diminish after 4 iterations. Still useful for new backbone warm-starts |

---

## Hard Constraints & Known Pitfalls

```
NEVER submit GPU-dependent code       → GPU runs for only 1 minute on Kaggle
NEVER use ONNX export                 → `onnxruntime` unavailable on Kaggle (no-internet); use PyTorch `.pt` checkpoints
NEVER use FP16                        → NaN risk; use BF16 (safe for EfficientNet/RegNetY/NFNet)
NEVER use torch.compile               → Incompatible with checkpoint portability across environments
NEVER use OpenVINO                    → ~2× speedup but −0.01 accuracy; eca_nfnet_l0 fails conversion
DO NOT randomly sample MixUp weight   → Fixed 0.5 is optimal (validated by 2025 winner)
DO NOT add Stochastic Depth in Stage 1 → Only helps in self-training (Noisy Student)
DO NOT use raw pseudo-labels in iteration 3+ → Apply power transform to reduce noise
DO NOT include only same-stage models in ensemble → Multi-stage diversity is key
DO NOT use soundscapes without quality filtering → Weight by prediction confidence
DO NOT trust local val alone          → Val-LB delta can be significant; always submit to check LB
DO NOT use relative paths in shell scripts → Brittle; always use absolute paths (SIFD lesson)
DO NOT use secondary labels as positive targets → Mask their loss (unknown window); +0.01 LB
DO NOT add Aves/BirdAves to final ensemble without LB gate → Looked good on public, hurt private
DO NOT cap-less import XC external data → Cap at 500 records/class to avoid class imbalance
ALWAYS validate on train_soundscapes_labels.csv → Closest proxy to test domain
ALWAYS git tag code state after each LB submission → Makes it easy to recover winning configs
ALWAYS upload models to Kaggle Dataset before final submission
ALWAYS ensure ≥10 samples per class before training → Duplicate rare clips to meet floor
NEVER use Perch MLP as standalone LB scorer → 0.90 local val but 0.49 LB due to domain gap
NEVER train Perch KD with ×10 oversample → catastrophic forgetting; use ×2–3 and mask missing species
NEVER use BatchNorm in MLP on frozen embeddings → running stats from training distribution ≠ inference distribution
NEVER trust local val alone for Perch MLP → huge domain gap between focal clips and test soundscapes
NEVER train a PCEN model on pseudo-labels generated by a log-mel model → distribution mismatch causes bell-curve overfitting (peak ep7, decline thereafter); PCEN requires PCEN pseudo-labels
NEVER fine-tune a BCE-trained model with ASL γ−≥2 on BCE-generated pseudo-labels → immediate degradation; ep1 peak = starting checkpoint value, then monotone decline; use γ−≤0.5 or apply ASL to focal-only labels
NEVER use fractional γ in AsymmetricLoss without clamping (1−pt) ≥ 0 in BF16 → BF16 rounding can make (1−pt) slightly negative; (−ε)^0.5 = NaN while (−ε)^2 = ε² (safe); fix: `.clamp(min=0.0).pow(gamma)` — already patched in `src/self_train.py`
```

---

## Prioritized Action Plan

### #0 ✅ — Fix CLAUDE.md (5 min) — *Done 2026-03-15*
CLAUDE.md currently references the **wrong** log directory (`DeepPastChallengeTranslateAkkadianEnglish/log`). Fix before any other work:
```bash
# Edit CLAUDE.md: change all occurrences of
# /home/swatson/work/MachineLearning/kaggle/DeepPastChallengeTranslateAkkadianEnglish
# to
# /home/swatson/work/MachineLearning/kaggle/BirdCLEF
```

### #1 ✅ — Setup & Data Download — *Done 2026-03-15*
```bash
conda activate kaggle
cd /home/swatson/work/MachineLearning/kaggle/BirdCLEF
mkdir -p src data/raw data/processed models log scripts jupyter
kaggle competitions download -c birdclef-2026 -p data/raw/
cd data/raw && unzip birdclef-2026.zip
```

### #2 ✅ — EDA & Fold Assignment (~2 days) — *Done 2026-03-15*
- Understand class distribution, duration stats, species breakdown by taxonomy class
- Generate `data/processed/train_folds.csv` with 5-fold stratified split
- Identify species appearing ONLY in train_soundscapes (not train_audio) — these need special handling
- **Gate**: Confirm ≥ 1 sample per class per fold. Flag species with < 10 samples.
- **Results**: 35,549 clips, 206 species in train (234 in taxonomy). 25 species with <10 clips. 28 species (all 25 insect sonotypes + 2 amphibians + 1 amphibian) have zero training audio — soundscape pseudo-labeling required. 5 folds of ~7,110 clips each. Report → `data/processed/eda_report.txt`.

### #3 ✅ — Google Perch Quick Baseline (~1 day) — *Done 2026-03-16*
Downloaded Perch v4 TF SavedModel (10932 classes, 92MB weights) from TF Hub. 158/162 competition Aves species mapped (4 unmapped: palhor3, strher2, wesfie1, y00678). Uploaded to Kaggle dataset `stevewatson999/birdclef2026-perch`. Notebook `birdclef2026-perch-baseline.ipynb` pushed to `stevewatson999/birdclef-2026-perch-baseline`. Awaiting LB score.  
**Gate**: Establish any LB score. Tag submission `Perch_baseline`.

### #4 ✅ — Build Core Pipeline (~3 days) — *Done 2026-03-16*
Wrote `src/config.py`, `src/utils.py`, `src/dataset.py`, `src/model.py`, `src/train.py`, `scripts/train_stage1.sh`.  
SED model (EfficientNet-B0 + GEM pool + Conv1d attention), BF16 training, mel in DataLoader workers (spawn context to avoid torchaudio fork deadlock). Smoke test: fold 0, 1 epoch, val ROC-AUC=0.5279, time=3m56s.

### #5 ✅ — Stage 1 Training (~2 days training time) — *Done 2026-03-16*
Trained 5 folds of EfficientNet-B0 (15 epochs each, BF16, ~50 min total). Val ROC-AUC: F0=0.7601, F1=0.7500, F2=0.7295, F3=0.7339, F4=0.7305 → **mean 0.7408**.  
Note: Gate was ≥0.85; 0.7408 is below — however this is measured against `train_soundscapes_labels.csv` which is a harder domain shift than focal clips. Proceeding to Stage 1 submission to calibrate LB gap before deciding whether to iterate.

### #6 ✅ — Hard Negative Mining (post Stage 1) — *Done 2026-03-16*
Ran `src/evaluate.py` (5-fold ensemble) on `train_soundscapes_labels.csv` (66 files, 1478 windows, 75 species with positive labels).  
**Ensemble Macro ROC-AUC: 0.7636** (75/234 species evaluated; 159 species have no positive labels in train soundscapes).  
Worst species: insect sonotypes (`47158sonXX`) dominate — AUC as low as 0.2416. Bird hard negatives: `strher2` (0.3860), `25073` (0.4108), `326272` (0.4701), `67107` (0.4836).  
Outputs: `data/processed/hard_species_stage1.txt`, `per_species_auc_stage1.csv`, `eval_stage1_predictions.csv`.  
Apply 4× sample weight for worst-30 species in Stage 2+.

### #7 ✅ — Pseudo-label Generation — *Done 2026-03-17*
Ran `src/pseudo_label.py` (5-fold ensemble) on all 10,658 train soundscapes.  
Runtime: 72m 52s. Output: 127,896 rows, 188/234 species with max_prob > 0.5, mean max prob = 0.760.  
Files: `data/processed/pseudo_labels_v1.csv`, `data/processed/pseudo_labels_v1_weights.csv`.

### #8 ✅ — Self-training Iteration 1 — *Done 2026-03-17*
Noisy Student self-training: focal + pseudo_labels_v1.csv (power=1.0) + warm-start from Stage 1.  
30 epochs × 5 folds, EfficientNet-B0, seed 42. ~13 hours.  
Per-fold best val ROC-AUC: F0=0.8252 F1=0.8310 F2=0.8283 F3=0.8263 F4=0.8246  
**Ensemble val ROC-AUC: 0.8395** (Stage 1 was 0.7636, +0.076 gain).  
**Gate**: Submit to Kaggle LB. Tag `SED_B0_SelfTrain1_<score>`.

### #9 🔄 — Multi-iterative pseudo-labeling (iterations 2–4)
Each iteration: generate new pseudo-labels → apply power transform → retrain with larger models  
**Gate after each**: submit to LB, check improvement. Stop when delta < 0.001.

- ✅ **Iteration 2** (EffB0-v2, pseudo_labels_v1, power=1.5, leakage fixed): val=0.7858, **LB=0.762** — *Done 2026-03-19* (`Bird_0.762`)
- ✅ **Pseudo-labels v2 generated**: 127,104 segments, `data/processed/pseudo_labels_v2.csv` — *Done 2026-03-19*
- ✅ **Iteration 3** (EffB0-v3 + RegNetY016-v1, pseudo_labels_v2, power=1.5): val≈0.7899 (EffB0), **LB=0.762** — *Done 2026-03-20*
  - EffB0-v3 (warm-start from v2): fold0=0.7845, fold1=0.8038, fold2=0.7821, fold3=0.8004, fold4=0.7788 → avg **0.7899**
  - RegNetY016-v1 (fresh init): fold0=0.7401, fold1=0.7564, fold2=0.7544, fold3/4 pending → avg ~**0.750** (gap ~0.040 vs EffB0)
  - **Root cause**: RegNetY is training from scratch in Pass 2 while EffB0 had 3 warm-start generations (v1→v2→v3). Options for remediation:
    - **Option A (Warm-start RegNetY)**: For Iter 4, initialise RegNetY from its Stage 1 v1 checkpoints (`models/sed_regnety_016.tv2_in1k_fold*_seed42_v1.pt`) rather than from scratch, giving it the same warm-start advantage. Expected: close the ~0.04 gap.
    - **Option B (Different architecture)**: Replace RegNetY with EfficientNet-B3 or another larger backbone for Iter 4. EffNetB3 is ~2× the parameters of B0 with ~1.5× inference cost; likely outperforms RegNetY at same training budget.
- ✅ **Iteration 4** (Option A — warm-start RegNetY): **LB=0.769** — *Done 2026-03-21*
  - EffB0-v4: warm-start from v3, 20 epochs → fold avg **0.7958** (local val)
  - RegNetY016-v2: **warm-start from v1** (Option A), 20 epochs → fold avg **0.7758** (local val)
  - pseudo_labels_v3.csv: 10-model average of EffB0-v3 + RegNetY016-v1
  - **+0.007 over Iter 3** (0.762 → 0.769) — warm-starting RegNetY closed the gap and improved the ensemble
- ✅ **Iteration 5** (Option B — EfficientNet-B3 fresh init): *Done 2026-03-22*
  - EffB0-v5 (warm-start from v4) + EffB3-v1 (fresh init, `tf_efficientnet_b3.ns_jft_in1k`)
  - pseudo_labels_v4.csv: 10-model average of EffB0-v4 + RegNetY016-v2
  - EffB0-v5 fold avg: **0.799 val AUC** (best EffB0 generation)
  - EffB3-v1 fold avg: **0.737 val AUC** (fresh init, diverse backbone)
  - Inference: 5×EffB0-v5 + 5×EffB3-v1 ensemble (10 models)
  - Models uploaded to `stevewatson999/birdclef2026-sed-models` v2
  - SED-only v5 notebook (v5) submitted to Kaggle — **LB=0.682** ❌ (worse than 0.769 baseline)
  - **Root cause**: EffB3-v1 trained from scratch (fresh init) while EffB0-v5 had 5 warm-start generations. EffB3 fold avg 0.737 vs EffB0 0.799 — the 5×EffB3 models dragged the ensemble down vs the 10×EffB0 (or 5×EffB0 + 5×RegNetY-v2) ensemble. Lesson: **never include a large fresh-init backbone in the ensemble until it has caught up via warm-starting**.

### #10 ⬜ — Dedicated Insecta/Amphibia Model
Train on expanded Xeno-canto insect/amphibian data with hard negative emphasis

### #11 ⬜ — Final Ensemble Construction
Combine 6–7 models from multiple stages  
Apply model soup (checkpoint weight averaging: last 3 epochs within backbone)

### #12 ✅ — Inference Notebook — *Done 2026-03-21*
`jupyter/sed/birdclef2026-sed-inference.ipynb` — 10-model ensemble (5×EffB0-v5 + 5×EffB3-v1), PyTorch CPU sliding-window inference, builds submission CSV.

### #13 ⬜ — Inference Tuning
- Sweep `smoothing kernel shape`, `delta-shift TTA`, `power adjustment` on local val
- Target: +0.005–0.01 ROC-AUC from post-processing
- Tag final best submission `Final_ensemble_<score>`

---

## 🔴 STRATEGIC PIVOT (2026-03-21) — Path to 0.90+

**Context**: After studying the LB (top score 0.933, ours 0.769), the gap is structural but **not due to Perch**. We tested Perch MLP (5 iterations, 0.90 local val → ~0.49 LB) and Perch KD (caused SED v6 to degrade 0.80→0.74 val AUC via catastrophic forgetting). Both approaches failed for us. The remaining gap is due to missing **PCEN** (+0.049), **ASL loss** (+0.02–0.03), **Freq-MixStyle** (+0.01–0.03), and **model soup** — all pure SED improvements that don't depend on Perch.

### Technique Effectiveness (validated in competition experiments, March 19):
| Technique | Gain (soundscape val AUC) | Status in our pipeline |
|-----------|--------------------------|----------------------|
| Perch embeddings + MLP probe | ~+0.14 vs our SED (claimed) | ❌ FAILED — 0.90 local val, ~0.49 LB due to domain gap |
| ASL loss (γ−=4, γ+=0) vs BCE | +0.02–0.03 | ❌ Using BCE — **implement next** |
| PCEN normalization | +0.049 (critical!) | ❌ Using AmplitudeToDB — **implement next** |
| Soft pseudo labels from Perch KD (×10 oversample) | +0.039 (claimed) | ❌ FAILED — caused catastrophic forgetting in SED v6 |
| Dual loss (clip + frame, w=0.5/0.5) | slight + | ❌ Clip-only — implement |
| Hard pseudo soundscapes (×5 oversample) | +0.01–0.02 | ✅ (×5 weight, different form) |
| Circular shift augmentation (p=0.5) | +0.005–0.01 | ❌ Not applied — implement |
| Strong SpecAugment | +0.005 | ✅ Partial |
| Model soup (checkpoint averaging) | DG/SWAD benefit | ❌ Single best checkpoint — implement |
| Freq-MixStyle (frequency stats mixing) | +1–3pp soundscape val | ❌ Not implemented — implement |
| Ensemble: SED 70% + Perch TFLite 30% | WiSE-FT style | ❌ ABANDONED — Perch MLP too weak on test |
| Rating ≥3 filter | −0.047 | ✅ Correctly NOT applied |
| PCEN removal | −0.049 | ✅ (but we never had PCEN — **add it**) |

### Validated "Best Formula" (from Tom's v28, reaching LB 0.918):
- Backbone: `tf_efficientnet_b0.ns_jft_in1k`, GEM Pool p_init=3.0
- Loss: **ASL** (γ−=4, γ+=0, clip=0.05)
- LR: 5e-4 → cosine, warmup=3ep, Batch=32, Epochs=35
- Augmentation: Mixup α=0.5, **circ_shift p=0.5**, strong SpecAugment
- Data: 3 clips/file, soundscape labels ×5, hard pseudo ×5
- Domain generalization: **Freq-MixStyle** (p=0.5, alpha=0.1)
- Post-training: **model soup** (average all epoch checkpoints after warmup)
- ~~Final ensemble: SED (70%) + Perch TFLite (30%)~~ → **SED-only ensemble** (Perch MLP failed for us)

### Our Adapted Formula (removing failed Perch components):
- Same backbone, loss, LR, augmentation as above
- **PCEN** instead of AmplitudeToDB (biggest single gain: +0.049)
- **No Perch KD** — use self-training with own best-generation pseudo labels instead
- **Dual backbone ensemble**: EffB0 + EffB3 (diversity > single backbone)
- **Warm-start self-training**: each iteration warm-starts from previous best
- Expected ceiling: ~0.88–0.90 (without Perch KD, we lose ~0.04 from Tom's formula)

### #14 ❌ ABANDONED — Perch MLP Probe
**Status**: Failed. Perch MLP achieves 0.904 local val AUC but only ~0.49 on LB.
**Root cause**: Perch v2 embeddings are trained on clean focal recordings. Test soundscapes are noisy 60-second field recordings — massive domain gap. The shallow MLP has no temporal context to aggregate across windows. Additionally, 31/234 competition species are missing from Perch's vocabulary.
**Experiment results**: See "Perch MLP Dead End" section above for full v1-v5 experiment table.
**Lesson**: Never trust local val when train/test domain differs fundamentally.

### #15c ✅ COMPLETED (2026-03-26) — Single-fold PCEN Diagnostic

**Result**: Best fold-0 val ROC-AUC = **0.7214** (epoch 14 of 20). Training loss still declining at epoch 20 (0.0786, not plateaued). PCEN code reviewed — implementation is correct (IIR smoother matches Wang 2017 formula).

**Epoch curve**:
| Epoch | Train loss | Val ROC-AUC |
|-------|-----------|-------------|
| 1 | 0.1207 | 0.6835 |
| 2 | 0.1020 | 0.6989 ★ |
| 6 | 0.0870 | 0.7197 ★ |
| 14 | 0.0801 | **0.7214** ★ |
| 20 | 0.0786 | 0.7065 |

**Interpretation**: Below the 0.75 gate, but this is **under-training**, not a PCEN bug:
- Training loss was still declining at ep20 — convergence not reached in 20 epochs from scratch
- v4 had 4 generations of warm-start advantage; fair comparison needs 40 epochs
- PCEN (v8, BCE) already beat ASL γ−=4 (v7) by +0.0466 — ASL at γ−=4 was the real Stage 6 culprit
- IIR formula, denominator power, and `bias_r` subtraction all verified correct

**Checkpoint**: `models/sed_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_v8.pt` (epoch 14, 2026-03-26 22:48)

### #15d ❌ FAILED (2026-03-27) — Extend PCEN Ablation to 40 Epochs

**Result**: Killed at epoch 22. Best fold-0 val ROC-AUC = **0.7308** (epoch 7), then consistent decline to 0.6966 at ep22. More epochs will not help.

**Epoch curve (v9)**:
| Epoch | Val ROC-AUC |
|-------|-------------|
| 7 | **0.7308** ★ |
| 8 | 0.7108 |
| 11 | 0.6759 |
| 18 | 0.7003 |
| 22 | 0.6966 |

**Root cause: pseudo-label distribution mismatch.**

Both v8 (ep14 peak 0.7214) and v9 (ep7 peak 0.7308) show the same bell-curve pattern: early peak, then long decline. This is **not** under-training — it's overfitting to noisy pseudo-labels:
- Pseudo batches outnumber focal batches 4.4:1 (1986 vs 446/epoch). As LR decays, the model fine-tunes on pseudo-label noise rather than generalizing.
- `pseudo_labels_v4.csv` was scored by **log-mel models** on log-mel spectrograms. When a PCEN model is trained on these, it must produce the same predictions from differently normalized features. The pseudo-label probabilities encode log-mel activation patterns, not PCEN patterns — adding noise to every pseudo-label segment.
- PCEN normalizes out amplitude variation (background AGC). Things log-mel models found distinctive (loud consistent calls) may look subdued in PCEN, creating signal-label mismatch.

**Conclusion**: PCEN cannot be properly validated using log-mel-generated pseudo-labels. The correct path requires a two-stage approach: (1) train PCEN supervised on focal data only to generate PCEN pseudo-labels, then (2) self-train with PCEN.

**New lesson added to pitfalls**: Never train a model with a different input representation on pseudo-labels generated by a model with the original representation.

### #15e ❌ FAILED (2026-03-27) — ASL γ−=2 Warm-Start v4 Ablation (v10)

**Result**: Killed at epoch 9. Peak val_roc_auc = **0.7748** (epoch 1), then monotone decline. Gate FAILED (needed ≥ 0.79).

**Epoch curve**:
| Epoch | Train loss | Val ROC-AUC |
|-------|-----------|-------------|
| 1 | 0.0191 | **0.7748** ★ |
| 2 | 0.0177 | 0.7644 |
| 3 | 0.0175 | 0.7401 |
| 4 | 0.0174 | 0.7389 |
| 5 | 0.0172 | 0.7252 |
| 6 | 0.0170 | 0.7308 |
| 7 | 0.0169 | 0.7327 |
| 8 | 0.0168 | 0.7221 |
| 9 | 0.0168 | 0.7296 |

**Root cause: pseudo-label calibration mismatch + warm-start disruption.**

The warm-started v4 model peaks at epoch 1 (essentially its BCE-calibrated starting point) and then immediately degrades as ASL reshapes the gradient landscape:
- v4 was trained end-to-end with BCE; its decision boundary is calibrated for pseudo-labels with soft values in the 0.3–0.7 range
- ASL γ−=2 aggressively down-weights easy negatives — but pseudo-labels near 0.5 (uncertain) are treated as negatives and get suppressed by the focal weighting
- Result: gradient conflicts between ASL penalizing uncertain pseudo-label predictions vs BCE having learned to output those exact values
- Same bell-curve failure mode as PCEN v9, just steeper (ep1 peak vs ep7 peak for PCEN)

**New pitfall added**: Never fine-tune a BCE-trained model with ASL on pseudo-labels generated by the same BCE model — calibration conflict causes immediate degradation.

Script: `scripts/ablate_asl_fold0.sh`

---

### #15f ❌ FAILED (2026-03-27) — ASL γ−=0.5 Warm-Start v4 Ablation (v11)

**Result**: Killed at epoch 6. Peak val_roc_auc = **0.7917** (epoch 1), then decline. Same pattern as v10 (γ−=2). Gate FAILED.

**Epoch curve**:
| Epoch | Train loss | Val ROC-AUC |
|-------|-----------|-------------|
| 1 | 0.0382 | **0.7917** ★ |
| 2 | 0.0367 | 0.7894 |
| 3 | 0.0363 | 0.7720 |
| 4 | 0.0362 | 0.7679 |
| 5 | 0.0358 | 0.7695 |
| 6 | 0.0355 | 0.7784 |

**Conclusion: ASL + warm-start from BCE is fundamentally incompatible**, regardless of γ strength. Even γ−=0.5 (barely different from BCE) shows the same ep1-peak-then-decline pattern. The model's learned decision boundary is calibrated for BCE loss geometry; ASL of any strength pushes it off that optimum immediately.

**New lesson**: ASL must be used from scratch or with ASL-trained checkpoints. It cannot be dropped into a BCE warm-start pipeline.

---

### #15g ❌ FAILED (2026-03-27) — ASL Focal-Only, BCE Pseudo (v12)

**Result**: Killed after epoch 1. val_roc_auc = **0.7079** — massive regression from v4 baseline (0.7958).

**Root cause**: The audio is a 50/50 mix of focal + pseudo, but loss uses only focal labels. The model gets **penalized for correctly detecting species in the pseudo audio half** (they appear as false positives against focal-only labels). This is fundamentally broken — not just a calibration issue.

---

### ASL Ablation Summary (2026-03-27)

**All three ASL warm-start variants failed:**
| Experiment | γ− | Peak AUC | vs v4 baseline (0.7958) |
|-----------|-----|---------|------------------------|
| v10 (full ASL) | 2.0 | 0.7748 (ep1) | −0.021 |
| v11 (soft ASL) | 0.5 | 0.7917 (ep1) | −0.004 |
| v12 (focal-only) | 2.0 | 0.7079 (ep1) | −0.088 |

**Conclusion: ASL cannot be retrofit onto a BCE-trained warm-start pipeline.** To use ASL, one would need to train from scratch with ASL from epoch 1 — losing 4 generations of warm-start improvement. This makes ASL a net negative for our pipeline. **Keep BCE as the loss function.**

**Revised gap analysis**: The +0.02–0.03 claimed from ASL in the literature assumes training from scratch or ASL-native warm-starts. For our pipeline (5 generations of BCE warm-start), BCE is the correct loss. Focus on PCEN (#15h) as the primary path to close the LB gap.

---

### #15h ✅/❌ — PCEN Proper Path, fold-0 diagnostic (v13, v14)

**Hypothesis**: PCEN cannot be validated on pseudo-labels generated by a log-mel model. The correct validation path requires generating PCEN pseudo-labels first.

**3-step process**:
1. ✅ **Step 1 (focal-only, PCEN)** — DONE: Trained fold-0 from scratch, PCEN + BCE, focal data only. 20 epochs. **LB = 0.762** (single fold, single model, scratch) — validates PCEN strongly.
2. ✅ **Step 2 (generate PCEN pseudo-labels)** — DONE: `pseudo_labels_pcen_v1.csv` generated from fold-0-only v13.
3. ❌ **Step 3 (PCEN self-train, fold-0 only)** — DONE but **FAILED**: v14 self-train (warm-start v13, pcen_v1 pseudo-labels, 30ep) → 0.7494 local val (ep12), **0.749 LB** — regression vs v13 0.762 LB.

**Root cause of v14 failure**: `pseudo_labels_pcen_v1.csv` was generated from a single-fold-0 model only. A single fold's predictions on the unlabeled soundscapes are too noisy — the model hasn't seen 4/5 of the training distribution. The fix is to complete all 5 folds of v13 first, generate ensemble pseudo-labels, then self-train. See **#15i** below.

#### Step 1 Results — v13 (2026-03-27)

**Best fold-0 val ROC-AUC = 0.7188 (epoch 18)**. Below the 0.73 soft gate by 0.011.

| Epoch | Train loss | Val ROC-AUC |
|-------|-----------|-------------|
| 1 | 0.0567 | 0.5247 ★ |
| 5 | 0.0267 | 0.6109 ★ |
| 9 | 0.0227 | 0.6800 ★ |
| 13 | 0.0214 | 0.7107 ★ |
| 18 | 0.0192 | **0.7188** ★ |
| 20 | 0.0175 | 0.7154 |

**Analysis**: Comparable to v8 (0.7214, PCEN+pseudo) and slightly below v9 (0.7308, PCEN+pseudo). Focal-only has fewer training samples (28,606 clips vs ~160K with pseudo), so lower peak is expected. The key advantage is that this checkpoint's internal representations truly match PCEN feature space — enabling correct pseudo-label generation in Step 2.

**Checkpoint**: `models/sed_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_v13.pt`

**Decision**: Proceed to Step 2 despite missing the 0.73 gate. The whole point of the 3-step path is that Step 1 produces a PCEN-native teacher for pseudo-labeling, not a competitive standalone model. v8/v9 showed that pseudo-labels dramatically boost PCEN performance (v8 ep1 = 0.6835 vs v13 ep1 = 0.5247, because v8 had pseudo-label data). Step 3 with PCEN-native pseudo-labels should exceed 0.73.

#### v13 LB Result — 0.762 (2026-03-27)

**v13 LB = 0.762** — only −0.007 vs v4's 0.769 (10 models, 4 warm-start generations, pseudo-labels).

This is a breakthrough validation of PCEN:
- v13 = 1 model, 1 fold, from scratch, no pseudo-labels, 20 epochs
- v4 = 10 models (5×EffB0 + 5×RegNetY), 4 generations of warm-start, pseudo_labels_v3, log-mel
- The near-parity with a 10× simpler setup proves PCEN is strictly better than log-mel for soundscape generalization

**Implication**: A full 5-fold PCEN self-train with PCEN-native pseudo-labels should significantly exceed 0.769. Step 3 (v14) confirmed this is the right direction but requires the full 5-fold ensemble. See **#15i** for the corrected plan.

### #15i ❌ — PCEN 5-fold Stage 1 + Ensemble Pseudo-labels + 5-fold Self-train (v15)

**Result: v15 self-train LB = 0.754 ❌ — worse than single-fold v13 (0.762) and best (0.769).**

Self-train is consistently net-negative for PCEN models (v14: 0.749, v15: 0.754). The pseudo-labels (even from 5-fold ensemble) add noise that degrades performance.

**Steps completed**:
1. ✅ **Fold-0 v13** already existed
2. ✅ **Trained folds 1-4 as v13** (25ep, PCEN, BCE, scratch, focal-only) — val: F0=0.7479, F1=0.7355, F2=0.7317, F3=0.7361, F4=0.7209
3. ✅ **Generated `pseudo_labels_pcen_v2.csv`** from 5-fold v13 ensemble
4. ✅ **5-fold PCEN self-train as v15** — val: F0=0.7342, F1=0.7433, F2=0.7169, F3=0.7367, F4=0.7259 (mean 0.7314, WORSE than v13 mean 0.7344)
5. ✅ **Pushed v15 checkpoints + LB = 0.754 ❌**

**Diagnostic (in progress)**: Submitted v13 5-fold ensemble (no self-train) to isolate whether the self-train step is the problem or PCEN itself caps out below 0.769.

**Gates**:
- v15 fold-0 epoch 1 AUC ≥ 0.76 (warm-start quality check; v14 started at 0.697)
- Final LB ≥ 0.780 → proceed to 5-fold EffB3 (Stage 7 / #15b)
- Final LB ≥ 0.790 → regenerate pseudo-labels from v15 ensemble and run v17 self-train iteration

**Expected**: single fold v13 = 0.762 LB; 5-fold v15 ensemble should reach ~0.785–0.800 LB.

#### #15i Gate Decision Tree (evaluate after v15 LB result)

| #15i LB Result | Interpretation | Next Action |
|----------------|---------------|-------------|
| **>= 0.800** | Full win. PCEN 5-fold self-train validated. | Proceed to #17A (Freq-MixStyle ablation) |
| **0.780–0.799** | Moderate win. PCEN self-train works. | Proceed to #17A |
| **0.770–0.779** | Marginal win. | Proceed cautiously to #17A + start #16A (temporal smoothing) in parallel |
| **< 0.770** | Self-train still degrading. | **Diagnose**: (a) Submit 5-fold v13 ensemble directly (no self-train) — if v13 ensemble LB > v15, self-train step is the problem. (b) If v13 5-fold ensemble > 0.769, skip self-train and use v13 as PCEN baseline. |

### #15 ✅ COMPLETED (2026-03-25) / ❌ REGRESSED — SED Overhaul with PCEN + ASL

**This is now THE path forward.** Perch KD is dropped (caused catastrophic forgetting in v6 SED). Pure SED improvements only.

**Step 1 — Add PCEN** (`src/utils.py`):
- Replace AmplitudeToDB with Per-Channel Energy Normalization
- PCEN normalizes for recording device frequency response variability
- Removing PCEN costs −0.049 soundscape val AUC (critical!)
- Use `torchaudio.functional` PCEN or manual implementation

**Step 2 — Replace BCE with ASL** (`src/self_train.py`, `src/train.py`):
- AsymmetricLoss(γ−=4, γ+=0, clip=0.05)
- Hard negative downweighting focuses learning on genuine positives
- Expected gain: +0.02–0.03

**Step 3 — Add dual loss** (clip + frame, 50/50 weight)
- Clip-level loss for classification, frame-level loss for temporal localization
- Small but consistent improvement

**Step 4 — Add circ_shift augmentation** (circular time shift, p=0.5)
- Simulates varying call positions within chunks
- Expected gain: +0.005–0.01

**Step 5 — Add Freq-MixStyle** (`src/dataset.py`):
- Mix per-frequency-bin mean/std between spectrograms
- Simulates different ARU devices — domain generalization technique
- Expected gain: +0.01–0.03

**Step 6 — Add model soup** (save all epoch checkpoints, average after warmup):
- Average weights of epochs [warmup_ep:] → OOD robustness (SWAD/DiWA benefit)
- No extra inference cost

**Step 7 — Fix validation**: Switch from OOF (train_audio) to soundscape val AUC throughout.

**Step 8 — Retrain and submit**: Full self-training pipeline with all improvements applied.

Script: `scripts/train_stage6_overhaul.sh`  
New self-train version: v6 (break from v5 lineage, fresh "best formula" run)

#### Session Log — 2026-03-25

**Implemented (2026-03-25):**

| Step | File | Change | Expected gain |
|------|------|--------|--------------|
| 1 — PCEN | `src/utils.py` | Replaced `AmplitudeToDB` + min-max norm with `torchaudio.functional.pcen(gain=0.98, bias=2.0, power=0.5, b=0.025)`. Mel scaled ×2³¹ (float32 audio → PCEN expected range). | **+0.049** |
| 2 — ASL loss | `src/self_train.py` | `AsymmetricLoss(γ−=4, γ+=0, clip=0.05)` replaces `BCEWithLogitsLoss`. Asymmetric focal weighting hard-suppresses easy negatives. | **+0.02–0.03** |
| 3 — Dual loss | `src/self_train.py` | 50% clip-level ASL + 50% frame-level ASL. Frame labels broadcast clip labels → all time frames. Uses existing `frame_logits` from `BirdSEDModel`. | +0.005 |
| 4 — Freq-MixStyle | `src/self_train.py` | GPU batch aug (p=0.5, α=0.1): mix per-freq-bin mean/std between randomly paired batch samples. Domain-randomises ARU frequency response. | **+0.01–0.03** |
| 5 — Circ-shift pseudo | `src/self_train.py` | `np.roll` (±25%, p=0.5) added to `PseudoLabelDataset.__getitem__`. Focal data already had this via `BirdTrainDataset._apply_time_shift`. | +0.005 |
| 6 — Model soup | `src/self_train.py` | Post epoch `≥ soup_start_ep` (default 10): saves `*_ep{N}.pt`; after training averages all → `*_soup.pt`; deletes intermediates. Added `--soup-start-ep` CLI flag. | OOD robustness |
| 7 — Script | `scripts/train_stage6_overhaul.sh` | EffB0-v6, 35 epochs, warm-start from v4, pseudo_labels_v4.csv power=1.5, soup-start-ep=10. Pushes soup+best-epoch models to Kaggle dataset. | — |

**Strategy (original)**: Warm-start from EffB0-v4 (LB=0.769 source). PCEN changes the mel representation — model adapts in ~5 epochs via fine-tuning LR (1e-4). Model soup averages epochs 10–35 (26 checkpoints/fold) for SWAD-style OOD robustness on soundscapes.

**Expected ceiling**: ~0.88–0.90 LB (without Perch KD; ~0.04 below Tom’s 0.918 ceiling but a clear jump over our 0.769 best).

#### Session Log — 2026-03-26 — REGRESSION POST-MORTEM

**LB result**: 0.705 ❌ (down from 0.769). Root cause determined.

**Root cause analysis**:

| Issue | Description |
|-------|-------------|
| **Input distribution mismatch** | Folds 1–4 warm-started from fold-0 PCEN soup. But fold-0 itself trained from scratch with PCEN and peaked at only 0.675 — the warm-start just propagated a bad model. |
| **LR schedule wrong for from-scratch** | `CosineAnnealingWarmRestarts(T_0=5)` in a 35-epoch run resets the cosine every 5 epochs — too aggressive. Model repeatedly "forgets." From-scratch on new features needs a single cosine decay. |
| **Too many changes at once** | PCEN + ASL(γ−=4) + dual loss + Freq-MixStyle + circ-shift all simultaneously. ASL(γ−=4) suppresses gradients hard early when predictions are uncertain, likely breaking convergence. Val AUC oscillated ±0.05–0.08 per fold rather than improving smoothly. |
| **Wrong notebook version** | Script trained v7 but notebook was hard-coded to `_v6.pt`. v6 = PCEN trained after warm-starting from log-mel v4 — even worse convergence than v7. |

**Key lessons**:
1. Never warm-start when changing input features (log-mel → PCEN). Train from scratch OR keep consistent features.
2. Validate PCEN in isolation — single fold, BCE loss, nothing else changed.
3. Use `CosineAnnealingLR` (single decay, no restarts) for from-scratch runs with novel features.
4. ASL(γ−=4) may be too aggressive early — try γ−=2 or use BCE initially.
5. Always match the inference notebook’s checkpoint pattern to what the training script actually saves.

**Status**: Notebook restored to v4 config (log-mel, EffB0-v4 + RegNetY-v2), pushed as v17. **Confirmed 0.769 LB** ✅. Next: run `scripts/ablate_pcen_fold0.sh` diagnostic before any full 5-fold PCEN run.

### #15b ⬜ — Stage 7: EffB3 warm-start + larger backbone ensemble (~0.90+ LB)
**After Stage 6 LB result is confirmed.**

**Goal**: Add architecture diversity to the Stage 6 ensemble by training `tf_efficientnet_b3.ns_jft_in1k` with the full Stage 6 recipe (PCEN + ASL + Freq-MixStyle + model soup), warm-started from the Stage 5 EffB3-v1 checkpoints (which were trained from scratch and underperformed — they now have a starting point).

**Why this is the right next step:**
- Stage 6 gives us 5×EffB0-v7 soup models. Ensembling with 5×EffB3-v2 (warm-started, Stage 6 recipe) adds capacity AND architecture diversity.
- EffB3-v1 scored 0.737 val AUC from scratch. Warm-starting from v1 + PCEN/ASL recipe should bring it to ≥0.77 val AUC, making it a net positive in the ensemble.
- Past lesson (#9): Never include a fresh-init large backbone in the ensemble — always warm-start first.

**Steps:**
1. **Wait for Stage 6 LB**: Only proceed if Stage 6 LB ≥ 0.80 (confirms PCEN+ASL working).
2. **Generate pseudo-labels v5**: Run `src/pseudo_label.py` using Stage 6 EffB0-v7 soup models (5-fold ensemble). These will be substantially better pseudo-labels than v4.
3. **Train EffB3-v2** (5 folds, 35 epochs, `--version 2`):
   - Warm-start from `models/sed_tf_efficientnet_b3.ns_jft_in1k_fold*_seed42_v1*.pt`
   - Same hyperparameters as Stage 6: PCEN, ASL, Freq-MixStyle, model soup, pseudo_labels_v5.csv
   - Use `--soup-start-ep 15` (later start than v7 since warm-start already good)
4. **Also retrain EffB0-v8** warm-starting from v7 soup on pseudo_labels_v5.csv — improves teacher quality.
5. **Ensemble**: 5×EffB0-v8 soup + 5×EffB3-v2 soup (10 models total).
6. **Update inference notebook** for new model set.

**Script**: `scripts/train_stage7_b3.sh`

**Expected LB**: ~0.88–0.91 (EffB0 Stage 6 ~0.82–0.84 + EffB3 diversity + better pseudo-labels)

**Decision gate**: If Stage 6 LB < 0.79 (no improvement over 0.769), investigate why before proceeding — may indicate PCEN implementation bug or training issue.

### #16 ⬜ — Post-Processing (parallel track, zero retraining)

Modify only the inference notebook. Can run in parallel with any training step.

| # | Change | Expected Gain | Status |
|---|--------|--------------|--------|
| A | **Temporal smoothing**: `gaussian_filter1d(preds, sigma=1, axis=0)` per species per soundscape. Bird/amphibian calls persist >5s. | **+0.008 LB** (confirmed) | ✅ v22 notebook — 0.765→0.773 |
| B | **Site-based species prior**: test filenames contain site codes; build co-occurrence matrix from `train_soundscapes_labels.csv`, multiply `output × (1 + 0.1 × site_prior)`. | +0.005–0.015 LB | ⬜ |
| C | **Ensemble combination `0.6*mean + 0.4*max`** across folds: pure mean suppresses rare-species detections. | +0.002–0.005 LB | ⬜ |
| D | **Temperature scaling (per-class)**: fit `T` per class on val via `sigmoid(logit / T)`. | +0.005–0.01 LB | ⬜ |

**Order**: A first (largest, simplest), validate on local soundscape val, then B, C, D.

### #17 ⬜ — One-at-a-Time Ablations (after #15i LB confirmed)

Each step re-runs 5-fold self-train changing exactly ONE flag from the previous best config. Avoids the Stage 6 fiasco of changing everything at once.

| Step | Version | Change | Expected Gain | Gate |
|------|---------|--------|--------------|------|
| A | v16 | Enable `--freq-mixstyle` (remove `--no-freq-mixstyle`). Warm-start from v15. | +0.01–0.03 | LB >= v15 + 0.005 |
| B | v17 | Enable `--dual-loss` (remove `--no-dual-loss`). Warm-start from v16. | +0.005 | LB >= v16 |
| C | v18 | Enable `--soup-start-ep 10` (checkpoint averaging). Warm-start from v17. | +0.005–0.01 | LB >= v17 |

**Training time per step**: ~10-12 hours (5 folds × 30 epochs).

**If any step fails its gate**: disable that flag and proceed to the next ablation from the last known-good version.

### #19 ⬜ — Pseudo-Label Regeneration + Iteration (after #17 converges)

Regenerate `pseudo_labels_pcen_v3.csv` from the best v16/v17/v18 ensemble, run another self-train iteration (v19). Proven pattern from iterations 2–4 (0.752→0.762→0.769).

**Key**: Always use the full 5-fold ensemble for pseudo-label generation. Never single-fold (lesson from v14).

### #20 🔄 — Perch Inference Notebook (adapted from public 0.908 notebook)

**Previous Perch attempts (MLP probe, KD) all failed.** The breakthrough was studying the public 0.908 notebook (`yashanathaniel/birdclef-2026-perch-v2-0-908`) which uses a completely different approach:

**Architecture**: No deep learning. Perch v2 frozen SavedModel → raw logits + 1536-d embeddings → Bayesian site+hour priors → sklearn LogReg probes → dual temporal smoothing → temperature scaling.

**Key insights we were missing:**
1. **Use raw Perch logits directly** (not just embeddings) — logits are already species-discriminative
2. **Site + hour-of-day Bayesian priors** from `train_soundscapes_labels.csv` — huge signal (species are location/time-dependent)
3. **LogReg probes on soundscape data** (not MLP on focal clips) — trains on the same domain as test data
4. **Genus proxy** for unmapped amphibians — max logit across genus matches in Perch's label list
5. **Class-specific smoothing** — avg-neighbor for texture classes (frogs/insects, alpha=0.35), local-max for event classes (birds, alpha=0.15)
6. **Temperature scaling** (T=1.15) — better calibration for rare classes in macro-averaged ROC-AUC

**No local training required.** Everything runs at inference time on Kaggle CPU:
- Perch v2 SavedModel: frozen Google model (public Kaggle model)
- LogReg probes: fit on ~708 labeled soundscape windows (seconds with sklearn)
- Site/hour priors: computed from `train_soundscapes_labels.csv`
- PCA: fit on cached embeddings

**Notebook**: `jupyter/perch/birdclef2026-perch-inference.ipynb` → pushed as `stevewatson999/birdclef-2026-perch-inference` v1.

**Data sources** (all public, no local models needed):
- `google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1` (Perch v2)
- `jaejohn/perch-meta` (pre-cached embeddings, optional)
- `kdmitrie/bc26-tensorflow-2-20-0` (TF 2.20 wheel)
- `birdclef-2026` (competition data)

**Result**: **0.908 LB** ✅ — confirmed. Proceed to SED+Perch ensemble (#20b).

### #21 ⬜ — EffB3 Warm-Start (after PCEN pipeline stabilizes at LB >= 0.780)

Train `tf_efficientnet_b3.ns_jft_in1k` with the validated PCEN config, warm-started from EffB3-v1 checkpoints. Gate: fold-0 val AUC >= 0.76 before running remaining folds.

Ensemble: 5×EffB0-vN + 5×EffB3-v2 (10 models). Expected: +0.01–0.02 from architecture diversity.

### Revised Timeline (2026-03-30)
| Date | Action | Expected LB |
|------|--------|-------------|
| Mar 29 ✅ | #15i: PCEN 5-fold v15 self-train → 0.754 ❌; v13 5-fold → 0.765; v13+smoothing → **0.773** ✅ | 0.773 |
| Mar 30 ✅ | #20: Perch v2 inference notebook — **0.908 LB** | 0.908 |
| Mar 30 ❌ | #20b: SED+Perch inline ensemble — **timed out** (even 1-fold SED exceeds 90 min CPU) | — |
| Mar 30 ❌ | #22: LightGBM probes + larger PCA + wider temporal features | 0.904 ❌ |
| Mar 31 ✅ | #22b: Revert PCA to 32 — confirmed 0.908 baseline | 0.908 ✅ |
| Mar 31 | #23: TTA on Perch (neighbor-average TTA, v12) | **0.906 LB** ❌ | Neighbor-averaging embeddings+logits with adjacent windows hurts (−0.002 vs 0.908 baseline). Smoothing blurs per-window signal the probes need. |
| Apr 1–2 | #24: Pre-compute SED predictions as Kaggle dataset + rank-average blend | +0.005–0.01 |
| Apr 3–5 | #25: Per-class Platt calibration + deeper probes (small NN or XGBoost) | +0.005–0.01 |
| Apr 6–10 | #26: Retrain SED with SoftAUCLoss + multi-arch ensemble (pre-computed) | +0.01–0.02 on SED |
| Apr 11–15 | Iterate: tune blend weights, multi-round pseudo-labeling for SED | **Target: 0.94+** |
| May 27 | Entry deadline | Best submission locked |
| Jun 3 | Final submission deadline | — |

**Target**: 0.942+ LB. **Realistic ceiling**: Perch probes optimized ~0.92; + pre-computed SED ensemble ~0.93–0.94. Top competitor: 0.9334.

### Strategy Shift (2026-03-30)

**SED cannot run inline** on Kaggle CPU (even 1 fold times out). All SED contributions must be **pre-computed as Kaggle datasets**. The primary scorer is now Perch v2 with improved probes. SED serves as a supplementary signal via rank-averaged blending.

**Key insight from 2025 top solutions**: 1st place used SoftAUCLoss (directly optimizes ROC-AUC metric), multi-architecture ensemble (5 archs × 2 spectrogram types), and 4 rounds of pseudo-labeling to reach 0.933. 2nd place used ECA-NFNet-L0 + EfficientNetV2-S with focal BCE, 2-3 pseudo-label iterations, and rank-based blending.

### #22 ❌ — Upgrade Perch Probes (LightGBM + larger PCA + wider temporal features)

**Goal**: Replace LogReg probes with LightGBM; increase PCA dimensions; add wider temporal context features. All inference-time, no retraining needed. Pure notebook change.

**Changes**:
1. **LightGBM instead of LogReg**: non-linear, handles feature interactions, fast on CPU
2. **PCA dim**: grid search 32/64/128/256 (current: 32)
3. **Wider temporal features**: 3-window mean, 5-window mean, file-level max, variance — not just prev/next
4. **Per-class calibration**: Platt scaling per class instead of global T=1.15

**Result**: **0.904 LB** ❌ — regressed −0.004 from 0.908 baseline. PCA 64 overfits on ~708 labeled soundscape windows. LightGBM was grid-searched but `PROBE_TYPE` left as `logreg` in submit mode, so regression is purely from PCA 64 + wider temporal features.

**Lesson**: With only ~708 training windows across ~5 sites, increasing feature dimensionality hurts. Changes must be tested one-at-a-time.

### #22b ✅ — Revert PCA to 32 (confirm 0.908 baseline)

Reverted `PROBE_PCA_DIM` from 64 back to 32. **0.908 LB confirmed** — PCA 64 was the sole cause of regression.

### #23 ❌ — Test-Time Augmentation on Perch (Dead End)

**Goal**: Improve predictions by averaging nearby window signals. 2025 top solutions reported +0.01 from TTA.

**v10–v11 (failed)**: +2.5s time-shift with 2× Perch inference. Timed out — 2× Perch calls exceed 90-min CPU budget.

**v12 (failed, 0.906 LB)**: Neighbor-average TTA — zero extra Perch cost. Averaged each window's raw logits and embeddings with adjacent windows (prev, current, next) within the same 60s file. Result: −0.002 regression vs 0.908 baseline. Smoothing blurs per-window temporal signal that probes rely on.

**Conclusion**: TTA on Perch is a dead end — both 2× inference (timeout) and neighbor averaging (hurts score) fail. Revert to baseline.

### #24 ⬜ — Pre-Compute SED Predictions as Kaggle Dataset

**Goal**: Run SED inference in a separate GPU notebook, save predictions as CSV/NPZ dataset, attach to inference notebook for rank-averaged blending.

**Steps**:
1. Create a GPU-enabled Kaggle notebook that loads SED checkpoints + test soundscapes
2. Run full 5-fold SED inference (no time constraint on GPU)
3. Save predictions as Kaggle dataset (`stevewatson999/birdclef2026-sed-predictions`)
4. Attach dataset to Perch inference notebook
5. Rank-average blend: `final = 0.5 * rank(perch) + 0.5 * rank(sed)` (tune weights)

**Problem**: Hidden test files are only available in submission notebooks. Pre-computing SED on test data is not possible. **Workaround**: Run SED inline but with extreme optimizations (TFLite/ONNX conversion, single fold, reduced mel resolution) OR accept SED covers only train soundscapes (used for probe training features, not test predictions).

**Alternative**: Include SED logits as additional probe features — train LightGBM probes on Perch embeddings + SED predictions on train soundscapes only. The probes learn to use SED signal where available.

### #25A 🔄 — Dual-Probe Ensemble (LogReg + LightGBM Rank-Average)

**Goal**: Train both LogReg and LightGBM probes per class, rank-normalize each per class, average. Ensemble diversity without extra model or compute cost.

**Changes** (all in `jupyter/perch/birdclef2026-perch-inference.ipynb`):
1. `PROBE_TYPE = 'dual'` in Settings
2. Cell 52: trains both `probe_models_lr` and `probe_models_lgbm` for each class
3. Cell 55: removes neighbor-average TTA (#23 dead end), applies both probe sets, per-class `rankdata()` average
4. Cell 58: rank-averaged scores already in (0,1) — used directly as probabilities (no temperature scaling needed)

**Gate**: LB > 0.908 baseline.

### #26 ⬜ — Retrain SED with SoftAUCLoss + Multi-Architecture Ensemble

Train SED models using SoftAUCLoss (directly optimizes macro-averaged ROC-AUC). Add ECA-NFNet-L0 and EfficientNetV2-S for architecture diversity. Pre-compute predictions for blend.

---

## #18 ⬜ — Data Quality, Noisy Labels & Post-Processing Improvements

Low-risk, high-ROI changes that don't require architectural changes. Implement after Stage 6/7 LB is confirmed.

### 18A — Data Splitting Improvements

| # | Change | Impact | Status |
|---|--------|--------|--------|
| A | **Site-aware soundscape splits**: verify no soundscape site appears in both pseudo-label training and val. The current 66-file exclusion is correct — confirm site codes match test set sites. | Prevents silent leakage | ⬜ |
| B | **Log class counts per fold** after `MultilabelStratifiedKFold`: confirm all 234 classes have ≥1 sample in every fold. The minimum-10-per-class duplication should prevent this, but log it explicitly. | Ensures all classes have gradient signal | ⬜ |
| C | **Include `train_soundscapes_labels.csv` rows as hard training examples in folds 1–4** (not fold 0 val): expert-labeled soundscape data is gold standard and covers the actual test domain — currently only used for validation. | +signal on hard positives, domain bridging | ⬜ |
| D | **Hold out one recording site for OOD validation**: reserve one site entirely from training and pseudo-labeling, test final model on it before submission. Gives unbiased estimate of site-generalization. | More reliable stopping criterion | ⬜ |

### 18B — Noisy Label Handling

| # | Change | Impact | Status |
|---|--------|--------|--------|
| A | **Per-species pseudo-label thresholds**: only include pseudo-label windows where confidence > max(0.3, 2× average train frequency) for that species. Generic power=1.5 sharpens all species equally; rare species with low precision need a harder gate. | Reduces false-positive pseudo-labels for rare species | ⬜ |
| B | **Label smoothing = 0.05 on primary labels**: replace hard label `1.0` with `0.95`. Species clips are often mislabeled or partially audible; even 0.05 smoothing regularizes over-confident focal loss. | +0.005–0.01 val AUC | ⬜ |
| C | **Secondary labels at weight 0.1 (not 0)**: completely masking secondary labels discards real presence information. Weight 0.1 is small enough to not force predictions in any specific window, but prevents the model from suppressing confirmed secondary species. | Helps multi-species soundscapes | ⬜ |
| D | **iNat quality heuristic**: filter out iNat clips shorter than 3 seconds. No ratings exist, but duration is a useful proxy for recording quality. | Removes obviously bad training examples | ⬜ |
| E | **Confidence threshold for pseudo-label window inclusion**: only include a window if at least one species exceeds 0.5 confidence. Pure background windows add noise and class imbalance. | Cleaner pseudo-label distribution | ⬜ |

### 18C — Post-Processing Predictions (inference-time, zero retraining cost)

| # | Change | Impact | Status |
|---|--------|--------|--------|
| A | **Temporal smoothing within soundscape**: apply a 3-window Gaussian kernel over time (windows t-1, t, t+1) for each species per soundscape. Bird/amphibian calls persist longer than 5s. `scipy.ndimage.gaussian_filter1d(preds, sigma=1, axis=0)` per file. | **+0.008 LB** (confirmed) | ✅ Done — v22, 0.765→0.773 |
| B | **Site-based species prior**: test filenames contain site code (e.g., `S05`). Build a site-species co-occurrence matrix from `train_soundscapes_labels.csv`. At inference multiply model output by `(1 + 0.1 * site_prior)`. | +0.005–0.015 LB | ⬜ |
| C | **Temperature scaling (per-class)**: calibrate each species' output using the val set. Fit temperature `T` per class via `sigmoid(logit / T)` to minimize BCE on val. Uncalibrated models systematically mis-rank rare species. | +0.005–0.01 LB | ⬜ |
| D | **Ensemble combination `0.6*mean + 0.4*max` across folds**: pure mean suppresses rare-species predictions caught by only 1–2 folds. The max component preserves these detections. Test vs pure mean on soundscape val. | +0.002–0.005 LB | ⬜ |

**Recommended implementation order**: 18C-A (temporal smoothing) → 18C-B (site prior) → 18B-B (label smoothing) → 18B-A (per-species pseudo threshold). First two are inference-only changes with no retraining required.

---

## Perch Integration v2 — New Approach

### Why Previous Perch Attempts Failed

| Attempt | Result | Root Cause |
|---------|--------|-----------|
| Perch MLP standalone (v1–v5) | 0.90 local val → 0.49–0.72 LB | Shallow MLP on frozen embeddings cannot generalize from clean focal clips to noisy soundscapes. No temporal context (5s windows independently). 31/234 species missing from Perch vocabulary. |
| Perch KD for SED (v6) | −0.06 val AUC (0.80→0.74) | ×10 oversample of soundscape pseudo-labels caused catastrophic forgetting of focal data. 31 missing species got zero-signal KD labels, actively poisoning those classes. |
| SED 70% + Perch MLP 30% ensemble | 0.667 LB | Perch MLP branch (~0.49 LB) destroyed SED predictions. Both branches need to be individually strong for blending to work. |

### What We Learned

1. **Perch embeddings are powerful** — they encode rich species-discriminative features from massive pretraining
2. **Frozen Perch + shallow MLP fails on domain shift** — the MLP has no capacity to adapt to noisy soundscapes
3. **Perch as direct KD teacher is too noisy** — its soft labels on soundscapes are poorly calibrated (mean 87 species > 0.5 per window)
4. **31/234 competition species are missing from Perch** — any approach must handle these gracefully
5. **Output-level blending requires both branches to be strong** — Perch MLP at 0.49 LB poisons any ensemble

### New Strategy: Perch as Feature Augmentation to SED

Instead of using Perch as a standalone model or a teacher, **concatenate Perch embeddings as additional input features to the SED model**. The SED backbone processes mel spectrograms as before, but the SED head also receives Perch embedding features — giving it access to Perch's species knowledge while retaining full end-to-end gradient flow.

#### Option A: Perch Feature Concatenation (recommended first attempt)

```
Input: 20-second audio clip
    ├── Mel spectrogram (3, 224, 512) → CNN backbone → (B, C, T') feature map
    └── 4× Perch TFLite 5-sec windows → 4× 1536-d embeddings
           → Linear(1536, C) → (B, C, 4)
           → Interpolate to T' frames → (B, C, T')
    ↓
Concatenate along channel dim → (B, 2C, T')
    ↓
Existing SED head (Conv1d attention + classifier)
    ↓
clip_logits + frame_logits (as before)
```

**Key design decisions:**
- Perch embeddings are projected to match the CNN backbone's channel dimension via a learned `Linear(1536, C)` layer
- The 4 embeddings (one per 5-sec window in 20-sec chunk) are interpolated to match the backbone's temporal resolution
- Concatenation doubles the channel dimension going into the SED head — the attention and classifier Conv1d layers are widened accordingly
- **Gradient flows through the projection layer but NOT through Perch itself** (frozen TFLite inference)
- For the 31 missing species: Perch embeddings still encode acoustic features even if Perch's own classifier doesn't know the species — the SED head learns to use these features

**Advantages over previous approaches:**
- SED backbone still processes mel spectrograms end-to-end — no domain gap issue
- Perch features are auxiliary, not primary — if Perch gives garbage for a window, the SED backbone's features still carry the prediction
- Full gradient flow through the fusion layer — model learns *how* to use Perch features, not just blindly trusting them
- Works with PCEN mel — no representation conflict since Perch processes raw audio independently
- Missing species: SED head has full gradient signal from focal labels; Perch embedding dimension provides genus/family-level acoustic information even for unmapped sonotypes

**Risks:**
- Perch TFLite inference is slow (~150ms per 5-sec window on CPU). During training, need to pre-extract and cache embeddings (already done in `perch_v2/data/processed/perch_embeddings/`)
- Adds ~1536→C projection parameters — minor compared to backbone size
- Inference time on Kaggle: 4 extra TFLite calls per 20-sec chunk × 9 chunks per soundscape = 36 calls × ~150ms ≈ 5.4s per soundscape × 600 soundscapes = 54 min. Combined with SED inference (~14 min) = **68 min total** (within 90-min budget)

**Implementation plan:**

1. **Modify `src/model.py`**: Add `PerchFusionSEDModel` that wraps `BirdSEDModel` with a Perch embedding branch
2. **Modify `src/dataset.py`**: `BirdTrainDataset` loads pre-extracted Perch embeddings alongside mel spectrograms (from `perch_v2/data/processed/perch_embeddings/train_audio_pw/`)
3. **Modify `src/self_train.py`**: `PseudoLabelDataset` loads pre-extracted Perch embeddings for soundscape windows (from `perch_v2/data/processed/perch_embeddings/train_soundscapes/`)
4. **Single fold-0 diagnostic first**: Train fold-0 PCEN + Perch fusion, 25 epochs, compare val AUC to PCEN-only v13 (0.7479). Gate: must exceed 0.76.
5. **If gate passes**: Full 5-fold training + self-train cycle

#### Option B: Cross-Attention Fusion (if Option A underwhelms)

Instead of concatenation, use cross-attention where the SED backbone's temporal features attend to Perch embeddings:

```
SED features: (B, C, T')  →  Q
Perch embeddings: (B, 4, 1536) → Linear → (B, 4, C)  →  K, V
    ↓
MultiheadAttention(Q, K, V)  →  (B, C, T')
    ↓
Add & LayerNorm with residual from SED features
    ↓
SED head (unchanged)
```

This is more expressive but adds complexity. Only try if Option A fails to show improvement.

#### Option C: Perch-Guided Pseudo-Label Refinement (lowest risk, can run in parallel)

Instead of modifying the SED model, use Perch to **improve the quality of pseudo-labels**:

1. For each soundscape 5-sec window, compute both:
   - SED ensemble prediction (current approach)
   - Perch TFLite prediction (203/234 species)
2. For the 203 Perch-covered species: `refined_pseudo = 0.7 × SED_pred + 0.3 × Perch_pred`
3. For the 31 uncovered species: `refined_pseudo = SED_pred` (unchanged)
4. Apply power transform as usual

This improves pseudo-label quality without any model architecture changes. The 0.7/0.3 weighting trusts SED more (it's trained on the right data) while using Perch as a regularizer.

**Advantages**: Zero architecture changes, zero inference overhead, can be tested immediately with existing code.
**Risk**: If Perch predictions on soundscapes are as noisy as before (~87 species > 0.5 per window), this will add noise, not reduce it. Requires thresholding Perch predictions first (only include Perch signal where Perch confidence > 0.8).

### Perch Integration Sequencing

| Order | Approach | Risk | Expected Gain | Training Time |
|-------|----------|------|--------------|---------------|
| 1 | **Option C** (pseudo-label refinement) | Very low | +0.005–0.015 | 0 extra (just re-run pseudo-label step) |
| 2 | **Option A** (feature concatenation) | Medium | +0.03–0.06 | ~12h fold-0 diagnostic + ~50h full 5-fold |
| 3 | **Option B** (cross-attention) | High | +0.04–0.08 | Same as A + more debugging |

**Start with Option C** because it requires no code changes to the model — just modify `pseudo_label.py` to blend SED and Perch predictions. If Option C shows improvement, it validates that Perch signal is useful for soundscapes and motivates the deeper Option A integration.

### Pre-requisites (already satisfied)

- [x] Perch TFLite model exists: `perch_v2/models/perch_v2/model.tflite`
- [x] Per-window train_audio embeddings: `perch_v2/data/processed/perch_embeddings/train_audio_pw/`
- [x] Per-window soundscape embeddings: `perch_v2/data/processed/perch_embeddings/train_soundscapes/`
- [x] Species mapping (203/234 covered): implemented in `perch_v2/src/extract_embeddings.py`
- [ ] `PerchFusionSEDModel` in `src/model.py` (needed for Option A)
- [ ] Perch embedding loading in `src/dataset.py` and `src/self_train.py` (needed for Option A)
- [ ] Perch-blended pseudo-label script (needed for Option C)

### Timing Budget on Kaggle (Option A inference)

| Operation | Time |
|-----------|------|
| Load SED models (5 folds) | 1 min |
| Load Perch TFLite model | 0.5 min |
| Per soundscape: 9 chunks × mel + SED forward | ~2.5 sec |
| Per soundscape: 12 windows × Perch TFLite | ~1.8 sec |
| Total per soundscape | ~4.3 sec |
| 600 soundscapes | **43 min** |
| Overhead + CSV write | 5 min |
| **Total** | **~48 min** (42-min buffer) |

This fits well within the 90-min budget. The key insight: Perch TFLite processes 5-sec windows natively (no overlap needed), and we only need 12 calls per soundscape (not 36 — we don't need overlapping Perch windows, just one per 5-sec output segment).

---

## Timeline (updated 2026-03-28)

### Completed (Mar 13–28)
| Week | Phase | Goal |
|------|-------|------|
| Mar 13–16 | #0–#6 | Setup, EDA, Perch baseline, core pipeline, Stage 1 training |
| Mar 17–19 | #7–#8 | Pseudo-labeling + self-training iter 1–2 (LB 0.762) |
| Mar 20–21 | #9 | Multi-iterative pseudo-labeling iter 3–4 (LB 0.769) |
| Mar 22–24 | #14 | Perch MLP probe (failed on LB), Perch KD (failed) |
| Mar 25–26 | #15 | SED overhaul with PCEN+ASL (regressed), diagnostic ablations |
| Mar 27–28 | #15c–i | PCEN validation, ASL abandonment, PCEN 5-fold pipeline launched |

### Upcoming (Mar 29 – Jun 3)
| Mar 29 | #15i LB gate + #16A temporal smoothing | ~0.785–0.800 + post-proc |
| Mar 30–31 | #17A Freq-MixStyle ablation (v16) | +0.01–0.03 |
| Apr 1–3 | #17B–C Dual loss + model soup (v17, v18) | +0.005–0.015 |
| Apr 4–5 | #19 Pseudo-label regen + iteration (v19) | +0.005–0.01 |
| Apr 6–8 | #20 Option C: Perch pseudo-label refinement | +0.005–0.015 |
| Apr 9–15 | #20 Option A: Perch feature fusion (if C succeeds) | +0.03–0.06 |
| Apr 16–18 | #21 EffB3 warm-start backbone diversity | +0.01–0.02 |
| Apr 19+ | Iterate, tune, final ensemble | Target: 0.85–0.90 |
| May 27 | Entry deadline | Best submission locked |
| Jun 3 | Final submission deadline | — |
