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
| Best Kaggle LB score | 0.590 (Perch zero-shot baseline, 2026-03-16) |
| Best local val ROC-AUC | 0.7408 (Stage 1 SED EfficientNet-B0, 5-fold mean, 2026-03-16) |
| Architecture | SED — EfficientNet-B0 + GEM pool + Conv1d attention |
| Training script | src/train.py, scripts/train_stage1.sh |
| Model dataset | `stevewatson999/birdclef2026-models` (to create) |
| Submission notebook | `jupyter/birdclef2026-inference.ipynb` (to create) |

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
10. [Phase 4 — Ensemble & ONNX Export](#phase-4--ensemble--onnx-export)
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
│   ├── export_onnx.py             # ONNX export for CPU inference
│   ├── evaluate.py                # Local ROC-AUC evaluation against train_soundscapes_labels.csv
│   └── utils.py                   # Audio loading, mel spectrogram, padding helpers
├── scripts/
│   ├── train_stage1.sh            # Run full 5-fold Stage 1 training
│   ├── pseudo_label_soundscapes.sh
│   ├── self_train_stage2.sh
│   └── export_ensemble_onnx.sh
├── models/                        # Saved model weights (gitignored)
│   ├── stage1_effb0_fold0/
│   │   ├── best.pth
│   │   └── last.pth
│   ├── ...
│   └── ensemble_onnx/
│       └── model.onnx
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
onnxruntime            # CPU inference on Kaggle
pandas, numpy, scikit-learn
torch-audiomentations  # GPU-accelerated PitchShift, TimeShift, AddBackgroundNoise
tensorboard            # optional
```

### Hardware (local training)
- NVIDIA GB10 (Blackwell, 128 GB unified memory, 273 GB/s LPDDR5X)
- Training in **BF16** — BF16 delivers 92 TFLOPS (vs 46 TFLOPS FP32) and avoids NaN issues (unlike FP16). EfficientNet/RegNetY/NFNet are all safe under BF16. See `optimize.md` in the Akkadian project for full reference.
- Inference ONNX on CPU (to match Kaggle constraint)

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
> **Note**: `torch.compile` is used during **training** for speed. For ONNX export, compile the model AFTER training; export the un-compiled weights. `torch.compile` and `torch.onnx.export` are incompatible in the same session.

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
Stage 1 → Pseudo-labels v1 → Self-train iteration 1 → Pseudo-labels v2 → Self-train iter 2 → ...
```

| Iteration | Power transform | Add backbones | Expected score boost |
|-----------|----------------|---------------|---------------------|
| 1 | 1.0 (raw) | EffB0, RegNetY008 | +3–4 pts |
| 2 | 1.0 / 0.65 | + EffB3, RegNetY016 | +1 pt |
| 3 | 1.0 / 0.55 | + EffB4, ECA-NFNet | +0.9 pt |
| 4 | 1.0 / 0.60 | Same | +0.3 pt |

**Decision gate**: Check LB before iteration 5. If no improvement → stop.

### 3.1 Dedicated Insecta/Amphibia Model

Train a **separate model** using Xeno-canto data for expanded Insecta/Amphibia species:
- Target classes: all Insecta + Amphibia from taxonomy.csv
- Additional classes: extra species from same families downloaded from Xeno-canto
- Model: EfficientNet-B0 (deeper models didn't help in 2025)
- Training: 40 epochs, BS=128
- At inference: run for all species, insert non-zero predictions only for Insecta/Amphibia columns
- Expected gain: +0.002–0.003 ROC-AUC

---

## Phase 4 — Ensemble & ONNX Export

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

### 4.3 ONNX Export (`src/export_onnx.py`)

```bash
python src/export_onnx.py \
    --models models/stage1_effb0_fold0/best.pth \
             models/stage2_effb3/best.pth \
             ... \
    --output models/ensemble_onnx/
```

ONNX export makes CPU inference 2–3× faster vs native PyTorch:
```python
# Export each model to ONNX (no quantization — quantization may hurt accuracy)
torch.onnx.export(
    model,
    dummy_input,      # (1, 3, 224, 512)
    output_path,
    dynamic_axes={"input": {0: "batch_size"}}
)
```

**Note**: `torch.compile` is safe for inference but NOT compatible with ONNX export pathway. Use ONNX for Kaggle CPU inference.

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
| ONNX (not PyTorch) | Strongly preferred | 2–3× faster on CPU |
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
| `SED_B0_Stage1` | EfficientNet-B0 | 1 | 0.7408 (mean) | — | Fold scores: F0=0.7601 F1=0.7500 F2=0.7295 F3=0.7339 F4=0.7305; 15 epochs BF16, 2026-03-16 |

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
| **ONNX export 2–3× faster CPU** | Essential for 90-min constraint | Always export to ONNX for submission |
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

---

## Hard Constraints & Known Pitfalls

```
NEVER submit GPU-dependent code       → GPU runs for only 1 minute on Kaggle
NEVER skip ONNX export                → PyTorch too slow for 90-min CPU limit
NEVER use FP16                        → NaN risk; use BF16 (safe for EfficientNet/RegNetY/NFNet)
NEVER use torch.compile during ONNX export → Incompatible; export from un-compiled model
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

### #7 ⬜ — Pseudo-label Generation (~4 hours)
Run inference on all unlabeled `train_soundscapes`  
Save confidence-weighted pseudo-labels to `data/processed/pseudo_labels_v1.csv`

### #8 ⬜ — Self-training Iteration 1 (~2 days)
Train Noisy Student: focal + pseudo-labeled data + hard negative upsampling  
**Gate**: LB improvement +3–4 pts vs Stage 1. Tag `SED_B0_SelfTrain1_<score>`.

### #9 ⬜ — Multi-iterative pseudo-labeling (iterations 2–4)
Each iteration: generate new pseudo-labels → apply power transform → retrain with larger models  
**Gate after each**: submit to LB, check improvement. Stop when delta < 0.001.

### #10 ⬜ — Dedicated Insecta/Amphibia Model
Train on expanded Xeno-canto insect/amphibian data with hard negative emphasis

### #11 ⬜ — Final Ensemble Construction
Combine 6–7 models from multiple stages  
Apply model soup (checkpoint weight averaging: last 3 epochs within backbone)

### #12 ⬜ — ONNX Export & Inference Notebook
Export all models (without torch.compile active) → build submission notebook → time locally (should be ≤60 min) → submit

### #13 ⬜ — Inference Tuning
- Sweep `smoothing kernel shape`, `delta-shift TTA`, `power adjustment` on local val
- Target: +0.005–0.01 ROC-AUC from post-processing
- Tag final best submission `Final_ensemble_<score>`

---

## Timeline

| Week | Phase | Goal |
|------|-------|------|
| Mar 13–16 | #0–#2 | Fix CLAUDE.md, setup, data download, EDA |
| Mar 17–18 | #3 | Google Perch baseline submission |
| Mar 19–21 | #4 | Core pipeline (config, dataset, model, utils) |
| Mar 22 – Apr 4 | #5–#6 | Stage 1 training + hard neg mining |
| Apr 5–11 | #7–#8 | Pseudo-labeling + self-training iter 1 |
| Apr 12 – May 2 | #9 | Multi-iterative pseudo-labeling (iterations 2–4) |
| May 3–16 | #10–#11 | Insecta model + final ensemble construction |
| May 17 – May 27 | #12–#13 | ONNX, notebook, inference tuning |
| May 27 – Jun 3 | Buffer | Final tweaks, best submission selection |
