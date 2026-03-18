---
description: "Use when writing or editing dataset loading, audio preprocessing, mel spectrogram, augmentations, or MixUp. Covers all audio pipeline parameters and augmentation settings."
applyTo: "src/dataset.py, src/utils.py"
---

# Dataset & Audio Pipeline Guidelines

## Audio Pipeline Parameters

| Parameter | Value |
|-----------|-------|
| Sample rate | 32 kHz (no resampling needed — recordings are already 32 kHz) |
| Chunk duration | 20 seconds = 640,000 samples |
| Mel bins (`n_mels`) | 224 |
| FFT size (`n_fft`) | 4096 |
| Hop length | 1252 → output width ≈ 512 |
| Frequency range | 0–16,000 Hz |
| Dynamic range | 80 dB (top_db) |
| Input tensor | (B, 3, 224, 512) — 3-channel by repeating mel spectrogram |

## MixUp (critical details)

```python
# Normalize by absmax before mixing — NOT peak normalize
w1 = w1 / (np.abs(w1).max() + 1e-8)
w2 = w2 / (np.abs(w2).max() + 1e-8)
mixed = 0.5 * w1 + 0.5 * w2          # Fixed 0.5 weight — NOT random beta
labels = np.maximum(label1, label2)   # Element-wise max — NOT average
```

## Augmentations

| Augmentation | Probability | Notes |
|---|---|---|
| Background noise injection | p=0.3–0.5 | Mix real env noise (freefield1010, warblrb, birdvox) at low gain |
| PitchShift | p=0.3 | ±2 semitones via `torch_audiomentations` on GPU |
| TimeShift | p=0.5 | ±25% via `torch_audiomentations` on GPU |
| SpecAugment time mask | p=0.3 | Applied to mel spectrogram |
| SpecAugment freq mask | p=0.3 | Applied to mel spectrogram |
| Random gain | p=0.5 | ±6 dB |

## Data Filtering

- **iNat clips**: no quality filter — iNaturalist clips have no ratings
- **XC clips**: prefer rating ≥ 3.5
- **Minimum 10 samples per class**: duplicate rare clips before fold splitting to ensure all folds have representation
- **Cap external XC data at 500 records/class** to avoid class imbalance

## Class Labels

- **Insect sonotypes** (e.g., `47158son16`) are valid class labels — treat as unique species, not as aggregated groups
- Labels come from `data/raw/train.csv` (`primary_label`, `secondary_labels`)
- Expert-labeled soundscape labels: `data/raw/train_soundscapes_labels.csv` (primary_label is semicolon-separated)
