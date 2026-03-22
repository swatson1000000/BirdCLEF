"""Central configuration for BirdCLEF+ 2026."""

import warnings
from functools import lru_cache
from pathlib import Path

# Suppress known-benign warnings that clutter logs on this hardware.
# GB10 is CUDA 12.1; PyTorch max is 12.0 — still runs fine.
warnings.filterwarnings(
    "ignore",
    message=r"Found GPU.*cuda capability.*Minimum and Maximum cuda capability",
    category=UserWarning,
)
# timm emits this when head keys are absent in a features_only backbone.
warnings.filterwarnings(
    "ignore",
    message=r"Unexpected keys.*found while loading pretrained weights",
    category=UserWarning,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT   = Path("/home/swatson/work/MachineLearning/kaggle/BirdCLEF")
RAW    = ROOT / "data" / "raw"
PROC   = ROOT / "data" / "processed"
MODELS = ROOT / "models"
LOG    = ROOT / "log"

# ── Audio pipeline ─────────────────────────────────────────────────────────────
SAMPLE_RATE   = 32_000
DURATION      = 20                        # seconds per training chunk
CHUNK_SAMPLES = SAMPLE_RATE * DURATION    # 640 000

N_MELS     = 224
N_FFT      = 4096
HOP_LENGTH = 1252                         # → 512 time frames for 20 s at 32 kHz
F_MIN      = 0
F_MAX      = 16_000
TOP_DB     = 80

# ── Model ──────────────────────────────────────────────────────────────────────
# Backbone priority: efficientnet_b0 → efficientvit_b0 → regnety_008 → b3 → b4 → eca_nfnet_l0
BACKBONE  = "tf_efficientnet_b0.ns_jft_in1k"
N_CLASSES = 234    # all taxonomy species, including insect sonotypes

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE   = 64
NUM_WORKERS  = 8
EPOCHS       = 15
LR           = 5e-4
LR_MIN       = 1e-6
WEIGHT_DECAY = 1e-4
T_0          = 5        # CosineAnnealingWarmRestarts period
N_FOLDS      = 5
SEED         = 42

# ── Augmentation probabilities ─────────────────────────────────────────────────
GAIN_PROB           = 0.5
GAIN_DB_RANGE       = 6.0      # ±6 dB
BG_NOISE_PROB       = 0.3
PITCH_SHIFT_PROB    = 0.3
TIME_SHIFT_PROB     = 0.5
SPEC_TIME_MASK_PROB = 0.3
SPEC_FREQ_MASK_PROB = 0.3


@lru_cache(maxsize=1)
def get_species_list() -> list:
    """Sorted list of all 234 taxonomy species labels (stable ordering)."""
    import pandas as pd
    taxonomy = pd.read_csv(RAW / "taxonomy.csv")
    return sorted(taxonomy["primary_label"].astype(str).tolist())


@lru_cache(maxsize=1)
def get_species_index() -> dict:
    """Map species label string → integer index."""
    return {sp: i for i, sp in enumerate(get_species_list())}
