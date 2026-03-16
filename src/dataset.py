"""Training dataset for BirdCLEF+ 2026.

Each sample returns:
  mel           : (3, N_MELS, T) float32 tensor  — mel spectrogram computed in worker
  labels        : (N_CLASSES,) float32 tensor
  secondary_mask: (N_CLASSES,) float32 tensor     — 0 at secondary label positions
"""

import ast
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config
from config import (
    RAW, SAMPLE_RATE, CHUNK_SAMPLES,
    GAIN_PROB, GAIN_DB_RANGE, BG_NOISE_PROB, TIME_SHIFT_PROB,
    get_species_index,
)
from utils import load_audio, pad_or_crop, waveform_to_mel


def _parse_secondary(value) -> list:
    """Parse the secondary_labels column (stored as a stringified Python list)."""
    if pd.isna(value) or str(value).strip() in ("", "[]"):
        return []
    try:
        return [str(x) for x in ast.literal_eval(str(value))]
    except Exception:
        return [x.strip().strip("'\"")
                for x in str(value).strip("[]").split(",") if x.strip()]


class BirdTrainDataset(Dataset):
    """BirdCLEF+ 2026 training dataset.

    Mel spectrogram computation runs inside DataLoader workers so it overlaps
    with GPU forward/backward, hiding its latency.

    CPU augmentations applied: random gain, time shift (roll), background
    noise injection, and fixed-0.5-weight MixUp with element-wise-max labels.
    SpecAugment (FreqMask / TimeMask) is applied in the training loop on GPU.

    Args:
        df                    : DataFrame from train_folds.csv (pre-filtered to train folds)
        augment               : whether to apply augmentations
        bg_noise_dir          : directory of background noise files (.ogg/.wav)
        min_samples_per_class : rare-species floor; under-represented species are
                                oversampled by row duplication before training
    """

    def __init__(
        self,
        df: pd.DataFrame,
        augment: bool = True,
        bg_noise_dir: "Path | None" = None,
        min_samples_per_class: int = 10,
    ):
        self.augment   = augment
        self.sp2idx    = get_species_index()
        self.n_classes = config.N_CLASSES

        # Duplicate rare-species rows to reach the minimum floor
        counts = df["primary_label"].astype(str).value_counts()
        rare   = set(counts[counts < min_samples_per_class].index)
        extras = []
        for sp in rare:
            sp_df  = df[df["primary_label"].astype(str) == sp]
            needed = min_samples_per_class - len(sp_df)
            extras.append(sp_df.sample(needed, replace=True, random_state=42))
        if extras:
            df = pd.concat([df] + extras, ignore_index=True)

        self.df = df.reset_index(drop=True)

        # Background noise file list (graceful if dir absent)
        self.bg_files: list = []
        if bg_noise_dir is not None:
            bg_dir = Path(bg_noise_dir)
            if bg_dir.exists():
                self.bg_files = (
                    list(bg_dir.rglob("*.ogg")) + list(bg_dir.rglob("*.wav"))
                )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _build_targets(self, row) -> tuple:
        """Return (labels, secondary_mask) arrays of shape (N_CLASSES,).

        secondary_mask is 1 for ALL positions EXCEPT secondary labels, where
        it is 0.  This causes secondary-label positions to be ignored in the
        BCE loss while still keeping their target values at 1 in labels.
        """
        labels = np.zeros(self.n_classes, dtype=np.float32)
        mask   = np.ones(self.n_classes,  dtype=np.float32)   # default: include

        primary = str(row["primary_label"])
        if primary in self.sp2idx:
            labels[self.sp2idx[primary]] = 1.0
            # mask already 1

        for sp in _parse_secondary(row.get("secondary_labels", "[]")):
            sp = str(sp)
            if sp in self.sp2idx:
                labels[self.sp2idx[sp]] = 1.0
                mask[self.sp2idx[sp]]   = 0.0   # mask out secondary

        return labels, mask

    def _load_waveform(self, row) -> np.ndarray:
        path = RAW / "train_audio" / str(row["filename"])
        wav  = load_audio(path)
        return pad_or_crop(wav, CHUNK_SAMPLES, random_crop=self.augment)

    def _apply_time_shift(self, wav: np.ndarray) -> np.ndarray:
        """Random circular time shift by up to ±25% of clip length."""
        if random.random() < TIME_SHIFT_PROB:
            max_shift = int(0.25 * len(wav))
            shift = random.randint(-max_shift, max_shift)
            wav = np.roll(wav, shift)
        return wav

    def _apply_gain(self, wav: np.ndarray) -> np.ndarray:
        if random.random() < GAIN_PROB:
            db  = random.uniform(-GAIN_DB_RANGE, GAIN_DB_RANGE)
            wav = wav * (10.0 ** (db / 20.0))
        return wav

    def _add_bg_noise(self, wav: np.ndarray) -> np.ndarray:
        if not self.bg_files or random.random() >= BG_NOISE_PROB:
            return wav
        bg_path = random.choice(self.bg_files)
        try:
            bg   = load_audio(bg_path)
            bg   = pad_or_crop(bg, CHUNK_SAMPLES, random_crop=True)
            gain = random.uniform(0.05, 0.15)
            wav  = wav + gain * bg
        except Exception:
            pass
        return wav

    def _mixup(
        self,
        wav1: np.ndarray,
        labels1: np.ndarray,
        mask1: np.ndarray,
        idx2: int,
    ) -> tuple:
        """Fixed 0.5/0.5 waveform MixUp.

        Both waveforms are normalised by their absolute maximum before mixing
        to prevent clipping.  Labels are merged with element-wise max.  The
        secondary_mask union keeps 1 wherever either sample's primary label is.
        """
        row2   = self.df.iloc[idx2]
        wav2   = self._load_waveform(row2)
        labels2, mask2 = self._build_targets(row2)

        w1 = wav1 / (np.abs(wav1).max() + 1e-8)
        w2 = wav2 / (np.abs(wav2).max() + 1e-8)

        mixed  = 0.5 * w1 + 0.5 * w2
        labels = np.maximum(labels1, labels2)
        mask   = np.maximum(mask1,   mask2)    # union of primary positions
        return mixed, labels, mask

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row    = self.df.iloc[idx]
        wav    = self._load_waveform(row)
        labels, mask = self._build_targets(row)

        if self.augment:
            wav = self._apply_gain(wav)
            wav = self._apply_time_shift(wav)
            wav = self._add_bg_noise(wav)
            wav, labels, mask = self._mixup(
                wav, labels, mask, random.randint(0, len(self) - 1)
            )

        mel = waveform_to_mel(wav)
        return (
            mel,
            torch.from_numpy(labels),
            torch.from_numpy(mask),
        )
