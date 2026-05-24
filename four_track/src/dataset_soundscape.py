"""Soundscape training dataset for Track A2.

Loads segments from train_soundscapes/*.ogg with multi-hot labels from
train_soundscapes_labels.csv. Each sample returns the same (mel, labels,
secondary_mask) triple as BirdTrainDataset for seamless ConcatDataset use.

secondary_mask is all-ones (no masking) since soundscape annotations are
treated as multi-label ground truth — we don't have secondary-label
distinction here.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config
from config import (
    RAW, SAMPLE_RATE, CHUNK_SAMPLES,
    GAIN_PROB, GAIN_DB_RANGE, TIME_SHIFT_PROB,
    get_species_index,
)
from utils import load_audio, pad_or_crop, waveform_to_mel


def _parse_time(s: str) -> int:
    """Convert HH:MM:SS to integer seconds."""
    h, m, sec = str(s).split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec)


class SoundscapeTrainDataset(Dataset):
    """Train dataset over labeled train_soundscapes segments.

    Each segment is a [start, end] slice of a 60s soundscape file, padded or
    random-cropped to CHUNK_SAMPLES (20s). Labels are multi-hot from the
    semi-colon-separated primary_label column in train_soundscapes_labels.csv.

    Args:
        labels_csv : path to train_soundscapes_labels.csv
        soundscape_dir : directory containing the .ogg files
        augment : whether to apply gain + time-shift augmentations
        oversample : repeat the dataset this many times (to balance with
                     the much larger focal-clip dataset in ConcatDataset)
    """

    def __init__(
        self,
        labels_csv: Path | str,
        soundscape_dir: Path | str,
        augment: bool = True,
        oversample: int = 1,
    ):
        self.soundscape_dir = Path(soundscape_dir)
        self.augment = augment
        self.sp2idx = get_species_index()
        self.n_classes = config.N_CLASSES

        df = pd.read_csv(labels_csv)
        # Parse labels eagerly so __getitem__ is fast
        records = []
        for _, row in df.iterrows():
            label_vec = np.zeros(self.n_classes, dtype=np.float32)
            for sp in str(row["primary_label"]).split(";"):
                sp = sp.strip()
                if sp in self.sp2idx:
                    label_vec[self.sp2idx[sp]] = 1.0
            records.append({
                "filename": str(row["filename"]),
                "start_sec": _parse_time(row["start"]),
                "end_sec": _parse_time(row["end"]),
                "labels": label_vec,
            })
        self.records = records
        self.oversample = max(1, oversample)
        self._base_len = len(records)

    def __len__(self) -> int:
        return self._base_len * self.oversample

    def __getitem__(self, idx: int):
        rec = self.records[idx % self._base_len]
        path = self.soundscape_dir / rec["filename"]

        try:
            wav = load_audio(path)
            s = int(rec["start_sec"] * SAMPLE_RATE)
            e = int(rec["end_sec"] * SAMPLE_RATE)
            segment = wav[s:e] if e <= len(wav) else wav[s:]
            segment = pad_or_crop(segment, CHUNK_SAMPLES,
                                  random_crop=self.augment)
        except Exception as ex:
            # Fallback: silence
            segment = np.zeros(CHUNK_SAMPLES, dtype=np.float32)

        if self.augment:
            # Random gain
            if random.random() < GAIN_PROB:
                db = random.uniform(-GAIN_DB_RANGE, GAIN_DB_RANGE)
                segment = segment * (10.0 ** (db / 20.0))
            # Random time shift
            if random.random() < TIME_SHIFT_PROB:
                max_shift = int(0.25 * len(segment))
                shift = random.randint(-max_shift, max_shift)
                segment = np.roll(segment, shift)

        mel = waveform_to_mel(segment)
        labels = torch.from_numpy(rec["labels"])
        mask = torch.ones(self.n_classes, dtype=torch.float32)  # no masking

        return mel, labels, mask
