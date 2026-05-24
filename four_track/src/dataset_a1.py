"""Path-aware extension of BirdTrainDataset for four_track A1.

Routes file lookups by the `collection` column:
  - default      → data/raw/train_audio/<filename>            (legacy behavior)
  - "AnuraSet"   → four_track/data/external/anuraset_focal/<filename>

`filename` in train_folds.csv is always a relative path (e.g.
"555146/AS_INCT17_20191205_191500_0000000.ogg" for AnuraSet rows, or
"22961/iNat1216197.ogg" for legacy iNat rows).

Optional AnuraSet-specific background mixup (§14.10.16): when
`mixin_df` is supplied, each AnuraSet row is force-mixed with
probability `mixin_p` against a random row sampled from `mixin_df`
(RMS-matched additive, α ~ U(*mixin_alpha_range*), peak-limited).
The mixin's own labels are discarded. Non-AnuraSet rows are
unaffected.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE       = Path(__file__).resolve().parent
FT_ROOT    = HERE.parent
ROOT       = FT_ROOT.parent
PARENT_SRC = ROOT / "src"

if str(PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(PARENT_SRC))

from config import RAW, CHUNK_SAMPLES, SAMPLE_RATE            # noqa: E402
from dataset import BirdTrainDataset                          # noqa: E402
from utils import load_audio, pad_or_crop                     # noqa: E402

ANURA_FOCAL  = FT_ROOT / "data" / "external" / "anuraset_focal"
BC25_SS_DIR  = RAW / "birdclef_2025" / "train_soundscapes"
BC26_SS_DIR  = RAW / "train_soundscapes"
PSEUDO_WIN_S = 5   # pseudo windows are 5 s slices of the source soundscape


def _pick_rms_window(
    wav: np.ndarray,
    target_len: int,
    top_k: int = 3,
    stride_s: float = 5.0,
) -> int:
    """Return an offset into `wav` biased toward high-RMS windows.

    Samples uniformly from the top-`top_k` candidate starts (spaced
    `stride_s` seconds apart) ranked by window RMS. Keeps some
    stochasticity vs. a purely deterministic argmax, so the model
    still sees diverse crops per focal file across epochs.
    """
    n = len(wav)
    if n <= target_len:
        return 0
    stride = max(1, int(stride_s * SAMPLE_RATE))
    starts = list(range(0, n - target_len + 1, stride))
    if not starts:
        return 0
    sq = wav.astype(np.float64) ** 2
    csum = np.concatenate(([0.0], np.cumsum(sq)))
    rms = [float(csum[s + target_len] - csum[s]) for s in starts]
    k = min(top_k, len(starts))
    top_idx = sorted(range(len(rms)), key=lambda i: -rms[i])[:k]
    return starts[random.choice(top_idx)]


class BirdTrainDatasetA1(BirdTrainDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        augment: bool = True,
        bg_noise_dir=None,
        min_samples_per_class: int = 10,
        mixin_df: "pd.DataFrame | None" = None,
        mixin_p: float = 0.7,
        mixin_alpha_range: tuple = (0.2, 0.5),
        rms_select: bool = False,
    ):
        super().__init__(
            df, augment=augment,
            bg_noise_dir=bg_noise_dir,
            min_samples_per_class=min_samples_per_class,
        )
        self.mixin_df = (
            mixin_df.reset_index(drop=True) if mixin_df is not None else None
        )
        self.mixin_p = float(mixin_p)
        self.mixin_alpha_range = tuple(mixin_alpha_range)
        self.rms_select = bool(rms_select)

    def _build_targets(self, row) -> tuple:
        # A2 pseudo-labels (BC2026_SS_PSEUDO) carry a `pseudo_positive_labels`
        # column with semicolon-separated species codes (multi-label). Apply
        # mask=1 across all classes so BCE applies (the parent's secondary-label
        # path masks them OUT, which is wrong for pseudo-labels we want to LEARN).
        coll = str(row.get("collection", "") or "")
        if coll == "BC2026_SS_PSEUDO":
            labels = np.zeros(self.n_classes, dtype=np.float32)
            mask = np.ones(self.n_classes, dtype=np.float32)
            pos = str(row.get("pseudo_positive_labels", "") or "")
            for sp in pos.split(";"):
                sp = sp.strip()
                if sp and sp in self.sp2idx:
                    labels[self.sp2idx[sp]] = 1.0
            return labels, mask
        return super()._build_targets(row)

    def _load_waveform(self, row) -> np.ndarray:
        coll = str(row.get("collection", "") or "")
        if coll == "AnuraSet":
            path = ANURA_FOCAL / str(row["filename"])
            wav = pad_or_crop(load_audio(path), CHUNK_SAMPLES,
                              random_crop=self.augment)
        elif coll == "BC2025":
            path = RAW / "birdclef_2025" / "train_audio" / str(row["filename"])
            wav = pad_or_crop(load_audio(path), CHUNK_SAMPLES,
                              random_crop=self.augment)
        elif coll == "BC2025_SS_PSEUDO":
            # Pseudo-labelled 5 s slice of a BC2025 soundscape; keep the
            # window fixed (label is for this specific 5 s), then tile to
            # CHUNK_SAMPLES via pad_or_crop (which loops short clips).
            path = BC25_SS_DIR / str(row["filename"])
            full = load_audio(path)
            ws   = int(row["pseudo_window_start"])
            s    = ws * SAMPLE_RATE
            seg  = full[s: s + PSEUDO_WIN_S * SAMPLE_RATE]
            if len(seg) == 0:                                 # file short/corrupt
                seg = full
            wav = pad_or_crop(seg, CHUNK_SAMPLES, random_crop=False)
        elif coll == "BC2026_SS_PSEUDO":
            # A2 pseudo-labelled 5 s slice of a BC2026 train_soundscape;
            # same handling as BC2025_SS_PSEUDO but for the current comp.
            path = BC26_SS_DIR / str(row["filename"])
            full = load_audio(path)
            ws   = int(row["pseudo_window_start"])
            s    = ws * SAMPLE_RATE
            seg  = full[s: s + PSEUDO_WIN_S * SAMPLE_RATE]
            if len(seg) == 0:
                seg = full
            wav = pad_or_crop(seg, CHUNK_SAMPLES, random_crop=False)
        else:
            path = RAW / "train_audio" / str(row["filename"])
            full = load_audio(path)
            if self.rms_select and self.augment:
                start = _pick_rms_window(full, CHUNK_SAMPLES)
                seg = full[start: start + CHUNK_SAMPLES]
                wav = pad_or_crop(seg, CHUNK_SAMPLES, random_crop=False)
            else:
                wav = pad_or_crop(full, CHUNK_SAMPLES,
                                  random_crop=self.augment)

        if (coll == "AnuraSet" and self.augment
                and self.mixin_df is not None
                and random.random() < self.mixin_p):
            wav = self._apply_background_mixup(wav)
        return wav

    def _apply_background_mixup(self, wav: np.ndarray) -> np.ndarray:
        """Additive RMS-matched mixup against a random mixin-pool clip."""
        mx_row = self.mixin_df.iloc[random.randrange(len(self.mixin_df))]
        mx_path = RAW / "train_audio" / str(mx_row["filename"])
        try:
            mx_wav = pad_or_crop(load_audio(mx_path), CHUNK_SAMPLES,
                                 random_crop=True)
        except Exception:
            # Missing or unreadable mixin file — skip mixup gracefully
            return wav

        r_wav = float(np.sqrt(np.mean(wav ** 2)))
        r_mx  = float(np.sqrt(np.mean(mx_wav ** 2)))
        if r_wav < 1e-6 or r_mx < 1e-6:
            return wav
        mx_wav = mx_wav * (r_wav / r_mx)

        alpha = random.uniform(*self.mixin_alpha_range)
        out = wav + alpha * mx_wav
        peak = float(np.abs(out).max())
        if peak > 0.99:
            out = out * (0.99 / peak)
        return out.astype(np.float32, copy=False)
