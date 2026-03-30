"""Noisy Student self-training for BirdCLEF+ 2026.

Mixes focal training clips with pseudo-labelled soundscape chunks using
fixed-0.5-weight MixUp.  The pseudo-labelled dataset is built from
data/processed/pseudo_labels_v1.csv (or a later version with power transform).

Usage:
    python -u src/self_train.py --fold 0 \
        --pseudo-csv data/processed/pseudo_labels_v1.csv \
        [--backbone tf_efficientnet_b0.ns_jft_in1k] \
        [--init-ckpt models/sed_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42.pt] \
        [--epochs 30] [--seed 42] [--pseudo-power 1.0] [--version 1]

The --init-ckpt flag warm-starts from Stage 1 weights (recommended).
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio.transforms as T
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent))

# (warning filters are set globally in config.py)

import config
from config import (
    RAW, PROC, MODELS,
    SAMPLE_RATE, CHUNK_SAMPLES, N_MELS,
    BATCH_SIZE, LR, LR_MIN, WEIGHT_DECAY, T_0,
    SPEC_TIME_MASK_PROB, SPEC_FREQ_MASK_PROB,
    get_species_list, get_species_index,
)
from dataset import BirdTrainDataset
from model import BirdSEDModel
from utils import load_audio, pad_or_crop, waveform_to_mel

# ── Optional GPU augmentations ─────────────────────────────────────────────────
try:
    from torch_audiomentations import PitchShift, Shift as TimeShift
    _HAS_AUDIOMENTATIONS = True
except Exception:
    _HAS_AUDIOMENTATIONS = False

SEGMENT_SEC = 5   # pseudo-label window size matches competition metric window

# Freq-MixStyle parameters
FREQ_MIXSTYLE_PROB  = 0.5
FREQ_MIXSTYLE_ALPHA = 0.1


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification.

    Downweights easy negatives (γ−=4) while leaving positives unmodified
    (γ+=0). The clip parameter shifts the hard-mining threshold by 0.05.
    Expected gain over BCE: +0.02–0.03 soundscape val AUC.

    Reference: "Asymmetric Loss For Multi-Label Classification"
               (Ben-Baruch et al., 2021)
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip      = clip
        self.eps       = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Per-element ASL. x: logits (..., C), y: targets (..., C) ∈ [0, 1]."""
        # Cast to float32 for numerically stable loss — BF16 autocast can produce
        # 0^0 = NaN via tensor.pow(tensor) path (exp(0*log(0)=nan)) when a confident
        # positive prediction rounds xs_pos to exactly 1.0 and y=1 → gamma=0.
        # Loss computation is cheap; backbone forward pass stays in BF16.
        x = x.float()
        y = y.float()

        xs_pos = torch.sigmoid(x)
        xs_neg = 1.0 - xs_pos

        # Asymmetric probability clipping (shift hard-mining boundary)
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        loss = (
            y       * torch.log(xs_pos.clamp(min=self.eps)) +
            (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        )

        # Asymmetric focusing.
        # The focusing weight (1-pt)^gamma is treated as a CONSTANT multiplier
        # (no gradient flows through it). This follows the official ASL paper
        # implementation and avoids NaN gradients from 0^0 / 0^(fractional):
        #   - torch.where backward leaks NaN gradient from unselected branch
        #   - tensor.pow(tensor) at 0^ε computes exp(ε*log(0))=exp(-inf)=0 forward
        #     but gradient is exp(-inf)*log(0)=0*(-inf)=nan backward
        # Disabling grad here means only the log-probability terms are differentiated.
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            with torch.no_grad():
                pt    = xs_pos * y + xs_neg * (1 - y)
                gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
                focusing = torch.pow((1.0 - pt).clamp(min=1e-8), gamma)
            loss *= focusing

        return -loss


def freq_mixstyle(
    mels: torch.Tensor,
    alpha: float = FREQ_MIXSTYLE_ALPHA,
) -> torch.Tensor:
    """Freq-MixStyle: mix per-frequency-bin mean/std across randomly paired samples.

    mels: (B, 3, F, T) on GPU.
    Simulates different ARU recording-device frequency responses.
    Expected gain: +0.01–0.03 soundscape val AUC.
    """
    B = mels.size(0)
    if B < 2:
        return mels

    # Per-frequency-bin statistics over time axis → (B, 3, F, 1)
    mu  = mels.mean(dim=3, keepdim=True)
    sig = mels.std(dim=3, keepdim=True).clamp(min=1e-7)

    # Normalise to zero mean, unit std
    x_norm = (mels - mu) / sig

    # Random convex combination of (mu, sig) from a shuffled partner
    lam  = torch.from_numpy(
        np.random.beta(alpha, alpha, size=(B, 1, 1, 1)).astype(np.float32)
    ).to(mels.device)
    perm      = torch.randperm(B, device=mels.device)
    mixed_mu  = lam * mu  + (1.0 - lam) * mu[perm]
    mixed_sig = lam * sig + (1.0 - lam) * sig[perm]

    return x_norm * mixed_sig + mixed_mu


# ── Pseudo-label dataset ───────────────────────────────────────────────────────

class PseudoLabelDataset(Dataset):
    """Each item is a 5-second soundscape segment with soft pseudo-labels.

    Returns (mel, labels) — no secondary_mask since these are soft labels.
    """

    def __init__(
        self,
        pseudo_csv: Path,
        soundscapes_dir: Path,
        species_list: list,
        power: float = 1.0,
        augment: bool = True,
    ):
        self.soundscapes_dir = soundscapes_dir
        self.species_list    = species_list
        self.n_classes       = len(species_list)
        self.power           = power
        self.augment         = augment

        df = pd.read_csv(pseudo_csv)
        # Keep only rows where the file actually exists
        exists_mask = df["filename"].apply(
            lambda fn: (soundscapes_dir / fn).exists()
        )
        df = df[exists_mask]

        # Exclude expert-labeled val files to prevent leakage
        val_labels_path = config.RAW / "train_soundscapes_labels.csv"
        if val_labels_path.exists():
            val_files = set(pd.read_csv(val_labels_path)["filename"].unique())
            before = len(df)
            df = df[~df["filename"].isin(val_files)]
            print(f"  Excluded {before - len(df)} rows from {len(val_files)} val files (leakage prevention)")

        self.df = df.reset_index(drop=True)
        print(f"  PseudoLabelDataset: {len(self.df)} segments from "
              f"{self.df['filename'].nunique()} files (power={power})")

        # Soft label matrix (apply power transform to sharpen/smooth)
        prob_cols       = [c for c in species_list if c in self.df.columns]
        missing         = set(species_list) - set(prob_cols)
        if missing:
            print(f"  [warn] {len(missing)} species missing in pseudo CSV — set to 0")
        self._probs = np.zeros((len(self.df), self.n_classes), dtype=np.float32)
        for i, sp in enumerate(species_list):
            if sp in self.df.columns:
                self._probs[:, i] = self.df[sp].values

        if power != 1.0:
            self._probs = np.power(self._probs, power)

        # Cache file waveforms per unique filename (loaded lazily)
        self._wav_cache: dict = {}

    def _get_waveform(self, filename: str) -> np.ndarray:
        if filename not in self._wav_cache:
            self._wav_cache[filename] = load_audio(
                self.soundscapes_dir / filename
            )
            # Evict cache if it grows too large (keep ≤ 200 files)
            if len(self._wav_cache) > 200:
                evict = next(iter(self._wav_cache))
                del self._wav_cache[evict]
        return self._wav_cache[filename]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row      = self.df.iloc[idx]
        waveform = self._get_waveform(row["filename"])

        start_samp = int(row["start_time"] * SAMPLE_RATE)
        end_samp   = int(row["end_time"]   * SAMPLE_RATE)
        seg        = waveform[start_samp:end_samp]
        seg        = pad_or_crop(seg, CHUNK_SAMPLES, random_crop=self.augment)

        if self.augment and random.random() < config.TIME_SHIFT_PROB:
            shift = random.randint(-int(0.25 * CHUNK_SAMPLES), int(0.25 * CHUNK_SAMPLES))
            seg   = np.roll(seg, shift)

        labels = self._probs[idx].copy()

        mel = waveform_to_mel(seg)
        return mel, torch.from_numpy(labels)


# ── Validation ─────────────────────────────────────────────────────────────────

def _parse_time(t: str) -> int:
    h, m, sec = str(t).split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec)


def build_soundscape_val(sp2idx: dict) -> tuple:
    df          = pd.read_csv(RAW / "train_soundscapes_labels.csv")
    n_classes   = config.N_CLASSES
    soundsc_dir = RAW / "train_soundscapes"

    val_mels   = []
    val_labels = np.zeros((len(df), n_classes), dtype=np.float32)

    for i, row in df.iterrows():
        t_start = _parse_time(row["start"])
        t_end   = _parse_time(row["end"])
        path    = soundsc_dir / str(row["filename"])
        try:
            wav     = load_audio(path)
            s, e    = int(t_start * SAMPLE_RATE), int(t_end * SAMPLE_RATE)
            segment = wav[s:e] if e <= len(wav) else wav[s:]
            segment = pad_or_crop(segment, CHUNK_SAMPLES, random_crop=False)
            mel     = waveform_to_mel(segment)
        except Exception as ex:
            print(f"  [warn] {row['filename']} @ {t_start}s: {ex}")
            mel = torch.zeros(3, N_MELS, 512)
        val_mels.append(mel)
        for sp in str(row["primary_label"]).split(";"):
            sp = sp.strip()
            if sp in sp2idx:
                val_labels[i, sp2idx[sp]] = 1.0

    return val_mels, val_labels


@torch.no_grad()
def validate(model: nn.Module, val_mels: list, val_labels: np.ndarray,
             device: torch.device, batch_size: int = 32) -> float:
    model.eval()
    all_probs = []
    for i in range(0, len(val_mels), batch_size):
        batch = torch.stack(val_mels[i: i + batch_size]).to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(batch)
        all_probs.append(torch.sigmoid(out["clip_logits"]).float().cpu().numpy())
    probs   = np.concatenate(all_probs, axis=0)
    present = val_labels.sum(axis=0) > 0
    if present.sum() == 0:
        return 0.0
    try:
        return float(roc_auc_score(
            val_labels[:, present], probs[:, present], average="macro"
        ))
    except ValueError:
        return 0.0


# ── Mixed-batch collation ──────────────────────────────────────────────────────

def noisy_student_mixup(
    focal_batch: tuple,
    pseudo_batch: tuple,
) -> tuple:
    """Combine a focal batch and a pseudo-label batch using fixed-0.5 MixUp.

    focal_batch  : (mels, labels, sec_mask) — from BirdTrainDataset
    pseudo_batch : (mels, labels)           — from PseudoLabelDataset

    Returns (mixed_mels, mixed_labels, mixed_mask) ready for loss.
    The secondary_mask is 1 (all included) for pseudo positions since the
    pseudo-labels are already confident soft targets.
    """
    f_mel, f_labels, f_mask = focal_batch
    p_mel, p_labels         = pseudo_batch

    # Normalise by absolute max per sample
    f_max = f_mel.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
    p_max = p_mel.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)

    mixed_mel    = 0.5 * (f_mel / f_max) + 0.5 * (p_mel / p_max)
    mixed_labels = torch.maximum(f_labels, p_labels)
    mixed_mask   = f_mask   # inherit focal mask (pseudo positions = all 1 already)

    return mixed_mel, mixed_labels, mixed_mask


# ── Training ───────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_fold(
    fold: int,
    backbone: str,
    epochs: int,
    seed: int,
    pseudo_csv: Path,
    pseudo_power: float,
    init_ckpt: "Path | None",
    version: int,
    soup_start_ep: int = 10,
    use_bce: bool = False,
    asl_gamma_neg: float = 4.0,
    use_freq_mixstyle: bool = True,
    use_dual_loss: bool = True,
    lr_schedule: str = "warm_restarts",
    asl_focal_only: bool = False,
) -> None:
    """Train one fold of the SED model.

    use_bce           : use BCEWithLogitsLoss instead of ASL (simpler, diagnostic)
    asl_gamma_neg     : gamma_neg for AsymmetricLoss (default 4.0; try 2.0 for softer suppression)
    use_freq_mixstyle : apply Freq-MixStyle batch augmentation
    use_dual_loss     : 50% clip-level + 50% frame-level loss; False = clip only
    lr_schedule       : 'warm_restarts' (CosineAnnealingWarmRestarts, default for
                        warm-start) or 'cosine' (CosineAnnealingLR single decay,
                        recommended for from-scratch runs with changed features)
    """
    set_seed(seed)

    device = torch.device("cuda")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cudnn.benchmark = True

    species_list = get_species_list()
    sp2idx       = get_species_index()

    # ── Focal dataset ──────────────────────────────────────────────────────────
    df       = pd.read_csv(PROC / "train_folds.csv")
    train_df = df[df["fold"] != fold].reset_index(drop=True)

    focal_ds = BirdTrainDataset(train_df, augment=True)
    focal_dl = DataLoader(
        focal_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        multiprocessing_context="spawn",
    )
    print(f"Fold {fold}: {len(focal_ds)} focal clips, {len(focal_dl)} batches/epoch")

    # ── Pseudo-label dataset ───────────────────────────────────────────────────
    soundscapes_dir = RAW / "train_soundscapes"
    pseudo_ds = PseudoLabelDataset(
        pseudo_csv=pseudo_csv,
        soundscapes_dir=soundscapes_dir,
        species_list=species_list,
        power=pseudo_power,
        augment=True,
    )

    # Weighted sampler: high-confidence soundscapes sampled more often
    weights_path = pseudo_csv.parent / pseudo_csv.name.replace(".csv", "_weights.csv")
    if weights_path.exists():
        wdf        = pd.read_csv(weights_path).set_index("filename")
        seg_weights = pseudo_ds.df["filename"].map(
            lambda fn: float(wdf.loc[fn, "sampler_weight"]) if fn in wdf.index else 1.0
        ).values.astype(np.float32)
        sampler = WeightedRandomSampler(
            weights=seg_weights, num_samples=len(pseudo_ds), replacement=True
        )
        shuffle_pseudo = False
    else:
        sampler        = None
        shuffle_pseudo = True

    pseudo_dl = DataLoader(
        pseudo_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=shuffle_pseudo if sampler is None else False,
        num_workers=max(2, config.NUM_WORKERS // 2),
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        multiprocessing_context="spawn",
    )
    print(f"  {len(pseudo_ds)} pseudo-label segments, {len(pseudo_dl)} batches/epoch")

    # ── Validation ─────────────────────────────────────────────────────────────
    print("Building soundscape validation set …")
    val_mels, val_labels = build_soundscape_val(sp2idx)
    n_present = int((val_labels.sum(axis=0) > 0).sum())
    print(f"  {len(val_mels)} segments, {n_present} species present")

    # ── Model ──────────────────────────────────────────────────────────────────
    freq_mask = T.FrequencyMasking(freq_mask_param=27).to(device)
    time_mask = T.TimeMasking(time_mask_param=64).to(device)

    model = BirdSEDModel(backbone_name=backbone).to(device)

    if init_ckpt is not None and init_ckpt.exists():
        state = torch.load(init_ckpt, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        print(f"  Warm-started from {init_ckpt}")
        # Fine-tuning from warm start: lower LR + single cosine decay (no restarts)
        # NOTE: only valid when init checkpoint used the *same* input pipeline
        # (e.g. log-mel → log-mel).  Mixing pipelines causes epoch-1 peak.
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR * 0.2, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=LR_MIN
        )
    else:
        print("  Training from scratch (no --init-ckpt provided)")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        if lr_schedule == "cosine":
            # Single cosine decay — recommended when feature pipeline changed
            # (e.g. log-mel → PCEN).  Avoids repeated LR spikes that prevent
            # convergence on an unfamiliar input distribution.
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=LR_MIN
            )
            print("  LR schedule: CosineAnnealingLR (single decay)")
        else:
            # Default: warm restarts (same as Stage 1 train.py)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=T_0, eta_min=LR_MIN
            )
            print("  LR schedule: CosineAnnealingWarmRestarts (warm restarts)")

    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    if use_bce:
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        print("  Loss: BCEWithLogitsLoss")
    else:
        criterion = AsymmetricLoss(gamma_neg=asl_gamma_neg, gamma_pos=0.0, clip=0.05)
        print(f"  Loss: AsymmetricLoss (gamma_neg={asl_gamma_neg}, gamma_pos=0, clip=0.05)")
    focal_asl_criterion = AsymmetricLoss(gamma_neg=asl_gamma_neg, gamma_pos=0.0, clip=0.05) if asl_focal_only else None
    if asl_focal_only:
        print(f"  ASL focal-only: ENABLED (gamma_neg={asl_gamma_neg}) — BCE for pseudo mixing only")
    if not use_freq_mixstyle:
        print("  Freq-MixStyle: DISABLED")
    if not use_dual_loss:
        print("  Dual loss: DISABLED (clip-only)")

    MODELS.mkdir(parents=True, exist_ok=True)
    save_path = MODELS / f"sed_{backbone}_fold{fold}_seed{seed}_v{version}.pt"

    # ── Training loop ──────────────────────────────────────────────────────────
    best_auc        = 0.0
    pseudo_iter     = iter(pseudo_dl)
    soup_ckpt_paths = []      # per-epoch checkpoints saved for model soup

    for epoch in range(1, epochs + 1):
        epoch_start  = time.time()
        model.train()
        running_loss = 0.0

        for focal_batch in focal_dl:
            # Fetch pseudo batch (cycle if exhausted)
            try:
                pseudo_batch = next(pseudo_iter)
            except StopIteration:
                pseudo_iter  = iter(pseudo_dl)
                pseudo_batch = next(pseudo_iter)

            if asl_focal_only:
                # Mix audio for Noisy Student regularization but compute ASL loss
                # against focal labels only (avoids calibration conflict with
                # soft BCE-generated pseudo-labels).
                f_mel, f_labels, f_sec_mask = focal_batch
                p_mel, _p_labels            = pseudo_batch
                f_max = f_mel.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
                p_max = p_mel.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
                mels     = (0.5 * (f_mel / f_max) + 0.5 * (p_mel / p_max))
                labels   = f_labels
                sec_mask = f_sec_mask
            else:
                mels, labels, sec_mask = noisy_student_mixup(focal_batch, pseudo_batch)

            mels     = mels.to(device, non_blocking=True)
            labels   = labels.to(device, non_blocking=True)
            sec_mask = sec_mask.to(device, non_blocking=True)

            # SpecAugment (GPU)
            if random.random() < SPEC_FREQ_MASK_PROB:
                mels = freq_mask(mels)
            if random.random() < SPEC_TIME_MASK_PROB:
                mels = time_mask(mels)

            # Freq-MixStyle (GPU) — mix per-freq-bin stats to simulate device diversity
            if use_freq_mixstyle and random.random() < FREQ_MIXSTYLE_PROB:
                mels = freq_mixstyle(mels)

            optimizer.zero_grad()

            # Select loss criterion: focal-only ASL overrides the shared criterion
            active_criterion = focal_asl_criterion if asl_focal_only else criterion

            with autocast_ctx:
                out = model(mels)

                # Clip-level loss
                clip_loss = (active_criterion(out["clip_logits"], labels) * sec_mask).mean()

                if use_dual_loss:
                    # Frame-level loss — broadcast clip labels to all time frames
                    frame_labels = labels.unsqueeze(1).expand_as(out["frame_logits"])
                    frame_mask   = sec_mask.unsqueeze(1).expand_as(out["frame_logits"])
                    frame_loss   = (active_criterion(out["frame_logits"], frame_labels) * frame_mask).mean()
                    loss = 0.5 * clip_loss + 0.5 * frame_loss
                else:
                    loss = clip_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        avg_loss = running_loss / len(focal_dl)
        val_auc  = validate(model, val_mels, val_labels, device)

        elapsed     = int(time.time() - epoch_start)
        mins, secs  = divmod(elapsed, 60)
        best_marker = ""
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)
            best_marker = " ★ BEST"

        # Save epoch checkpoint for model soup (post-warmup epochs only)
        if soup_start_ep > 0 and epoch >= soup_start_ep:
            soup_ep_path = MODELS / f"{save_path.stem}_ep{epoch}.pt"
            torch.save(model.state_dict(), soup_ep_path)
            soup_ckpt_paths.append(soup_ep_path)

        print("=" * 40)
        print(
            f"Epoch {epoch:2d}/{epochs}: "
            f"train_loss={avg_loss:.4f}  "
            f"val_roc_auc={val_auc:.4f}  "
            f"time={mins}m {secs:02d}s  "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
            f"{best_marker}"
        )
        print("=" * 40)

    # ── Model soup: average all post-warmup epoch checkpoints ─────────────────
    if soup_ckpt_paths:
        print(f"\nBuilding model soup from {len(soup_ckpt_paths)} epoch checkpoints …")
        soup_state = None
        for p in soup_ckpt_paths:
            state = torch.load(p, map_location="cpu", weights_only=True)
            if soup_state is None:
                soup_state = {k: v.float() for k, v in state.items()}
            else:
                for k in soup_state:
                    soup_state[k] += state[k].float()
        for k in soup_state:
            soup_state[k] /= len(soup_ckpt_paths)
        soup_save_path = MODELS / f"{save_path.stem}_soup.pt"
        torch.save(soup_state, soup_save_path)
        print(f"Model soup saved ({len(soup_ckpt_paths)} checkpoints) → {soup_save_path}")
        for p in soup_ckpt_paths:
            p.unlink(missing_ok=True)

    print(f"\nFold {fold} complete. Best val ROC-AUC: {best_auc:.4f}")
    print(f"Best-epoch model → {save_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="BirdCLEF+ 2026 Noisy Student self-training")
    parser.add_argument("--fold",         type=int,   default=0)
    parser.add_argument("--backbone",     type=str,   default=config.BACKBONE)
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--seed",         type=int,   default=config.SEED)
    parser.add_argument("--pseudo-csv",   type=Path,  default=PROC / "pseudo_labels_v1.csv")
    parser.add_argument("--pseudo-power", type=float, default=1.0,
                        help="Power transform exponent applied to pseudo-label probs (>1 sharpens)")
    parser.add_argument("--init-ckpt",    type=Path,  default=None,
                        help="Stage 1 checkpoint to warm-start from")
    parser.add_argument("--version",      type=int,   default=1,
                        help="Output filename version suffix (v1, v2 …)")
    parser.add_argument("--soup-start-ep", type=int, default=10,
                        help="Start saving epoch checkpoints for model soup after this epoch (0=disabled)")
    parser.add_argument("--use-bce", action="store_true", default=False,
                        help="Use BCEWithLogitsLoss instead of ASL (diagnostic/ablation)")
    parser.add_argument("--asl-gamma-neg", type=float, default=4.0,
                        help="gamma_neg for AsymmetricLoss (default 4.0; use 2.0 for softer suppression)")
    parser.add_argument("--no-freq-mixstyle", action="store_true", default=False,
                        help="Disable Freq-MixStyle augmentation (diagnostic/ablation)")
    parser.add_argument("--no-dual-loss", action="store_true", default=False,
                        help="Use clip-level loss only; disable frame-level dual loss")
    parser.add_argument("--asl-focal-only", action="store_true", default=False,
                        help="Apply ASL only to focal labels; pseudo used for audio mixing only "
                             "(avoids calibration conflict with BCE-generated pseudo-labels)")
    parser.add_argument("--lr-schedule", type=str, default="warm_restarts",
                        choices=["warm_restarts", "cosine"],
                        help="LR scheduler: warm_restarts (default) or cosine (single decay, "
                             "recommended for from-scratch on changed feature pipeline)")
    args = parser.parse_args()

    train_one_fold(
        fold=args.fold,
        backbone=args.backbone,
        epochs=args.epochs,
        seed=args.seed,
        pseudo_csv=args.pseudo_csv,
        pseudo_power=args.pseudo_power,
        init_ckpt=args.init_ckpt,
        version=args.version,
        soup_start_ep=args.soup_start_ep,
        use_bce=args.use_bce,
        asl_gamma_neg=args.asl_gamma_neg,
        use_freq_mixstyle=not args.no_freq_mixstyle,
        use_dual_loss=not args.no_dual_loss,
        lr_schedule=args.lr_schedule,
        asl_focal_only=args.asl_focal_only,
    )


if __name__ == "__main__":
    main()
