"""Stage 1 supervised training for BirdCLEF+ 2026.

Usage:
    python -u src/train.py --fold 0 [--backbone tf_efficientnet_b0.ns_jft_in1k]
                                    [--epochs 15] [--seed 42]

GPU augmentation pipeline (applied before mel computation):
  1. torch_audiomentations PitchShift  (p=0.3, ±2 semitones)
  2. torch_audiomentations Shift       (p=0.5, ±25% time shift)
  3. torchaudio MelSpectrogram + AmplitudeToDB
  4. torchaudio FrequencyMasking + TimeMasking (SpecAugment)
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
from torch.utils.data import DataLoader

# Add src/ to path so imports work when run from project root
sys.path.insert(0, str(Path(__file__).parent))

# Suppress benign hardware/version mismatch warning on GB10 (sm_121 > PyTorch max sm_120)
# (warning filters are set globally in config.py)

import config
from config import (
    RAW, PROC, MODELS,
    SAMPLE_RATE, CHUNK_SAMPLES, N_MELS, N_FFT, HOP_LENGTH, F_MIN, F_MAX, TOP_DB,
    BATCH_SIZE, LR, LR_MIN, WEIGHT_DECAY, T_0,
    PITCH_SHIFT_PROB, TIME_SHIFT_PROB, SPEC_TIME_MASK_PROB, SPEC_FREQ_MASK_PROB,
    get_species_index,
)
from dataset import BirdTrainDataset
from model import BirdSEDModel
from utils import load_audio, pad_or_crop, waveform_to_mel

# ── Optional GPU augmentations via torch_audiomentations ──────────────────────
try:
    from torch_audiomentations import PitchShift, Shift as TimeShift
    _HAS_AUDIOMENTATIONS = True
except Exception:
    _HAS_AUDIOMENTATIONS = False
    print("[warn] torch_audiomentations unavailable — PitchShift/TimeShift skipped")


# ── Validation data ────────────────────────────────────────────────────────────

def build_soundscape_val(sp2idx: dict) -> tuple:
    """Precompute validation mel tensors and labels from train_soundscapes_labels.csv.

    Returns:
        val_mels  : list of (3, N_MELS, T) float32 tensors
        val_labels: (N_segments, N_CLASSES) float32 numpy array
    """
    df = pd.read_csv(RAW / "train_soundscapes_labels.csv")
    n_classes   = config.N_CLASSES
    soundsc_dir = RAW / "train_soundscapes"

    def _parse_time(s: str) -> int:
        h, m, sec = str(s).split(":")
        return int(h) * 3600 + int(m) * 60 + int(sec)

    val_mels   = []
    val_labels = np.zeros((len(df), n_classes), dtype=np.float32)

    for i, row in df.iterrows():
        t_start = _parse_time(row["start"])
        t_end   = _parse_time(row["end"])

        path = soundsc_dir / str(row["filename"])
        try:
            wav     = load_audio(path)
            s, e    = int(t_start * SAMPLE_RATE), int(t_end * SAMPLE_RATE)
            segment = wav[s:e] if e <= len(wav) else wav[s:]
            segment = pad_or_crop(segment, CHUNK_SAMPLES, random_crop=False)
            mel     = waveform_to_mel(segment)   # (3, N_MELS, T) CPU float32
        except Exception as ex:
            print(f"  [warn] skipping {row['filename']} @ {t_start}s: {ex}")
            mel = torch.zeros(3, N_MELS, 512)

        val_mels.append(mel)

        for sp in str(row["primary_label"]).split(";"):
            sp = sp.strip()
            if sp in sp2idx:
                val_labels[i, sp2idx[sp]] = 1.0

    return val_mels, val_labels


@torch.no_grad()
def validate(
    model: nn.Module,
    val_mels: list,
    val_labels: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> float:
    """Compute macro-averaged ROC-AUC on validation soundscape segments."""
    model.eval()
    all_probs = []

    for i in range(0, len(val_mels), batch_size):
        batch = torch.stack(val_mels[i: i + batch_size]).to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(batch)
        probs = torch.sigmoid(out["clip_logits"]).float().cpu().numpy()
        all_probs.append(probs)

    probs   = np.concatenate(all_probs, axis=0)   # (N, N_CLASSES)
    present = val_labels.sum(axis=0) > 0
    if present.sum() == 0:
        return 0.0
    try:
        return float(roc_auc_score(
            val_labels[:, present], probs[:, present], average="macro"
        ))
    except ValueError:
        return 0.0


# ── Training ──────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_fold(fold: int, backbone: str, epochs: int, seed: int, version: "int | None" = None) -> None:
    set_seed(seed)

    device = torch.device("cuda")

    # BF16 / SDPA setup (Blackwell GB10)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cudnn.benchmark = True   # fixed input shapes

    # ── Data ──────────────────────────────────────────────────────────────────
    df       = pd.read_csv(PROC / "train_folds.csv")
    train_df = df[df["fold"] != fold].reset_index(drop=True)

    train_ds = BirdTrainDataset(train_df, augment=True)
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        # 'spawn' avoids fork-after-threads hang (torchaudio OpenMP threads
        # leave locks in inherited state; spawn starts workers clean)
        multiprocessing_context="spawn",
    )
    print(f"Fold {fold}: {len(train_ds)} training clips, {len(train_dl)} batches/epoch")

    sp2idx = get_species_index()
    print("Building soundscape validation set …")
    val_mels, val_labels = build_soundscape_val(sp2idx)
    n_present = int((val_labels.sum(axis=0) > 0).sum())
    print(f"  {len(val_mels)} segments, {n_present} species present in validation")

    # ── Model ─────────────────────────────────────────────────────────────────
    # SpecAugment transforms (GPU)
    freq_mask = T.FrequencyMasking(freq_mask_param=27).to(device)
    time_mask = T.TimeMasking(time_mask_param=64).to(device)

    model   = BirdSEDModel(backbone_name=backbone).to(device)
    # NOTE: torch.compile is disabled — GB10 (sm_121) exceeds PyTorch's max
    # supported sm_120, causing silent hangs during compiled kernel execution.
    # BF16 + SDPA still provide significant speedups without compile.
    # model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, eta_min=LR_MIN
    )

    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    bce_fn       = nn.BCEWithLogitsLoss(reduction="none")

    MODELS.mkdir(parents=True, exist_ok=True)
    vtag = f"_v{version}" if version is not None else ""
    save_path = MODELS / f"sed_{backbone}_fold{fold}_seed{seed}{vtag}.pt"

    # ── Training loop ─────────────────────────────────────────────────────────
    best_auc = 0.0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        for mels, labels, sec_mask in train_dl:
            mels     = mels.to(device, non_blocking=True)
            labels   = labels.to(device, non_blocking=True)
            sec_mask = sec_mask.to(device, non_blocking=True)

            # SpecAugment on GPU (mel already computed by workers)
            if random.random() < SPEC_FREQ_MASK_PROB:
                mels = freq_mask(mels)
            if random.random() < SPEC_TIME_MASK_PROB:
                mels = time_mask(mels)

            optimizer.zero_grad()

            with autocast_ctx:
                out      = model(mels)
                loss_per = bce_fn(out["clip_logits"], labels)    # (B, N_CLASSES)
                loss     = (loss_per * sec_mask).mean()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        avg_loss = running_loss / len(train_dl)
        val_auc  = validate(model, val_mels, val_labels, device)

        elapsed  = int(time.time() - epoch_start)
        mins, s  = divmod(elapsed, 60)

        best_marker = ""
        if val_auc > best_auc:
            best_auc = val_auc
            # Save state dict for ONNX compatibility
            torch.save(model.state_dict(), save_path)
            best_marker = " ★ BEST"

        print("=" * 40)
        print(
            f"Epoch {epoch:2d}/{epochs}: "
            f"train_loss={avg_loss:.4f}  "
            f"val_roc_auc={val_auc:.4f}  "
            f"time={mins}m {s:02d}s  "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
            f"{best_marker}"
        )
        print("=" * 40)

    print(f"\nFold {fold} complete. Best val ROC-AUC: {best_auc:.4f}")
    print(f"Model saved → {save_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="BirdCLEF+ 2026 Stage 1 training")
    parser.add_argument("--fold",     type=int, default=0,              help="Validation fold (0–4)")
    parser.add_argument("--backbone", type=str, default=config.BACKBONE, help="timm backbone name")
    parser.add_argument("--epochs",   type=int, default=config.EPOCHS,  help="Number of epochs")
    parser.add_argument("--seed",     type=int, default=config.SEED,    help="Random seed")
    parser.add_argument("--version",  type=int, default=None,
                        help="Version tag for checkpoint filename (e.g. 13 → _v13)")
    args = parser.parse_args()

    train_one_fold(
        fold=args.fold,
        backbone=args.backbone,
        epochs=args.epochs,
        seed=args.seed,
        version=args.version,
    )


if __name__ == "__main__":
    main()
