"""Track A1 SED training — PCEN + ASL + Frequency-MixStyle EffNet-B0.

Trains the four_track A1 branch on `train_audio` focal clips. This script
follows the same data / mel pipeline as the parent `BirdCLEF/src/train.py`
(loaded by import) and adds:

  - Asymmetric Loss (ASL) instead of vanilla BCE — see losses.py
  - Frequency MixStyle hook on the EfficientNet backbone — see model_a1.py
  - 5-fold capable wrapper (drive multiple folds in one nohup run)
  - 25 default epochs (vs 15 in legacy)
  - Output checkpoints under four_track/models/a1/
  - Logs follow the four_track CLAUDE.md per-epoch convention

Usage:
    # Single fold
    python -u src/train_a1.py --fold 0

    # All five folds sequentially
    python -u src/train_a1.py --folds 0,1,2,3,4

    # Smoke test (1 fold, 1 epoch, 1 train batch + 1 val batch)
    python -u src/train_a1.py --fold 0 --smoke-test
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

# ── Path wiring ───────────────────────────────────────────────────────────────
HERE       = Path(__file__).resolve().parent          # four_track/src/
FT_ROOT    = HERE.parent                               # four_track/
ROOT       = FT_ROOT.parent                            # BirdCLEF/
PARENT_SRC = ROOT / "src"                              # BirdCLEF/src/

# Make parent BirdCLEF/src/ importable for the legacy data / mel utilities.
if str(PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(PARENT_SRC))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config  # parent BirdCLEF/src/config.py — paths, mel hyperparameters
from config import (
    RAW, PROC,
    SAMPLE_RATE, CHUNK_SAMPLES, N_MELS,
    BATCH_SIZE, LR, LR_MIN, WEIGHT_DECAY, T_0,
    SPEC_TIME_MASK_PROB, SPEC_FREQ_MASK_PROB,
    get_species_index,
)
from dataset import BirdTrainDataset
from utils import load_audio, pad_or_crop, waveform_to_mel

# Local A1 modules
from losses import AsymmetricLossOptimized, HybridBceAsl
from model_a1 import BirdSEDModelA1

# Output paths INSIDE the four_track workspace
A1_MODELS_DIR = FT_ROOT / "models" / "a1"
A1_LOG_DIR    = FT_ROOT / "log"


# ── Validation: parent train_soundscapes_labels.csv ───────────────────────────

def build_soundscape_val(sp2idx: dict) -> tuple:
    """Same precompute as parent train.py — kept here so the four_track A1
    pipeline doesn't import the parent train module (avoids side-effects).
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
        path    = soundsc_dir / str(row["filename"])
        try:
            wav     = load_audio(path)
            s, e    = int(t_start * SAMPLE_RATE), int(t_end * SAMPLE_RATE)
            segment = wav[s:e] if e <= len(wav) else wav[s:]
            segment = pad_or_crop(segment, CHUNK_SAMPLES, random_crop=False)
            mel     = waveform_to_mel(segment)
        except Exception as ex:
            print(f"  [warn] skipping {row['filename']} @ {t_start}s: {ex}", flush=True)
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
    max_batches: int | None = None,
) -> float:
    model.eval()
    all_probs = []
    n_done = 0
    for i in range(0, len(val_mels), batch_size):
        batch = torch.stack(val_mels[i: i + batch_size]).to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(batch)
        all_probs.append(torch.sigmoid(out["clip_logits"]).float().cpu().numpy())
        n_done += 1
        if max_batches is not None and n_done >= max_batches:
            break

    probs = np.concatenate(all_probs, axis=0)
    # If we truncated for smoke test, shrink labels to match
    if probs.shape[0] < val_labels.shape[0]:
        val_labels = val_labels[: probs.shape[0]]
    present = val_labels.sum(axis=0) > 0
    if present.sum() == 0:
        return 0.0
    try:
        return float(roc_auc_score(
            val_labels[:, present], probs[:, present], average="macro"
        ))
    except ValueError:
        return 0.0


# ── Trainer ───────────────────────────────────────────────────────────────────

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
    loss_name: str,
    mixstyle_p: float,
    smoke_test: bool,
    val_cache: tuple | None = None,
) -> tuple:
    """Train a single fold and return (best_auc, save_path, val_cache)."""
    set_seed(seed + fold)

    device = torch.device("cuda")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cudnn.benchmark = True

    # ── Data ──────────────────────────────────────────────────────────────────
    df       = pd.read_csv(PROC / "train_folds.csv")
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    if smoke_test:
        train_df = train_df.head(BATCH_SIZE * 2)  # ~2 batches

    train_ds = BirdTrainDataset(train_df, augment=True)
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0 if smoke_test else config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=not smoke_test,
        multiprocessing_context=None if smoke_test else "spawn",
    )
    print(f"Fold {fold}: {len(train_ds)} clips, {len(train_dl)} batches/epoch", flush=True)

    sp2idx = get_species_index()
    if val_cache is None:
        print("Building soundscape validation set …", flush=True)
        val_mels, val_labels = build_soundscape_val(sp2idx)
        val_cache = (val_mels, val_labels)
    else:
        val_mels, val_labels = val_cache
    n_present = int((val_labels.sum(axis=0) > 0).sum())
    print(f"  {len(val_mels)} val segments, {n_present} species present", flush=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    freq_mask = T.FrequencyMasking(freq_mask_param=27).to(device)
    time_mask = T.TimeMasking(time_mask_param=64).to(device)

    model = BirdSEDModelA1(
        backbone_name=backbone,
        mixstyle_p=mixstyle_p,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, eta_min=LR_MIN
    )

    if loss_name == "asl":
        loss_fn = AsymmetricLossOptimized(reduction="none")
    elif loss_name == "hybrid":
        loss_fn = HybridBceAsl(bce_weight=0.5)
    elif loss_name == "bce":
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    else:
        raise ValueError(f"Unknown loss '{loss_name}'")

    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    A1_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = A1_MODELS_DIR / f"a1_{backbone}_fold{fold}_seed{seed}_{loss_name}.pt"

    # ── Training loop ─────────────────────────────────────────────────────────
    best_auc = 0.0
    for epoch in range(1, epochs + 1):
        epoch_start  = time.time()
        model.train()
        running_loss = 0.0
        n_seen       = 0

        for batch_idx, (mels, labels, sec_mask) in enumerate(train_dl):
            mels     = mels.to(device, non_blocking=True)
            labels   = labels.to(device, non_blocking=True)
            sec_mask = sec_mask.to(device, non_blocking=True)

            if random.random() < SPEC_FREQ_MASK_PROB:
                mels = freq_mask(mels)
            if random.random() < SPEC_TIME_MASK_PROB:
                mels = time_mask(mels)

            optimizer.zero_grad()
            with autocast_ctx:
                out      = model(mels)
                loss_per = loss_fn(out["clip_logits"], labels)   # (B, N_CLASSES)
                loss     = (loss_per * sec_mask).mean()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += loss.item()
            n_seen       += 1

            if smoke_test and batch_idx >= 0:
                # One batch is enough for the smoke test
                break

        scheduler.step()

        avg_loss = running_loss / max(n_seen, 1)
        val_auc  = validate(
            model, val_mels, val_labels, device,
            max_batches=1 if smoke_test else None,
        )

        elapsed = int(time.time() - epoch_start)
        mins, s = divmod(elapsed, 60)

        best_marker = ""
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)
            best_marker = " ★ BEST"

        print("=" * 40, flush=True)
        print(
            f"Fold {fold}  Epoch {epoch:2d}/{epochs}: "
            f"train_loss={avg_loss:.4f}  "
            f"val_roc_auc={val_auc:.4f}  "
            f"time={mins}m {s:02d}s  "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
            f"{best_marker}",
            flush=True,
        )
        print("=" * 40, flush=True)

    print(f"\nFold {fold} complete. Best val ROC-AUC: {best_auc:.4f}", flush=True)
    print(f"Checkpoint → {save_path}\n", flush=True)
    return best_auc, save_path, val_cache


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Track A1 SED training (PCEN+ASL+MixStyle)")
    p.add_argument("--fold",       type=int, default=None,
                   help="Single fold to train (mutually exclusive with --folds)")
    p.add_argument("--folds",      type=str, default=None,
                   help="Comma-separated list of folds to train, e.g. 0,1,2,3,4")
    p.add_argument("--backbone",   type=str, default=config.BACKBONE)
    p.add_argument("--epochs",     type=int, default=25)
    p.add_argument("--seed",       type=int, default=config.SEED)
    p.add_argument("--loss",       type=str, default="asl",
                   choices=["bce", "asl", "hybrid"])
    p.add_argument("--mixstyle-p", type=float, default=0.5,
                   help="MixStyle activation probability (0 disables)")
    p.add_argument("--smoke-test", action="store_true",
                   help="1 fold, 1 epoch, 1 train batch, 1 val batch — verifies wiring")
    args = p.parse_args()

    if args.smoke_test:
        folds = [args.fold if args.fold is not None else 0]
        epochs = 1
    else:
        if args.fold is not None and args.folds is not None:
            sys.exit("--fold and --folds are mutually exclusive")
        if args.folds is not None:
            folds = [int(x) for x in args.folds.split(",")]
        elif args.fold is not None:
            folds = [args.fold]
        else:
            folds = [0, 1, 2, 3, 4]
        epochs = args.epochs

    print("=" * 60, flush=True)
    print("Track A1 — PCEN + ASL + FreqMixStyle EffNet-B0 SED", flush=True)
    print(f"  backbone   : {args.backbone}", flush=True)
    print(f"  folds      : {folds}", flush=True)
    print(f"  epochs     : {epochs}", flush=True)
    print(f"  loss       : {args.loss}", flush=True)
    print(f"  mixstyle_p : {args.mixstyle_p}", flush=True)
    print(f"  smoke_test : {args.smoke_test}", flush=True)
    print(f"  models →   : {A1_MODELS_DIR}", flush=True)
    print("=" * 60, flush=True)

    val_cache = None  # share precomputed val mels across folds
    fold_results = []
    t0 = time.time()
    for f in folds:
        best_auc, save_path, val_cache = train_one_fold(
            fold=f,
            backbone=args.backbone,
            epochs=epochs,
            seed=args.seed,
            loss_name=args.loss,
            mixstyle_p=args.mixstyle_p,
            smoke_test=args.smoke_test,
            val_cache=val_cache,
        )
        fold_results.append((f, best_auc, save_path))

    elapsed = int(time.time() - t0)
    h, rem = divmod(elapsed, 3600)
    m, s   = divmod(rem, 60)

    print("=" * 60, flush=True)
    print(f"Track A1 run complete  total time: {h}h {m:02d}m {s:02d}s", flush=True)
    for f, auc, path in fold_results:
        print(f"  fold {f}: best val_roc_auc = {auc:.4f}  → {path.name}", flush=True)
    if len(fold_results) > 1:
        mean_auc = float(np.mean([a for _, a, _ in fold_results]))
        print(f"  mean fold val_roc_auc = {mean_auc:.4f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
