"""L1 cross-arch noisy student — ProtoSSM teacher → A1 EffNet-B0 student.

Sibling of `train_a1.py`. Does not modify the baseline pipeline.

Differences from baseline A1:
  - Soft targets from §10 ProtoSSM teacher merged per-clip via element-wise
    max(hard_multihot, teacher_soft). See new_plan.md §14.9.2.
  - BCE loss by default (not ASL — ASL's γ_neg mis-weights soft targets;
    see §14.9.3).
  - mixstyle_p=0.7 default (up from 0.5) and SpecAugment probs scaled ×1.2
    to add noise for the "noisy" side of Noisy Student (§14.9.4).
  - train_df filtered to retained=True clips from
    `c2_pseudo_labels_kagglefeat/pseudo_labels.parquet` (33,516 of 35,549).
  - Checkpoints → `four_track/models/a1_ns/` (baseline a1/ untouched).

Usage:
    python -u src/train_a1_noisy_student.py --fold 0 --epochs 25
    python -u src/train_a1_noisy_student.py --fold 0 --smoke-test
    python -u src/train_a1_noisy_student.py --folds 1,2,4 --epochs 25

Kill gate (§14.9.6):
    fold-0 val_roc_auc ≥ 0.7514 (baseline 0.7414 + 0.010) OR LB ≥ 0.935.
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

if str(PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(PARENT_SRC))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config
from config import (
    RAW, PROC,
    SAMPLE_RATE, CHUNK_SAMPLES, N_MELS,
    BATCH_SIZE, LR, LR_MIN, WEIGHT_DECAY, T_0,
    get_species_index,
)
from dataset import BirdTrainDataset
from utils import load_audio, pad_or_crop, waveform_to_mel

from model_a1 import BirdSEDModelA1

# Reuse validation helpers from baseline A1 by import.
from train_a1 import build_soundscape_val, validate, set_seed

# Output paths
A1_NS_MODELS_DIR = FT_ROOT / "models" / "a1_ns"
PSEUDO_DIR       = FT_ROOT / "data" / "processed" / "c2_pseudo_labels_kagglefeat"
PSEUDO_NPZ       = PSEUDO_DIR / "pseudo_soft_labels.npz"
PSEUDO_PARQUET   = PSEUDO_DIR / "pseudo_labels.parquet"

# Noise-lever overrides (§14.9.4). Baseline values come from config.
NS_MIXSTYLE_P_DEFAULT = 0.7
NS_SPEC_MASK_SCALE    = 1.2


# ── Pseudo-label loading ──────────────────────────────────────────────────────

def load_soft_lookup() -> dict:
    """Build {stem: soft_label_vec} for retained clips only.

    Returns a dict of length 33,516 (retained count). Stems not present
    here are dropped from the training set at NoisyStudentDataset init.
    """
    if not PSEUDO_NPZ.exists() or not PSEUDO_PARQUET.exists():
        sys.exit(
            f"Missing pseudo-label artifacts:\n"
            f"  NPZ     : {PSEUDO_NPZ} ({'ok' if PSEUDO_NPZ.exists() else 'MISSING'})\n"
            f"  parquet : {PSEUDO_PARQUET} ({'ok' if PSEUDO_PARQUET.exists() else 'MISSING'})"
        )

    z = np.load(PSEUDO_NPZ)
    stems = z["stems"].astype(str)
    soft  = z["soft_labels"].astype(np.float32)   # (N, 234)
    assert soft.shape == (len(stems), config.N_CLASSES), (
        f"soft_labels shape {soft.shape} != ({len(stems)}, {config.N_CLASSES})"
    )

    pq  = pd.read_parquet(PSEUDO_PARQUET)
    retained_stems = set(pq.loc[pq["retained"], "stem"].astype(str).tolist())

    lookup = {}
    for i, s in enumerate(stems):
        if s in retained_stems:
            lookup[s] = soft[i]

    print(
        f"Pseudo-labels: {len(stems)} stems in NPZ, "
        f"{len(retained_stems)} retained in parquet, "
        f"{len(lookup)} built into lookup.",
        flush=True,
    )
    return lookup


# ── Noisy-student dataset ─────────────────────────────────────────────────────

class NoisyStudentDataset(BirdTrainDataset):
    """BirdTrainDataset with teacher soft labels max-merged into targets.

    The base class's `_mixup` already element-wise-maxes labels1/labels2, so
    after our override MixUp still behaves correctly: both mixed samples
    carry their merged (hard ∨ soft) targets, and MixUp's max-merge yields
    the intended "union of both mixed samples' teacher + hard signals."
    """

    def __init__(
        self,
        df: pd.DataFrame,
        soft_lookup: dict,
        augment: bool = True,
        bg_noise_dir: "Path | None" = None,
        min_samples_per_class: int = 10,
    ):
        # Filter df to rows whose stem is in soft_lookup (retained=True set).
        stems = df["filename"].map(lambda f: Path(str(f)).stem)
        keep_mask = stems.isin(soft_lookup.keys())
        n_kept = int(keep_mask.sum())
        n_drop = int((~keep_mask).sum())
        print(
            f"NS dataset: kept {n_kept} of {len(df)} clips "
            f"(dropped {n_drop} not in retained soft-label set).",
            flush=True,
        )
        df_kept = df.loc[keep_mask].reset_index(drop=True)

        super().__init__(
            df_kept,
            augment=augment,
            bg_noise_dir=bg_noise_dir,
            min_samples_per_class=min_samples_per_class,
        )
        # After super().__init__ the class may have duplicated rare-species
        # rows, but the stems of duplicated rows still exist in soft_lookup
        # (rare species get duplicated from within kept set), so the lookup
        # stays valid post-oversample.
        self.soft_lookup = soft_lookup

    def _build_targets(self, row) -> tuple:
        labels, mask = super()._build_targets(row)
        stem = Path(str(row["filename"])).stem
        soft = self.soft_lookup.get(stem)
        if soft is not None:
            labels = np.maximum(labels, soft)
        return labels, mask


# ── Trainer ───────────────────────────────────────────────────────────────────

def train_one_fold(
    fold: int,
    backbone: str,
    epochs: int,
    seed: int,
    loss_name: str,
    mixstyle_p: float,
    spec_mask_scale: float,
    soft_lookup: dict,
    smoke_test: bool,
    val_cache: tuple | None = None,
) -> tuple:
    set_seed(seed + fold)

    device = torch.device("cuda")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cudnn.benchmark = True

    # SpecAug probabilities — noise-lever override (§14.9.4).
    spec_freq_prob = min(1.0, config.SPEC_FREQ_MASK_PROB * spec_mask_scale)
    spec_time_prob = min(1.0, config.SPEC_TIME_MASK_PROB * spec_mask_scale)

    # ── Data ──────────────────────────────────────────────────────────────────
    df       = pd.read_csv(PROC / "train_folds.csv")
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    if smoke_test:
        train_df = train_df.head(BATCH_SIZE * 2)

    train_ds = NoisyStudentDataset(train_df, soft_lookup=soft_lookup, augment=True)
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
    print(
        f"Fold {fold}: {len(train_ds)} clips, {len(train_dl)} batches/epoch",
        flush=True,
    )

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

    # BCE is the correct default for Noisy Student (soft Bernoulli targets);
    # ASL/hybrid left available for ablation only.
    if loss_name == "bce":
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    elif loss_name == "asl":
        from losses import AsymmetricLossOptimized
        loss_fn = AsymmetricLossOptimized(reduction="none")
    elif loss_name == "hybrid":
        from losses import HybridBceAsl
        loss_fn = HybridBceAsl(bce_weight=0.5)
    else:
        raise ValueError(f"Unknown loss '{loss_name}'")

    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    A1_NS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = (
        A1_NS_MODELS_DIR
        / f"a1_ns_{backbone}_fold{fold}_seed{seed}_{loss_name}.pt"
    )

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

            if random.random() < spec_freq_prob:
                mels = freq_mask(mels)
            if random.random() < spec_time_prob:
                mels = time_mask(mels)

            optimizer.zero_grad()
            with autocast_ctx:
                out      = model(mels)
                loss_per = loss_fn(out["clip_logits"], labels)
                loss     = (loss_per * sec_mask).mean()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += loss.item()
            n_seen       += 1

            if smoke_test and batch_idx >= 0:
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
    p = argparse.ArgumentParser(
        description="L1 cross-arch noisy student (ProtoSSM → A1 EffNet-B0)"
    )
    p.add_argument("--fold",       type=int, default=None)
    p.add_argument("--folds",      type=str, default=None,
                   help="Comma-separated list, e.g. 1,2,4")
    p.add_argument("--backbone",   type=str, default=config.BACKBONE)
    p.add_argument("--epochs",     type=int, default=25)
    p.add_argument("--seed",       type=int, default=config.SEED)
    p.add_argument("--loss",       type=str, default="bce",
                   choices=["bce", "asl", "hybrid"],
                   help="BCE is the correct default for NS; others are ablation.")
    p.add_argument("--mixstyle-p", type=float, default=NS_MIXSTYLE_P_DEFAULT)
    p.add_argument("--spec-mask-scale", type=float, default=NS_SPEC_MASK_SCALE,
                   help="Multiplier for SPEC_{FREQ,TIME}_MASK_PROB (§14.9.4).")
    p.add_argument("--smoke-test", action="store_true")
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
            # Baseline has folds 0,1,2,4 on disk; mirror that for ensemble parity.
            folds = [0, 1, 2, 4]
        epochs = args.epochs

    print("=" * 60, flush=True)
    print("L1 — Cross-arch noisy student (ProtoSSM → A1 EffNet-B0)", flush=True)
    print(f"  backbone        : {args.backbone}", flush=True)
    print(f"  folds           : {folds}", flush=True)
    print(f"  epochs          : {epochs}", flush=True)
    print(f"  loss            : {args.loss}", flush=True)
    print(f"  mixstyle_p      : {args.mixstyle_p}", flush=True)
    print(f"  spec_mask_scale : {args.spec_mask_scale}", flush=True)
    print(f"  smoke_test      : {args.smoke_test}", flush=True)
    print(f"  models →        : {A1_NS_MODELS_DIR}", flush=True)
    print("=" * 60, flush=True)

    soft_lookup = load_soft_lookup()

    val_cache = None
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
            spec_mask_scale=args.spec_mask_scale,
            soft_lookup=soft_lookup,
            smoke_test=args.smoke_test,
            val_cache=val_cache,
        )
        fold_results.append((f, best_auc, save_path))

    elapsed = int(time.time() - t0)
    h, rem = divmod(elapsed, 3600)
    m, s   = divmod(rem, 60)

    print("=" * 60, flush=True)
    print(f"L1 NS run complete  total time: {h}h {m:02d}m {s:02d}s", flush=True)
    for f, auc, path in fold_results:
        print(f"  fold {f}: best val_roc_auc = {auc:.4f}  → {path.name}", flush=True)
    if len(fold_results) > 1:
        mean_auc = float(np.mean([a for _, a, _ in fold_results]))
        print(f"  mean fold val_roc_auc = {mean_auc:.4f}", flush=True)

    # Remind about the kill gate so log is self-contained.
    print("  kill gate: fold-0 val ≥ 0.7514 (baseline 0.7414 + 0.010) OR LB ≥ 0.935",
          flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
