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
import gc
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
from dataset import BirdTrainDataset  # noqa: F401  (kept for legacy callers)
from utils import load_audio, pad_or_crop, waveform_to_mel

# Local A1 modules
from dataset_a1 import BirdTrainDatasetA1
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


def _load_pretrained_backbone(model: nn.Module, ckpt_path: Path) -> None:
    """Load backbone weights from a pretrained ckpt; drop class-dependent head keys.

    Used by L2 multi-year pretraining: the pretrained ckpt was trained on a
    union class space (~400 classes) but the finetune model has 234 classes,
    so the final 1x1 convs in `cls_conv` and `att_conv[4]` shape-mismatch
    and must be reinitialized fresh.
    """
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    drop = {"cls_conv.weight", "cls_conv.bias",
            "att_conv.4.weight", "att_conv.4.bias"}
    head_keys = [k for k in state if k in drop]
    for k in head_keys:
        state.pop(k)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  [init-from] {ckpt_path}", flush=True)
    print(f"  [init-from] dropped head keys: {sorted(head_keys)}", flush=True)
    print(f"  [init-from] missing in ckpt (head reinit): {sorted(missing)}", flush=True)
    if unexpected:
        print(f"  [init-from] unexpected (ignored): {sorted(unexpected)}", flush=True)


def train_one_fold(
    fold: int,
    backbone: str,
    epochs: int,
    seed: int,
    loss_name: str,
    mixstyle_p: float,
    smoke_test: bool,
    val_cache: tuple | None = None,
    init_from: Path | None = None,
    ft_recipe: str = "gentle",
    anuraset_mixup: bool = False,
    save_all_epochs: bool = False,
    multi_layer_gem: bool = False,
    rms_select: bool = False,
    swa: bool = False,
    swa_start_frac: float = 0.65,
    swa_lr: float = 4e-4,
    bg_noise_dir: Path | None = None,
    pseudo_manifest: Path | None = None,
    ckpt_suffix: str = "",
) -> tuple:
    """Train a single fold and return (best_auc, save_path, val_cache).

    When `swa=True` (§14.14.12 P5), PyTorch SWA activates at
    `ceil(swa_start_frac * epochs)`: SWALR replaces the cosine warm-restart
    schedule, and an `AveragedModel` accumulates the per-epoch weights.
    After the final epoch we run `update_bn` once and validate + save the
    SWA-averaged model as the fold checkpoint (suffix `_swa`).
    """
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

    # A2: optionally concat soundscape pseudo-labels (BC2026_SS_PSEUDO rows).
    # These are training-only (fold=-1, never validated) and use the
    # pseudo_positive_labels column for multi-label targets.
    if pseudo_manifest is not None and not smoke_test:
        pseudo_df = pd.read_csv(pseudo_manifest)
        # Align columns: pseudo CSV has fewer columns than train_folds; fill missing.
        for col in train_df.columns:
            if col not in pseudo_df.columns:
                pseudo_df[col] = np.nan
        # Bring in pseudo_positive_labels even though it's not in train_folds.
        # BirdTrainDatasetA1._build_targets reads it from row directly.
        all_cols = list(train_df.columns) + (
            ["pseudo_positive_labels", "pseudo_window_start"]
            if "pseudo_positive_labels" not in train_df.columns
            else []
        )
        pseudo_df = pseudo_df[[c for c in all_cols if c in pseudo_df.columns]]
        train_df = pd.concat([train_df, pseudo_df], ignore_index=True)
        print(f"  [pseudo] +{len(pseudo_df)} BC2026_SS_PSEUDO rows  "
              f"(total train={len(train_df)})", flush=True)

    mixin_df = None
    if anuraset_mixup:
        pre_path = PROC / "train_folds_pre_anuraset.csv"
        if not pre_path.exists():
            raise FileNotFoundError(
                f"--anuraset-mixup requires {pre_path} (the pre-AnuraSet snapshot)"
            )
        pre_df = pd.read_csv(pre_path)
        # Pantanal bbox — exclude rows recorded inside this region so the
        # mixin pool carries non-Pantanal acoustic backgrounds only.
        PAN_LAT = (-22.0, -14.0)
        PAN_LON = (-62.0, -54.0)
        in_pan = (
            pre_df["latitude"].between(*PAN_LAT)
            & pre_df["longitude"].between(*PAN_LON)
        )
        # Rows with missing lat/long are kept (unknown-location prior ≈ global).
        in_pan = in_pan.fillna(False)
        mixin_df = pre_df[~in_pan].reset_index(drop=True)
        print(
            f"  [mixup] mixin pool: {len(mixin_df)} / {len(pre_df)} rows "
            f"(excluded {int(in_pan.sum())} Pantanal-bbox rows)",
            flush=True,
        )

    train_ds = BirdTrainDatasetA1(
        train_df,
        augment=True,
        bg_noise_dir=bg_noise_dir,
        mixin_df=mixin_df,
        rms_select=rms_select,
    )
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
    if bg_noise_dir is not None:
        print(
            f"  [bg-noise] dir={bg_noise_dir} files={len(train_ds.bg_files)} "
            f"BG_NOISE_PROB={config.BG_NOISE_PROB}",
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
        multi_layer_gem=multi_layer_gem,
    ).to(device)

    if init_from is not None:
        _load_pretrained_backbone(model, init_from)

    # When finetuning from a pretrained ckpt, use a gentler recipe:
    #   lr = 1e-4 (vs 5e-4) + 2-epoch linear warmup + single-cycle cosine.
    # This avoids the warm-restart spikes that clobber transferred features.
    # Pass --ft-recipe production to opt out and use the standard A1 recipe.
    if init_from is not None and ft_recipe == "gentle":
        ft_lr     = 1e-4
        ft_warmup = 2
        optimizer = torch.optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=WEIGHT_DECAY)
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=ft_warmup
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - ft_warmup, 1), eta_min=LR_MIN
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[ft_warmup]
        )
        print(f"  [finetune-recipe] lr={ft_lr}  warmup={ft_warmup} ep  "
              f"cosine→{LR_MIN}  (no warm restarts)", flush=True)
    else:
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
    elif loss_name == "ce":
        # §14.11.10 T2.1: softmax cross-entropy on normalized multi-hot targets.
        # Targets are treated as a probability distribution (labels/labels.sum).
        # sec_mask is NOT applied — secondaries stay in the target distribution,
        # consistent with 1st-place 2024 recipe that weights all labels.
        loss_fn = nn.CrossEntropyLoss(reduction="none")
    else:
        raise ValueError(f"Unknown loss '{loss_name}'")

    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    A1_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_suffix = "_swa" if swa else ""
    if init_from is not None and ft_recipe == "production":
        save_suffix += "_prodft"
    save_suffix += ckpt_suffix
    save_path = A1_MODELS_DIR / f"a1_{backbone}_fold{fold}_seed{seed}_{loss_name}{save_suffix}.pt"

    swa_model: torch.optim.swa_utils.AveragedModel | None = None
    swa_scheduler: torch.optim.swa_utils.SWALR | None = None
    swa_start_epoch = max(1, int(np.ceil(swa_start_frac * epochs))) if swa else None
    if swa:
        print(f"  [SWA] enabled  swa_start_epoch={swa_start_epoch}/{epochs}  "
              f"swa_lr={swa_lr}", flush=True)

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
                if loss_name == "ce":
                    tgt_sum  = labels.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                    tgt_prob = labels / tgt_sum
                    loss_per = loss_fn(out["clip_logits"], tgt_prob)   # (B,)
                    loss     = loss_per.mean()
                else:
                    loss_per = loss_fn(out["clip_logits"], labels)     # (B, N_CLASSES)
                    loss     = (loss_per * sec_mask).mean()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += loss.item()
            n_seen       += 1

            if smoke_test and batch_idx >= 0:
                # One batch is enough for the smoke test
                break

        if swa and swa_model is None and epoch >= swa_start_epoch:
            swa_model = torch.optim.swa_utils.AveragedModel(model)
            swa_scheduler = torch.optim.swa_utils.SWALR(
                optimizer, swa_lr=swa_lr, anneal_epochs=2, anneal_strategy="linear"
            )
            print(f"  [SWA] activated at epoch {epoch} — SWALR replaces cosine "
                  f"schedule, anneal over 2 ep to swa_lr={swa_lr}", flush=True)

        if swa_scheduler is not None:
            swa_scheduler.step()
            swa_model.update_parameters(model)
        else:
            scheduler.step()

        avg_loss = running_loss / max(n_seen, 1)
        val_auc  = validate(
            model, val_mels, val_labels, device,
            max_batches=1 if smoke_test else None,
        )

        elapsed = int(time.time() - epoch_start)
        mins, s = divmod(elapsed, 60)

        # In SWA mode we intentionally do NOT save per-epoch — the final
        # SWA-averaged ckpt (post-update_bn) is the only authoritative
        # artifact; per-epoch raw-model val_auc is logged as a progress
        # signal but has no BN-refresh on the averaged weights.
        best_marker = ""
        if not swa and val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)
            best_marker = " ★ BEST"

        if save_all_epochs:
            # §14.11.10.8 T2.3 ckpt soup: persist per-epoch state_dicts so a
            # post-training averaging pass can build the soup candidate.
            soup_dir = A1_MODELS_DIR / "_soup" / f"fold{fold}_{loss_name}_seed{seed}"
            soup_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"epoch": epoch, "val_roc_auc": val_auc,
                 "state_dict": model.state_dict()},
                soup_dir / f"epoch{epoch:02d}.pt",
            )

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

        # GB10 unified-memory hygiene — see four_track/CLAUDE.md
        gc.collect()
        torch.cuda.empty_cache()

    if swa and swa_model is not None:
        bn_t0 = time.time()
        print(f"\n[SWA] running update_bn over train loader…", flush=True)
        swa_model.train()
        with torch.no_grad(), autocast_ctx:
            for mels, _labels, _sec in train_dl:
                mels = mels.to(device, non_blocking=True)
                swa_model(mels)
                if smoke_test:
                    break
        swa_model.eval()
        print(f"[SWA] update_bn done in {time.time()-bn_t0:.1f}s", flush=True)

        swa_val_auc = validate(
            swa_model, val_mels, val_labels, device,
            max_batches=1 if smoke_test else None,
        )
        torch.save(swa_model.module.state_dict(), save_path)
        best_auc = swa_val_auc
        print(f"\n[SWA] averaged model val_roc_auc = {swa_val_auc:.4f}", flush=True)
        print(f"[SWA] saved SWA ckpt → {save_path}", flush=True)

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
                   choices=["bce", "asl", "hybrid", "ce"])
    p.add_argument("--mixstyle-p", type=float, default=0.5,
                   help="MixStyle activation probability (0 disables)")
    p.add_argument("--smoke-test", action="store_true",
                   help="1 fold, 1 epoch, 1 train batch, 1 val batch — verifies wiring")
    p.add_argument("--init-from", type=Path, default=None,
                   help="Path to a pretrained ckpt; loads backbone, drops 234-class head")
    p.add_argument("--ft-recipe", choices=["gentle", "production"], default="gentle",
                   help="When --init-from is set: 'gentle' uses lr=1e-4 + 2-ep warmup + "
                        "single cosine (preserves features); 'production' uses lr=5e-4 + "
                        "warm restarts (full A1 recipe, may clobber transferred features)")
    p.add_argument("--anuraset-mixup", action="store_true",
                   help="L5b salvage (§14.10.16): background-mixup AnuraSet rows with a "
                        "non-Pantanal mixin pool built from train_folds_pre_anuraset.csv")
    p.add_argument("--bg-noise-dir", type=Path, default=None,
                   help="§21 Option 1: directory of background-noise audio (.ogg/.wav) "
                        "for waveform-level mixing during training. When set, "
                        "BirdTrainDatasetA1 will mix focal clips with random bg files "
                        "at BG_NOISE_PROB (config.py) and gain 0.05-0.15.")
    p.add_argument("--save-all-epochs", action="store_true",
                   help="T2.3 ckpt soup (§14.11.10.8): save state_dict after every epoch "
                        "to models/a1/_soup/fold{F}_{loss}_seed{seed}/epoch{E:02d}.pt")
    p.add_argument("--multi-layer-gem", action="store_true",
                   help="M2 probe (§14.14.1): concat EffNet blocks 3+4 after per-scale "
                        "GeM freq-pool and adaptive time-pool; heads scale to C_3 + C_4")
    p.add_argument("--rms-select", action="store_true",
                   help="P8 probe (§14.14.3): replace random 20s crop with sample from "
                        "top-3 RMS-ranked candidate windows (5s stride) per focal file")
    p.add_argument("--swa", action="store_true",
                   help="P5 (§14.14.12): PyTorch SWA with AveragedModel + SWALR "
                        "+ update_bn pass at end. Saves *_swa.pt")
    p.add_argument("--swa-start-frac", type=float, default=0.65,
                   help="Fraction of epochs before SWA activates (default 0.65)")
    p.add_argument("--pseudo-manifest", type=Path, default=None,
                   help="A2: CSV of BC2026 soundscape pseudo-labels "
                        "(from src/a2_build_pseudo_manifest.py). Concatenated "
                        "into train_df; rows use pseudo_positive_labels for "
                        "multi-label targets.")
    p.add_argument("--swa-lr", type=float, default=4e-4,
                   help="LR held by SWALR during the averaging window (default 4e-4)")
    p.add_argument("--ckpt-suffix", type=str, default="",
                   help="Extra suffix appended to ckpt filename "
                        "(e.g. '_l4' to differentiate L4 round-2 from base recipe). "
                        "If --smoke-test is also set, '_smoke' is appended after this.")
    args = p.parse_args()

    if args.smoke_test and not args.swa:
        folds = [args.fold if args.fold is not None else 0]
        epochs = 1
    elif args.smoke_test and args.swa:
        folds = [args.fold if args.fold is not None else 0]
        epochs = args.epochs  # respect user's --epochs for SWA smoke
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
    print(f"  init_from  : {args.init_from}", flush=True)
    print(f"  ft_recipe  : {args.ft_recipe}", flush=True)
    print(f"  anuraset_mixup : {args.anuraset_mixup}", flush=True)
    print(f"  multi_layer_gem: {args.multi_layer_gem}", flush=True)
    print(f"  rms_select : {args.rms_select}", flush=True)
    print(f"  swa        : {args.swa}  start_frac={args.swa_start_frac}  "
          f"lr={args.swa_lr}", flush=True)
    print(f"  pseudo     : {args.pseudo_manifest or '(disabled)'}", flush=True)
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
            init_from=args.init_from,
            ft_recipe=args.ft_recipe,
            anuraset_mixup=args.anuraset_mixup,
            save_all_epochs=args.save_all_epochs,
            multi_layer_gem=args.multi_layer_gem,
            rms_select=args.rms_select,
            swa=args.swa,
            swa_start_frac=args.swa_start_frac,
            swa_lr=args.swa_lr,
            bg_noise_dir=args.bg_noise_dir,
            pseudo_manifest=args.pseudo_manifest,
            ckpt_suffix=args.ckpt_suffix,
        )
        fold_results.append((f, best_auc, save_path))
        gc.collect()
        torch.cuda.empty_cache()

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
