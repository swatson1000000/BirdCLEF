"""Track B2 SED training — PCEN + hybrid loss + ConvNeXt-tiny.

Trains the §14.16 Track B2 branch: ConvNeXt-tiny backbone + A1-style
SED head on the same `train_audio` substrate as A1. Purpose: architectural
diversity from A1 (EffNet-B0) and from both Perch consumers (ProtoSSM, B1)
for downstream fusion in the production notebook's Cell 37b.

Mirrors `train_a1.py` with:
  - Model class: BirdSEDModelB2 (model_b2.py)
  - Output dir: four_track/models/b2/
  - Default backbone: convnext_tiny.fb_in22k_ft_in1k

Dropped from A1 (not applicable to ConvNeXt or not in Phase 1 scope):
  - --init-from (L2 pretraining was EffNet-specific)
  - --multi-layer-gem (A1 M2 probe was EffNet-specific)
  - --save-all-epochs / --swa (T2.3 soup + P5 SWA both killed on A1's
    cosine warm-restart schedule; not revisiting on B2 until baseline
    B2 5-fold is established)

Usage:
    python -u src/train_b2.py --fold 0 --smoke-test
    python -u src/train_b2.py --folds 0,1,2,3,4 --loss hybrid
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

# Local modules
from dataset_a1 import BirdTrainDatasetA1
from losses import AsymmetricLossOptimized, HybridBceAsl
from model_b2 import BirdSEDModelB2

# Output paths INSIDE the four_track workspace
B2_MODELS_DIR = FT_ROOT / "models" / "b2"
B2_LOG_DIR    = FT_ROOT / "log"


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
    anuraset_mixup: bool = False,
    rms_select: bool = False,
    resume_path: Path | None = None,
    resume_epoch: int = 0,
    resume_best_auc: float = 0.0,
) -> tuple:
    """Train a single B2 fold and return (best_auc, save_path, val_cache).

    When `resume_path` is provided, model weights are loaded from that
    checkpoint, training starts at `resume_epoch + 1`, the cosine warm-
    restart scheduler is fast-forwarded by `resume_epoch` steps, and
    `best_auc` is initialised to `resume_best_auc` so the saved file is
    only overwritten by a strictly better validation score.
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

    model = BirdSEDModelB2(
        backbone_name=backbone,
        mixstyle_p=mixstyle_p,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, eta_min=LR_MIN
    )

    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")
        if not (1 <= resume_epoch < epochs):
            raise ValueError(
                f"--resume-epoch must satisfy 1 <= N < epochs ({epochs}); got {resume_epoch}"
            )
        ckpt = torch.load(resume_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        for _ in range(resume_epoch):
            scheduler.step()
        print(
            f"  [resume] loaded {resume_path}",
            flush=True,
        )
        print(
            f"  [resume] start_epoch={resume_epoch + 1}/{epochs}  "
            f"best_auc={resume_best_auc:.4f}  scheduler stepped {resume_epoch}×  "
            f"(missing={len(missing)}, unexpected={len(unexpected)})",
            flush=True,
        )
        if missing:
            print(f"  [resume] missing keys: {sorted(missing)[:5]}...", flush=True)
        if unexpected:
            print(f"  [resume] unexpected keys: {sorted(unexpected)[:5]}...", flush=True)

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

    B2_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = B2_MODELS_DIR / f"b2_{backbone.split('.')[0]}_fold{fold}_seed{seed}_{loss_name}.pt"

    # ── Training loop ─────────────────────────────────────────────────────────
    best_auc = resume_best_auc if resume_path is not None else 0.0
    start_epoch = (resume_epoch + 1) if resume_path is not None else 1
    for epoch in range(start_epoch, epochs + 1):
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

        # GB10 unified-memory hygiene: drop cached CUDA blocks + Python refs
        # before the next epoch's allocations so fragmentation doesn't
        # accumulate across the 5-fold × 25-epoch loop.
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nFold {fold} complete. Best val ROC-AUC: {best_auc:.4f}", flush=True)
    print(f"Checkpoint → {save_path}\n", flush=True)
    return best_auc, save_path, val_cache


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Track B2 SED training (ConvNeXt-tiny)")
    p.add_argument("--fold",       type=int, default=None,
                   help="Single fold to train (mutually exclusive with --folds)")
    p.add_argument("--folds",      type=str, default=None,
                   help="Comma-separated list of folds, e.g. 0,1,2,3,4")
    p.add_argument("--backbone",   type=str,
                   default=BirdSEDModelB2.DEFAULT_BACKBONE)
    p.add_argument("--epochs",     type=int, default=25)
    p.add_argument("--seed",       type=int, default=config.SEED)
    p.add_argument("--loss",       type=str, default="hybrid",
                   choices=["bce", "asl", "hybrid"])
    p.add_argument("--mixstyle-p", type=float, default=0.5,
                   help="MixStyle activation probability (0 disables)")
    p.add_argument("--smoke-test", action="store_true",
                   help="1 fold, 1 epoch, 1 train batch, 1 val batch — verifies wiring")
    p.add_argument("--anuraset-mixup", action="store_true",
                   help="Kept for interface parity with train_a1; not used in B2 Phase 1")
    p.add_argument("--rms-select", action="store_true",
                   help="Kept for interface parity with train_a1; not used in B2 Phase 1")
    p.add_argument("--resume", type=Path, default=None,
                   help="Checkpoint to resume the FIRST fold from (state_dict .pt). "
                        "Subsequent folds in --folds train from scratch.")
    p.add_argument("--resume-epoch", type=int, default=None,
                   help="Last completed epoch in the resumed checkpoint (1-indexed). "
                        "Required with --resume.")
    p.add_argument("--resume-best-auc", type=float, default=None,
                   help="Best val_roc_auc achieved so far in the resumed run. "
                        "Required with --resume; new ckpt is saved only if a later "
                        "epoch beats this value.")
    args = p.parse_args()

    if args.resume is not None:
        if args.resume_epoch is None or args.resume_best_auc is None:
            sys.exit("--resume requires both --resume-epoch and --resume-best-auc")
        if args.smoke_test:
            sys.exit("--resume is not compatible with --smoke-test")

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
    print("Track B2 — ConvNeXt-tiny SED", flush=True)
    print(f"  backbone   : {args.backbone}", flush=True)
    print(f"  folds      : {folds}", flush=True)
    print(f"  epochs     : {epochs}", flush=True)
    print(f"  loss       : {args.loss}", flush=True)
    print(f"  mixstyle_p : {args.mixstyle_p}", flush=True)
    print(f"  smoke_test : {args.smoke_test}", flush=True)
    print(f"  models →   : {B2_MODELS_DIR}", flush=True)
    if args.resume is not None:
        print(f"  resume     : {args.resume}  "
              f"(epoch={args.resume_epoch}, best_auc={args.resume_best_auc:.4f}, "
              f"first fold only)", flush=True)
    print("=" * 60, flush=True)

    val_cache = None
    fold_results = []
    t0 = time.time()
    for i, f in enumerate(folds):
        is_first = (i == 0)
        best_auc, save_path, val_cache = train_one_fold(
            fold=f,
            backbone=args.backbone,
            epochs=epochs,
            seed=args.seed,
            loss_name=args.loss,
            mixstyle_p=args.mixstyle_p,
            smoke_test=args.smoke_test,
            val_cache=val_cache,
            anuraset_mixup=args.anuraset_mixup,
            rms_select=args.rms_select,
            resume_path=args.resume if is_first else None,
            resume_epoch=args.resume_epoch if (is_first and args.resume is not None) else 0,
            resume_best_auc=args.resume_best_auc if (is_first and args.resume is not None) else 0.0,
        )
        fold_results.append((f, best_auc, save_path))
        gc.collect()
        torch.cuda.empty_cache()

    elapsed = int(time.time() - t0)
    h, rem = divmod(elapsed, 3600)
    m, s   = divmod(rem, 60)

    print("=" * 60, flush=True)
    print(f"Track B2 run complete  total time: {h}h {m:02d}m {s:02d}s", flush=True)
    for f, auc, path in fold_results:
        print(f"  fold {f}: best val_roc_auc = {auc:.4f}  → {path.name}", flush=True)
    if len(fold_results) > 1:
        mean_auc = float(np.mean([a for _, a, _ in fold_results]))
        print(f"  mean fold val_roc_auc = {mean_auc:.4f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
