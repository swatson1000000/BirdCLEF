"""Bundle 1 NS — Sydorskyi-recipe noisy-student finetune.

Sibling of train_a1.py / train_a1_noisy_student.py. Distinct from the
killed L1 NS recipe (max-merge hard, 100% pseudo, ProtoSSM teacher) —
this one follows Sydorskyi BC2025 2nd-place's actual recipe per
`reference_bc2025_winners_writeups.md`:

  - Pseudo NPZ from src/pseudo_emit_sydorskyi.py (already-filtered:
    chunks with max prob >= 0.5 kept, per-class probs < 0.1 zeroed).
  - Per train-step: 40% prob a sample is drawn from pseudo set;
    60% from real focal hard-label set.
  - MixUp at audio level: with prob --mixup-p, sum waveforms and
    sum-clip targets to [0,1]. No 0.5/0.5 weighting (Sydorskyi).
  - Soft targets only (BCE on soft Bernoulli — ASL/hybrid mis-weights
    soft, see L1 NS post-mortem).
  - Standard A1 augmentations otherwise (gain, time shift, BG noise,
    SpecAug GPU-side).
  - --init-from accepts a pretrained-V2-S backbone ckpt (Bundle 1
    Phase 5.1 output) as init for finetune.

Bundle 1 sequencing:
  Phase 5.2 NS iter-1: teacher = L2-redux-pretrained V2-S → emit pseudos
                        → train with this script
  Phase 5.3 NS iter-2: teacher = iter-1 student → emit pseudos
                        → train with this script (same code, new --pseudo-npz)

Usage:
    # Smoke
    python -u src/train_a1_ns_sydorskyi.py --fold 0 --smoke-test \\
        --pseudo-npz data/pseudo_iter1_smoke.npz \\
        --pseudo-audio-dir ../data/raw/birdclef_2025/train_soundscapes \\
        --backbone tf_efficientnetv2_s.in21k_ft_in1k

    # Phase 5.2 NS iter-1, fold 0, init from L2-redux pretrain
    python -u src/train_a1_ns_sydorskyi.py --fold 0 --epochs 25 \\
        --pseudo-npz data/pseudo_iter1.npz \\
        --pseudo-audio-dir ../data/raw/birdclef_2025/train_soundscapes \\
        --backbone tf_efficientnetv2_s.in21k_ft_in1k \\
        --init-from models/l2_redux/l2_redux_best_tf_efficientnetv2_s_in21k_ft_in1k_with_xc.pt

    # 4-fold sweep on DT (sequential per 4:1 rule, N=4)
    python -u src/train_a1_ns_sydorskyi.py --folds 0,1,2,4 --epochs 25 \\
        --pseudo-npz data/pseudo_iter1.npz \\
        --pseudo-audio-dir ../data/raw/birdclef_2025/train_soundscapes \\
        --backbone tf_efficientnetv2_s.in21k_ft_in1k \\
        --init-from models/l2_redux/l2_redux_best_tf_efficientnetv2_s_in21k_ft_in1k_with_xc.pt
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
from torch.utils.data import DataLoader, Dataset

# ── Path wiring ───────────────────────────────────────────────────────────────
HERE       = Path(__file__).resolve().parent
FT_ROOT    = HERE.parent
ROOT       = FT_ROOT.parent
PARENT_SRC = ROOT / "src"

if str(PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(PARENT_SRC))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config  # noqa: E402
from config import (  # noqa: E402
    RAW, PROC,
    SAMPLE_RATE, CHUNK_SAMPLES, N_MELS,
    BATCH_SIZE, LR, LR_MIN, WEIGHT_DECAY, T_0,
    SPEC_TIME_MASK_PROB, SPEC_FREQ_MASK_PROB,
    GAIN_PROB, GAIN_DB_RANGE, TIME_SHIFT_PROB,
    get_species_index,
)
from utils import load_audio, pad_or_crop, waveform_to_mel  # noqa: E402

from model_a1 import BirdSEDModelA1  # noqa: E402
from train_a1 import (                # noqa: E402
    build_soundscape_val, validate, set_seed, _load_pretrained_backbone,
)

A1_NS_MODELS_DIR = FT_ROOT / "models" / "a1_ns_sydorskyi"


# ── Combined dataset ──────────────────────────────────────────────────────────

class SydorskyiNSDataset(Dataset):
    """Unified real+pseudo dataset with Sydorskyi-style audio-level MixUp.

    Real branch: focal BC2026 train_audio clips with hard one-hot labels.
    Pseudo branch: chunks emitted by pseudo_emit_sydorskyi.py with soft
                   targets (already filtered by max>=0.5 / zero<0.1).

    Each __getitem__:
      1. Roll dice: pseudo with prob --pseudo-prob, else real.
      2. Load chosen sample's waveform + target.
      3. Apply gain + time-shift (CPU augment).
      4. With prob --mixup-p, draw a random partner (ANY source),
         element-wise sum waveforms, element-wise sum-clip targets [0,1].
      5. Convert to mel.

    Length = len(real_df). One epoch = one pass over real focal clips.
    """

    def __init__(self,
                 real_df: pd.DataFrame,
                 pseudo_npz_path: Path,
                 pseudo_audio_dir: Path,
                 augment: bool,
                 pseudo_prob: float,
                 mixup_p: float,
                 min_samples_per_class: int = 10):
        self.augment      = augment
        self.pseudo_prob  = float(pseudo_prob)
        self.mixup_p      = float(mixup_p)
        self.sp2idx       = get_species_index()
        self.n_classes    = config.N_CLASSES
        self.pseudo_audio_dir = Path(pseudo_audio_dir)

        # Real-focal upsampling (matches BirdTrainDataset behavior).
        counts = real_df["primary_label"].astype(str).value_counts()
        rare = set(counts[counts < min_samples_per_class].index)
        extras = []
        for sp in rare:
            sp_df = real_df[real_df["primary_label"].astype(str) == sp]
            needed = min_samples_per_class - len(sp_df)
            extras.append(sp_df.sample(needed, replace=True, random_state=42))
        if extras:
            real_df = pd.concat([real_df] + extras, ignore_index=True)
        self.real_df = real_df.reset_index(drop=True)

        z = np.load(pseudo_npz_path, allow_pickle=True)
        self.p_clip_names = z["clip_names"].astype(str)        # (P,)
        self.p_chunk_idx  = z["chunk_idx"].astype(np.int32)    # (P,)
        self.p_soft       = z["soft_labels"].astype(np.float32) # (P, K)
        assert self.p_soft.shape[1] == self.n_classes, (
            f"pseudo soft_labels shape {self.p_soft.shape} != (*, {self.n_classes})"
        )
        self.n_pseudo = len(self.p_clip_names)
        if self.n_pseudo == 0:
            sys.exit(f"Pseudo NPZ {pseudo_npz_path} contains 0 chunks.")

    def __len__(self) -> int:
        return len(self.real_df)

    # ── Real branch ───────────────────────────────────────────────────────────
    def _load_real(self, idx: int):
        row = self.real_df.iloc[idx]
        path = RAW / "train_audio" / str(row["filename"])
        wav  = load_audio(path)
        wav  = pad_or_crop(wav, CHUNK_SAMPLES, random_crop=self.augment)

        labels = np.zeros(self.n_classes, dtype=np.float32)
        primary = str(row["primary_label"])
        if primary in self.sp2idx:
            labels[self.sp2idx[primary]] = 1.0
        # Secondary labels handled as soft 1s (no mask — Sydorskyi recipe
        # uses equal-weight secondary, see ablation enhancements bundle).
        sec = row.get("secondary_labels", "[]")
        try:
            import ast
            for sp in ast.literal_eval(str(sec)):
                if str(sp) in self.sp2idx:
                    labels[self.sp2idx[str(sp)]] = 1.0
        except Exception:
            pass
        return wav.astype(np.float32, copy=False), labels

    # ── Pseudo branch ─────────────────────────────────────────────────────────
    def _load_pseudo(self, p_idx: int):
        rel = str(self.p_clip_names[p_idx])
        c   = int(self.p_chunk_idx[p_idx])
        path = self.pseudo_audio_dir / rel
        wav  = load_audio(path)
        s    = c * CHUNK_SAMPLES
        e    = s + CHUNK_SAMPLES
        chunk = wav[s:e]
        chunk = pad_or_crop(chunk, CHUNK_SAMPLES, random_crop=False)
        soft  = self.p_soft[p_idx].copy()
        return chunk.astype(np.float32, copy=False), soft

    # ── CPU augmentation (single waveform) ────────────────────────────────────
    def _aug_wave(self, wav: np.ndarray) -> np.ndarray:
        if not self.augment:
            return wav
        if random.random() < GAIN_PROB:
            db = random.uniform(-GAIN_DB_RANGE, GAIN_DB_RANGE)
            wav = wav * (10.0 ** (db / 20.0))
        if random.random() < TIME_SHIFT_PROB:
            max_shift = int(0.25 * len(wav))
            wav = np.roll(wav, random.randint(-max_shift, max_shift))
        return wav

    def _draw_random_pair(self):
        """Sydorskyi mixup: draw any-source partner (40/60 same as parent dice)."""
        if random.random() < self.pseudo_prob:
            return self._load_pseudo(random.randrange(self.n_pseudo))
        return self._load_real(random.randrange(len(self.real_df)))

    def __getitem__(self, idx: int):
        if random.random() < self.pseudo_prob:
            wav, labels = self._load_pseudo(random.randrange(self.n_pseudo))
        else:
            wav, labels = self._load_real(idx)

        wav = self._aug_wave(wav)

        if self.augment and random.random() < self.mixup_p:
            wav2, labels2 = self._draw_random_pair()
            wav2 = self._aug_wave(wav2)
            wav = wav + wav2                            # element-wise sum
            labels = np.clip(labels + labels2, 0.0, 1.0)
            # Optional gentle peak-normalize to keep mel within sane range.
            peak = float(np.abs(wav).max())
            if peak > 1.0:
                wav = wav / peak

        mel = waveform_to_mel(wav)
        return mel, torch.from_numpy(labels.astype(np.float32))


# ── Trainer ───────────────────────────────────────────────────────────────────

def train_one_fold(
    fold: int,
    backbone: str,
    epochs: int,
    seed: int,
    mixstyle_p: float,
    pseudo_prob: float,
    mixup_p: float,
    pseudo_npz: Path,
    pseudo_audio_dir: Path,
    init_from: Path | None,
    smoke_test: bool,
    val_cache: tuple | None = None,
) -> tuple:
    set_seed(seed + fold)

    device = torch.device("cuda")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cudnn.benchmark = True

    df       = pd.read_csv(PROC / "train_folds.csv")
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    if smoke_test:
        train_df = train_df.head(BATCH_SIZE * 2)

    train_ds = SydorskyiNSDataset(
        train_df,
        pseudo_npz_path=pseudo_npz,
        pseudo_audio_dir=pseudo_audio_dir,
        augment=True,
        pseudo_prob=pseudo_prob,
        mixup_p=mixup_p,
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
    print(f"Fold {fold}: {len(train_ds)} clips/epoch (real focal pool), "
          f"{train_ds.n_pseudo} pseudo chunks available, "
          f"{len(train_dl)} batches/epoch", flush=True)

    sp2idx = get_species_index()
    if val_cache is None:
        print("Building soundscape validation set …", flush=True)
        val_mels, val_labels = build_soundscape_val(sp2idx)
        val_cache = (val_mels, val_labels)
    else:
        val_mels, val_labels = val_cache

    freq_mask = T.FrequencyMasking(freq_mask_param=27).to(device)
    time_mask = T.TimeMasking(time_mask_param=64).to(device)

    model = BirdSEDModelA1(
        backbone_name=backbone,
        mixstyle_p=mixstyle_p,
    ).to(device)

    if init_from is not None:
        _load_pretrained_backbone(model, init_from)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, eta_min=LR_MIN
    )
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    A1_NS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bb_tag = backbone.replace("/", "_").replace(".", "_")
    save_path = A1_NS_MODELS_DIR / f"a1_ns_syd_{bb_tag}_fold{fold}_seed{seed}.pt"

    best_auc = 0.0
    for epoch in range(1, epochs + 1):
        epoch_start  = time.time()
        model.train()
        running_loss = 0.0
        n_seen       = 0

        for batch_idx, (mels, labels) in enumerate(train_dl):
            mels   = mels.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if random.random() < SPEC_FREQ_MASK_PROB:
                mels = freq_mask(mels)
            if random.random() < SPEC_TIME_MASK_PROB:
                mels = time_mask(mels)

            optimizer.zero_grad()
            with autocast_ctx:
                out  = model(mels)
                loss = loss_fn(out["clip_logits"], labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += loss.item()
            n_seen       += 1

            if smoke_test and batch_idx >= 0:
                break

        scheduler.step()

        avg_loss = running_loss / max(n_seen, 1)
        val_auc = validate(
            model, val_mels, val_labels, device,
            max_batches=1 if smoke_test else None,
        )

        elapsed = int(time.time() - epoch_start)
        mins, s = divmod(elapsed, 60)

        best_marker = ""
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "state_dict": model.state_dict(),
                "best_auc":   best_auc,
                "epoch":      epoch,
                "backbone":   backbone,
                "fold":       fold,
                "seed":       seed,
                "init_from":  str(init_from) if init_from else None,
                "pseudo_npz": str(pseudo_npz),
            }, save_path)
            best_marker = " ★ BEST"

        print("=" * 40, flush=True)
        print(
            f"Fold {fold}  Epoch {epoch:2d}/{epochs}: "
            f"train_loss={avg_loss:.4f}  "
            f"val_roc_auc={val_auc:.4f}  "
            f"time={mins}m {s:02d}s  "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
            f"{best_marker}", flush=True,
        )
        print("=" * 40, flush=True)

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nFold {fold} complete. Best val ROC-AUC: {best_auc:.4f}", flush=True)
    print(f"Checkpoint → {save_path}\n", flush=True)
    return best_auc, save_path, val_cache


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Bundle 1 NS — Sydorskyi recipe")
    p.add_argument("--fold",       type=int, default=None)
    p.add_argument("--folds",      type=str, default=None,
                   help="Comma-separated list, e.g. 0,1,2,4")
    p.add_argument("--backbone",   type=str, default=config.BACKBONE)
    p.add_argument("--epochs",     type=int, default=25)
    p.add_argument("--seed",       type=int, default=getattr(config, "SEED", 42))
    p.add_argument("--mixstyle-p", type=float, default=0.5)

    p.add_argument("--pseudo-npz",       type=Path, required=True,
                   help="NPZ written by src/pseudo_emit_sydorskyi.py")
    p.add_argument("--pseudo-audio-dir", type=Path, required=True,
                   help="Audio root used at pseudo-emit time (relpaths in NPZ "
                        "resolve under this)")

    p.add_argument("--pseudo-prob", type=float, default=0.4,
                   help="Sydorskyi: 40% per train-step is from pseudo set")
    p.add_argument("--mixup-p",     type=float, default=0.5,
                   help="Audio-level MixUp probability (Sydorskyi sum + clip)")

    p.add_argument("--init-from",   type=Path, default=None,
                   help="Pretrained-backbone ckpt (Bundle 1 Phase 5.1 output)")

    p.add_argument("--smoke-test",  action="store_true")
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
            folds = [0, 1, 2, 4]
        epochs = args.epochs

    print("=" * 60, flush=True)
    print("Bundle 1 NS — Sydorskyi recipe", flush=True)
    print(f"  backbone        : {args.backbone}", flush=True)
    print(f"  folds           : {folds}", flush=True)
    print(f"  epochs          : {epochs}", flush=True)
    print(f"  pseudo_npz      : {args.pseudo_npz}", flush=True)
    print(f"  pseudo_audio_dir: {args.pseudo_audio_dir}", flush=True)
    print(f"  pseudo_prob     : {args.pseudo_prob}", flush=True)
    print(f"  mixup_p         : {args.mixup_p}", flush=True)
    print(f"  mixstyle_p      : {args.mixstyle_p}", flush=True)
    print(f"  init_from       : {args.init_from}", flush=True)
    print(f"  models →        : {A1_NS_MODELS_DIR}", flush=True)
    print("=" * 60, flush=True)

    val_cache = None
    fold_results = []
    t0 = time.time()
    for f in folds:
        best_auc, save_path, val_cache = train_one_fold(
            fold=f,
            backbone=args.backbone,
            epochs=epochs,
            seed=args.seed,
            mixstyle_p=args.mixstyle_p,
            pseudo_prob=args.pseudo_prob,
            mixup_p=args.mixup_p,
            pseudo_npz=args.pseudo_npz,
            pseudo_audio_dir=args.pseudo_audio_dir,
            init_from=args.init_from,
            smoke_test=args.smoke_test,
            val_cache=val_cache,
        )
        fold_results.append((f, best_auc, save_path))
        gc.collect()
        torch.cuda.empty_cache()

    elapsed = int(time.time() - t0)
    h, rem = divmod(elapsed, 3600)
    m, s   = divmod(rem, 60)

    print("=" * 60, flush=True)
    print(f"Sydorskyi NS run complete  total time: {h}h {m:02d}m {s:02d}s", flush=True)
    for f, auc, path in fold_results:
        print(f"  fold {f}: best val_roc_auc = {auc:.4f}  → {path.name}", flush=True)
    if len(fold_results) > 1:
        mean_auc = float(np.mean([a for _, a, _ in fold_results]))
        print(f"  mean fold val_roc_auc = {mean_auc:.4f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
