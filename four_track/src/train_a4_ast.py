"""Track A4 — AST (Audio Spectrogram Transformer) fine-tune on BC2026.

A different architecture family from the EffNet-based A1/A2 (Track A4 sits
parallel to A1 in the four-track plan). Pretrained on AudioSet
(MIT/ast-finetuned-audioset-10-10-0.4593, 86.6M params), with the 527-class
AudioSet head replaced by a 234-class BC2026 head.

Inputs are AST-format kaldi fbank features (1024 frames × 128 mels) on a 10s
window at 16 kHz, computed inside the DataLoader workers via
torchaudio.compliance.kaldi.fbank.

Designed for a single-fold gate probe (fold 0 → broader-pool OOF vs A2 fold-0
anchor 0.8094). 5-fold scaling is gated on the result, not the default.

Usage:
  python -u src/train_a4_ast.py --fold 0 --epochs 10 \\
    --pseudo-manifest data/processed/a2_pseudo_manifest.csv \\
    --batch-size 16 --lr 5e-5
"""

from __future__ import annotations

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
import torchaudio
import torchaudio.compliance.kaldi as ta_kaldi
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
FT_ROOT = HERE.parent
ROOT = FT_ROOT.parent

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config  # parent BirdCLEF/src/config.py
from config import RAW, PROC, SAMPLE_RATE, CHUNK_SAMPLES, get_species_index  # noqa: E402
from utils import load_audio, pad_or_crop  # noqa: E402
from dataset_a1 import BirdTrainDatasetA1  # noqa: E402
from losses import AsymmetricLossOptimized  # noqa: E402
from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification  # noqa: E402


# ── AST constants ──────────────────────────────────────────────────────────────
AST_MODEL_ID  = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_SR        = 16_000
AST_WIN_S     = 10            # AST native 10s window
AST_WIN_LEN   = AST_SR * AST_WIN_S   # 160,000 samples post-resample
AST_N_MELS    = 128
AST_T_FRAMES  = 1024
# AST normalization stats (from AudioSet pretraining)
AST_MEAN      = -4.2677393
AST_STD       = 4.5689974

A4_MODELS_DIR = FT_ROOT / "models" / "a4"

# Training defaults (single-fold first cut)
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR         = 5e-5
DEFAULT_WD         = 1e-4
DEFAULT_EPOCHS     = 10
WARMUP_EPOCHS      = 1


# ── Audio → AST features ──────────────────────────────────────────────────────

_resampler_32k_to_16k = torchaudio.transforms.Resample(
    orig_freq=SAMPLE_RATE, new_freq=AST_SR
)


def wav_32k_to_ast_features(wav: np.ndarray, random_crop: bool) -> torch.Tensor:
    """Convert a (CHUNK_SAMPLES,) 32kHz numpy waveform to AST (T=1024, F=128)
    fbank features ready for ASTForAudioClassification.

    Steps:
      1. Crop / random-crop to a 10-second 32kHz slice.
      2. Resample 32k → 16k.
      3. Apply kaldi fbank (matches AST pretrain settings: 128 mels,
         25 ms window, 10 ms hop, hanning window).
      4. Pad/truncate to 1024 frames.
      5. Normalize with AST mean/std.
    """
    target_32k_len = AST_SR * AST_WIN_S * (SAMPLE_RATE // AST_SR)  # = 320,000
    if len(wav) > target_32k_len:
        if random_crop:
            offset = random.randint(0, len(wav) - target_32k_len)
        else:
            offset = (len(wav) - target_32k_len) // 2
        wav = wav[offset: offset + target_32k_len]
    elif len(wav) < target_32k_len:
        # Tile to fill
        reps = -(-target_32k_len // len(wav))
        wav = np.tile(wav, reps)[:target_32k_len]

    wav_t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    wav_t = _resampler_32k_to_16k(wav_t)  # (1, 160_000)

    fbank = ta_kaldi.fbank(
        wav_t,
        sample_frequency=AST_SR,
        window_type="hanning",
        num_mel_bins=AST_N_MELS,
        frame_length=25.0,
        frame_shift=10.0,
    )  # (n_frames ~ 998, 128)

    if fbank.shape[0] < AST_T_FRAMES:
        pad = AST_T_FRAMES - fbank.shape[0]
        fbank = nn.functional.pad(fbank, (0, 0, 0, pad))
    else:
        fbank = fbank[:AST_T_FRAMES]

    fbank = (fbank - AST_MEAN) / (AST_STD * 2)
    return fbank  # (1024, 128)


# ── Dataset ───────────────────────────────────────────────────────────────────

class BirdTrainDatasetA4(BirdTrainDatasetA1):
    """Same row-loading semantics as A1 (focal + AnuraSet + BC2025 +
    BC2026_SS_PSEUDO) but returns AST features instead of (3, 224, 512) mels.
    Mixup/augmentation hooks reuse parent BirdTrainDataset's CPU-side logic.
    """

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        wav = self._load_waveform(row)
        labels, mask = self._build_targets(row)

        if self.augment:
            wav = self._apply_gain(wav)
            wav = self._apply_time_shift(wav)
            wav = self._add_bg_noise(wav)
            wav, labels, mask = self._mixup(
                wav, labels, mask, random.randint(0, len(self) - 1)
            )

        input_values = wav_32k_to_ast_features(wav, random_crop=self.augment)
        return (
            input_values,                   # (1024, 128)
            torch.from_numpy(labels),       # (234,)
            torch.from_numpy(mask),         # (234,)
        )


def build_ast_soundscape_val(sp2idx: dict) -> tuple:
    """1478-window labeled soundscape pool, AST-feature-formatted.
    Mirrors train_a1.build_soundscape_val but produces AST inputs.
    """
    df = pd.read_csv(RAW / "train_soundscapes_labels.csv")
    n_classes = config.N_CLASSES
    soundsc_dir = RAW / "train_soundscapes"

    def _parse_time(s: str) -> int:
        h, m, sec = str(s).split(":")
        return int(h) * 3600 + int(m) * 60 + int(sec)

    val_feats = []
    val_labels = np.zeros((len(df), n_classes), dtype=np.float32)

    for i, row in df.iterrows():
        t_start = _parse_time(row["start"])
        t_end = _parse_time(row["end"])
        path = soundsc_dir / str(row["filename"])
        try:
            wav = load_audio(path)
            s, e = int(t_start * SAMPLE_RATE), int(t_end * SAMPLE_RATE)
            segment = wav[s:e] if e <= len(wav) else wav[s:]
            segment = pad_or_crop(segment, CHUNK_SAMPLES, random_crop=False)
            feats = wav_32k_to_ast_features(segment, random_crop=False)
        except Exception as ex:
            print(f"  [warn] skipping {row['filename']} @ {t_start}s: {ex}",
                  flush=True)
            feats = torch.zeros(AST_T_FRAMES, AST_N_MELS)
        val_feats.append(feats)

        for sp in str(row["primary_label"]).split(";"):
            sp = sp.strip()
            if sp in sp2idx:
                val_labels[i, sp2idx[sp]] = 1.0

    return val_feats, val_labels


@torch.no_grad()
def validate_ast(
    model: nn.Module,
    val_feats: list,
    val_labels: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
    max_batches: int | None = None,
) -> float:
    model.eval()
    all_probs = []
    n_done = 0
    for i in range(0, len(val_feats), batch_size):
        batch = torch.stack(val_feats[i: i + batch_size]).to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_values=batch)
        all_probs.append(torch.sigmoid(out.logits).float().cpu().numpy())
        n_done += 1
        if max_batches is not None and n_done >= max_batches:
            break
    probs = np.concatenate(all_probs, axis=0)
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


# ── Model ─────────────────────────────────────────────────────────────────────

def make_ast_model(num_classes: int = 234) -> nn.Module:
    """Load AST pretrained + replace 527-class head with a num_classes head."""
    model = ASTForAudioClassification.from_pretrained(AST_MODEL_ID)
    in_dim = model.classifier.dense.in_features
    model.classifier.dense = nn.Linear(in_dim, num_classes)
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_fold(
    fold: int,
    epochs: int,
    seed: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    smoke_test: bool,
    pseudo_manifest: Path | None,
) -> tuple:
    set_seed(seed + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # ── Data ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(PROC / "train_folds.csv")
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    if smoke_test:
        train_df = train_df.head(batch_size * 2)

    if pseudo_manifest is not None and not smoke_test:
        pseudo_df = pd.read_csv(pseudo_manifest)
        for col in train_df.columns:
            if col not in pseudo_df.columns:
                pseudo_df[col] = np.nan
        all_cols = list(train_df.columns) + (
            ["pseudo_positive_labels", "pseudo_window_start"]
            if "pseudo_positive_labels" not in train_df.columns
            else []
        )
        pseudo_df = pseudo_df[[c for c in all_cols if c in pseudo_df.columns]]
        train_df = pd.concat([train_df, pseudo_df], ignore_index=True)
        print(f"  [pseudo] +{len(pseudo_df)} BC2026_SS_PSEUDO rows  "
              f"(total train={len(train_df)})", flush=True)

    train_ds = BirdTrainDatasetA4(train_df, augment=True)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if smoke_test else config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=not smoke_test,
        multiprocessing_context=None if smoke_test else "spawn",
    )
    print(f"Fold {fold}: {len(train_ds)} clips, {len(train_dl)} batches/epoch",
          flush=True)

    sp2idx = get_species_index()
    print("Building AST soundscape val …", flush=True)
    t0 = time.time()
    val_feats, val_labels = build_ast_soundscape_val(sp2idx)
    n_present = int((val_labels.sum(axis=0) > 0).sum())
    print(f"  {len(val_feats)} val segments, {n_present} species present  "
          f"({time.time()-t0:.1f}s)", flush=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Loading AST pretrained ({AST_MODEL_ID}) …", flush=True)
    model = make_ast_model(num_classes=config.N_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  AST params: {n_params/1e6:.1f} M", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - WARMUP_EPOCHS, 1), eta_min=lr * 0.05
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched],
        milestones=[WARMUP_EPOCHS],
    )

    loss_fn = AsymmetricLossOptimized(reduction="none")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    A4_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = A4_MODELS_DIR / f"a4_ast_fold{fold}_seed{seed}_asl.pt"

    # ── Training loop ─────────────────────────────────────────────────────────
    best_auc = 0.0
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        n_seen = 0

        for batch_idx, (feats, labels, sec_mask) in enumerate(train_dl):
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            sec_mask = sec_mask.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast_ctx:
                out = model(input_values=feats)
                loss_per = loss_fn(out.logits, labels)
                loss = (loss_per * sec_mask).mean()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            n_seen += 1

            if smoke_test and batch_idx >= 0:
                break

        scheduler.step()

        avg_loss = running_loss / max(n_seen, 1)
        val_auc = validate_ast(
            model, val_feats, val_labels, device,
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

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nFold {fold} complete. Best val ROC-AUC: {best_auc:.4f}", flush=True)
    print(f"Checkpoint → {save_path}\n", flush=True)
    return best_auc, save_path


def main() -> None:
    p = argparse.ArgumentParser(description="Track A4 AST fine-tune (single-fold gate)")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WD)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--pseudo-manifest", type=str, default=None,
                   help="CSV of BC2026_SS_PSEUDO rows (A2 pseudo manifest)")
    args = p.parse_args()

    t0 = time.time()
    print(f"AST run: fold={args.fold} epochs={args.epochs} seed={args.seed} "
          f"bs={args.batch_size} lr={args.lr} wd={args.weight_decay} "
          f"smoke_test={args.smoke_test} "
          f"pseudo={'yes' if args.pseudo_manifest else 'no'}",
          flush=True)

    best_auc, save_path = train_one_fold(
        fold=args.fold,
        epochs=args.epochs,
        seed=args.seed,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        smoke_test=args.smoke_test,
        pseudo_manifest=Path(args.pseudo_manifest) if args.pseudo_manifest else None,
    )

    print("=" * 60, flush=True)
    print(f"Track A4 (AST) run complete  total time: "
          f"{int((time.time()-t0)/3600)}h "
          f"{int((time.time()-t0)%3600/60)}m "
          f"{int((time.time()-t0)%60)}s",
          flush=True)
    print(f"  fold {args.fold}: best val_roc_auc = {best_auc:.4f}  → {save_path.name}",
          flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
