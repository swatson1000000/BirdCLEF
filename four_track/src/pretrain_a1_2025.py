"""L2 pretraining — A1 EffNet-B0 on BirdCLEF-2025 focal audio.

Forks `src/train_a1.py`'s mel pipeline, model, and loss but trains on the
**union of 2025 + 2026 class space** so the backbone sees broader acoustic
diversity. The classifier head is discarded by the finetune step (which
re-inits a 234-class head); only backbone/mixstyle/gem/att-conv-non-final
weights transfer.

Usage:
    # Smoke test (1 epoch, ~2 batches) — Stage 1 gate per new_plan.md §14.10.5
    python -u src/pretrain_a1_2025.py --epochs 1 --smoke-test

    # Full pretrain (10 epochs)
    python -u src/pretrain_a1_2025.py --epochs 10
"""

import argparse
import json
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
from sklearn.model_selection import GroupShuffleSplit
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
    SAMPLE_RATE, CHUNK_SAMPLES, N_MELS,
    BATCH_SIZE, WEIGHT_DECAY,
    SPEC_TIME_MASK_PROB, SPEC_FREQ_MASK_PROB,
    GAIN_PROB, GAIN_DB_RANGE, TIME_SHIFT_PROB,
)
from utils import load_audio, pad_or_crop, waveform_to_mel  # noqa: E402

from model_a1 import BirdSEDModelA1  # noqa: E402

# ── 2025 data layout ──────────────────────────────────────────────────────────
DATA_2025      = ROOT / "data" / "raw" / "birdclef_2025"
TRAIN_AUDIO_25 = DATA_2025 / "train_audio"
TAX_2025       = DATA_2025 / "taxonomy.csv"
TRAIN_CSV_25   = DATA_2025 / "train.csv"

TAX_2026       = ROOT / "data" / "raw" / "taxonomy.csv"

# ── Output paths ──────────────────────────────────────────────────────────────
PRETRAIN_DIR   = FT_ROOT / "models" / "a1_pretrained_2025"
UNION_JSON     = FT_ROOT / "data" / "processed" / "union_2025_2026_classes.json"


# ── Union class space ─────────────────────────────────────────────────────────

def build_union_class_list() -> list:
    """Return sorted union of 2025 + 2026 primary_label species; cache to JSON."""
    if UNION_JSON.exists():
        with UNION_JSON.open() as f:
            return json.load(f)

    if not TAX_2025.exists():
        sys.exit(f"Missing {TAX_2025}. Extract birdclef-2025.zip first.")
    if not TAX_2026.exists():
        sys.exit(f"Missing {TAX_2026}.")

    sp_25 = set(pd.read_csv(TAX_2025)["primary_label"].astype(str))
    sp_26 = set(pd.read_csv(TAX_2026)["primary_label"].astype(str))
    union = sorted(sp_25 | sp_26)

    UNION_JSON.parent.mkdir(parents=True, exist_ok=True)
    with UNION_JSON.open("w") as f:
        json.dump(union, f)
    print(f"  Built union class list: |2025|={len(sp_25)}  |2026|={len(sp_26)}  "
          f"|union|={len(union)}  → {UNION_JSON}", flush=True)
    return union


# ── 2025 dataset ──────────────────────────────────────────────────────────────

class Pretrain2025Dataset(Dataset):
    """BirdCLEF-2025 focal-clip dataset over the 2025+2026 union class space.

    Mirrors `BirdTrainDataset` (parent BirdCLEF/src/dataset.py) but:
      - reads audio from `data/raw/birdclef_2025/train_audio/`
      - targets are one-hot over the union class space
      - secondary labels NOT used (2025's `train.csv` has them but we keep
        the pretrain simple — secondary handling is a finetune-only nuance)
      - no MixUp, no background noise (additive complexity not needed for
        a pretrain whose head will be discarded)
    """

    def __init__(self, df: pd.DataFrame, sp2idx: dict, augment: bool = True):
        self.df        = df.reset_index(drop=True)
        self.sp2idx    = sp2idx
        self.n_classes = len(sp2idx)
        self.augment   = augment

    def __len__(self) -> int:
        return len(self.df)

    def _load_waveform(self, row) -> np.ndarray:
        path = TRAIN_AUDIO_25 / str(row["filename"])
        wav  = load_audio(path)
        return pad_or_crop(wav, CHUNK_SAMPLES, random_crop=self.augment)

    def _build_targets(self, row) -> np.ndarray:
        labels = np.zeros(self.n_classes, dtype=np.float32)
        primary = str(row["primary_label"])
        if primary in self.sp2idx:
            labels[self.sp2idx[primary]] = 1.0
        return labels

    def _augment(self, wav: np.ndarray) -> np.ndarray:
        if random.random() < GAIN_PROB:
            db  = random.uniform(-GAIN_DB_RANGE, GAIN_DB_RANGE)
            wav = wav * (10.0 ** (db / 20.0))
        if random.random() < TIME_SHIFT_PROB:
            max_shift = int(0.25 * len(wav))
            wav = np.roll(wav, random.randint(-max_shift, max_shift))
        return wav

    def __getitem__(self, idx: int):
        row    = self.df.iloc[idx]
        wav    = self._load_waveform(row)
        labels = self._build_targets(row)

        if self.augment:
            wav = self._augment(wav)

        mel = waveform_to_mel(wav)
        mask = np.ones(self.n_classes, dtype=np.float32)  # no secondary mask in pretrain
        return mel, torch.from_numpy(labels), torch.from_numpy(mask)


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: nn.Module, loader: DataLoader, device: torch.device,
    max_batches: int | None = None,
) -> tuple:
    """Macro ROC-AUC on the held-out 2025 val split."""
    model.eval()
    all_probs, all_labels = [], []
    for i, (mels, labels, _mask) in enumerate(loader):
        mels = mels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(mels)
        all_probs.append(torch.sigmoid(out["clip_logits"]).float().cpu().numpy())
        all_labels.append(labels.numpy())
        if max_batches is not None and i + 1 >= max_batches:
            break

    probs  = np.concatenate(all_probs,  axis=0)
    labels = np.concatenate(all_labels, axis=0)
    present = labels.sum(axis=0) > 0
    n_present = int(present.sum())
    if n_present == 0:
        return 0.0, 0
    try:
        auc = float(roc_auc_score(
            labels[:, present], probs[:, present], average="macro"
        ))
    except ValueError:
        auc = 0.0
    return auc, n_present


# ── Training ──────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    p = argparse.ArgumentParser(description="L2 — A1 pretrain on BirdCLEF-2025")
    p.add_argument("--epochs",     type=int, default=10)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--lr-min",     type=float, default=1e-6)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--mixstyle-p", type=float, default=0.5)
    p.add_argument("--val-frac",   type=float, default=0.05,
                   help="Stratified val fraction off 2025 train.csv")
    p.add_argument("--smoke-test", action="store_true",
                   help="1 epoch, 2 train batches, 1 val batch — wiring check")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cudnn.benchmark = True

    # ── Class space ───────────────────────────────────────────────────────────
    union = build_union_class_list()
    sp2idx_union = {sp: i for i, sp in enumerate(union)}
    n_classes_union = len(union)

    # ── 2025 train.csv → split ────────────────────────────────────────────────
    if not TRAIN_CSV_25.exists():
        sys.exit(f"Missing {TRAIN_CSV_25}. Extract birdclef-2025.zip first.")
    df = pd.read_csv(TRAIN_CSV_25)
    df["primary_label"] = df["primary_label"].astype(str)
    # Drop rows whose primary_label is not in the union (defensive — should not happen)
    df = df[df["primary_label"].isin(sp2idx_union)].reset_index(drop=True)
    # Group-based split by `author` (recordist) prevents same-recordist clips
    # from appearing on both sides — random stratified split caused leaky val
    # (observed: 0.99 pretrain val AUC → 0.66 downstream finetune val).
    df["author"] = df["author"].fillna("__unknown__").astype(str)
    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_frac,
                            random_state=args.seed)
    tr_idx, vl_idx = next(gss.split(df, groups=df["author"]))
    train_df = df.iloc[tr_idx].reset_index(drop=True)
    val_df   = df.iloc[vl_idx].reset_index(drop=True)
    n_val_species  = val_df["primary_label"].nunique()
    n_shared_auth  = len(set(train_df["author"]) & set(val_df["author"]))
    print(f"  [split] grouped by author: "
          f"train={len(train_df)} ({train_df['author'].nunique()} authors)  "
          f"val={len(val_df)} ({val_df['author'].nunique()} authors)  "
          f"val_species={n_val_species}  shared_authors={n_shared_auth}",
          flush=True)

    if args.smoke_test:
        train_df = train_df.head(args.batch_size * 2)
        val_df   = val_df.head(args.batch_size)

    print("=" * 60, flush=True)
    print("L2 pretrain — BirdCLEF-2025 → A1 EffNet-B0 backbone", flush=True)
    print(f"  union classes  : {n_classes_union}  "
          f"(2025+2026; head is discarded by finetune)", flush=True)
    print(f"  train clips    : {len(train_df)}", flush=True)
    print(f"  val   clips    : {len(val_df)}", flush=True)
    print(f"  epochs         : {args.epochs}", flush=True)
    print(f"  lr / lr_min    : {args.lr} / {args.lr_min}", flush=True)
    print(f"  batch / nworker: {args.batch_size} / {args.num_workers}", flush=True)
    print(f"  mixstyle_p     : {args.mixstyle_p}", flush=True)
    print(f"  smoke_test     : {args.smoke_test}", flush=True)
    print(f"  → ckpt         : {PRETRAIN_DIR}", flush=True)
    print("=" * 60, flush=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = Pretrain2025Dataset(train_df, sp2idx_union, augment=True)
    val_ds   = Pretrain2025Dataset(val_df,   sp2idx_union, augment=False)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0 if args.smoke_test else args.num_workers,
        pin_memory=True, drop_last=True,
        persistent_workers=not args.smoke_test,
        multiprocessing_context=None if args.smoke_test else "spawn",
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0 if args.smoke_test else max(2, args.num_workers // 2),
        pin_memory=True, drop_last=False,
        persistent_workers=not args.smoke_test,
        multiprocessing_context=None if args.smoke_test else "spawn",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    freq_mask = T.FrequencyMasking(freq_mask_param=27).to(device)
    time_mask = T.TimeMasking(time_mask_param=64).to(device)

    model = BirdSEDModelA1(
        backbone_name=config.BACKBONE,
        n_classes=n_classes_union,
        mixstyle_p=args.mixstyle_p,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr_min,
    )
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PRETRAIN_DIR / "a1_pretrained_2025.pt"

    # ── Training loop ─────────────────────────────────────────────────────────
    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
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
                loss_per = loss_fn(out["clip_logits"], labels)
                loss     = (loss_per * sec_mask).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += loss.item()
            n_seen       += 1

            if args.smoke_test and batch_idx + 1 >= 2:
                break

        scheduler.step()

        avg_loss = running_loss / max(n_seen, 1)
        val_auc, n_present = validate(
            model, val_dl, device,
            max_batches=1 if args.smoke_test else None,
        )

        elapsed = int(time.time() - epoch_start)
        mins, s = divmod(elapsed, 60)

        best_marker = ""
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "state_dict":   model.state_dict(),
                "union_classes": union,
                "epoch":        epoch,
                "val_auc":      val_auc,
            }, save_path)
            best_marker = " ★ BEST"

        print("=" * 40, flush=True)
        print(
            f"L2 Pretrain  Epoch {epoch:2d}/{args.epochs}: "
            f"train_loss={avg_loss:.4f}  "
            f"val_roc_auc={val_auc:.4f} (n_present={n_present})  "
            f"time={mins}m {s:02d}s  "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
            f"{best_marker}",
            flush=True,
        )
        print("=" * 40, flush=True)

    print(f"\nL2 pretrain complete. Best val ROC-AUC: {best_auc:.4f}", flush=True)
    print(f"Checkpoint → {save_path}\n", flush=True)


if __name__ == "__main__":
    main()
