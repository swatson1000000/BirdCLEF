"""iNatSounds 2024 pretrain — A1 EffNet-B0 backbone on multi-taxa iNat audio.

Per Bundle 2 plan §14.22.9.5 step 5 (corrected mechanism: transfer-learning
hypothesis, NOT direct co-training — 89% of clips are for species NOT in
BC2026, so we're betting on encoder-level acoustic-feature transfer).

Differences vs pretrain_l2_redux.py (the closest sibling):
- Manifest source: `inat_manifest.csv` produced by build_inat_manifest.py
  (loaded directly; we don't re-walk the iNat directory tree).
- Class space: 5,569 iNat species via sorted(scientific_name) — NOT
  BC2026 primary_label. Backbone weights transfer; head is discarded.
- Split: iNat's official train/val (137K / 45K clips). NOT GroupShuffleSplit
  by author — iNat's split is already curated and we trust it.
- Sampling: WeightedRandomSampler with 1/n_per_class weights. CRITICAL
  for non-Aves balance — without it the encoder sees 82% Aves clips and
  the +0.27 Insecta probe-v3 ceiling becomes structurally unreachable.
- Audio root: passed via --inat-root; files at {inat_root}/{file_path}.

Usage on deepthought:
    # Wiring smoke test (1 ep, 2 train batches, 1 val batch)
    python -u src/pretrain_inat_sounds.py \\
        --inat-root /home/swatson/work/kaggle/BirdCLEF/four_track/data/external/inat_sounds_2024 \\
        --epochs 1 --smoke-test

    # Phase 5 — full 50-epoch pretrain (Bundle 2 dispatch)
    python -u src/pretrain_inat_sounds.py \\
        --inat-root /home/swatson/work/kaggle/BirdCLEF/four_track/data/external/inat_sounds_2024 \\
        --epochs 50

Output ckpt at PRETRAIN_DIR / 'inat_best_<backbone>.pt' with iNat species
list embedded as a buffer for downstream verification.
"""

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ── Path wiring (mirrors pretrain_l2_redux.py) ────────────────────────────────
FT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FT_ROOT.parent / "src"))   # parent BirdCLEF/src for config

import config  # noqa: E402
from config import (  # noqa: E402
    BACKBONE, BATCH_SIZE, CHUNK_SAMPLES, GAIN_DB_RANGE, GAIN_PROB,
    SPEC_FREQ_MASK_PROB, SPEC_TIME_MASK_PROB, TIME_SHIFT_PROB, WEIGHT_DECAY,
)

sys.path.insert(0, str(FT_ROOT / "src"))  # four_track/src for our model + utils
from utils import load_audio, pad_or_crop, waveform_to_mel  # noqa: E402
from model_a1 import BirdSEDModelA1  # noqa: E402


PRETRAIN_DIR = FT_ROOT / "models" / "pretrain_inat"
SPECIES_JSON = FT_ROOT / "data" / "processed" / "inat_species.json"


# ── Manifest + species list ──────────────────────────────────────────────────

def load_inat_manifest(inat_root: Path) -> pd.DataFrame:
    """Load the manifest produced by build_inat_manifest.py.

    Returns DataFrame with columns:
        split, file_path, inat_cat_id, scientific_name, common_name,
        tax_class, kingdom, phylum, order, family, genus,
        bc2026_primary_label, bc2026_class_name, audio_dir_name
    """
    manifest_path = inat_root / "inat_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest missing: {manifest_path} — run build_inat_manifest.py first"
        )
    df = pd.read_csv(manifest_path, dtype=str)
    df["abs_path"] = df["file_path"].apply(lambda p: str(inat_root / p))
    print(f"  [manifest] loaded {len(df)} rows from {manifest_path}", flush=True)
    return df


def build_species_list(manifest: pd.DataFrame) -> list:
    """Sorted union of iNat scientific names. Cached to JSON for finetune."""
    SPECIES_JSON.parent.mkdir(parents=True, exist_ok=True)
    species = sorted(manifest["scientific_name"].astype(str).unique())
    with open(SPECIES_JSON, "w") as f:
        json.dump(species, f)
    print(f"  [species] {len(species)} unique iNat species → {SPECIES_JSON}",
          flush=True)
    return species


# ── Dataset ──────────────────────────────────────────────────────────────────

class INatSoundsDataset(Dataset):
    """iNat 2024 audio clips for B0 encoder pretrain.

    Single-label (annotations are 1:1 with audio_id per the COCO-format JSON).
    No secondary labels, no MixUp, no background noise — minimal design,
    matching pretrain_l2_redux.

    NOTE: iNat clips are .wav, sample rates vary by recording. `load_audio`
    handles resampling to config.SAMPLE_RATE (32 kHz).
    """

    def __init__(self, df: pd.DataFrame, sp2idx: dict, augment: bool = True,
                 mixup_prob: float = 0.0,
                 mel_cache_dir: "Path | None" = None,
                 mel_cache_split: str = "train"):
        self.df = df.reset_index(drop=True)
        self.sp2idx = sp2idx
        self.n_classes = len(sp2idx)
        self.augment = augment
        self.mixup_prob = mixup_prob
        self.mel_cache_dir = mel_cache_dir
        self.mel_cache_split = mel_cache_split

    def __len__(self) -> int:
        return len(self.df)

    def _load_waveform(self, row) -> np.ndarray:
        path = Path(row["abs_path"])
        try:
            wav = load_audio(path)
            if wav.size == 0:
                raise ValueError("zero-length waveform after load_audio")
            return pad_or_crop(wav, CHUNK_SAMPLES, random_crop=self.augment)
        except Exception as exc:
            print(f"  [skip-bad] {path.name}: {type(exc).__name__}: {exc}",
                  flush=True)
            return np.zeros(CHUNK_SAMPLES, dtype=np.float32)

    def _build_targets(self, row) -> np.ndarray:
        labels = np.zeros(self.n_classes, dtype=np.float32)
        sci = str(row["scientific_name"])
        if sci in self.sp2idx:
            labels[self.sp2idx[sci]] = 1.0
        return labels

    def _augment(self, wav: np.ndarray) -> np.ndarray:
        if random.random() < GAIN_PROB:
            db = random.uniform(-GAIN_DB_RANGE, GAIN_DB_RANGE)
            wav = wav * (10.0 ** (db / 20.0))
        if random.random() < TIME_SHIFT_PROB:
            max_shift = int(0.25 * len(wav))
            wav = np.roll(wav, random.randint(-max_shift, max_shift))
        return wav

    def _mixup(self, wav1: np.ndarray, labels1: np.ndarray, idx2: int) -> tuple:
        row2 = self.df.iloc[idx2]
        wav2 = self._load_waveform(row2)
        labels2 = self._build_targets(row2)
        w1 = wav1 / (np.abs(wav1).max() + 1e-8)
        w2 = wav2 / (np.abs(wav2).max() + 1e-8)
        mixed = 0.5 * w1 + 0.5 * w2
        labels = np.maximum(labels1, labels2)
        return mixed.astype(np.float32, copy=False), labels

    def _load_cached_mel(self, idx: int) -> "torch.Tensor | None":
        """Return the pre-cached mel for this row, or None if cache miss.

        Skips wav-level augmentation (gain, time-shift) and MixUp — those are
        traded for the speed of a single fp16 mel read. Spec augmentation
        (freq/time masking) is still applied per-batch on GPU.
        """
        if self.mel_cache_dir is None:
            return None
        cache_path = self.mel_cache_dir / f"{self.mel_cache_split}_{idx:07d}.npy"
        if not cache_path.exists():
            return None
        mel_np = np.load(cache_path)  # fp16
        return torch.from_numpy(mel_np.astype(np.float32, copy=False))

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        labels = self._build_targets(row)
        mask = np.ones(self.n_classes, dtype=np.float32)

        # Fast path: read pre-cached mel directly. Skips wav augmentation and
        # MixUp; only spec augmentation (applied on GPU in train loop) remains.
        cached_mel = self._load_cached_mel(idx)
        if cached_mel is not None:
            return cached_mel, torch.from_numpy(labels), torch.from_numpy(mask)

        # Slow path: load audio, augment, compute mel from scratch.
        wav = self._load_waveform(row)
        if self.augment:
            wav = self._augment(wav)
            if self.mixup_prob > 0.0 and random.random() < self.mixup_prob:
                wav, labels = self._mixup(wav, labels, random.randint(0, len(self) - 1))
        mel = waveform_to_mel(wav)
        return mel, torch.from_numpy(labels), torch.from_numpy(mask)


def make_balanced_sampler(df: pd.DataFrame, sp2idx: dict) -> WeightedRandomSampler:
    """1/n_per_class weighted sampling. Required for iNat's 82% Aves bias.

    With uniform sampling, Insecta classes (745/5569 = 13% by species but
    only 6.7% by clip count) would see ~5x fewer epoch samples than Aves.
    This sampler equalizes per-species expected draws.
    """
    counts = df["scientific_name"].value_counts()
    per_row_weight = df["scientific_name"].map(lambda s: 1.0 / counts[s]).to_numpy()
    # Treat as class-balanced: each species drawn equiprobably, regardless
    # of clip count. The 1/n weighting achieves this exactly.
    return WeightedRandomSampler(
        weights=per_row_weight, num_samples=len(df), replacement=True,
    )


# ── Validation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device,
             max_batches: int | None = None) -> tuple:
    """Macro ROC-AUC on iNat val split."""
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

    probs = np.concatenate(all_probs, axis=0)
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


# ── Loss ─────────────────────────────────────────────────────────────────────

def focal_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor,
                          gamma: float = 2.0) -> torch.Tensor:
    """Per-element focal-BCE with logits (no reduction). Same as L2-redux."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = targets * p + (1 - targets) * (1 - p)
    focal_weight = (1 - pt) ** gamma
    return focal_weight * bce


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    p = argparse.ArgumentParser(description="iNatSounds 2024 — B0 encoder pretrain")
    p.add_argument("--inat-root", type=Path, required=True,
                   help="Directory containing inat_manifest.csv + train/ + val/")
    p.add_argument("--epochs", type=int, default=50,
                   help="50 for full Bundle 2 pretrain; 1 for smoke")
    p.add_argument("--lr", type=float, default=2.5e-4,
                   help="Sydorskyi-style lr (vs L2 attempt 1's 1e-3)")
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixstyle-p", type=float, default=0.5)
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Focal-BCE γ; set 0 for plain BCE")
    p.add_argument("--save-every", type=int, default=10,
                   help="Save backbone-only ckpt every N epochs (default 10)")
    p.add_argument("--smoke-test", action="store_true",
                   help="1 epoch, 2 train batches, 1 val batch — wiring check")
    p.add_argument("--backbone", type=str, default=BACKBONE,
                   help="timm backbone name (default config.BACKBONE = B0)")
    p.add_argument("--resume", type=Path, default=None,
                   help="Path to a prior ckpt; loads state_dict and best_auc")
    p.add_argument("--start-epoch", type=int, default=None,
                   help="Override resume epoch (default: ckpt['epoch']). "
                        "Loop runs (start_epoch + 1) ... epochs inclusive.")
    p.add_argument("--natural-sampling", action="store_true",
                   help="Disable WeightedRandomSampler (1/n_per_class balanced) and "
                        "use natural shuffle. Better matches BC2026 finetune target "
                        "frequency distribution (Aves-heavy).")
    p.add_argument("--mixup-prob", type=float, default=0.0,
                   help="Apply A1-style 0.5/0.5 waveform MixUp with element-wise-max "
                        "labels at this probability per sample. Set 0 to disable "
                        "(backward compat). Disabled automatically on cache hits.")
    p.add_argument("--mel-cache-dir", type=Path, default=None,
                   help="Pre-cached fp16 mels (one .npy per row). When set, the "
                        "train dataset reads cached mels at split-and-index "
                        "{cache_dir}/train_{idx:07d}.npy, bypassing audio load + "
                        "mel-compute. Val falls back to raw audio unless val "
                        "files also exist. Generate the cache via "
                        "src/build_inat_mel_cache.py.")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cudnn.benchmark = True

    # ── Class space ──────────────────────────────────────────────────────────
    manifest = load_inat_manifest(args.inat_root)
    species = build_species_list(manifest)
    sp2idx = {sp: i for i, sp in enumerate(species)}
    n_classes = len(species)

    train_df = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_df = manifest[manifest["split"] == "val"].reset_index(drop=True)

    # Per-tax-class breakdown for the log.
    tax_class_dist = train_df["tax_class"].value_counts().to_dict()

    if args.smoke_test:
        # Override batch_size so train forward+backward actually runs on the
        # tiny partition. Original spec is "1-step on 8 clips"; we use bs=8
        # and 2 batches for both forward+backward exercise.
        args.batch_size = 8
        # Use a small balanced subset so the smoke covers Insecta + Amphibia +
        # Aves rather than only Aves (which would be most of a head-batch sample).
        # Need n_per_class * 5 (tax classes) > batch_size to actually fill batches
        # under WeightedRandomSampler num_samples = len(df).
        smoke_n_per_class = 5  # 25 train rows, 25 val rows — > batch_size=8
        train_df = (
            train_df.groupby("tax_class", group_keys=False)
            .apply(lambda g: g.head(smoke_n_per_class))
            .reset_index(drop=True)
        )
        val_df = (
            val_df.groupby("tax_class", group_keys=False)
            .apply(lambda g: g.head(smoke_n_per_class))
            .reset_index(drop=True)
        )

    print("=" * 60, flush=True)
    print(f"iNatSounds pretrain → backbone {args.backbone}", flush=True)
    print(f"  iNat species : {n_classes}", flush=True)
    print(f"  total clips  : {len(manifest)}", flush=True)
    by_tax = manifest["tax_class"].value_counts().to_dict()
    print(f"  by tax_class : " + "  ".join(f"{k}={v}" for k, v in sorted(by_tax.items())),
          flush=True)
    print(f"  train clips  : {len(train_df)}", flush=True)
    print(f"  val   clips  : {len(val_df)}", flush=True)
    print(f"  epochs       : {args.epochs}", flush=True)
    print(f"  lr / lr_min  : {args.lr} / {args.lr_min}", flush=True)
    print(f"  batch / nwk  : {args.batch_size} / {args.num_workers}", flush=True)
    print(f"  mixstyle_p   : {args.mixstyle_p}", flush=True)
    print(f"  focal_gamma  : {args.focal_gamma}", flush=True)
    print(f"  save_every   : {args.save_every}", flush=True)
    print(f"  smoke_test   : {args.smoke_test}", flush=True)
    print(f"  → ckpt       : {PRETRAIN_DIR}", flush=True)
    print("=" * 60, flush=True)

    # ── Datasets + samplers ──────────────────────────────────────────────────
    train_ds = INatSoundsDataset(train_df, sp2idx, augment=True,
                                  mixup_prob=args.mixup_prob,
                                  mel_cache_dir=args.mel_cache_dir,
                                  mel_cache_split="train")
    val_ds = INatSoundsDataset(val_df, sp2idx, augment=False,
                                mel_cache_dir=args.mel_cache_dir,
                                mel_cache_split="val")

    if args.natural_sampling:
        train_sampler = None
        sampler_desc = "natural shuffle (no class balancing)"
    else:
        train_sampler = make_balanced_sampler(train_df, sp2idx)
        sampler_desc = "WeightedRandomSampler 1/n_per_class (balanced)"
    print(f"  sampler      : {sampler_desc}", flush=True)
    print(f"  mixup_prob   : {args.mixup_prob}", flush=True)
    print(f"  mel_cache_dir: {args.mel_cache_dir}", flush=True)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=0 if args.smoke_test else args.num_workers,
        pin_memory=True,
        drop_last=not args.smoke_test,
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

    # ── Model ────────────────────────────────────────────────────────────────
    freq_mask = T.FrequencyMasking(freq_mask_param=27).to(device)
    time_mask = T.TimeMasking(time_mask_param=64).to(device)

    model = BirdSEDModelA1(
        backbone_name=args.backbone,
        n_classes=n_classes,
        mixstyle_p=args.mixstyle_p,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr_min,
    )
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    bb_tag = args.backbone.replace("/", "_").replace(".", "_")
    save_tag_parts = []
    if args.natural_sampling:
        save_tag_parts.append("natfreq")
    if args.mixup_prob > 0.0:
        save_tag_parts.append(f"mixup{int(round(args.mixup_prob * 100))}")
    if args.mel_cache_dir is not None:
        save_tag_parts.append("melcache")
    save_suffix = ("_" + "_".join(save_tag_parts)) if save_tag_parts else ""
    save_path = PRETRAIN_DIR / f"inat_best_{bb_tag}{save_suffix}.pt"

    # ── Optional resume ──────────────────────────────────────────────────────
    start_epoch = 0
    best_auc = 0.0
    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"--resume ckpt missing: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        missing, unexpected = model.load_state_dict(
            ckpt["state_dict"], strict=False,
        )
        if missing or unexpected:
            print(f"  [resume] missing={len(missing)} unexpected={len(unexpected)}",
                  flush=True)
        start_epoch = (
            args.start_epoch if args.start_epoch is not None
            else int(ckpt.get("epoch", 0))
        )
        best_auc = float(ckpt.get("val_auc", 0.0))
        for _ in range(start_epoch):
            scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]
        print(
            f"  [resume] from {args.resume}\n"
            f"           start_epoch={start_epoch} (loop runs {start_epoch + 1}..{args.epochs})\n"
            f"           best_auc={best_auc:.4f}  next_lr={cur_lr:.2e}",
            flush=True,
        )

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        n_seen = 0

        for batch_idx, (mels, labels, sec_mask) in enumerate(train_dl):
            mels = mels.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            sec_mask = sec_mask.to(device, non_blocking=True)

            if random.random() < SPEC_FREQ_MASK_PROB:
                mels = freq_mask(mels)
            if random.random() < SPEC_TIME_MASK_PROB:
                mels = time_mask(mels)

            optimizer.zero_grad()
            with autocast_ctx:
                out = model(mels)
                loss_per = focal_bce_with_logits(
                    out["clip_logits"], labels, gamma=args.focal_gamma,
                )
                loss = (loss_per * sec_mask).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += loss.item()
            n_seen += 1

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
                "state_dict": model.state_dict(),
                "inat_species": species,
                "epoch": epoch,
                "val_auc": val_auc,
                "n_classes": n_classes,
                "tax_class_dist": tax_class_dist,
                "backbone": args.backbone,
            }, save_path)
            best_marker = " ★ BEST"

        if args.save_every > 0 and epoch % args.save_every == 0:
            ep_path = PRETRAIN_DIR / f"inat_backbone_{bb_tag}_e{epoch:02d}.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "inat_species": species,
                "epoch": epoch,
                "val_auc": val_auc,
                "n_classes": n_classes,
                "backbone": args.backbone,
            }, ep_path)
            print(f"  [save] backbone ckpt → {ep_path}", flush=True)

        print("=" * 40, flush=True)
        print(
            f"iNat Pretrain  Epoch {epoch:2d}/{args.epochs}: "
            f"train_loss={avg_loss:.4f}  "
            f"val_roc_auc={val_auc:.4f} (n_present={n_present})  "
            f"time={mins}m {s:02d}s  "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
            f"{best_marker}",
            flush=True,
        )
        print("=" * 40, flush=True)

        # GPU memory hygiene per four_track/CLAUDE.md.
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\niNat pretrain complete. Best val ROC-AUC: {best_auc:.4f}", flush=True)
    print(f"Best ckpt → {save_path}\n", flush=True)


if __name__ == "__main__":
    main()
