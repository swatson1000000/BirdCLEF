"""L2-redux pretrain — A1 EffNet-B0 backbone on multi-source BirdCLEF focal Aves.

Sources combined: BirdCLEF-2023 + BirdCLEF-2024 + BirdCLEF-2025 train_audio.
~70K Aves recordings spanning ~600 species (counts logged at runtime).

Per new_plan.md §14.17.15.7.2:
  - Phase 2b smoke: --epochs 5, gate via encoder probe on BC2026 val.
  - Phase 4 full:   --epochs 50 (post-Phase-2c XC bulk merge).

Single-split GroupShuffleSplit by author (matches L2 attempt 1's anti-leakage
fix). The classifier head is discarded by finetune; only backbone weights
transfer. Output ckpt embeds the union class list as a buffer for finetune-
side verification.

**Loss = focal-BCE (γ=2).** Deliberate deviation from L2 attempt 1's plain
BCE — Sydorskyi's BC2025 +0.046 ablation delta was on focal-BCE, and
Phase 4 will scale to ~7,591 long-tailed classes where focal weighting
matters even more. See new_plan.md §14.17.15 for the focal vs plain
decision rationale.

Usage:
    # Wiring smoke test (1 ep, 2 train batches)
    python -u src/pretrain_l2_redux.py --epochs 1 --smoke-test

    # Phase 2b — 5-epoch smoke pretrain on BC-historic (no XC bulk)
    python -u src/pretrain_l2_redux.py --epochs 5

    # Phase 4 — full 50-epoch pretrain after Phase 2c merges XC bulk in
    python -u src/pretrain_l2_redux.py --epochs 50
"""

import argparse
import json
import random
import subprocess
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

# ── Source layout ─────────────────────────────────────────────────────────────
SOURCES = [
    # BC2023 + BC2024 use `train_metadata.csv` as their per-clip metadata file;
    # BC2025 switched to `train.csv`. Schema (primary_label, author, filename
    # relative to train_audio/) is identical across all three.
    {
        "comp_id":   "birdclef-2023",
        "audio_dir": ROOT / "data" / "external" / "birdclef_history" / "birdclef-2023" / "train_audio",
        "csv_name":  "train_metadata.csv",
        "csv_path":  ROOT / "data" / "external" / "birdclef_history" / "birdclef-2023" / "train_metadata.csv",
    },
    {
        "comp_id":   "birdclef-2024",
        "audio_dir": ROOT / "data" / "external" / "birdclef_history" / "birdclef-2024" / "train_audio",
        "csv_name":  "train_metadata.csv",
        "csv_path":  ROOT / "data" / "external" / "birdclef_history" / "birdclef-2024" / "train_metadata.csv",
    },
    {
        "comp_id":   "birdclef-2025",
        "audio_dir": ROOT / "data" / "raw" / "birdclef_2025" / "train_audio",
        "csv_name":  "train.csv",
        "csv_path":  ROOT / "data" / "raw" / "birdclef_2025" / "train.csv",
    },
]

XC_BULK_DIR  = ROOT / "data" / "external" / "xenocanto_bulk"

SPECIES_JSON = FT_ROOT / "data" / "processed" / "l2_redux_aves_species.json"
MANIFEST_CSV = FT_ROOT / "data" / "processed" / "l2_redux_manifest.csv"

# Separate cache when XC bulk is merged in — keeps pre-XC manifest reproducible.
SPECIES_JSON_WITH_XC = FT_ROOT / "data" / "processed" / "l2_redux_aves_species_with_xc.json"
MANIFEST_CSV_WITH_XC = FT_ROOT / "data" / "processed" / "l2_redux_manifest_with_xc.csv"

PRETRAIN_DIR = FT_ROOT / "models" / "l2_redux"


# ── Manifest + species list ───────────────────────────────────────────────────

def fetch_train_csv(comp_id: str, csv_name: str, csv_path: Path) -> None:
    """Pull a per-clip metadata CSV (BC2023/24 = train_metadata.csv,
    BC2025 = train.csv) from a Kaggle competition if missing locally."""
    if csv_path.exists():
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [fetch] {comp_id}/{csv_name} missing — pulling via kaggle CLI...", flush=True)
    cmd = [
        "kaggle", "competitions", "download",
        "-c", comp_id, "-f", csv_name,
        "--path", str(csv_path.parent), "--force",
    ]
    subprocess.run(cmd, check=True)
    # CLI writes <csv_name>.zip OR <csv_name> directly depending on size; handle both.
    zipped = csv_path.parent / f"{csv_name}.zip"
    if zipped.exists():
        subprocess.run(["unzip", "-o", "-q", str(zipped), "-d", str(csv_path.parent)], check=True)
        zipped.unlink()
    if not csv_path.exists():
        sys.exit(f"  [fetch] failed to obtain {csv_path}")


def _walk_xc_bulk() -> list:
    """Walk XC bulk dir, return rows (abs_path, primary_label, source_comp, author).

    Per-species author lookup comes from the species' _meta.json (XC v3 API
    response, where 'rec' is the recordist). Skip species dirs without
    _meta.json (download still in flight for that species).
    """
    if not XC_BULK_DIR.exists():
        sys.exit(f"  [manifest] --include-xc-bulk set but XC bulk dir missing: {XC_BULK_DIR}")
    rows = []
    n_species_complete = 0
    for sp_dir in sorted(XC_BULK_DIR.iterdir()):
        if not sp_dir.is_dir():
            continue
        meta_path = sp_dir / "_meta.json"
        if not meta_path.exists():
            continue
        with meta_path.open() as f:
            recs = json.load(f)
        author_lookup = {f"XC{r.get('id')}": (r.get("rec") or "__unknown__") for r in recs}
        primary_label = sp_dir.name
        n_added_before = len(rows)
        for ogg in sorted(sp_dir.glob("XC*.ogg")):
            if ogg.stat().st_size == 0:
                continue
            stem = ogg.stem
            rows.append({
                "abs_path":      str(ogg),
                "primary_label": primary_label,
                "source_comp":   "xenocanto-bulk",
                "author":        author_lookup.get(stem, "__unknown__"),
            })
        if len(rows) > n_added_before:
            n_species_complete += 1
    print(f"  [manifest] xenocanto-bulk: +{len(rows)} clips across "
          f"{n_species_complete} species", flush=True)
    return rows


def build_manifest(include_xc_bulk: bool = False) -> pd.DataFrame:
    """Build (abs_path, primary_label, source_comp, author) manifest across all sources.

    Cached at MANIFEST_CSV (or MANIFEST_CSV_WITH_XC if include_xc_bulk).
    Author column comes from each comp's train.csv; files not in the CSV
    (defensive — rare) get author='__unknown__'.
    """
    cache_path = MANIFEST_CSV_WITH_XC if include_xc_bulk else MANIFEST_CSV
    if cache_path.exists():
        return pd.read_csv(cache_path)

    rows = []
    for src in SOURCES:
        comp_id = src["comp_id"]
        audio_dir = src["audio_dir"]
        csv_path = src["csv_path"]
        if not audio_dir.exists():
            sys.exit(f"  [manifest] missing audio dir: {audio_dir}")

        # Build author lookup from CSV (all three comps use 'filename' as relpath
        # under train_audio/ and 'author' as recordist column).
        fetch_train_csv(comp_id, src["csv_name"], csv_path)
        train_df = pd.read_csv(csv_path)
        train_df["filename"] = train_df["filename"].astype(str)
        train_df["author"]   = train_df["author"].fillna("__unknown__").astype(str)
        author_lookup = dict(zip(train_df["filename"], train_df["author"]))

        # Walk train_audio/<species>/<file>.ogg.
        n_before = len(rows)
        for sp_dir in sorted(audio_dir.iterdir()):
            if not sp_dir.is_dir():
                continue
            primary_label = sp_dir.name
            for ogg in sorted(sp_dir.glob("*.ogg")):
                relname = f"{primary_label}/{ogg.name}"
                rows.append({
                    "abs_path":      str(ogg),
                    "primary_label": primary_label,
                    "source_comp":   comp_id,
                    "author":        author_lookup.get(relname, "__unknown__"),
                })
        n_added = len(rows) - n_before
        print(f"  [manifest] {comp_id}: +{n_added} clips", flush=True)

    if include_xc_bulk:
        rows.extend(_walk_xc_bulk())

    df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"  [manifest] total {len(df)} clips, {df['primary_label'].nunique()} species "
          f"→ {cache_path}", flush=True)
    return df


def build_species_list(manifest: pd.DataFrame, include_xc_bulk: bool = False) -> list:
    """Sorted union of primary_labels across the manifest. Cached to JSON."""
    cache_path = SPECIES_JSON_WITH_XC if include_xc_bulk else SPECIES_JSON
    if cache_path.exists():
        with cache_path.open() as f:
            return json.load(f)
    species = sorted(manifest["primary_label"].astype(str).unique())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(species, f)
    print(f"  [species] {len(species)} unique Aves codes → {cache_path}", flush=True)
    return species


# ── Dataset ───────────────────────────────────────────────────────────────────

class L2ReduxDataset(Dataset):
    """Multi-source BirdCLEF focal-clip dataset for L2-redux pretrain.

    Reads (abs_path, primary_label) rows from the manifest. One-hot over the
    L2-redux Aves union (~600 species across BC2023+2024+2025). No secondary
    labels (pretrain head is discarded), no MixUp, no background noise — same
    minimal design as Pretrain2025Dataset (the L2 attempt 1 baseline).
    """

    def __init__(self, df: pd.DataFrame, sp2idx: dict, augment: bool = True):
        self.df        = df.reset_index(drop=True)
        self.sp2idx    = sp2idx
        self.n_classes = len(sp2idx)
        self.augment   = augment

    def __len__(self) -> int:
        return len(self.df)

    def _load_waveform(self, row) -> np.ndarray:
        wav = load_audio(Path(row["abs_path"]))
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

        mel  = waveform_to_mel(wav)
        mask = np.ones(self.n_classes, dtype=np.float32)
        return mel, torch.from_numpy(labels), torch.from_numpy(mask)


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device,
             max_batches: int | None = None) -> tuple:
    """Macro ROC-AUC on the held-out group-split val."""
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

def focal_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor,
                          gamma: float = 2.0) -> torch.Tensor:
    """Per-element focal-BCE with logits (no reduction).

    Sydorskyi BC2025 used `use_bce_focal_loss=True` for the EffNetV2-S
    pretrain that yielded the +0.046 ablation delta. With γ=2, easy-positive
    and easy-negative examples (where the model is already confident) get
    down-weighted by `(1 − pt)^2`, focusing gradient on hard / rare-class
    samples. Critical at the 7,591-class long tail; useful at our 600-class
    scale too.
    """
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
    p = argparse.ArgumentParser(description="L2-redux — A1 pretrain on BC2023+24+25")
    p.add_argument("--epochs",      type=int, default=5,
                   help="5 for Phase 2b smoke; 50 for Phase 4 full")
    p.add_argument("--lr",          type=float, default=2.5e-4,
                   help="Sydorskyi's value 2.5e-4 (vs L2 attempt 1 1e-3)")
    p.add_argument("--lr-min",      type=float, default=1e-6)
    p.add_argument("--batch-size",  type=int, default=BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--mixstyle-p",  type=float, default=0.5)
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Focal-BCE γ (Sydorskyi value 2.0; set 0 for plain BCE)")
    p.add_argument("--val-frac",    type=float, default=0.05)
    p.add_argument("--save-every",  type=int, default=0,
                   help="If >0, save backbone-only ckpt every N epochs (Phase 4)")
    p.add_argument("--smoke-test",  action="store_true",
                   help="1 epoch, 2 train batches, 1 val batch — wiring check")
    p.add_argument("--backbone",    type=str, default=config.BACKBONE,
                   help="timm backbone name (default config.BACKBONE = B0)")
    p.add_argument("--include-xc-bulk", action="store_true",
                   help="Merge data/external/xenocanto_bulk/ into the manifest "
                        "(Bundle 1 Phase 5.1 = 819K-rec / 7,489-species pretrain)")
    args = p.parse_args()

    set_seed(args.seed)
    import gc  # for per-epoch hygiene
    device = torch.device("cuda")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cudnn.benchmark = True

    # ── Class space ───────────────────────────────────────────────────────────
    manifest = build_manifest(include_xc_bulk=args.include_xc_bulk)
    species  = build_species_list(manifest, include_xc_bulk=args.include_xc_bulk)
    sp2idx   = {sp: i for i, sp in enumerate(species)}
    n_classes = len(species)

    # ── Group-split by author ─────────────────────────────────────────────────
    df = manifest.copy()
    df["primary_label"] = df["primary_label"].astype(str)
    df["author"]        = df["author"].fillna("__unknown__").astype(str)

    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed)
    tr_idx, vl_idx = next(gss.split(df, groups=df["author"]))
    train_df = df.iloc[tr_idx].reset_index(drop=True)
    val_df   = df.iloc[vl_idx].reset_index(drop=True)

    n_val_species = val_df["primary_label"].nunique()
    n_shared_auth = len(set(train_df["author"]) & set(val_df["author"]))

    print(f"  [split] grouped by author: "
          f"train={len(train_df)} ({train_df['author'].nunique()} authors)  "
          f"val={len(val_df)} ({val_df['author'].nunique()} authors)  "
          f"val_species={n_val_species}  shared_authors={n_shared_auth}", flush=True)

    if args.smoke_test:
        train_df = train_df.head(args.batch_size * 2)
        val_df   = val_df.head(args.batch_size)

    print("=" * 60, flush=True)
    print(f"L2-redux pretrain — multi-source BC focal Aves → backbone {args.backbone}", flush=True)
    print(f"  union classes  : {n_classes}", flush=True)
    print(f"  total clips    : {len(manifest)}", flush=True)
    by_src = manifest["source_comp"].value_counts().to_dict()
    print(f"  by source      : " + "  ".join(f"{k}={v}" for k, v in sorted(by_src.items())),
          flush=True)
    print(f"  train clips    : {len(train_df)}", flush=True)
    print(f"  val   clips    : {len(val_df)}", flush=True)
    print(f"  epochs         : {args.epochs}", flush=True)
    print(f"  lr / lr_min    : {args.lr} / {args.lr_min}", flush=True)
    print(f"  batch / nworker: {args.batch_size} / {args.num_workers}", flush=True)
    print(f"  mixstyle_p     : {args.mixstyle_p}", flush=True)
    print(f"  focal_gamma    : {args.focal_gamma}  "
          f"({'plain BCE' if args.focal_gamma == 0 else 'focal-BCE'})", flush=True)
    print(f"  save_every     : {args.save_every}", flush=True)
    print(f"  smoke_test     : {args.smoke_test}", flush=True)
    print(f"  → ckpt         : {PRETRAIN_DIR}", flush=True)
    print("=" * 60, flush=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = L2ReduxDataset(train_df, sp2idx, augment=True)
    val_ds   = L2ReduxDataset(val_df,   sp2idx, augment=False)

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
        backbone_name=args.backbone,
        n_classes=n_classes,
        mixstyle_p=args.mixstyle_p,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=WEIGHT_DECAY)
    # Sydorskyi recipe uses CosineAnnealingWarmRestarts(T_0 = N_EPOCHS * len_train);
    # for our shorter-horizon smoke that collapses to plain CosineAnnealingLR
    # over the full run.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr_min,
    )
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    bb_tag = args.backbone.replace("/", "_").replace(".", "_")
    xc_tag = "_with_xc" if args.include_xc_bulk else ""
    save_path = PRETRAIN_DIR / f"l2_redux_best_{bb_tag}{xc_tag}.pt"

    # ── Training loop ─────────────────────────────────────────────────────────
    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        n_seen = 0

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
                loss_per = focal_bce_with_logits(
                    out["clip_logits"], labels, gamma=args.focal_gamma,
                )
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
                "state_dict":     model.state_dict(),
                "union_classes":  species,
                "epoch":          epoch,
                "val_auc":        val_auc,
                "source_comps":   sorted(manifest["source_comp"].unique().tolist()),
                "backbone":       args.backbone,
                "include_xc_bulk": args.include_xc_bulk,
            }, save_path)
            best_marker = " ★ BEST"

        # Per-N-epoch backbone-only ckpt for Phase 4 finetune init.
        if args.save_every > 0 and epoch % args.save_every == 0:
            ep_path = PRETRAIN_DIR / f"l2_redux_backbone_{bb_tag}{xc_tag}_e{epoch:02d}.pt"
            torch.save({
                "state_dict":     model.state_dict(),
                "union_classes":  species,
                "epoch":          epoch,
                "val_auc":        val_auc,
                "source_comps":   sorted(manifest["source_comp"].unique().tolist()),
                "backbone":       args.backbone,
                "include_xc_bulk": args.include_xc_bulk,
            }, ep_path)
            print(f"  [save] backbone ckpt → {ep_path}", flush=True)

        print("=" * 40, flush=True)
        print(
            f"L2-redux Pretrain  Epoch {epoch:2d}/{args.epochs}: "
            f"train_loss={avg_loss:.4f}  "
            f"val_roc_auc={val_auc:.4f} (n_present={n_present})  "
            f"time={mins}m {s:02d}s  "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
            f"{best_marker}",
            flush=True,
        )
        print("=" * 40, flush=True)

        # GPU memory hygiene — per four_track/CLAUDE.md.
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nL2-redux pretrain complete. Best val ROC-AUC: {best_auc:.4f}", flush=True)
    print(f"Best ckpt → {save_path}\n", flush=True)


if __name__ == "__main__":
    main()
