"""L2-redux encoder linear probe — Phase 2b gate.

Per new_plan.md §14.17.15.7.2: freeze the EffNet-B0 backbone (and GeM pool),
re-initialize only the head surface (`cls_conv` + `att_conv`) to 234 classes,
train 5 epochs on BC2026 fold-0 train data with hybrid loss, evaluate val_v2
macro-AUC. The "linear probe" framing is loose — strictly we're training the
SED head (1×1 cls conv + small attn conv) on top of a frozen encoder, which
is the standard way to evaluate encoder transfer in SED architectures.

Two complementary runs are needed for a meaningful gate:

    # Baseline — ImageNet-init backbone, head trained 5 ep
    python -u src/probe_l2_redux_encoder.py --tag imagenet

    # Treatment — L2-redux backbone, head trained 5 ep (same data, same head)
    python -u src/probe_l2_redux_encoder.py --tag l2redux \
        --init-from models/l2_redux/l2_redux_best.pt

Decision rule: if (l2redux − imagenet) macro-AUC ≥ 0.01 on val_v2,
green-light Phase 2c XC bulk download. Below that, the encoder isn't
teaching enough generic acoustic features to be worth the multi-day
download.

Output: `models/l2_redux/probe_<tag>_log.json` with per-epoch metrics.
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
import torchaudio.transforms as T
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

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
    BATCH_SIZE, WEIGHT_DECAY,
    SPEC_TIME_MASK_PROB, SPEC_FREQ_MASK_PROB,
    get_species_index,
)
from utils import load_audio, pad_or_crop, waveform_to_mel  # noqa: E402

from dataset_a1 import BirdTrainDatasetA1  # noqa: E402
from losses import HybridBceAsl  # noqa: E402
from model_a1 import BirdSEDModelA1  # noqa: E402

PROBE_DIR = FT_ROOT / "models" / "l2_redux"


# ── Backbone state-dict loader (mirrors train_a1.py:_load_pretrained_backbone)
def _load_pretrained_backbone(model: nn.Module, ckpt_path: Path) -> None:
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


# ── Backbone freeze ───────────────────────────────────────────────────────────

PROBE_PARAM_PREFIXES = ("cls_conv.", "att_conv.")  # everything else is frozen

def freeze_backbone(model: nn.Module) -> tuple:
    """Freeze backbone + GeM pool + MixStyle; leave cls_conv + att_conv trainable.

    Returns (n_trainable, n_total) for logging.
    """
    n_total = 0
    n_trainable = 0
    for name, param in model.named_parameters():
        n_total += param.numel()
        if any(name.startswith(pfx) for pfx in PROBE_PARAM_PREFIXES):
            param.requires_grad = True
            n_trainable += param.numel()
        else:
            param.requires_grad = False
    return n_trainable, n_total


def set_backbone_eval(model: nn.Module) -> None:
    """Put the frozen modules into eval() so BN running stats don't update.

    Probe modules (cls_conv + att_conv) stay in train() so their internal
    BN (in att_conv) accumulates fresh stats on the BC2026 distribution.
    """
    # Default to eval, then flip the probe submodules back to train.
    model.eval()
    for name, mod in model.named_modules():
        if any(name.startswith(pfx[:-1]) for pfx in PROBE_PARAM_PREFIXES):
            mod.train()


# ── Validation: parent train_soundscapes_labels.csv (val_v2 substrate) ────────

def build_soundscape_val(sp2idx: dict) -> tuple:
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
def validate(model: nn.Module, val_mels: list, val_labels: np.ndarray,
             device: torch.device, batch_size: int = 32) -> float:
    model.eval()
    all_probs = []
    for i in range(0, len(val_mels), batch_size):
        batch = torch.stack(val_mels[i: i + batch_size]).to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(batch)
        all_probs.append(torch.sigmoid(out["clip_logits"]).float().cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    present = val_labels.sum(axis=0) > 0
    if present.sum() == 0:
        return 0.0
    try:
        return float(roc_auc_score(
            val_labels[:, present], probs[:, present], average="macro"
        ))
    except ValueError:
        return 0.0


# ── Probe trainer ─────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    p = argparse.ArgumentParser(description="L2-redux encoder linear probe")
    p.add_argument("--tag", required=True,
                   help="Run tag, e.g. 'imagenet' or 'l2redux' — used in log filename")
    p.add_argument("--init-from", type=Path, default=None,
                   help="Path to pretrained backbone ckpt (omit for ImageNet init)")
    p.add_argument("--fold",        type=int,   default=0,
                   help="Held-out fold for val (data not used; val_v2 is the gate)")
    p.add_argument("--epochs",      type=int,   default=5)
    p.add_argument("--lr",          type=float, default=1e-3,
                   help="Probe-side LR — head only, no decay schedule")
    p.add_argument("--batch-size",  type=int,   default=BATCH_SIZE)
    p.add_argument("--num-workers", type=int,   default=config.NUM_WORKERS)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--mixstyle-p",  type=float, default=0.0,
                   help="MixStyle disabled by default for probe (frozen backbone)")
    p.add_argument("--smoke-test",  action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cudnn.benchmark = True

    # ── Data: BC2026 fold-non-{fold} train data ───────────────────────────────
    df       = pd.read_csv(PROC / "train_folds.csv")
    train_df = df[df["fold"] != args.fold].reset_index(drop=True)
    if args.smoke_test:
        train_df = train_df.head(args.batch_size * 2)

    sp2idx = get_species_index()
    n_classes = len(sp2idx)
    assert n_classes == config.N_CLASSES == 234

    train_ds = BirdTrainDatasetA1(train_df, augment=True)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0 if args.smoke_test else args.num_workers,
        pin_memory=True, drop_last=True,
        persistent_workers=not args.smoke_test,
        multiprocessing_context=None if args.smoke_test else "spawn",
    )

    print("=" * 60, flush=True)
    print(f"L2-redux encoder linear probe — tag={args.tag}", flush=True)
    print(f"  init_from      : {args.init_from or '(ImageNet default)'}", flush=True)
    print(f"  classes        : {n_classes}", flush=True)
    print(f"  train clips    : {len(train_df)}  (fold ≠ {args.fold})", flush=True)
    print(f"  epochs         : {args.epochs}", flush=True)
    print(f"  lr             : {args.lr}  (head only)", flush=True)
    print(f"  batch / nworker: {args.batch_size} / {args.num_workers}", flush=True)
    print(f"  mixstyle_p     : {args.mixstyle_p}  (probe → 0)", flush=True)
    print(f"  smoke_test     : {args.smoke_test}", flush=True)
    print("=" * 60, flush=True)

    # ── Build soundscape val (same protocol as train_a1.py) ───────────────────
    print("  [val] building val_v2 mel cache from train_soundscapes_labels.csv ...", flush=True)
    t0 = time.time()
    val_mels, val_labels = build_soundscape_val(sp2idx)
    print(f"  [val] {len(val_mels)} clips cached in {time.time()-t0:.1f}s", flush=True)

    # ── Model + freeze ────────────────────────────────────────────────────────
    freq_mask = T.FrequencyMasking(freq_mask_param=27).to(device)
    time_mask = T.TimeMasking(time_mask_param=64).to(device)

    model = BirdSEDModelA1(
        backbone_name=config.BACKBONE,
        n_classes=n_classes,
        mixstyle_p=args.mixstyle_p,
    ).to(device)

    if args.init_from is not None:
        if not args.init_from.exists():
            sys.exit(f"  [init-from] missing ckpt: {args.init_from}")
        _load_pretrained_backbone(model, args.init_from)

    n_train, n_total = freeze_backbone(model)
    print(f"  [freeze] trainable params: {n_train:,} / {n_total:,} "
          f"({100.0 * n_train / n_total:.2f}%)", flush=True)

    # Probe-only optimizer over the trainable param subset.
    probe_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(probe_params, lr=args.lr, weight_decay=WEIGHT_DECAY)
    loss_fn = HybridBceAsl(bce_weight=0.5)  # match train_a1.py default
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = PROBE_DIR / f"probe_{args.tag}_log.json"
    history = []

    # ── Training loop ─────────────────────────────────────────────────────────
    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        set_backbone_eval(model)  # backbone frozen + BN running stats locked
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
                loss_per = loss_fn(out["clip_logits"], labels)
                loss     = (loss_per * sec_mask).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(probe_params, 5.0)
            optimizer.step()

            running_loss += loss.item()
            n_seen       += 1

            if args.smoke_test and batch_idx + 1 >= 2:
                break

        avg_loss = running_loss / max(n_seen, 1)
        val_auc  = validate(model, val_mels, val_labels, device,
                            batch_size=args.batch_size)

        elapsed = int(time.time() - epoch_start)
        mins, s = divmod(elapsed, 60)

        best_marker = ""
        if val_auc > best_auc:
            best_auc = val_auc
            best_marker = " ★ BEST"

        history.append({
            "epoch":      epoch,
            "train_loss": avg_loss,
            "val_v2_auc": val_auc,
            "best_so_far": val_auc == best_auc,
        })

        print("=" * 40, flush=True)
        print(
            f"L2-redux Probe[{args.tag}]  Epoch {epoch:2d}/{args.epochs}: "
            f"train_loss={avg_loss:.4f}  "
            f"val_v2_auc={val_auc:.4f}  "
            f"time={mins}m {s:02d}s  "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
            f"{best_marker}",
            flush=True,
        )
        print("=" * 40, flush=True)

        # GPU memory hygiene per CLAUDE.md.
        gc.collect()
        torch.cuda.empty_cache()

    summary = {
        "tag":         args.tag,
        "init_from":   str(args.init_from) if args.init_from else None,
        "best_val_v2": best_auc,
        "history":     history,
        "fold":        args.fold,
        "epochs":      args.epochs,
        "lr":          args.lr,
    }
    with log_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nProbe[{args.tag}] complete. Best val_v2 AUC: {best_auc:.4f}", flush=True)
    print(f"Log → {log_path}\n", flush=True)


if __name__ == "__main__":
    main()
