"""Sydorskyi-recipe OOF soft pseudo-label emission for Bundle 1 NS.

Per `reference_bc2025_winners_writeups.md` (Sydorskyi BC2025 2nd place):
  - Mean-ensemble teachers, OOF-fold-aware (each chunk's pseudo from
    teachers NOT trained on that chunk's fold).
  - Keep chunks where max prob >= --keep-thresh (default 0.5).
  - Zero per-class probs below --zero-thresh (default 0.1).
  - Soft targets (no argmax, no thresholding past the per-class zero).

Multi-machine sharding: --shard-total N --shard-id i runs disjoint
slices of the clip list. For Bundle 1 NS iter-1, target audio = BC2025
unlabeled train_soundscapes/. Pseudo emission is I/O-dominated, so the
GB10 (skynet) vs RTX-4080 (deepthought) per-epoch gap shrinks; sharded
launches across both machines are within the 4:1 rule's I/O-bound
exception (see four_track/CLAUDE.md).

Usage:
    # Smoke (1 clip, BS=4) — proves wiring on existing focal-only V2-S ckpts
    python -u src/pseudo_emit_sydorskyi.py \\
        --teacher-ckpts models/a1/a1_tf_efficientnetv2_s.in21k_ft_in1k_fold0_seed42_hybrid.pt \\
        --teacher-folds 0 \\
        --backbone tf_efficientnetv2_s.in21k_ft_in1k \\
        --target-audio-dir ../data/raw/birdclef_2025/train_soundscapes \\
        --output-npz data/pseudo_iter1_smoke.npz \\
        --smoke-test

    # Full BC2025 SS pseudo emission, sharded across 2 machines
    # On DT (shard 0):
    python -u src/pseudo_emit_sydorskyi.py \\
        --teacher-ckpts models/a1_l2/v2s_fold0.pt models/a1_l2/v2s_fold1.pt \\
                        models/a1_l2/v2s_fold2.pt models/a1_l2/v2s_fold4.pt \\
        --teacher-folds 0 1 2 4 \\
        --backbone tf_efficientnetv2_s.in21k_ft_in1k \\
        --target-audio-dir ../data/raw/birdclef_2025/train_soundscapes \\
        --output-npz data/pseudo_iter1_shard0of2.npz \\
        --shard-total 2 --shard-id 0
    # On skynet (shard 1): same but --shard-id 1
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
from config import SAMPLE_RATE, CHUNK_SAMPLES, N_CLASSES  # noqa: E402
from utils import load_audio, pad_or_crop, waveform_to_mel  # noqa: E402

from model_a1 import BirdSEDModelA1  # noqa: E402


def _hash_to_fold(name: str, k: int) -> int:
    """Deterministic name -> fold-id in [0, k). Used when no CSV fold map given."""
    h = hashlib.sha1(name.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % k


def build_fold_lookup(target_dir: Path, folds_csv: Path | None,
                      filename_col: str, fold_col: str,
                      teacher_folds: list[int]) -> dict[str, int]:
    """Map each clip filename (basename) -> fold id.

    If folds_csv given, look up by filename_col. Otherwise hash filename
    into the available teacher fold pool.
    """
    if folds_csv is not None:
        df = pd.read_csv(folds_csv)
        df[filename_col] = df[filename_col].astype(str)
        # Some folds csvs use full relpath; we key by basename for robustness.
        out = {}
        for _, row in df.iterrows():
            base = Path(str(row[filename_col])).name
            out[base] = int(row[fold_col])
        return out

    # Hash mode — deterministic over the teacher fold pool.
    pool = sorted(teacher_folds)
    out = {}
    for f in sorted(target_dir.rglob("*.ogg")):
        rel = str(f.relative_to(target_dir))
        out[rel] = pool[_hash_to_fold(rel, len(pool))]
    return out


# ── Dataset over chunks ───────────────────────────────────────────────────────

class ChunkDataset(Dataset):
    """Yield (mel, clip_id, chunk_idx, fold_assign) for every CHUNK_SAMPLES window
    in every audio file under target_dir.

    Files shorter than CHUNK_SAMPLES yield exactly one (zero-padded) chunk.
    """

    def __init__(self, target_dir: Path, fold_lookup: dict[str, int],
                 shard_total: int = 1, shard_id: int = 0,
                 limit_clips: int | None = None):
        self.target_dir = target_dir
        all_files = sorted(p for p in target_dir.rglob("*.ogg")
                           if p.stat().st_size > 0)
        # Shard at the *clip* level so chunks of one file stay co-located;
        # avoids audio reloads in non-trivial worker setups.
        clip_files = [f for i, f in enumerate(all_files)
                      if i % shard_total == shard_id]
        if limit_clips is not None:
            clip_files = clip_files[:limit_clips]

        # Pre-compute (file, n_chunks). Audio length probe via librosa would be
        # ideal but adds load; cheaper to load-then-chunk in __getitem__ and
        # cache n_chunks here using soundfile metadata.
        import soundfile as sf  # local import — only needed here
        index = []
        for f in clip_files:
            try:
                info = sf.info(str(f))
                n_samples = int(info.frames * SAMPLE_RATE / info.samplerate) \
                            if info.samplerate != SAMPLE_RATE else int(info.frames)
            except Exception:
                n_samples = CHUNK_SAMPLES   # treat unreadable as 1 chunk
            n_chunks = max(1, (n_samples + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES)
            for c in range(n_chunks):
                index.append((f, c))

        self.index       = index
        self.fold_lookup = fold_lookup

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        f, c = self.index[i]
        wav  = load_audio(f)
        s    = c * CHUNK_SAMPLES
        e    = s + CHUNK_SAMPLES
        chunk = wav[s:e]
        chunk = pad_or_crop(chunk, CHUNK_SAMPLES, random_crop=False)
        mel   = waveform_to_mel(chunk)            # (3, N_MELS, T)
        rel   = str(f.relative_to(self.target_dir))
        fold  = self.fold_lookup.get(rel, -1)
        return mel, rel, int(c), int(fold)


def _collate(batch):
    mels   = torch.stack([b[0] for b in batch])
    names  = [b[1] for b in batch]
    chunks = torch.tensor([b[2] for b in batch], dtype=torch.int32)
    folds  = torch.tensor([b[3] for b in batch], dtype=torch.int8)
    return mels, names, chunks, folds


# ── Teacher loading ───────────────────────────────────────────────────────────

def load_teacher(ckpt_path: Path, backbone: str, n_classes: int,
                 device: torch.device) -> nn.Module:
    """Build BirdSEDModelA1 with the requested backbone and load ckpt state."""
    model = BirdSEDModelA1(
        backbone_name=backbone,
        n_classes=n_classes,
        mixstyle_p=0.0,                 # disable MixStyle at inference
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [load] {ckpt_path.name}: missing keys: {sorted(missing)[:5]}"
              f"{' ...' if len(missing) > 5 else ''}", flush=True)
    if unexpected:
        print(f"  [load] {ckpt_path.name}: unexpected keys: {sorted(unexpected)[:5]}"
              f"{' ...' if len(unexpected) > 5 else ''}", flush=True)
    model.eval()
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_batch(teachers: dict[int, nn.Module], mels: torch.Tensor,
                folds: torch.Tensor) -> torch.Tensor:
    """OOF mean-ensemble over teachers != fold for each item.

    teachers: {fold_id: model}.
    mels: (B, 3, N_MELS, T) on device.
    folds: (B,) int on cpu.

    Returns (B, n_classes) sigmoid probs on cpu.
    """
    teacher_folds = sorted(teachers.keys())
    # Forward pass through each teacher once over the full batch (cheaper than
    # per-item teacher selection). Then OOF-mask in numpy.
    all_probs = []
    for tf in teacher_folds:
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = teachers[tf](mels)
        probs = torch.sigmoid(out["clip_logits"]).float().cpu().numpy()  # (B, K)
        all_probs.append(probs)
    stack = np.stack(all_probs, axis=0)               # (T, B, K)
    folds_np = folds.numpy()                          # (B,)
    # Mask: per item, exclude rows where teacher_fold == clip_fold.
    out = np.zeros((stack.shape[1], stack.shape[2]), dtype=np.float32)
    for bi in range(stack.shape[1]):
        keep = [ti for ti, tf in enumerate(teacher_folds) if tf != folds_np[bi]]
        if not keep:
            keep = list(range(len(teacher_folds)))    # safety: no OOF teacher → use all
        out[bi] = stack[keep, bi].mean(axis=0)
    return torch.from_numpy(out)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Sydorskyi-recipe OOF pseudo emit")
    p.add_argument("--teacher-ckpts", nargs="+", required=True, type=Path,
                   help="One or more fold ckpt paths")
    p.add_argument("--teacher-folds", nargs="+", required=True, type=int,
                   help="Fold id per ckpt (parallel to --teacher-ckpts)")
    p.add_argument("--backbone", required=True, type=str,
                   help="timm backbone name shared across all teacher ckpts")
    p.add_argument("--target-audio-dir", required=True, type=Path)
    p.add_argument("--output-npz", required=True, type=Path)

    p.add_argument("--folds-csv",      type=Path, default=None,
                   help="CSV mapping clip basename -> fold (else hash mode)")
    p.add_argument("--filename-col",   type=str,  default="filename")
    p.add_argument("--fold-col",       type=str,  default="fold")

    p.add_argument("--keep-thresh",    type=float, default=0.5,
                   help="Sydorskyi: drop chunks where max(prob) < this")
    p.add_argument("--zero-thresh",    type=float, default=0.1,
                   help="Sydorskyi: zero per-class probs below this")

    p.add_argument("--n-classes",      type=int,  default=N_CLASSES)
    p.add_argument("--batch-size",     type=int,  default=64)
    p.add_argument("--num-workers",    type=int,  default=config.NUM_WORKERS)
    p.add_argument("--shard-total",    type=int,  default=1)
    p.add_argument("--shard-id",       type=int,  default=0)
    p.add_argument("--limit-clips",    type=int,  default=None,
                   help="Process only first N clips (for smoke / partial runs)")
    p.add_argument("--device",         type=str,  default="cuda")
    p.add_argument("--smoke-test",     action="store_true",
                   help="--limit-clips=1 --batch-size=4 --num-workers=0")
    args = p.parse_args()

    if args.smoke_test:
        args.limit_clips = 1
        args.batch_size  = 4
        args.num_workers = 0

    if len(args.teacher_ckpts) != len(args.teacher_folds):
        sys.exit("--teacher-ckpts and --teacher-folds must have the same length")
    if not (0 <= args.shard_id < args.shard_total):
        sys.exit(f"--shard-id {args.shard_id} not in [0, {args.shard_total})")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cudnn.benchmark = True

    print("=" * 60, flush=True)
    print("Sydorskyi pseudo emit", flush=True)
    print(f"  backbone        : {args.backbone}", flush=True)
    print(f"  teachers (fold->ckpt):", flush=True)
    for f, c in zip(args.teacher_folds, args.teacher_ckpts):
        print(f"    fold {f}: {c.name}", flush=True)
    print(f"  target dir      : {args.target_audio_dir}", flush=True)
    print(f"  fold map        : {'CSV ' + str(args.folds_csv) if args.folds_csv else 'hash'}",
          flush=True)
    print(f"  shard           : {args.shard_id+1}/{args.shard_total}", flush=True)
    print(f"  keep / zero thr : {args.keep_thresh} / {args.zero_thresh}", flush=True)
    print(f"  output          : {args.output_npz}", flush=True)
    print(f"  device          : {device}", flush=True)
    print("=" * 60, flush=True)

    fold_lookup = build_fold_lookup(
        args.target_audio_dir, args.folds_csv,
        args.filename_col, args.fold_col,
        args.teacher_folds,
    )

    ds = ChunkDataset(
        args.target_audio_dir, fold_lookup,
        shard_total=args.shard_total, shard_id=args.shard_id,
        limit_clips=args.limit_clips,
    )
    print(f"  shard chunks    : {len(ds)} (clips x avg_chunks_per_clip)", flush=True)

    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=_collate, drop_last=False,
        persistent_workers=(args.num_workers > 0),
        multiprocessing_context="spawn" if args.num_workers > 0 else None,
    )

    teachers: dict[int, nn.Module] = {}
    for f, c in zip(args.teacher_folds, args.teacher_ckpts):
        print(f"  [load] fold {f} <- {c}", flush=True)
        teachers[f] = load_teacher(c, args.backbone, args.n_classes, device)

    kept_names, kept_chunks, kept_folds, kept_softs = [], [], [], []
    n_seen = n_kept = 0
    t0 = time.time()
    for bi, (mels, names, chunks, folds) in enumerate(dl):
        mels = mels.to(device, non_blocking=True)
        probs = infer_batch(teachers, mels, folds).numpy()    # (B, K)

        # Sydorskyi filter: keep chunk iff max prob >= keep-thresh.
        max_per_chunk = probs.max(axis=1)
        keep_mask = max_per_chunk >= args.keep_thresh
        n_seen += probs.shape[0]
        n_kept += int(keep_mask.sum())
        if not keep_mask.any():
            continue

        kept = probs[keep_mask].copy()
        # Per-class zero below zero-thresh (in place after copy).
        kept[kept < args.zero_thresh] = 0.0

        kept_softs.append(kept.astype(np.float32))
        kept_names.extend([names[i] for i in range(len(names)) if keep_mask[i]])
        kept_chunks.append(chunks.numpy()[keep_mask])
        kept_folds.append(folds.numpy()[keep_mask])

        if (bi + 1) % 25 == 0 or bi == 0:
            elapsed = time.time() - t0
            print(f"  [emit] batch {bi+1}/{len(dl)}  "
                  f"seen={n_seen}  kept={n_kept} ({100.*n_kept/max(n_seen,1):.1f}%)  "
                  f"elapsed={elapsed:.1f}s", flush=True)

    if not kept_softs:
        sys.exit(f"No chunks passed --keep-thresh={args.keep_thresh}; nothing to write.")

    soft_arr   = np.concatenate(kept_softs, axis=0).astype(np.float32)
    chunk_arr  = np.concatenate(kept_chunks, axis=0).astype(np.int32)
    fold_arr   = np.concatenate(kept_folds, axis=0).astype(np.int8)
    name_arr   = np.array(kept_names, dtype=object)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        clip_names    = name_arr,
        chunk_idx     = chunk_arr,
        fold_assign   = fold_arr,
        soft_labels   = soft_arr,
        keep_thresh   = np.float32(args.keep_thresh),
        zero_thresh   = np.float32(args.zero_thresh),
        teacher_folds = np.array(sorted(teachers.keys()), dtype=np.int8),
        backbone      = args.backbone,
    )
    print(f"\n  [write] {args.output_npz}", flush=True)
    print(f"  total chunks seen : {n_seen}", flush=True)
    print(f"  chunks kept       : {n_kept} ({100.*n_kept/max(n_seen,1):.1f}%)", flush=True)
    print(f"  output shape      : soft_labels {soft_arr.shape}, "
          f"unique clips {len(set(name_arr))}", flush=True)


if __name__ == "__main__":
    main()
