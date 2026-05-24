"""Phase 2b+2c of val-v2 rebuild — assemble the two-channel val NPZ.

Consumes:
  - train_soundscapes_labels.csv  (ground truth for 66 annotated files)
  - data/processed/val_v2/perch_new500.npz  (new Perch run, 6000 segments)
  - data/raw/train_soundscapes/*.ogg  (audio source for mel extraction)
  - data/processed/train_folds.csv  (fold-0 focal holdout)
  - data/raw/train_audio/*  (audio source for focal holdout)

Produces:
  - data/processed/val_v2/val_v2_soundscape.npz  (Channel A, primary gate)
  - data/processed/val_v2/val_v2_focal.npz       (Channel B, per-species diag)

Design:
  Channel A (soundscape)
    * Start with the 739 unique GT segments (dedup 2× rows in CSV).
    * Add every 5-s segment from the 500 new Perch files whose Perch score
      vector has any entry > τ=8.0 (logit). Labels are the τ-mask.
    * Optional per-species cap (--cap-per-species) uses a rarity-greedy
      keep rule; disabled by default in v1.
    * Extract mels via the parent `waveform_to_mel` pipeline so the NPZ is
      drop-in for `train_a1.build_soundscape_val`.

  Channel B (focal holdout)
    * From fold-0 of train_folds.csv, pick 1 clip per species that has ≥2
      fold-0 clips (so ≥1 stays in training for other folds).
    * Single-label, clean primary_label only.
    * Center-5s crop → pad to CHUNK_SAMPLES → mel.
    * Used only when diagnosing fold-0 models.

Usage:
    python -u src/rebuild_val_build.py
    python -u src/rebuild_val_build.py --tau 7.0  # lower precision, more labels
    python -u src/rebuild_val_build.py --cap-per-species 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
    get_species_index,
)
from utils import load_audio, pad_or_crop, waveform_to_mel  # noqa: E402


SOUNDSCAPE_DIR  = config.RAW / "train_soundscapes"
TRAIN_AUDIO_DIR = config.RAW / "train_audio"
GT_CSV          = config.RAW / "train_soundscapes_labels.csv"
FOLDS_CSV       = ROOT / "data" / "processed" / "train_folds.csv"

OUT_DIR      = FT_ROOT / "data" / "processed" / "val_v2"
PERCH_NEW    = OUT_DIR / "perch_new500.npz"
OUT_SCAPE    = OUT_DIR / "val_v2_soundscape.npz"
OUT_FOCAL    = OUT_DIR / "val_v2_focal.npz"

WINDOW_SEC = 5  # size of each val segment (before padding to CHUNK_SAMPLES)


# ─────────────────────────────────────────────────────────────────────────────
# Channel A assembly
# ─────────────────────────────────────────────────────────────────────────────

def _parse_hms(s: str) -> int:
    h, m, sec = str(s).split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec)


def load_gt_records(sp2idx: dict) -> list[dict]:
    df = pd.read_csv(GT_CSV)
    df["start_sec"] = df["start"].apply(_parse_hms)
    df = df.drop_duplicates(["filename", "start_sec"]).reset_index(drop=True)
    n_classes = len(sp2idx)

    records = []
    for _, r in df.iterrows():
        vec = np.zeros(n_classes, dtype=np.float32)
        for sp in str(r["primary_label"]).split(";"):
            sp = sp.strip()
            if sp in sp2idx:
                vec[sp2idx[sp]] = 1.0
        records.append({
            "filename":  str(r["filename"]),
            "start_sec": int(r["start_sec"]),
            "labels":    vec,
            "source":    "gt",
        })
    return records


def load_perch_new_records(tau: float) -> list[dict]:
    if not PERCH_NEW.exists():
        raise SystemExit(f"[error] missing {PERCH_NEW} — run rebuild_val_perch.py first")

    data = np.load(PERCH_NEW, allow_pickle=True)
    scores = data["scores"].astype(np.float32)        # (N, 234)
    fnames = data["filename"].astype(str)              # (N,)
    secs   = data["start_sec"].astype(np.int32)        # (N,)

    labels = (scores > tau).astype(np.float32)         # (N, 234)
    nonempty = labels.sum(axis=1) > 0
    n_total    = len(fnames)
    n_kept     = int(nonempty.sum())
    n_dropped  = n_total - n_kept
    print(f"[perch-new] segments total: {n_total}", flush=True)
    print(f"[perch-new] ≥1 label at τ={tau}: {n_kept}  (dropped silent: {n_dropped})",
          flush=True)
    print(f"[perch-new] total emitted labels: {int(labels[nonempty].sum())}",
          flush=True)
    print(f"[perch-new] species with ≥1 emission: "
          f"{int((labels[nonempty].sum(axis=0) > 0).sum())}",
          flush=True)

    records = []
    for i in np.where(nonempty)[0]:
        records.append({
            "filename":  fnames[i],
            "start_sec": int(secs[i]),
            "labels":    labels[i],
            "source":    "perch_new",
        })
    return records


def cap_per_species(records: list[dict], cap: int, seed: int = 42) -> list[dict]:
    if cap <= 0 or not records:
        return records
    n_classes = len(records[0]["labels"])
    counts    = np.zeros(n_classes, dtype=np.int32)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))

    kept = []
    for i in order:
        rec = records[i]
        sp_idx = np.nonzero(rec["labels"])[0]
        if len(sp_idx) == 0:
            continue
        # keep if at least one of its species is still under cap
        if np.any(counts[sp_idx] < cap):
            kept.append(rec)
            counts[sp_idx] += 1

    # restore deterministic order by (filename, start_sec, source)
    kept.sort(key=lambda r: (r["source"], r["filename"], r["start_sec"]))
    print(f"[cap] per-species cap={cap}: "
          f"{len(records)} → {len(kept)} records",
          flush=True)
    return kept


def extract_soundscape_mels(records: list[dict]) -> np.ndarray:
    """Mirror train_a1.build_soundscape_val mel pipeline:
       load_audio → slice 5s window → pad_or_crop → waveform_to_mel."""
    n = len(records)
    print(f"[mels] extracting {n} soundscape mels …", flush=True)
    out = np.zeros((n, 3, N_MELS, 512), dtype=np.float16)

    # Group by file to reduce re-reads.
    by_file: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        by_file.setdefault(r["filename"], []).append(i)

    t0 = time.time()
    n_done = 0
    for fname, idxs in by_file.items():
        path = SOUNDSCAPE_DIR / fname
        try:
            wav = load_audio(path)
        except Exception as ex:
            print(f"  [warn] audio load failed {fname}: {ex}", flush=True)
            n_done += len(idxs)
            continue

        for i in idxs:
            start = records[i]["start_sec"] * SAMPLE_RATE
            end   = start + WINDOW_SEC * SAMPLE_RATE
            if end > len(wav):
                seg = wav[start:]
            else:
                seg = wav[start:end]
            seg = pad_or_crop(seg, CHUNK_SAMPLES, random_crop=False)
            mel = waveform_to_mel(seg).to(torch.float16).numpy()
            out[i] = mel
            n_done += 1

        if n_done % 200 == 0 or n_done == n:
            rate = n_done / max(time.time() - t0, 1e-6)
            print(f"  [{n_done:>4}/{n}]  {rate:.1f} mels/s", flush=True)

    return out


def build_channel_a(args) -> None:
    sp2idx = get_species_index()
    gt_recs    = load_gt_records(sp2idx)
    perch_recs = load_perch_new_records(args.tau)

    # GT files (66) and perch-new files (500) are disjoint by construction.
    all_recs = gt_recs + perch_recs
    print(f"[combine] GT records: {len(gt_recs)}  "
          f"perch-new records: {len(perch_recs)}  "
          f"total: {len(all_recs)}",
          flush=True)

    all_recs = cap_per_species(all_recs, args.cap_per_species)

    mels = extract_soundscape_mels(all_recs)
    labels    = np.stack([r["labels"]   for r in all_recs]).astype(np.float32)
    filenames = np.array([r["filename"]  for r in all_recs], dtype=object)
    starts    = np.array([r["start_sec"] for r in all_recs], dtype=np.int32)
    sources   = np.array([r["source"]    for r in all_recs], dtype=object)

    n_species = int((labels.sum(axis=0) > 0).sum())
    print(f"\n[channel-a] final: {len(all_recs)} segments,  "
          f"{n_species} species present,  "
          f"mels {mels.nbytes/1e9:.2f} GB",
          flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_SCAPE,
        mels=mels,
        labels=labels,
        filename=filenames,
        start_sec=starts,
        source=sources,
        tau=np.float32(args.tau),
        cap=np.int32(args.cap_per_species),
    )
    print(f"[channel-a] wrote → {OUT_SCAPE}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Channel B assembly (focal holdout, fold 0)
# ─────────────────────────────────────────────────────────────────────────────

def build_channel_b() -> None:
    sp2idx = get_species_index()
    n_classes = len(sp2idx)

    df = pd.read_csv(FOLDS_CSV)
    print(f"[channel-b] train_folds.csv: {len(df)} rows, "
          f"{df['primary_label'].nunique()} species, "
          f"fold=0 rows: {(df['fold']==0).sum()}",
          flush=True)

    fold0 = df[df["fold"] == 0].copy()
    # Require ≥2 fold-0 clips so holding out 1 keeps ≥1 for training.
    keep_species = fold0.groupby("primary_label").size()
    keep_species = set(keep_species[keep_species >= 2].index.tolist())
    fold0 = fold0[fold0["primary_label"].isin(keep_species)]
    picked = (
        fold0
        .groupby("primary_label", group_keys=False)
        .sample(n=1, random_state=42)
        .reset_index(drop=True)
    )
    print(f"[channel-b] picked {len(picked)} clips "
          f"({picked['primary_label'].nunique()} species, "
          f"≥2-clip species only)", flush=True)

    n = len(picked)
    mels    = np.zeros((n, 3, N_MELS, 512), dtype=np.float16)
    labels  = np.zeros((n, n_classes), dtype=np.float32)
    fns     = []
    sp_code = []

    t0 = time.time()
    for i, row in picked.iterrows():
        rel = row["filename"]
        path = TRAIN_AUDIO_DIR / rel
        if not path.exists():
            alt = FT_ROOT.parent / "data" / "external" / "birdclef_2025" / "train_audio" / rel
            path = alt if alt.exists() else path
        try:
            wav = load_audio(path)
            # center-5s crop; if shorter than 5s, use full clip
            want = WINDOW_SEC * SAMPLE_RATE
            if len(wav) > want:
                start = (len(wav) - want) // 2
                seg = wav[start:start + want]
            else:
                seg = wav
            seg = pad_or_crop(seg, CHUNK_SAMPLES, random_crop=False)
            mels[i] = waveform_to_mel(seg).to(torch.float16).numpy()
        except Exception as ex:
            print(f"  [warn] {rel}: {ex}", flush=True)
            mels[i] = 0.0

        sp = str(row["primary_label"])
        if sp in sp2idx:
            labels[i, sp2idx[sp]] = 1.0
        fns.append(rel)
        sp_code.append(sp)

        if (i + 1) % 50 == 0 or (i + 1) == n:
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            print(f"  [{i+1:>3}/{n}]  {rate:.1f} mels/s", flush=True)

    n_species = int((labels.sum(axis=0) > 0).sum())
    print(f"\n[channel-b] final: {n} clips,  "
          f"{n_species} species,  "
          f"mels {mels.nbytes/1e9:.2f} GB",
          flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_FOCAL,
        mels=mels,
        labels=labels,
        filename=np.array(fns, dtype=object),
        primary_label=np.array(sp_code, dtype=object),
    )
    print(f"[channel-b] wrote → {OUT_FOCAL}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tau", type=float, default=8.0,
                   help="Perch logit threshold for new files (default 8.0 → prec≥0.92)")
    p.add_argument("--cap-per-species", type=int, default=0,
                   help="Max retained records per species (0 = no cap)")
    p.add_argument("--skip-channel-a", action="store_true")
    p.add_argument("--skip-channel-b", action="store_true")
    args = p.parse_args()

    if not args.skip_channel_a:
        print("=" * 60, flush=True)
        print("Channel A — soundscape val (primary gate)", flush=True)
        print("=" * 60, flush=True)
        build_channel_a(args)

    if not args.skip_channel_b:
        print("\n" + "=" * 60, flush=True)
        print("Channel B — focal holdout (per-species diagnostic)", flush=True)
        print("=" * 60, flush=True)
        build_channel_b()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
