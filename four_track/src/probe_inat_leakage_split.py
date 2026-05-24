"""Leakage-split AUC probe for the iNat pretrain backbone.

Decomposes iNat val macro-AUC into:
  - leaked subset: val clips where (rights_holder, category_id) appears in train
  - unleaked subset: val clips where the (recordist, species) pair is NOT in train
  - full: all val clips (the headline number printed by pretrain_inat_sounds.py)

Use: post-epoch-7 sanity check to distinguish honest plateau (both subsets
flatten together) from leakage-saturation (full keeps climbing because leaked
is approaching 1.0 while unleaked has flattened).

Runs CPU-only by default to avoid contending with the running pretrain on the
deepthought GPU. Subsamples val (--n-clips, default 8000) for speed; macro-AUC
on this size is statistically tight.

Usage on deepthought (CPU, while pretrain is training on GPU):
    python -u four_track/src/probe_inat_leakage_split.py \\
        --ckpt /mnt/mytoshiba/MachineLearning/_runon/BirdCLEF/four_track/models/pretrain_inat/inat_best_tf_efficientnet_b0_ns_jft_in1k.pt \\
        --inat-root /home/swatson/work/MachineLearning/kaggle/BirdCLEF/four_track/data/external/inat_sounds_2024 \\
        --n-clips 8000
"""

import argparse
import json
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

FT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FT_ROOT.parent / "src"))
import config  # noqa: E402
from config import BACKBONE  # noqa: E402

sys.path.insert(0, str(FT_ROOT / "src"))
from utils import load_audio, pad_or_crop, waveform_to_mel  # noqa: E402
from model_a1 import BirdSEDModelA1  # noqa: E402


def load_inat_split_jsons(inat_root: Path):
    """Load train.json + val.json from the iNat tarballs."""
    out = {}
    for split in ("train", "val"):
        tgz = inat_root / f"{split}.json.tar.gz"
        if not tgz.exists():
            raise FileNotFoundError(tgz)
        with tarfile.open(tgz, "r:gz") as tf:
            member = next(m for m in tf.getmembers() if m.name.endswith(".json"))
            out[split] = json.load(tf.extractfile(member))
    return out


def build_pair_index(split_json):
    """Index: (rights_holder, category_id) pairs and audio_id → metadata."""
    a2cat = {ann["audio_id"]: ann["category_id"]
             for ann in split_json["annotations"]}
    a2rh = {a["id"]: (a.get("rights_holder", "") or "")
            for a in split_json["audio"]}
    a2fp = {a["id"]: a["file_name"] for a in split_json["audio"]}
    pairs = set()
    for aid, cat in a2cat.items():
        rh = a2rh.get(aid, "")
        pairs.add((rh, cat))
    fp2meta = {a2fp[aid]: {"rights_holder": a2rh[aid],
                          "category_id": a2cat.get(aid)}
               for aid in a2cat}
    return pairs, fp2meta


class ValProbeDataset(Dataset):
    """Single-label iNat val for AUC probing. No augmentation."""

    def __init__(self, df: pd.DataFrame, sp2idx: dict):
        self.df = df.reset_index(drop=True)
        self.sp2idx = sp2idx
        self.n_classes = len(sp2idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav = load_audio(Path(row["abs_path"]))
        wav = pad_or_crop(wav, config.CHUNK_SAMPLES, random_crop=False)
        mel = waveform_to_mel(wav)
        labels = np.zeros(self.n_classes, dtype=np.float32)
        sci = str(row["scientific_name"])
        if sci in self.sp2idx:
            labels[self.sp2idx[sci]] = 1.0
        return mel, torch.from_numpy(labels)


def macro_auc_subset(probs, labels, mask):
    """Macro-AUC over species with ≥1 positive in the row-subset."""
    if mask.sum() == 0:
        return float("nan"), 0
    sub_labels = labels[mask]
    sub_probs = probs[mask]
    present = sub_labels.sum(axis=0) > 0
    n_present = int(present.sum())
    if n_present == 0:
        return float("nan"), 0
    try:
        auc = float(roc_auc_score(
            sub_labels[:, present], sub_probs[:, present], average="macro"))
    except ValueError:
        auc = float("nan")
    return auc, n_present


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--inat-root", type=Path, required=True)
    p.add_argument("--n-clips", type=int, default=8000,
                   help="stratified subsample of val (4k leaked + 4k unleaked)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cpu",
                   help="cpu (default) or cuda — use cpu when GPU is busy")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── Load JSONs + build leak labels ─────────────────────────────────────
    print(f"[probe] loading iNat train.json + val.json", flush=True)
    splits = load_inat_split_jsons(args.inat_root)
    train_pairs, _ = build_pair_index(splits["train"])
    _, val_fp2meta = build_pair_index(splits["val"])
    print(f"  train (recordist, species) pairs: {len(train_pairs)}", flush=True)

    # ── Load manifest, build species index ─────────────────────────────────
    manifest = pd.read_csv(args.inat_root / "inat_manifest.csv", dtype=str)
    manifest["abs_path"] = manifest["file_path"].apply(
        lambda fp: str(args.inat_root / fp))
    species = sorted(manifest["scientific_name"].astype(str).unique())
    sp2idx = {sp: i for i, sp in enumerate(species)}

    val = manifest[manifest["split"] == "val"].copy().reset_index(drop=True)
    val["leak"] = val["file_path"].map(
        lambda fp: (val_fp2meta.get(fp, {}).get("rights_holder", ""),
                    val_fp2meta.get(fp, {}).get("category_id")) in train_pairs)

    # ── Stratified subsample ────────────────────────────────────────────────
    half = args.n_clips // 2
    leaked = val[val.leak].sample(min(half, val.leak.sum()),
                                  random_state=args.seed)
    unleaked = val[~val.leak].sample(min(half, (~val.leak).sum()),
                                     random_state=args.seed)
    sample = pd.concat([leaked, unleaked]).reset_index(drop=True)
    print(f"[probe] sampled {len(sample)} val clips "
          f"({len(leaked)} leaked, {len(unleaked)} unleaked)", flush=True)

    # ── Model ───────────────────────────────────────────────────────────────
    print(f"[probe] loading ckpt: {args.ckpt}", flush=True)
    model = BirdSEDModelA1(
        backbone_name=BACKBONE, n_classes=len(species),
        mixstyle_p=0.0,
    )
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for key in ("state_dict", "model"):
            if key in state:
                state = state[key]
                break
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if len(missing) > 20:
        raise RuntimeError(
            f"refused to probe with {len(missing)} missing keys — "
            f"ckpt structure not understood; first missing: {missing[:5]}"
        )
    device = torch.device(args.device)
    model = model.to(device).eval()

    # ── Inference ───────────────────────────────────────────────────────────
    ds = ValProbeDataset(sample, sp2idx)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=False)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for i, (mels, labels) in enumerate(dl):
            mels = mels.to(device)
            out = model(mels)
            probs = torch.sigmoid(out["clip_logits"]).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
            if (i + 1) % 25 == 0:
                print(f"  batch {i+1}/{len(dl)}", flush=True)

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    leak_mask = sample["leak"].to_numpy()

    # ── Three AUCs ──────────────────────────────────────────────────────────
    full_auc, full_n = macro_auc_subset(probs, labels, np.ones(len(labels), dtype=bool))
    leak_auc, leak_n = macro_auc_subset(probs, labels, leak_mask)
    unleak_auc, unleak_n = macro_auc_subset(probs, labels, ~leak_mask)

    print()
    print("=" * 60)
    print("iNat leakage-split macro-AUC")
    print("=" * 60)
    print(f"  ckpt:      {args.ckpt}")
    print(f"  n_clips:   {len(sample)} (leaked={leak_mask.sum()}, "
          f"unleaked={(~leak_mask).sum()})")
    print(f"  FULL  AUC: {full_auc:.4f}  (n_present={full_n})")
    print(f"  LEAK  AUC: {leak_auc:.4f}  (n_present={leak_n})")
    print(f"  UNLEAK AUC:{unleak_auc:.4f}  (n_present={unleak_n})")
    print(f"  GAP (leak - unleak): {leak_auc - unleak_auc:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
