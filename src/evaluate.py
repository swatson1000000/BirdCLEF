"""Evaluate Stage 1 ensemble on expert-labeled train soundscapes.

Computes per-species ROC-AUC against data/raw/train_soundscapes_labels.csv
(the same domain as the competition test set) and writes:

  data/processed/hard_species_stage1.txt    — worst-30 species by AUC
  data/processed/per_species_auc_stage1.csv — full per-species AUC table
  data/processed/eval_stage1_predictions.csv — raw prediction scores

Usage:
  python src/evaluate.py [--backbone NAME] [--seed N] [--folds 0,1,2,3,4]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

import config
from model import BirdSEDModel
from utils import load_audio, pad_or_crop, waveform_to_mel


# ── Model helpers ──────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Path, device: torch.device) -> BirdSEDModel:
    model = BirdSEDModel(
        backbone_name=config.BACKBONE,
        n_classes=config.N_CLASSES,
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ── Label parsing ──────────────────────────────────────────────────────────────

def parse_hms(t: str) -> float:
    """Parse HH:MM:SS or MM:SS string to total seconds (float)."""
    parts = str(t).strip().split(":")
    parts = [int(p) for p in parts]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return float(parts[0])


# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_segment(
    models: list,
    waveform: np.ndarray,
    start_s: float,
    end_s: float,
    device: torch.device,
    autocast_ctx,
) -> np.ndarray:
    """Ensemble predict a (start_s, end_s) window → (N_CLASSES,) probs."""
    start_samp = int(start_s * config.SAMPLE_RATE)
    end_samp   = int(end_s   * config.SAMPLE_RATE)
    seg = waveform[start_samp:end_samp]
    seg = pad_or_crop(seg, config.CHUNK_SAMPLES, random_crop=False)

    mel = waveform_to_mel(seg).unsqueeze(0).to(device)  # (1, 3, N_MELS, T)

    probs_acc = np.zeros(config.N_CLASSES, dtype=np.float64)
    for m in models:
        with autocast_ctx:
            out = m(mel)
        probs_acc += torch.sigmoid(out["clip_logits"]).float().cpu().numpy()[0]
    return (probs_acc / len(models)).astype(np.float32)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default=config.BACKBONE)
    parser.add_argument("--seed",     type=int, default=config.SEED)
    parser.add_argument("--folds",    default=None,
                        help="Comma-separated fold indices, e.g. '0,1,2,3,4'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)

    # ── Species metadata ───────────────────────────────────────────────────────
    species_list = config.get_species_list()
    species_idx  = config.get_species_index()
    n_classes    = len(species_list)
    print(f"Species: {n_classes}")

    # ── Load models ────────────────────────────────────────────────────────────
    fold_ids = list(range(config.N_FOLDS))
    if args.folds is not None:
        fold_ids = [int(f) for f in args.folds.split(",")]

    print("Loading checkpoints …")
    models = []
    for fold in fold_ids:
        ckpt = config.MODELS / f"sed_{args.backbone}_fold{fold}_seed{args.seed}.pt"
        if not ckpt.exists():
            print(f"  [WARN] not found: {ckpt}")
            continue
        m = load_model(ckpt, device)
        models.append(m)
        print(f"  fold {fold} loaded ✓")
    assert models, "No checkpoints found — check backbone/seed/fold args"
    print(f"Ensemble size: {len(models)} models")

    # ── Load ground-truth labels ───────────────────────────────────────────────
    labels_path = config.RAW / "train_soundscapes_labels.csv"
    labels_df   = pd.read_csv(labels_path).reset_index(drop=True)
    n_rows      = len(labels_df)
    n_files     = labels_df["filename"].nunique()
    print(f"Labels: {n_rows} windows across {n_files} soundscape files")

    # ── Build ground-truth matrix ──────────────────────────────────────────────
    gt_matrix = np.zeros((n_rows, n_classes), dtype=np.float32)
    for i, row in labels_df.iterrows():
        for sp in str(row["primary_label"]).split(";"):
            sp = sp.strip()
            if sp in species_idx:
                gt_matrix[i, species_idx[sp]] = 1.0

    # ── Run inference ──────────────────────────────────────────────────────────
    pred_matrix    = np.zeros((n_rows, n_classes), dtype=np.float32)
    soundscapes_dir = config.RAW / "train_soundscapes"
    missing_files  = 0

    t0     = time.time()
    groups = labels_df.groupby("filename", sort=False)

    for fname, grp in tqdm(groups, total=n_files, desc="Soundscapes"):
        audio_path = soundscapes_dir / fname
        if not audio_path.exists():
            missing_files += 1
            continue

        waveform = load_audio(audio_path)

        for row_idx, row in grp.iterrows():
            start_s = parse_hms(row["start"])
            end_s   = parse_hms(row["end"])
            pred_matrix[row_idx] = predict_segment(
                models, waveform, start_s, end_s, device, autocast_ctx
            )

    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    print(f"\nInference complete — {mins}m {secs:02d}s  ({missing_files} files missing)")

    # ── Per-species ROC-AUC ────────────────────────────────────────────────────
    per_species_auc = {}
    skipped = 0
    for i, sp in enumerate(species_list):
        gt_col = gt_matrix[:, i]
        n_pos  = int(gt_col.sum())
        if n_pos == 0 or n_pos == n_rows:
            skipped += 1
            continue
        auc = roc_auc_score(gt_col, pred_matrix[:, i])
        per_species_auc[sp] = float(auc)

    sorted_auc  = sorted(per_species_auc.items(), key=lambda x: x[1])
    macro_auc   = float(np.mean(list(per_species_auc.values())))
    n_evaluated = len(per_species_auc)

    print(f"\n{'='*56}")
    print(f"Stage 1 Ensemble — Macro ROC-AUC: {macro_auc:.4f}")
    print(f"Evaluated: {n_evaluated}/{n_classes} species  (skipped {skipped} with no positive labels)")
    print(f"{'='*56}")

    print("\n── Worst 30 species ────────────────────────────────────")
    for sp, auc in sorted_auc[:30]:
        print(f"  {sp:<22s}  {auc:.4f}")

    print("\n── Best 30 species ─────────────────────────────────────")
    for sp, auc in sorted_auc[-30:][::-1]:
        print(f"  {sp:<22s}  {auc:.4f}")

    # ── Save outputs ───────────────────────────────────────────────────────────
    config.PROC.mkdir(parents=True, exist_ok=True)

    # Hard species (worst 30)
    hard_path = config.PROC / "hard_species_stage1.txt"
    with open(hard_path, "w") as f:
        f.write(f"# Worst-30 species by val ROC-AUC (Stage 1, {len(models)}-fold ensemble)\n")
        f.write(f"# Backbone: {args.backbone}  Seed: {args.seed}\n")
        f.write(f"# Macro ROC-AUC: {macro_auc:.4f}\n")
        for sp, auc in sorted_auc[:30]:
            f.write(f"{sp}\t{auc:.4f}\n")
    print(f"\nSaved hard species list  → {hard_path}")

    # Full per-species AUC table
    auc_df   = pd.DataFrame(sorted_auc, columns=["species", "val_roc_auc"])
    auc_path = config.PROC / "per_species_auc_stage1.csv"
    auc_df.to_csv(auc_path, index=False)
    print(f"Saved per-species AUC    → {auc_path}")

    # Raw prediction matrix
    pred_df = pd.DataFrame(pred_matrix, columns=species_list)
    pred_df.insert(0, "filename", labels_df["filename"].values)
    pred_df.insert(1, "start",    labels_df["start"].values)
    pred_df.insert(2, "end",      labels_df["end"].values)
    pred_path = config.PROC / "eval_stage1_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved raw predictions    → {pred_path}")


if __name__ == "__main__":
    main()
