"""Generate pseudo-labels for all train_soundscapes using the Stage 1 ensemble.

For each soundscape:
  1. Slide a 20-second window with 5-second stride over the full audio.
  2. Run the N-fold ensemble → frame_logits (T' backbone frames per window).
  3. Split T' frames into 4 groups (one per 5-second sub-segment).
  4. Take max over frames within each sub-segment → per-segment prediction.
  5. Average predictions from overlapping windows per 5-second segment.

Output: data/processed/pseudo_labels_v1.csv
  Columns: filename, start_time, end_time, <species_0> … <species_233>

Usage:
  python src/pseudo_label.py [--backbone NAME] [--seed N] [--folds 0,1,2,3,4]
                              [--output PATH] [--version 1]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

import config
from model import BirdSEDModel
from utils import load_audio, pad_or_crop, waveform_to_mel

SEGMENT_SEC       = 5                           # pseudo-label window size
WINDOW_SEC        = 20                          # inference chunk = CHUNK_SAMPLES
STRIDE_SEC        = 5                           # sliding window stride
N_SEGS_PER_WINDOW = WINDOW_SEC // STRIDE_SEC   # 4 sub-segments per window


# ── Model helpers ──────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Path, device: torch.device, backbone_name: str = None) -> BirdSEDModel:
    model = BirdSEDModel(
        backbone_name=backbone_name or config.BACKBONE,
        n_classes=config.N_CLASSES,
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_soundscape(
    models: list,
    waveform: np.ndarray,
    device: torch.device,
    autocast_ctx,
) -> np.ndarray:
    """Pseudo-label a full soundscape via sliding 20-second window.

    Returns (n_segments, N_CLASSES) float32 array of ensemble-averaged,
    max-over-frames probabilities for each 5-second segment.
    """
    total_samples = len(waveform)
    total_sec     = total_samples / config.SAMPLE_RATE
    n_segments    = int(np.ceil(total_sec / SEGMENT_SEC))

    pred_sum   = np.zeros((n_segments, config.N_CLASSES), dtype=np.float64)
    pred_count = np.zeros(n_segments, dtype=np.int32)

    window_start_sec = 0.0
    while window_start_sec < total_sec:
        start_samp = int(window_start_sec * config.SAMPLE_RATE)
        end_samp   = start_samp + config.CHUNK_SAMPLES
        chunk = waveform[start_samp:min(end_samp, total_samples)]
        chunk = pad_or_crop(chunk, config.CHUNK_SAMPLES, random_crop=False)

        mel = waveform_to_mel(chunk).unsqueeze(0).to(device)  # (1, 3, N_MELS, T)

        # Ensemble: accumulate frame probabilities across folds
        frame_probs_acc = None
        for m in models:
            with autocast_ctx:
                out = m(mel)
            # frame_logits: (1, T', N_CLASSES)
            fp = torch.sigmoid(out["frame_logits"]).float().cpu().numpy()[0]
            if frame_probs_acc is None:
                frame_probs_acc = fp.copy()
            else:
                frame_probs_acc += fp
        frame_probs_acc /= len(models)   # (T', N_CLASSES)

        T_prime         = frame_probs_acc.shape[0]
        frames_per_seg  = max(1, T_prime // N_SEGS_PER_WINDOW)
        first_seg_idx   = int(window_start_sec / SEGMENT_SEC)

        for sub in range(N_SEGS_PER_WINDOW):
            seg_idx = first_seg_idx + sub
            if seg_idx >= n_segments:
                break
            f_start = sub * frames_per_seg
            f_end   = (sub + 1) * frames_per_seg if sub < N_SEGS_PER_WINDOW - 1 else T_prime
            seg_max = frame_probs_acc[f_start:f_end].max(axis=0)   # (N_CLASSES,)
            pred_sum[seg_idx]   += seg_max
            pred_count[seg_idx] += 1

        window_start_sec += STRIDE_SEC

    # Normalise by number of contributing windows
    mask = pred_count > 0
    pred_sum[mask] /= pred_count[mask, None]

    return pred_sum.astype(np.float32)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-labels for train_soundscapes")
    parser.add_argument("--backbone", default=config.BACKBONE)
    parser.add_argument("--seed",     type=int, default=config.SEED)
    parser.add_argument("--folds",    default=None,
                        help="Comma-separated fold indices, e.g. '0,1,2,3,4'")
    parser.add_argument("--version",  type=int, default=1,
                        help="Output version suffix (default: 1 → pseudo_labels_v1.csv)")
    parser.add_argument("--ckpt-version", default=None,
                        help="Checkpoint version suffix, e.g. '2' → sed_*_v2.pt (default: no suffix)")
    parser.add_argument("--output",   default=None,
                        help="Override output path (default: data/processed/pseudo_labels_vN.csv)")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else \
        config.PROC / f"pseudo_labels_v{args.version}.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)

    species_list = config.get_species_list()
    n_classes    = len(species_list)
    print(f"Species: {n_classes}")

    # ── Load models ────────────────────────────────────────────────────────────
    fold_ids = list(range(config.N_FOLDS))
    if args.folds is not None:
        fold_ids = [int(f) for f in args.folds.split(",")]

    print("Loading checkpoints …")
    models = []
    vsuffix = f"_v{args.ckpt_version}" if args.ckpt_version else ""
    for fold in fold_ids:
        ckpt = config.MODELS / f"sed_{args.backbone}_fold{fold}_seed{args.seed}{vsuffix}.pt"
        if not ckpt.exists():
            print(f"  [WARN] not found: {ckpt}")
            continue
        m = load_model(ckpt, device, backbone_name=args.backbone)
        models.append(m)
        print(f"  fold {fold} loaded ✓")
    assert models, "No checkpoints found — check --backbone, --seed, --folds args"
    print(f"Ensemble: {len(models)} models")

    # ── Find soundscape files ──────────────────────────────────────────────────
    soundscapes_dir = config.RAW / "train_soundscapes"
    ogg_files       = sorted(soundscapes_dir.glob("*.ogg"))
    print(f"Soundscape files: {len(ogg_files)}")

    # ── Generate pseudo-labels ─────────────────────────────────────────────────
    rows = []
    t0   = time.time()

    for audio_path in tqdm(ogg_files, desc="Pseudo-labeling"):
        waveform  = load_audio(audio_path)
        total_sec = len(waveform) / config.SAMPLE_RATE

        probs = predict_soundscape(models, waveform, device, autocast_ctx)
        # probs: (n_segments, N_CLASSES)

        for seg_idx, seg_probs in enumerate(probs):
            start_t = seg_idx * SEGMENT_SEC
            end_t   = start_t + SEGMENT_SEC
            if start_t >= total_sec:
                break
            row = {
                "filename":   audio_path.name,
                "start_time": start_t,
                "end_time":   min(end_t, total_sec),
            }
            for sp, prob in zip(species_list, seg_probs):
                row[sp] = float(prob)
            rows.append(row)

    elapsed      = time.time() - t0
    mins, secs   = divmod(int(elapsed), 60)
    print(f"\nInference complete — {mins}m {secs:02d}s")

    # ── Save output ────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    config.PROC.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    n_files    = df["filename"].nunique()
    n_rows     = len(df)
    max_probs  = df[species_list].max(axis=0)
    active_sp  = int((max_probs > 0.5).sum())
    mean_max   = float(max_probs.mean())

    print(f"Saved {n_rows} rows ({n_files} files) → {output_path}")
    print(f"  Species with max_prob > 0.5 : {active_sp}/{n_classes}")
    print(f"  Mean max prob per species   : {mean_max:.4f}")

    # Sampler weight hint (sum of max probs per soundscape → used by self_train.py)
    weight_df = df.groupby("filename")[species_list].max().sum(axis=1).reset_index()
    weight_df.columns = ["filename", "sampler_weight"]
    weight_path = output_path.parent / f"pseudo_labels_v{args.version}_weights.csv"
    weight_df.to_csv(weight_path, index=False)
    print(f"Sampler weights → {weight_path}")


if __name__ == "__main__":
    main()
