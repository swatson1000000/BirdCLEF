"""§21 Option 1 closeout — eval bg-noise fold-0 on the 304-window clean OOF.

Loads:
  - bg-noise fold-0 ckpt (single-fold) at
    models/a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_hybrid_prodft.pt
  - 1478-window soundscape pool (filenames, start_secs, y_true) from
    four_track/data/v4_5fold_soundscape_oof.npz
  - Reproduces the seed=42 0.8 train/val split → 14 held-out files / 304 windows

Computes:
  - macro AUC on 304 held-out windows (apples-to-apples vs v4 fold-0 alone = 0.8022)

Gate (per §21): must beat 0.8022 by ≥+0.005 → ≥0.8072 to earn 5-fold training.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
FT = HERE.parent
ROOT = FT.parent

# Parent src on path for shared modules
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(FT / "src"))

import config  # noqa: E402
from config import get_species_index  # noqa: E402
from utils import load_audio, pad_or_crop, waveform_to_mel  # noqa: E402
from model_a1 import BirdSEDModelA1  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RAW = ROOT / "data" / "raw" / "train_soundscapes"
CKPT = FT / "models" / "a1" / "a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_hybrid_prodft.pt"


def reproduce_split() -> set:
    """Reproduce perch_v2 seed=42 0.8 split → return held-out 14 files."""
    df = pd.read_csv(ROOT / "data" / "raw" / "train_soundscapes_labels.csv")
    files = sorted(df["filename"].unique())
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(files))
    n_train = max(1, int(len(files) * 0.8))
    return set(np.array(files)[idx[n_train:]])


def main() -> None:
    print(f"=== §21 Option 1 (bg-noise) fold-0 304-window OOF eval ===", flush=True)
    print(f"ckpt: {CKPT}", flush=True)
    print(f"device: {DEVICE}", flush=True)

    # 1. Load 1478 OOF reference
    oof = np.load(FT / "data" / "v4_5fold_soundscape_oof.npz", allow_pickle=True)
    filenames = np.array([str(f) for f in oof["filenames"]])
    start_secs = oof["start_sec"]
    y_true = oof["y_true"]

    # 2. Build held-out mask
    held = reproduce_split()
    mask = np.array([fn in held for fn in filenames])
    print(f"\nheld-out windows: {mask.sum()} of {len(mask)}", flush=True)

    # 3. Load model
    sp2idx = get_species_index()
    n_classes = config.N_CLASSES
    model = BirdSEDModelA1(n_classes=n_classes)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE).eval()
    print("model loaded", flush=True)

    # 4. Build mel for each held-out window and run forward
    print("\nscoring 304 held-out windows ...", flush=True)
    probs = np.zeros((mask.sum(), n_classes), dtype=np.float32)
    y_held = y_true[mask]
    fns_held = filenames[mask]
    secs_held = start_secs[mask]

    # Cache audio per file
    audio_cache: dict = {}
    chunk = config.CHUNK_SAMPLES
    sr = config.SAMPLE_RATE
    import time

    t0 = time.time()
    with torch.no_grad():
        for i, (fn, ss) in enumerate(zip(fns_held, secs_held)):
            if fn not in audio_cache:
                audio_cache[fn] = load_audio(RAW / str(fn))
            full = audio_cache[fn]
            s_idx = int(ss) * sr
            e_idx = s_idx + chunk
            wav = full[s_idx:e_idx]
            wav = pad_or_crop(wav, chunk, random_crop=False)
            mel = waveform_to_mel(wav)  # (n_mels, T) — may be ndarray or Tensor
            if isinstance(mel, torch.Tensor):
                mel_t = mel.unsqueeze(0).to(DEVICE)
            else:
                mel_t = torch.from_numpy(mel).unsqueeze(0).to(DEVICE)
            out = model(mel_t)
            # BirdSEDModelA1 returns dict with clip_logits (B, n_classes)
            logits = out["clip_logits"]
            probs[i] = torch.sigmoid(logits).cpu().numpy()[0]
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{mask.sum()} ({time.time()-t0:.1f}s)", flush=True)

    print(f"scoring done in {time.time()-t0:.1f}s", flush=True)

    # 5. Macro AUC
    def macro_auc(y, p):
        aucs = []
        for c in range(y.shape[1]):
            if y[:, c].sum() > 0 and y[:, c].sum() < len(y):
                aucs.append(roc_auc_score(y[:, c], p[:, c]))
        return float(np.mean(aucs)), len(aucs)

    auc, n_c = macro_auc(y_held, probs)
    print(f"\n=== RESULT ===", flush=True)
    print(f"bg-noise fold-0 alone on 304 held-out: {auc:.4f}  ({n_c} classes)", flush=True)
    print(f"", flush=True)
    print(f"§21 gate:  ≥ 0.8072 (v4 fold-0 0.8022 + 0.005)", flush=True)
    print(f"PASS GATE: {'YES' if auc >= 0.8072 else 'NO'}", flush=True)
    print(f"", flush=True)
    print(f"Apples-to-apples vs other 304-held-out fold-0 results:", flush=True)
    print(f"  v4 fold-0 alone:           0.8022", flush=True)
    print(f"  bg-noise fold-0 alone:     {auc:.4f}   (Δ vs v4: {auc-0.8022:+.4f})", flush=True)

    np.savez_compressed(
        FT / "data" / "bgnoise_fold0_304oof.npz",
        probs=probs.astype(np.float32),
        y_true=y_held,
        filenames=fns_held,
        start_sec=secs_held,
        auc=auc,
    )
    print(f"\nsaved: four_track/data/bgnoise_fold0_304oof.npz", flush=True)


if __name__ == "__main__":
    main()
