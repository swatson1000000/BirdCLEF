"""L4 step 1: Generate L3-prec 5-fold pseudo-labels on unlabeled train_soundscapes.

Mirrors the production A2 pipeline (a2_pseudo_label_a1.py) but uses the
L3-precursor 5-fold ckpts (seed123_ce) as the teacher instead of A1's
JIT bundle. The L3-prec ensemble at broader-pool 0.8700 is +0.030 stronger
than the A1 teacher that A2 used; the L4 hypothesis is that round-2
pseudo from a stronger SED teacher breaks the A2-self-anchor that
ruined A3 (per plan §678).

Differences from a2_pseudo_label_a1.py:
- Loads raw BirdSEDModelA1 state_dicts (seed123_ce), not JIT bundles.
- Uses utils.waveform_to_mel + pad_or_crop (matches train_a1.py exactly).
- Output naming: l3prec_pseudo_soundscape.npz (downstream
  a2_build_pseudo_manifest.py --probs reads this directly).

Output: data/processed/l3prec_pseudo_soundscape.npz with keys
  filenames  (n_windows,) <U
  start_sec  (n_windows,) int32
  probs      (n_windows, 234) float32  — sigmoid-mean across 5 L3-prec folds

Run via:
  python -u src/l4_emit_pseudo_l3prec.py 2>&1 | tee log/l4_emit_$(date +%Y%m%d_%H%M%S).log
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
FT_ROOT = HERE.parent
ROOT = FT_ROOT.parent
PARENT_SRC = ROOT / "src"

if str(PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(PARENT_SRC))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config  # noqa: E402
from config import RAW, SAMPLE_RATE, CHUNK_SAMPLES  # noqa: E402
from utils import load_audio, pad_or_crop, waveform_to_mel  # noqa: E402
from model_a1 import BirdSEDModelA1  # noqa: E402

CKPT_DIR = FT_ROOT / "models" / "a1"
CKPT_TEMPLATE = "a1_tf_efficientnet_b0.ns_jft_in1k_fold{f}_seed123_ce.pt"
FOLDS = [0, 1, 2, 3, 4]

TRAIN_SS_DIR = RAW / "train_soundscapes"
LABELS_CSV = RAW / "train_soundscapes_labels.csv"
OUT_PATH = FT_ROOT / "data" / "processed" / "l3prec_pseudo_soundscape.npz"

WINDOW_SEC = 5
WINDOWS_PER_FILE = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N unlabeled files (smoke test)")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    labels_df = pd.read_csv(LABELS_CSV)
    labeled = set(labels_df["filename"].unique())
    all_files = sorted(p.name for p in TRAIN_SS_DIR.glob("*.ogg"))
    unlabeled = sorted(f for f in all_files if f not in labeled)
    print(f"Total: {len(all_files)},  labeled: {len(labeled)},  "
          f"unlabeled: {len(unlabeled)}", flush=True)
    if args.limit:
        unlabeled = unlabeled[: args.limit]
        print(f"Limited to first {len(unlabeled)} files for smoke test", flush=True)

    print("Loading 5 L3-prec ckpts (seed123_ce) …", flush=True)
    models = []
    for f in FOLDS:
        p = CKPT_DIR / CKPT_TEMPLATE.format(f=f)
        assert p.exists(), f"missing ckpt: {p}"
        m = BirdSEDModelA1(backbone_name=config.BACKBONE, mixstyle_p=0.0)
        sd = torch.load(p, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        missing, unexpected = m.load_state_dict(sd, strict=False)
        if missing:
            print(f"  fold {f}: {len(missing)} missing keys", flush=True)
        if unexpected:
            print(f"  fold {f}: {len(unexpected)} unexpected keys "
                  f"(first 3: {sorted(unexpected)[:3]})", flush=True)
        m = m.to(device).eval()
        models.append(m)
        print(f"  fold {f}: {p.stat().st_size / 1e6:.1f} MB loaded", flush=True)

    n_files = len(unlabeled)
    n_rows = n_files * WINDOWS_PER_FILE
    probs = np.zeros((n_rows, config.N_CLASSES), dtype=np.float32)
    filenames_out = np.empty(n_rows, dtype=object)
    start_sec_out = np.zeros(n_rows, dtype=np.int32)

    print(f"Processing {n_files} files × {WINDOWS_PER_FILE} windows × "
          f"{len(models)} folds …", flush=True)
    t_start = time.time()
    row = 0
    for fi, fname in enumerate(unlabeled):
        try:
            wav = load_audio(TRAIN_SS_DIR / fname)
            if len(wav) < SAMPLE_RATE * 60:
                wav = np.pad(wav, (0, SAMPLE_RATE * 60 - len(wav)))
            elif len(wav) > SAMPLE_RATE * 60:
                wav = wav[: SAMPLE_RATE * 60]

            file_mels = []
            for w in range(WINDOWS_PER_FILE):
                s = w * WINDOW_SEC * SAMPLE_RATE
                e = s + WINDOW_SEC * SAMPLE_RATE
                seg = wav[s:e]
                seg = pad_or_crop(seg, CHUNK_SAMPLES, random_crop=False)
                file_mels.append(waveform_to_mel(seg))
            batch = torch.stack(file_mels).to(device)

            sig_sum = torch.zeros(
                (WINDOWS_PER_FILE, config.N_CLASSES),
                device=device, dtype=torch.float32,
            )
            with torch.no_grad():
                for m in models:
                    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                        out = m(batch)
                    sig_sum += torch.sigmoid(out["clip_logits"]).float()
            sig_mean = (sig_sum / len(models)).cpu().numpy()

            for w in range(WINDOWS_PER_FILE):
                probs[row + w] = sig_mean[w]
                filenames_out[row + w] = fname
                start_sec_out[row + w] = w * WINDOW_SEC
            row += WINDOWS_PER_FILE

            if (fi + 1) % 50 == 0 or fi == 0:
                el = time.time() - t_start
                eta = el / (fi + 1) * (n_files - fi - 1)
                print(f"  [{fi + 1}/{n_files}] {fname[:50]:<50} "
                      f"cum={el:.1f}s ({el / (fi + 1):.2f}s/file)  "
                      f"ETA {eta / 60:.1f}m", flush=True)
        except Exception as e:
            print(f"  [err] {fname}: {e}", flush=True)
            row += WINDOWS_PER_FILE

    el = time.time() - t_start
    print(f"\nDone. {n_files} files in {el:.1f}s ({el / n_files:.2f}s/file)",
          flush=True)

    valid = filenames_out != None  # noqa: E711
    print(f"Valid rows: {valid.sum()}/{n_rows}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        filenames=filenames_out[valid].astype("U64"),
        start_sec=start_sec_out[valid],
        probs=probs[valid],
    )
    print(f"Saved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)",
          flush=True)


if __name__ == "__main__":
    main()
