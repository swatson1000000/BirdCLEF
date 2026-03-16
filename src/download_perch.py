"""Download Google Perch (bird-vocalization-classifier) model and labels.

Run this script ONCE to download the TF SavedModel and eBird label list.
Then upload the resulting models/perch/ folder to a Kaggle dataset.

Usage:
    # In a Python environment with TensorFlow installed:
    pip install tensorflow tensorflow-hub

    cd /home/swatson/work/MachineLearning/kaggle/BirdCLEF
    python src/download_perch.py

Outputs:
    models/perch/model/        <- TF SavedModel (to upload to Kaggle dataset)
    models/perch/ebird2021.csv <- Perch label list (9736 eBird codes)

Kaggle dataset setup:
    1. Create a new dataset: kaggle datasets create \
           -p models/perch \
           --title "BirdCLEF 2026 Perch" \
           --user stevewatson999 \
           --public
    2. Use input path: /kaggle/input/birdclef2026-perch
"""

import sys
import urllib.request
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT / "models" / "perch"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PERCH_URL  = "https://tfhub.dev/google/bird-vocalization-classifier/4"
LABELS_OUT = OUT_DIR / "ebird2021.csv"
MODEL_OUT  = OUT_DIR / "model"

# Fallback label URLs (the chirp repo has moved the file around)
_LABEL_URLS = [
    "https://raw.githubusercontent.com/google-research/perch/main/chirp/data/class_lists/ebird2021.csv",
    "https://raw.githubusercontent.com/google-research/chirp/main/chirp/data/class_lists/ebird2021.csv",
    "https://raw.githubusercontent.com/google-research/perch/main/chirp/data/class_lists/world_birds.csv",
]


def download_labels() -> None:
    if LABELS_OUT.exists():
        print(f"Labels already exist: {LABELS_OUT}")
        return
    for url in _LABEL_URLS:
        try:
            print(f"Downloading label list from: {url}")
            urllib.request.urlretrieve(url, LABELS_OUT)
            print(f"  Saved → {LABELS_OUT}")
            return
        except Exception as e:
            print(f"  Failed ({e}), trying next …")
    raise RuntimeError("All label URLs failed. Extract labels from SavedModel assets instead.")


def extract_labels_from_model() -> None:
    """Fallback: extract label CSV from TF Hub module cache (assets/label.csv)."""
    import glob
    patterns = [
        "/tmp/tfhub_modules/*/assets/label.csv",
        str(Path.home() / ".cache" / "tfhub_modules" / "*" / "assets" / "label.csv"),
        str(MODEL_OUT / "assets" / "label.csv"),
    ]
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            import shutil
            shutil.copy(files[0], LABELS_OUT)
            print(f"  Extracted labels from TF Hub cache: {files[0]}")
            return
    print("  [warn] Label file not found — please copy assets/label.csv from the "
          "TF Hub module cache manually.")


def download_model() -> None:
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
    except ImportError:
        sys.exit(
            "TensorFlow / tensorflow-hub not installed.\n"
            "Run:  pip install tensorflow tensorflow-hub\n"
            "Then re-run this script."
        )

    if MODEL_OUT.exists() and any(MODEL_OUT.iterdir()):
        print(f"Model already exists: {MODEL_OUT}")
        return

    print(f"Downloading Perch SavedModel from TF Hub …")
    print(f"  URL:  {PERCH_URL}")
    print(f"  This may take a few minutes …")
    model = hub.load(PERCH_URL)

    print(f"Saving SavedModel → {MODEL_OUT}")
    tf.saved_model.save(model, str(MODEL_OUT))
    print(f"  Done.")


def verify() -> None:
    """Quick sanity check: load model + run dummy inference."""
    import numpy as np
    try:
        import tensorflow as tf
    except ImportError:
        print("TF not available — skipping verification")
        return

    print("\nVerifying model …")
    model = tf.saved_model.load(str(MODEL_OUT))
    dummy = tf.zeros([1, 160_000])          # 5s at 32kHz
    logits, embeddings = model.infer_tf(dummy)
    print(f"  logits shape:     {logits.shape}")        # (1, 9736)
    print(f"  embeddings shape: {embeddings.shape}")
    print("  Verification OK!")


if __name__ == "__main__":
    download_labels_ok = True
    try:
        download_labels()
    except RuntimeError as e:
        print(f"[warn] {e}")
        download_labels_ok = False

    download_model()

    if not download_labels_ok:
        extract_labels_from_model()

    verify()
    print(f"\nNext steps:")
    print(f"  1. Upload {OUT_DIR} to Kaggle dataset (see docstring for command)")
    print(f"  2. Attach dataset to your Kaggle notebook")
