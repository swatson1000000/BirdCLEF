"""Build the perch-train-audio-extract notebook from source cells.

Run from this directory:
    python build_notebook.py

Produces: birdclef2026-perch-train-audio-extract.ipynb
"""
from __future__ import annotations
import json
from pathlib import Path


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True) or [""],
    }


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True) or [""],
    }


CELL_0 = r'''# Cell 0 — Install TF 2.20 + ONNXRuntime (ONNX preferred; v5 = own ONNX dataset)
#
# v2/v3 (GPU) failed: TF 2.19 on Kaggle GPU image can't deserialize the
# JAX-exported XlaCallModule ops in either perch_v2/2 or perch_v2_cpu/1.
# v4 reverted to CPU runtime + TF 2.20, but brucewu's dataset failed to
# mount under /kaggle/input/ → silently fell back to slow TF SavedModel
# (54.81 h projected → over 9h CPU slot cap).
# v5 ships our own dataset (stevewatson999/birdclef-2026-perch-onnx)
# containing perch_v2_no_dft.onnx + onnxruntime+deps wheels.
import subprocess, sys, os
from pathlib import Path

# Debug: list what's actually mounted at /kaggle/input/
import sys as _sys
print("=== /kaggle/input/ contents ===", flush=True)
_sys.stdout.flush()
for entry in sorted(Path("/kaggle/input").iterdir()):
    if entry.is_dir():
        n = sum(1 for _ in entry.iterdir())
        print(f"  {entry.name}/  ({n} entries)", flush=True)
    else:
        print(f"  {entry.name}", flush=True)
print("===============================", flush=True)
_sys.stdout.flush()

# TF 2.20 wheels (needed for newer XlaCallModule support; Kaggle CPU image ships TF 2.18/2.19)
!pip install -q --no-deps /kaggle/input/notebooks/ashok205/tf-wheels/tf_wheels/tensorboard-2.20.0-py3-none-any.whl
!pip install -q --no-deps /kaggle/input/notebooks/ashok205/tf-wheels/tf_wheels/tensorflow-2.20.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# ONNXRuntime from our own bundled wheels (3-5x faster than TF on CPU)
_WHEEL_DIR = Path("/kaggle/input/birdclef-2026-perch-onnx/wheels")
_INSTALL_ONLY = {'onnxruntime', 'flatbuffers', 'protobuf', 'sympy', 'mpmath', 'packaging'}
if _WHEEL_DIR.exists():
    for whl in sorted(_WHEEL_DIR.glob('*.whl')):
        pkg_name = whl.name.split('-')[0].lower().replace('_', '-')
        if pkg_name in _INSTALL_ONLY or any(pkg_name.startswith(x) for x in _INSTALL_ONLY):
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-deps', '--quiet', str(whl)])
    print('ONNXRuntime wheels installed.')
else:
    print('WARNING: own ONNX wheel dir not found, will use TF-only path (slower)')
'''


CELL_1 = r'''# Cell 1 — Mode switch
#
# SMOKE = True  → extract first SMOKE_N clips only (fast sanity check + wall-time projection)
# SMOKE = False → extract all 35,549 train_audio focal clips
#
# PARTITION_ID / PARTITION_COUNT:
#   v5 smoke @ ONNXRuntime rate 0.67 clips/s → 14.81 h for 35,549 clips.
#   Kaggle CPU slot cap is 9 h → split into 3 partitions (≈5 h each) via
#   modular hash on sorted clip index:  keep clip i iff i % PARTITION_COUNT == PARTITION_ID.
#   Alphabetical-species sort keeps each partition roughly species-balanced.
#   Between pushes, edit PARTITION_ID only (0 → 1 → 2) and re-run `python build_notebook.py`.
SMOKE          = False
SMOKE_N        = 100         # unused when SMOKE=False
MAX_CLIPS      = None        # None = process everything in the partition
PARTITION_ID   = 2           # ← EDIT ME between pushes: 0, 1, 2
PARTITION_COUNT = 3

print(f"SMOKE           = {SMOKE}")
print(f"SMOKE_N         = {SMOKE_N}")
print(f"MAX_CLIPS       = {MAX_CLIPS}")
print(f"PARTITION_ID    = {PARTITION_ID}")
print(f"PARTITION_COUNT = {PARTITION_COUNT}")
'''


CELL_2 = r'''# Cell 2 — Imports and paths
import gc
import os
import re
import time
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

_WALL_START = time.time()

# Competition + model paths
BASE      = Path("/kaggle/input/competitions/birdclef-2026")
MODEL_DIR = Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1")

SR             = 32_000
WINDOW_SEC     = 5
WINDOW_SAMPLES = SR * WINDOW_SEC          # 160 000
MIN_TAIL       = WINDOW_SAMPLES // 4      # 1.25 s — matches perch_v2/src/extract_embeddings.py
MAX_DURATION_SEC  = 600                   # cap very long field recordings (same as extract_train_audio_c2.py)
MAX_LOAD_SAMPLES  = SR * MAX_DURATION_SEC

OUT_DIR   = Path("/kaggle/working")
_part_suffix = f"_p{PARTITION_ID}of{PARTITION_COUNT}" if PARTITION_COUNT > 1 else ""
OUT_NPZ   = OUT_DIR / f"full_train_audio_perch{_part_suffix}.npz"
OUT_META  = OUT_DIR / f"full_train_audio_meta{_part_suffix}.parquet"

print("TensorFlow:", tf.__version__)
print("Competition dir exists:", BASE.exists())
print("Model dir exists:      ", MODEL_DIR.exists())

# Locate ONNX (3-5x faster than TF SavedModel on CPU)
ONNX_PATH = None
for _cand in [
    Path("/kaggle/input/birdclef-2026-perch-onnx/perch_v2_no_dft.onnx"),
    Path("/kaggle/input/birdclef-2026-cvlb-assets-0911/perch_v2_no_dft.onnx"),
    Path("/kaggle/input/perch-meta/perch_v2_no_dft.onnx"),
]:
    if _cand.exists():
        ONNX_PATH = _cand
        break
print(f"ONNX_PATH: {ONNX_PATH} (exists={ONNX_PATH is not None and ONNX_PATH.exists()})")
'''


CELL_3 = r'''# Cell 3 — Load taxonomy + primary labels
taxonomy   = pd.read_csv(BASE / "taxonomy.csv")
sample_sub = pd.read_csv(BASE / "sample_submission.csv")

PRIMARY_LABELS = sample_sub.columns[1:].tolist()
N_CLASSES      = len(PRIMARY_LABELS)
label_to_idx   = {c: i for i, c in enumerate(PRIMARY_LABELS)}

taxonomy["primary_label"]    = taxonomy["primary_label"].astype(str)
taxonomy["scientific_name_lookup"] = taxonomy["scientific_name"]  # no manual synonyms

print(f"N_CLASSES = {N_CLASSES}")
print(f"taxonomy  = {taxonomy.shape}")
'''


CELL_4 = r'''# Cell 4 — Build Perch inferencer (ONNX preferred, TF fallback)
def build_perch_inferencer(model_dir, onnx_path):
    if onnx_path is not None and onnx_path.exists():
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            output_names = [o.name for o in session.get_outputs()]

            def _infer_onnx(batch_audio):
                outputs = session.run(output_names, {"inputs": batch_audio.astype(np.float32, copy=False)})
                vals = dict(zip(output_names, outputs))
                return (vals["label"].astype(np.float32, copy=False),
                        vals["embedding"].astype(np.float32, copy=False))

            print("[setup] using ONNXRuntime CPU for Perch")
            return _infer_onnx, "onnxruntime"
        except Exception as exc:
            print(f"[setup] ONNXRuntime unavailable, falling back to TF: {exc}")

    print("[setup] using TensorFlow SavedModel for Perch")
    birdclassifier = tf.saved_model.load(str(model_dir))
    _infer_fn = birdclassifier.signatures["serving_default"]

    def _infer_tf(batch_audio):
        outputs = _infer_fn(inputs=tf.convert_to_tensor(batch_audio))
        return (outputs["label"].numpy().astype(np.float32, copy=False),
                outputs["embedding"].numpy().astype(np.float32, copy=False))

    return _infer_tf, "tensorflow"


_perch_infer, _perch_backend = build_perch_inferencer(MODEL_DIR, ONNX_PATH)
print(f"Perch backend: {_perch_backend}")
'''


CELL_5 = r'''# Cell 5 — Build 234-class mapping + selective genus proxies
#
# Verbatim port of postproc notebook cell 3 (aka "Load Perch, mapping, and
# selective frog proxies"), trimmed to the pieces needed for feature
# extraction only (no active-class / texture grouping — those are only
# used for post-processing downstream).
bc_labels = (
    pd.read_csv(MODEL_DIR / "assets" / "labels.csv")
    .reset_index()
    .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
)

NO_LABEL_INDEX = len(bc_labels)

bc_lookup = bc_labels.rename(columns={"scientific_name": "scientific_name_lookup"})
mapping = taxonomy.merge(
    bc_lookup[["scientific_name_lookup", "bc_index"]],
    on="scientific_name_lookup",
    how="left",
)
mapping["bc_index"] = mapping["bc_index"].fillna(NO_LABEL_INDEX).astype(int)

label_to_bc_index = mapping.set_index("primary_label")["bc_index"]
BC_INDICES        = np.array([int(label_to_bc_index.loc[c]) for c in PRIMARY_LABELS], dtype=np.int32)

MAPPED_MASK       = BC_INDICES != NO_LABEL_INDEX
MAPPED_POS        = np.where(MAPPED_MASK)[0].astype(np.int32)
MAPPED_BC_INDICES = BC_INDICES[MAPPED_MASK].astype(np.int32)

CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
PROXY_TAXA     = {"Amphibia", "Insecta", "Aves"}

# Build automatic genus proxies for unmapped non-sonotypes
unmapped_df = mapping[mapping["bc_index"] == NO_LABEL_INDEX].copy()
unmapped_non_sonotype = unmapped_df[
    ~unmapped_df["primary_label"].astype(str).str.contains("son", na=False)
].copy()

def _genus_hits(sci):
    genus = str(sci).split()[0] if sci else ""
    if not genus:
        return None
    hits = bc_labels[
        bc_labels["scientific_name"].astype(str).str.match(rf"^{re.escape(genus)}\s", na=False)
    ]
    return hits

proxy_map: dict[str, list[int]] = {}
for _, row in unmapped_non_sonotype.iterrows():
    target = row["primary_label"]
    sci    = row["scientific_name"]
    hits   = _genus_hits(sci)
    if hits is not None and len(hits) > 0:
        proxy_map[target] = hits["bc_index"].astype(int).tolist()

SELECTED_PROXY_TARGETS = sorted([
    t for t in proxy_map.keys()
    if CLASS_NAME_MAP.get(t) in PROXY_TAXA
])
selected_proxy_pos_to_bc = {
    label_to_idx[t]: np.array(proxy_map[t], dtype=np.int32)
    for t in SELECTED_PROXY_TARGETS
}

proxy_counts = {cls: sum(1 for t in SELECTED_PROXY_TARGETS
                         if CLASS_NAME_MAP.get(t) == cls)
                for cls in PROXY_TAXA}

print(f"[mapping] direct-mapped: {MAPPED_MASK.sum()} / {N_CLASSES}")
print(f"[mapping] unmapped:      {(~MAPPED_MASK).sum()}")
print(f"[mapping] proxy targets: {proxy_counts}  (total {len(selected_proxy_pos_to_bc)})")
'''


CELL_6 = r'''# Cell 6 — Audio helpers (focal-clip aware)
#
# Differences from the soundscape path (postproc cell 5):
#   * Clips are variable-length (30 s focal clips up to ~60-min field recs)
#   * We cap reads at MAX_DURATION_SEC to protect runtime from very long files
#   * Non-overlapping 5-s windows, keep last partial iff ≥1.25 s (zero-padded)
#   * Clips shorter than 1.25 s get a single zero-padded window
#   * No site/hour metadata — only species + stem + window_idx

def load_mono_32k(path):
    """Read up to MAX_DURATION_SEC at the file's native sample rate, mono, 32 kHz."""
    with sf.SoundFile(str(path)) as f:
        native_sr  = f.samplerate
        max_frames = int(MAX_DURATION_SEC * native_sr)
        wav = f.read(frames=max_frames, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1) if wav.shape[1] > 1 else wav[:, 0]
    if native_sr != SR:
        import scipy.signal
        n_out = int(len(wav) * SR / native_sr)
        wav = scipy.signal.resample(wav, n_out).astype(np.float32)
    return wav


def clip_windows(wav):
    """Yield non-overlapping 5-s windows; keep last partial iff ≥1.25 s."""
    n = len(wav)
    if n < MIN_TAIL:
        # Very short clip — single zero-padded window
        yield np.pad(wav[:WINDOW_SAMPLES], (0, max(0, WINDOW_SAMPLES - n))).astype(np.float32)
        return
    for start in range(0, n, WINDOW_SAMPLES):
        chunk = wav[start:start + WINDOW_SAMPLES]
        if len(chunk) < MIN_TAIL:
            break
        if len(chunk) < WINDOW_SAMPLES:
            chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))
        yield chunk.astype(np.float32, copy=False)


def build_scores(logits_batch):
    """(T, 14795) → (T, 234) using MAPPED_POS + selective genus proxies (max-pool)."""
    T = logits_batch.shape[0]
    scores = np.zeros((T, N_CLASSES), dtype=np.float32)
    scores[:, MAPPED_POS] = logits_batch[:, MAPPED_BC_INDICES]
    for pos, bc_idx_arr in selected_proxy_pos_to_bc.items():
        scores[:, pos] = logits_batch[:, bc_idx_arr].max(axis=1)
    return scores


print("[helpers] ready — load_mono_32k, clip_windows, build_scores")
'''


CELL_7 = r'''# Cell 7 — Enumerate train_audio focal clips
audio_dir = BASE / "train_audio"
ogg_files = sorted(audio_dir.rglob("*.ogg"))
print(f"Found {len(ogg_files)} .ogg clips under {audio_dir}")

if SMOKE:
    ogg_files = ogg_files[:SMOKE_N]
    print(f"SMOKE mode → keeping first {len(ogg_files)} clips")

if PARTITION_COUNT > 1:
    _pre = len(ogg_files)
    ogg_files = [f for i, f in enumerate(ogg_files) if i % PARTITION_COUNT == PARTITION_ID]
    print(f"PARTITION {PARTITION_ID}/{PARTITION_COUNT}: {_pre} → {len(ogg_files)} clips")

if MAX_CLIPS is not None:
    ogg_files = ogg_files[:MAX_CLIPS]
    print(f"MAX_CLIPS  → trimmed to {len(ogg_files)} clips")
'''


CELL_8 = r'''# Cell 8 — Extract Perch features clip-by-clip
# Memory budget: ~(N_windows * 1536 f32) + (N_windows * 234 f32)
#   Smoke (100 clips × ~6 win)   → ~6 MB emb + ~0.6 MB scores → trivial
#   Full (35,549 clips × ~7 win) → ~1.5 GB emb + ~0.23 GB scores → fits in 16 GB Kaggle RAM
# We collect per-clip lists and concat once at the end.

clip_emb_list    = []
clip_scores_list = []
clip_meta_rows   = []

err_clips = []

t0 = time.time()
n_total_windows = 0

for i, ogg in enumerate(tqdm(ogg_files, desc="extract")):
    species = ogg.parent.name
    stem    = ogg.stem
    try:
        wav = load_mono_32k(ogg)
        clip_duration = float(len(wav)) / SR
        batch = np.stack(list(clip_windows(wav)), axis=0).astype(np.float32, copy=False)  # (T, 160000)
        logits, emb = _perch_infer(batch)        # (T, 14795), (T, 1536)
        scores = build_scores(logits)             # (T, 234)

        T = batch.shape[0]
        clip_emb_list.append(emb.astype(np.float32, copy=False))
        clip_scores_list.append(scores.astype(np.float32, copy=False))
        for t in range(T):
            clip_meta_rows.append({
                "row_id":             f"{stem}_{t}",
                "stem":               stem,
                "species":            species,
                "window_idx":         t,
                "clip_duration_sec":  clip_duration,
            })
        n_total_windows += T

        del wav, batch, logits, emb, scores
    except Exception as e:
        err_clips.append((str(ogg), repr(e)))
        print(f"  [ERR] {ogg}: {e}", flush=True)
        continue

    if (i + 1) % 100 == 0 or (i + 1) == len(ogg_files):
        dt  = time.time() - t0
        rate = (i + 1) / dt if dt > 0 else 0.0
        eta  = (len(ogg_files) - (i + 1)) / rate if rate > 0 else float("inf")
        print(f"  [{i + 1}/{len(ogg_files)}] windows={n_total_windows} "
              f"elapsed={dt/60:.1f}m rate={rate:.2f} clips/s ETA={eta/60:.1f}m",
              flush=True)

    if (i + 1) % 500 == 0:
        gc.collect()

print(f"\n[done] clips_ok={len(clip_emb_list)} clips_err={len(err_clips)} "
      f"total_windows={n_total_windows} elapsed={(time.time()-t0)/60:.1f}m")
'''


CELL_9 = r'''# Cell 9 — Write outputs (one big NPZ + one meta Parquet)
if len(clip_emb_list) == 0:
    raise RuntimeError("No successful extractions — refusing to write empty output.")

emb_full    = np.concatenate(clip_emb_list,    axis=0).astype(np.float32, copy=False)
scores_full = np.concatenate(clip_scores_list, axis=0).astype(np.float32, copy=False)
meta_full   = pd.DataFrame(clip_meta_rows)

print(f"emb_full:    {emb_full.shape}    {emb_full.dtype}")
print(f"scores_full: {scores_full.shape} {scores_full.dtype}")
print(f"meta_full:   {meta_full.shape}")

# Schema sanity checks (match what downstream consumers expect)
assert emb_full.shape[0]    == scores_full.shape[0] == len(meta_full)
assert emb_full.shape[1]    == 1536
assert scores_full.shape[1] == 234
assert set(meta_full.columns) >= {"row_id", "stem", "species", "window_idx", "clip_duration_sec"}

# Write
OUT_DIR.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT_NPZ, emb=emb_full, scores=scores_full)
meta_full.to_parquet(OUT_META, index=False)

print(f"\n[write] {OUT_NPZ}   ({OUT_NPZ.stat().st_size / 1e6:.1f} MB)")
print(f"[write] {OUT_META}   ({OUT_META.stat().st_size / 1e6:.2f} MB)")

# Error log
if err_clips:
    err_df = pd.DataFrame(err_clips, columns=["path", "error"])
    err_df.to_csv(OUT_DIR / "extract_errors.csv", index=False)
    print(f"[write] {OUT_DIR / 'extract_errors.csv'} ({len(err_df)} rows)")
'''


CELL_10 = r'''# Cell 10 — Summary + wall-time projection
wall = time.time() - _WALL_START
print(f"\nWall time: {wall/60:.2f} min ({wall:.1f} s)")
print(f"Clips processed: {len(clip_emb_list)}")
print(f"Windows total:   {scores_full.shape[0] if 'scores_full' in dir() else 0}")

if SMOKE:
    # Project full-run wall time from the smoke-test rate
    rate = len(clip_emb_list) / wall if wall > 0 else 0.0
    full_n = 35_549
    proj = full_n / rate if rate > 0 else float("inf")
    print("\n[projection] (linear extrapolation — caveat: long field-rec outliers)")
    print(f"  smoke clips: {len(clip_emb_list)}  @  rate {rate:.2f} clips/s")
    print(f"  projected full extraction: {proj/3600:.2f} h for {full_n} clips")
    if proj < 6 * 3600:
        print("  → OK: expected to fit in a single Kaggle CPU slot (9 h cap)")
    elif proj < 9 * 3600:
        print("  → TIGHT: consider 2-run species partition")
    elif proj < 18 * 3600:
        print("  → 2-run species partition recommended")
    else:
        print("  → OVER BUDGET: must partition into 3+ runs")
'''


CELLS = [
    md(
        "# BirdCLEF 2026 — Perch v2 feature extraction on `train_audio`\n"
        "\n"
        "Track C (§13 Phase 1) — extracts per-window Perch v2 embeddings and "
        "234-class logit-mapped scores for all 35,549 focal clips in "
        "`train_audio/`, using the **same Perch call-path** the postproc "
        "notebook uses on soundscapes. The output is a single NPZ + meta "
        "Parquet pair suitable for direct upload as a Kaggle dataset.\n"
        "\n"
        "**Run modes** (via env var `SMOKE`):\n"
        "- `SMOKE=1` (default): first 100 clips only → wall-time projection.\n"
        "- `SMOKE=0`: full extraction.\n"
    ),
    code(CELL_0),
    code(CELL_1),
    code(CELL_2),
    code(CELL_3),
    code(CELL_4),
    code(CELL_5),
    code(CELL_6),
    code(CELL_7),
    code(CELL_8),
    code(CELL_9),
    code(CELL_10),
]


NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


if __name__ == "__main__":
    out = Path(__file__).parent / "birdclef2026-perch-train-audio-extract.ipynb"
    out.write_text(json.dumps(NB, indent=1))
    print(f"wrote {out} ({out.stat().st_size} bytes)")
