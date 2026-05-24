"""Build the perch-train-soundscapes-extract notebook from source cells.

Run from this directory:
    python build_notebook.py

Produces: birdclef2026-perch-train-soundscapes-extract.ipynb
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

# ONNXRuntime from our own bundled wheels (3-5x faster than TF on CPU).
# Kaggle's mount layout varies — search for the wheels dir wherever it landed
# under /kaggle/input/.
_WHEEL_DIRS = list(Path("/kaggle/input").rglob("birdclef-2026-perch-onnx/wheels"))
_INSTALL_ONLY = {'onnxruntime', 'flatbuffers', 'protobuf', 'sympy', 'mpmath', 'packaging'}
if _WHEEL_DIRS:
    _WHEEL_DIR = _WHEEL_DIRS[0]
    print(f"Found ONNX wheels at: {_WHEEL_DIR}")
    for whl in sorted(_WHEEL_DIR.glob('*.whl')):
        pkg_name = whl.name.split('-')[0].lower().replace('_', '-')
        if pkg_name in _INSTALL_ONLY or any(pkg_name.startswith(x) for x in _INSTALL_ONLY):
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-deps', '--quiet', str(whl)])
    print('ONNXRuntime wheels installed.')
else:
    print('WARNING: own ONNX wheel dir not found, will use TF-only path (slower)')
    print('  searched: /kaggle/input/**/birdclef-2026-perch-onnx/wheels (rglob)')
'''


CELL_1 = r'''# Cell 1 — Mode switch
#
# SMOKE = True  → extract first SMOKE_N files only (fast sanity check + wall-time projection)
# SMOKE = False → extract all 10,592 train_soundscapes 60-s files (127,104 fixed 5-s windows)
#
# PARTITION_ID / PARTITION_COUNT:
#   Each soundscape file emits exactly 12 fixed 5-s windows. With ~10,592 files
#   total, the full extraction is ~127,104 Perch forwards — heavier than the
#   focal-clip workload, so we partition into 4 chunks (~2,648 files / ~31,776
#   windows each) to stay within the 9 h Kaggle CPU slot.
#   Partitioning is modular on the sorted file index:
#     keep file i iff i % PARTITION_COUNT == PARTITION_ID.
#   Between pushes, edit PARTITION_ID only (0 → 1 → 2 → 3) and re-run
#   `python build_notebook.py`.
SMOKE          = False
SMOKE_N        = 100         # unused when SMOKE=False
MAX_CLIPS      = None        # None = process everything in the partition
PARTITION_ID   = 1           # ← EDIT ME between pushes: 0, 1
PARTITION_COUNT = 2

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
OUT_NPZ   = OUT_DIR / f"full_train_soundscapes_perch{_part_suffix}.npz"
OUT_META  = OUT_DIR / f"full_train_soundscapes_meta{_part_suffix}.parquet"

print("TensorFlow:", tf.__version__)
print("Competition dir exists:", BASE.exists())
print("Model dir exists:      ", MODEL_DIR.exists())

# Locate ONNX (3-5x faster than TF SavedModel on CPU). Kaggle's mount layout
# varies — rglob for the .onnx file wherever it landed.
_ONNX_HITS = list(Path("/kaggle/input").rglob("perch_v2_no_dft.onnx"))
ONNX_PATH = _ONNX_HITS[0] if _ONNX_HITS else None
print(f"ONNX_PATH: {ONNX_PATH} (exists={ONNX_PATH is not None and ONNX_PATH.exists()})")
if not _ONNX_HITS:
    print('  no perch_v2_no_dft.onnx found anywhere under /kaggle/input')
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


CELL_6 = r'''# Cell 6 — Audio helpers (soundscape-aware)
#
# train_soundscapes files are fixed 60-s unlabeled .ogg @ 32 kHz mono.
# We emit exactly 12 non-overlapping 5-s windows per file at start_sec ∈
# {0, 5, 10, …, 55}. Short/long files are pad/truncated to exactly 60 s
# before windowing so the per-file window count is always 12.

SOUNDSCAPE_DURATION_SEC = 60
SOUNDSCAPE_FRAMES       = SR * SOUNDSCAPE_DURATION_SEC   # 1_920_000
WINDOWS_PER_FILE        = SOUNDSCAPE_DURATION_SEC // WINDOW_SEC  # 12

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


def soundscape_windows(wav):
    """Yield exactly WINDOWS_PER_FILE (=12) fixed 5-s windows starting at
    0, 5, 10, …, 55 s. Pads/truncates wav to exactly SOUNDSCAPE_FRAMES first
    so the count is invariant regardless of small per-file duration drift."""
    n = len(wav)
    if n < SOUNDSCAPE_FRAMES:
        wav = np.pad(wav, (0, SOUNDSCAPE_FRAMES - n))
    elif n > SOUNDSCAPE_FRAMES:
        wav = wav[:SOUNDSCAPE_FRAMES]
    for t in range(WINDOWS_PER_FILE):
        start = t * WINDOW_SAMPLES
        chunk = wav[start:start + WINDOW_SAMPLES]
        yield chunk.astype(np.float32, copy=False)


def build_scores(logits_batch):
    """(T, 14795) → (T, 234) using MAPPED_POS + selective genus proxies (max-pool)."""
    T = logits_batch.shape[0]
    scores = np.zeros((T, N_CLASSES), dtype=np.float32)
    scores[:, MAPPED_POS] = logits_batch[:, MAPPED_BC_INDICES]
    for pos, bc_idx_arr in selected_proxy_pos_to_bc.items():
        scores[:, pos] = logits_batch[:, bc_idx_arr].max(axis=1)
    return scores


print("[helpers] ready — load_mono_32k, soundscape_windows, build_scores")
'''


CELL_7 = r'''# Cell 7 — Enumerate train_soundscapes 60-s files (flat dir, no species subdirs)
audio_dir = BASE / "train_soundscapes"
ogg_files = sorted(audio_dir.glob("*.ogg"))
print(f"Found {len(ogg_files)} .ogg files under {audio_dir}")

if SMOKE:
    ogg_files = ogg_files[:SMOKE_N]
    print(f"SMOKE mode → keeping first {len(ogg_files)} files")

if PARTITION_COUNT > 1:
    _pre = len(ogg_files)
    ogg_files = [f for i, f in enumerate(ogg_files) if i % PARTITION_COUNT == PARTITION_ID]
    print(f"PARTITION {PARTITION_ID}/{PARTITION_COUNT}: {_pre} → {len(ogg_files)} files")

if MAX_CLIPS is not None:
    ogg_files = ogg_files[:MAX_CLIPS]
    print(f"MAX_CLIPS  → trimmed to {len(ogg_files)} files")
'''


CELL_8 = r'''# Cell 8 — Extract Perch features file-by-file (12 fixed 5-s windows per 60-s file)
# Memory budget: ~(N_windows * 1536 f32) + (N_windows * 234 f32)
#   Smoke (100 files × 12 win)    → ~7 MB emb + ~1 MB scores → trivial
#   Full (10,592 files × 12 win)  → ~780 MB emb + ~120 MB scores → fits in 16 GB Kaggle RAM
# We collect per-file lists and concat once at the end.

clip_emb_list    = []
clip_scores_list = []
clip_meta_rows   = []

err_clips = []

t0 = time.time()
n_total_windows = 0

for i, ogg in enumerate(tqdm(ogg_files, desc="extract")):
    filename = ogg.name
    try:
        wav = load_mono_32k(ogg)
        batch = np.stack(list(soundscape_windows(wav)), axis=0).astype(np.float32, copy=False)  # (12, 160000)
        logits, emb = _perch_infer(batch)        # (12, 14795), (12, 1536)
        scores = build_scores(logits)             # (12, 234)

        T = batch.shape[0]
        clip_emb_list.append(emb.astype(np.float32, copy=False))
        clip_scores_list.append(scores.astype(np.float32, copy=False))
        for t in range(T):
            start_sec = t * WINDOW_SEC
            clip_meta_rows.append({
                "row_id":      f"{filename}_{start_sec}",
                "filename":    filename,
                "start_sec":   start_sec,
                "window_idx":  t,
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
              f"elapsed={dt/60:.1f}m rate={rate:.2f} files/s ETA={eta/60:.1f}m",
              flush=True)

    if (i + 1) % 500 == 0:
        gc.collect()

print(f"\n[done] files_ok={len(clip_emb_list)} files_err={len(err_clips)} "
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
assert set(meta_full.columns) >= {"row_id", "filename", "start_sec", "window_idx"}

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
print(f"Files processed: {len(clip_emb_list)}")
print(f"Windows total:   {scores_full.shape[0] if 'scores_full' in dir() else 0}")

if SMOKE:
    # Project full-run wall time from the smoke-test rate
    rate = len(clip_emb_list) / wall if wall > 0 else 0.0
    full_n = 10_592
    proj = full_n / rate if rate > 0 else float("inf")
    print("\n[projection] (linear extrapolation — fixed 12 windows per 60-s file)")
    print(f"  smoke files: {len(clip_emb_list)}  @  rate {rate:.2f} files/s")
    print(f"  projected full extraction: {proj/3600:.2f} h for {full_n} files")
    if proj < 6 * 3600:
        print("  → OK: expected to fit in a single Kaggle CPU slot (9 h cap)")
    elif proj < 9 * 3600:
        print("  → TIGHT: consider 2-run partition")
    elif proj < 18 * 3600:
        print("  → 2-run partition recommended")
    else:
        print("  → OVER BUDGET: must partition into 3+ runs")
'''


CELLS = [
    md(
        "# BirdCLEF 2026 — Perch v2 feature extraction on `train_soundscapes`\n"
        "\n"
        "Extracts per-window Perch v2 embeddings (1536-d) and 234-class "
        "logit-mapped scores for all 10,592 unlabeled 60-s files in "
        "`train_soundscapes/`. Each file emits exactly 12 fixed 5-s windows "
        "(start_sec ∈ {0, 5, 10, …, 55}) → 127,104 windows total. Uses the "
        "**same Perch call-path** the postproc notebook uses on test "
        "soundscapes. The output is a single NPZ + meta Parquet pair suitable "
        "for direct upload as a Kaggle dataset and downstream consumption by "
        "ProtoSSM (avoiding the local-Perch feature poisoning issue).\n"
        "\n"
        "**Run modes** (via `SMOKE` constant in Cell 1):\n"
        "- `SMOKE=True` (default): first 100 files only → wall-time projection.\n"
        "- `SMOKE=False`: full extraction (partitioned into 4 chunks of "
        "~2,648 files each via `PARTITION_ID`).\n"
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
    out = Path(__file__).parent / "birdclef2026-perch-train-soundscapes-extract.ipynb"
    out.write_text(json.dumps(NB, indent=1))
    print(f"wrote {out} ({out.stat().st_size} bytes)")
