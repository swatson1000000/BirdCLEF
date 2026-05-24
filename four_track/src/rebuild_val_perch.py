"""Phase 2a of val-v2 rebuild — run Perch v2 on 500 new soundscape files.

Samples 500 files from `train_soundscapes/` that are NOT in the existing
59-overlap Perch cache, runs Perch v2 (TF SavedModel on GPU by default,
ONNX CPU fallback), and writes segment-level scores to a single NPZ so
`rebuild_val_build.py` can consume them without re-invoking Perch.

Deterministic:
  - Seed = 42 for file sampling.
  - File sort order fixed before sampling.
  - Output NPZ lists filenames + segment start-seconds in row order.

Cost estimate on the 6000-window run:
  - TF GPU (preferred): ~2-5 minutes total (audio I/O dominates).
  - ONNX CPU: ~15-30 minutes.

Usage:
    python -u src/rebuild_val_perch.py                # TF SavedModel (GPU)
    python -u src/rebuild_val_perch.py --backend onnx # ONNX CPU
    python -u src/rebuild_val_perch.py --n-files 100  # smoke test
    python -u src/rebuild_val_perch.py --dry-run      # list files only
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

warnings.filterwarnings("ignore")

# ── Path wiring ──────────────────────────────────────────────────────────────
HERE       = Path(__file__).resolve().parent
FT_ROOT    = HERE.parent
ROOT       = FT_ROOT.parent
PARENT_SRC = ROOT / "src"

if str(PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(PARENT_SRC))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Reuse the Perch mapping logic already validated in Track C1.
from extract_train_audio_c2 import build_mapping  # noqa: E402

SOUNDSCAPE_DIR = ROOT / "data" / "raw" / "train_soundscapes"
PERCH_CACHE    = FT_ROOT / "data" / "kaggle_perch_cache" / "full_perch_meta.parquet"

DEFAULT_ONNX       = ROOT / "data" / "external" / "birdclef-0911" / "perch_v2_no_dft.onnx"
DEFAULT_SAVEDMODEL = ROOT / "perch_v2" / "models" / "perch_v2"
DEFAULT_LABELS     = ROOT / "perch_v2" / "models" / "perch_v2" / "assets" / "labels.csv"
DEFAULT_TAXONOMY   = ROOT / "data" / "raw" / "taxonomy.csv"
DEFAULT_SAMPLE_SUB = ROOT / "data" / "raw" / "sample_submission.csv"

OUT_DIR = FT_ROOT / "data" / "processed" / "val_v2"
OUT_NPZ = OUT_DIR / "perch_new500.npz"

SR             = 32_000
WINDOW_SEC     = 5
WINDOW_SAMPLES = SR * WINDOW_SEC            # 160_000
CLIP_SEC       = 60                          # soundscape duration
CLIP_SAMPLES   = SR * CLIP_SEC               # 1_920_000
BATCH_SIZE     = 32
SEED           = 42


def _load_mono_32k(path: Path, max_samples: int = CLIP_SAMPLES) -> np.ndarray:
    with sf.SoundFile(str(path)) as f:
        native_sr = f.samplerate
        wav = f.read(frames=max_samples, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1) if wav.shape[1] > 1 else wav[:, 0]
    if native_sr != SR:
        import scipy.signal
        n_out = int(len(wav) * SR / native_sr)
        wav = scipy.signal.resample(wav, n_out).astype(np.float32)
    return wav


def _windows_fixed12(wav: np.ndarray) -> np.ndarray:
    """Return exactly 12 non-overlapping 5-s windows, zero-padded if short.

    Matches the existing Perch cache convention: 60-s soundscape → 12 windows.
    Soundscape files are 60 s by construction, but we pad defensively.
    """
    n_needed = 12 * WINDOW_SAMPLES
    if len(wav) < n_needed:
        wav = np.pad(wav, (0, n_needed - len(wav)))
    else:
        wav = wav[:n_needed]
    return wav.reshape(12, WINDOW_SAMPLES).astype(np.float32, copy=False)


def _build_scores(logits: np.ndarray,
                  mapped_pos: np.ndarray,
                  mapped_bc: np.ndarray,
                  proxy_pos_to_bc: dict) -> np.ndarray:
    T = logits.shape[0]
    n_classes = int(mapped_pos.max()) + 1
    # Safer: derive n_classes from the max across direct + proxy targets.
    if proxy_pos_to_bc:
        n_classes = max(n_classes, max(proxy_pos_to_bc.keys()) + 1)
    scores = np.zeros((T, n_classes), dtype=np.float32)
    scores[:, mapped_pos] = logits[:, mapped_bc]
    for pos, bc_idx_arr in proxy_pos_to_bc.items():
        scores[:, pos] = logits[:, bc_idx_arr].max(axis=1)
    return scores


def _pick_files(n_files: int) -> list[Path]:
    all_files = sorted(SOUNDSCAPE_DIR.glob("*.ogg"))
    if not all_files:
        raise SystemExit(f"[error] no .ogg in {SOUNDSCAPE_DIR}")

    # Exclude the 59 files already in the Perch cache.
    meta = pd.read_parquet(PERCH_CACHE)
    parts = meta["row_id"].str.extract(r"^(?P<stem>.+)_\d+$")
    cached_stems = set(parts["stem"].unique().tolist())
    candidates = [p for p in all_files if p.stem not in cached_stems]
    print(f"[pick] soundscape files total: {len(all_files)}", flush=True)
    print(f"[pick] already in Perch cache: {len(cached_stems)}", flush=True)
    print(f"[pick] candidates for new run: {len(candidates)}", flush=True)

    rng = random.Random(SEED)
    rng.shuffle(candidates)
    picked = sorted(candidates[:n_files])
    print(f"[pick] sampled (seed={SEED}): {len(picked)}", flush=True)
    return picked


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-files", type=int, default=500)
    p.add_argument("--backend", choices=["tf", "onnx"], default="onnx")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    picked = _pick_files(args.n_files)

    if args.dry_run:
        print("[dry-run] would process these files (first 10):", flush=True)
        for p_ in picked[:10]:
            print(f"  {p_.name}", flush=True)
        print(f"  … total {len(picked)}", flush=True)
        return 0

    # Build Perch → 234-class mapping.
    print(f"\n[mapping] building Perch → 234-class mapping", flush=True)
    n_classes, mapped_pos, mapped_bc, proxy_pos_to_bc = build_mapping(
        DEFAULT_LABELS, DEFAULT_TAXONOMY, DEFAULT_SAMPLE_SUB,
    )

    # Initialise Perch session (single process, no multiprocessing).
    if args.backend == "onnx":
        print(f"\n[session] ONNX from {DEFAULT_ONNX}", flush=True)
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        sess = ort.InferenceSession(str(DEFAULT_ONNX), sess_options=so,
                                    providers=["CPUExecutionProvider"])

        def _infer(batch: np.ndarray) -> np.ndarray:
            out = sess.run(None, {"inputs": batch.astype(np.float32, copy=False)})
            names = [o.name for o in sess.get_outputs()]
            return dict(zip(names, out))["label"].astype(np.float32, copy=False)
    else:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        print(f"\n[session] TF SavedModel from {DEFAULT_SAVEDMODEL}", flush=True)
        print(f"[session] TF GPUs visible: {len(gpus)}", flush=True)
        if gpus:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
        model = tf.saved_model.load(str(DEFAULT_SAVEDMODEL))
        sig = model.signatures["serving_default"]

        def _infer(batch: np.ndarray) -> np.ndarray:
            out = sig(inputs=tf.convert_to_tensor(batch))
            return out["label"].numpy().astype(np.float32, copy=False)

    # Accumulators.
    all_scores: list[np.ndarray] = []
    file_col: list[str]          = []
    sec_col:  list[int]          = []

    t0 = time.time()
    n_errs = 0
    for i, path in enumerate(picked):
        try:
            wav = _load_mono_32k(path)
            batch = _windows_fixed12(wav)  # (12, 160000)

            # Micro-batch if batch > BATCH_SIZE (not applicable at T=12, but general).
            logits_parts = []
            for s in range(0, batch.shape[0], BATCH_SIZE):
                logits_parts.append(_infer(batch[s:s + BATCH_SIZE]))
            logits = np.concatenate(logits_parts, axis=0)     # (12, 14795)
            scores = _build_scores(logits, mapped_pos, mapped_bc, proxy_pos_to_bc)
            # Pad scores to 234 if fewer cols came back.
            if scores.shape[1] < 234:
                pad = np.zeros((scores.shape[0], 234 - scores.shape[1]), dtype=np.float32)
                scores = np.concatenate([scores, pad], axis=1)

            all_scores.append(scores)
            file_col.extend([path.name] * 12)
            sec_col.extend([w * WINDOW_SEC for w in range(12)])  # start-sec: 0,5,10,...,55

            if (i + 1) % 25 == 0 or (i + 1) == len(picked):
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                eta  = (len(picked) - (i + 1)) / max(rate, 1e-6)
                print(f"  [{i+1:>4}/{len(picked)}] "
                      f"elapsed={elapsed/60:.1f}m  rate={rate:.1f} files/s  "
                      f"eta={eta/60:.1f}m",
                      flush=True)
        except Exception as ex:
            n_errs += 1
            print(f"  [warn] {path.name}: {ex}", flush=True)

    if not all_scores:
        print("[error] no segments produced", flush=True)
        return 1

    scores_arr = np.concatenate(all_scores, axis=0)  # (N, 234)
    file_arr   = np.array(file_col, dtype=object)
    sec_arr    = np.array(sec_col,  dtype=np.int32)

    print(f"\n[done] segments: {scores_arr.shape[0]}  "
          f"errors: {n_errs}  elapsed: {(time.time()-t0)/60:.1f}m",
          flush=True)

    np.savez(
        OUT_NPZ,
        scores=scores_arr,
        filename=file_arr,
        start_sec=sec_arr,
    )
    print(f"[done] wrote → {OUT_NPZ}  "
          f"(scores {scores_arr.shape}, {scores_arr.nbytes/1e6:.1f} MB)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
