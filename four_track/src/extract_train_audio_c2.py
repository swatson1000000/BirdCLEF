"""Track C1 — per-clip Perch v2 feature extraction for train_audio.

Produces the EXACT same (emb, scores) representation that the postproc
notebook feeds into ProtoSSM/B1 on soundscapes, but applied to the ~35K
train_audio focal clips. The output is the input format Track C2 will
consume to pseudo-label + retrain ProtoSSM.

Pipeline mirrors the postproc notebook cells 2, 3, 5:
  - Perch v2 via ONNXRuntime (preferred) or TF SavedModel (fallback) —
    NOT the TFLite path in perch_v2/src/extract_embeddings.py (different
    backend, different mapping logic, inappropriate for C2).
  - 234-class competition scores built by direct logit indexing at
    MAPPED_POS ← logits[:, MAPPED_BC_INDICES], plus automatic genus
    proxies for unmapped non-sonotype classes (Amphibia + Insecta + Aves),
    max-pooled over the proxy's aliasing Perch logits. Same logic as
    notebook cell 3's `selected_proxy_pos_to_bc`.
  - Non-overlapping 5-second windows; last partial window kept only if
    ≥1.25 s; short tail is zero-padded to 5 s before inference; clips
    shorter than 1.25 s get a single zero-padded window (matches the
    fallback in perch_v2/src/extract_embeddings.py).

Output layout (one .npz per clip, resume-friendly):
  four_track/data/processed/perch_train_audio_c2/
    <species>/
      <stem>.npz     # { emb: (T,1536) f32, scores: (T,234) f32 }
    manifest.csv     # species, stem, n_windows, path

Runtime: ~256K total windows (35,549 clips × mean ~7 windows). On CPU
with ONNXRuntime this is ~4-8 h with 8 workers. Resume-friendly via
`.exists()` short-circuit.

Usage (from four_track/ with kaggle conda env active):
    nohup python -u src/extract_train_audio_c2.py \
        > log/extract_train_audio_c2_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
FT_ROOT   = Path(__file__).resolve().parents[1]
ROOT      = FT_ROOT.parent
DEFAULT_AUDIO      = ROOT / "data" / "raw" / "train_audio"
DEFAULT_ONNX       = ROOT / "data" / "external" / "birdclef-0911" / "perch_v2_no_dft.onnx"
DEFAULT_SAVEDMODEL = ROOT / "perch_v2" / "models" / "perch_v2"
DEFAULT_LABELS     = ROOT / "perch_v2" / "models" / "perch_v2" / "assets" / "labels.csv"
DEFAULT_TAXONOMY   = ROOT / "data" / "raw" / "taxonomy.csv"
DEFAULT_SAMPLE_SUB = ROOT / "data" / "raw" / "sample_submission.csv"
DEFAULT_OUT        = FT_ROOT / "data" / "processed" / "perch_train_audio_c2"

SR            = 32_000
WINDOW_SEC    = 5
WINDOW_SAMPLES = SR * WINDOW_SEC   # 160 000
MIN_TAIL      = WINDOW_SAMPLES // 4  # 1.25 s — matches perch_v2/src/extract_embeddings.py

# Cap any single clip at the first MAX_DURATION_SEC of audio. Protects
# workers from blowing up on the handful of pathologically long field
# recordings in train_audio (up to ~60 min). 10 minutes = 120 five-second
# windows, which is far more than any focal clip needs for C2's purpose,
# and keeps per-worker RSS bounded.
MAX_DURATION_SEC  = 600
MAX_LOAD_SAMPLES  = SR * MAX_DURATION_SEC  # 19_200_000


# ── Mapping construction (mirrors notebook cell 3) ───────────────────────────

def build_mapping(labels_csv: Path, taxonomy_csv: Path, sample_sub_csv: Path):
    """Return (n_classes, mapped_pos, mapped_bc_indices, proxy_pos_to_bc).

    This is a verbatim port of notebook cell 3's mapping logic:
      - PRIMARY_LABELS = sample_submission.csv columns (sorted competition order)
      - BC_INDICES per competition class via scientific_name lookup into
        Perch's labels.csv
      - MAPPED_POS / MAPPED_BC_INDICES = the classes that have a direct
        Perch alias (one Perch index each)
      - selected_proxy_pos_to_bc = genus-level fallback for unmapped
        non-sonotype classes in {Amphibia, Insecta, Aves}, keyed by the
        *competition* class index and storing the set of Perch indices
        to max-pool over
    """
    sample_sub = pd.read_csv(sample_sub_csv)
    primary_labels = sample_sub.columns[1:].tolist()
    n_classes      = len(primary_labels)
    label_to_idx   = {c: i for i, c in enumerate(primary_labels)}

    taxonomy = pd.read_csv(taxonomy_csv)
    taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
    # No manual synonyms yet (mirrors notebook cell 3's empty MANUAL_SCIENTIFIC_NAME_MAP)
    taxonomy["scientific_name_lookup"] = taxonomy["scientific_name"]

    bc_labels = (
        pd.read_csv(labels_csv)
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

    bc_indices = np.array(
        [int(label_to_bc_index.loc[c]) for c in primary_labels],
        dtype=np.int32,
    )
    mapped_mask        = bc_indices != NO_LABEL_INDEX
    mapped_pos         = np.where(mapped_mask)[0].astype(np.int32)
    mapped_bc_indices  = bc_indices[mapped_mask].astype(np.int32)

    class_name_map = taxonomy.set_index("primary_label")["class_name"].to_dict()
    PROXY_TAXA     = {"Amphibia", "Insecta", "Aves"}

    # Genus proxies: for unmapped non-sonotype competition classes,
    # collect all Perch classes whose scientific_name starts with the
    # same genus token.
    unmapped_df = mapping[mapping["bc_index"] == NO_LABEL_INDEX].copy()
    unmapped_non_sonotype = unmapped_df[
        ~unmapped_df["primary_label"].astype(str).str.contains("son", na=False)
    ].copy()

    proxy_map: dict[str, list[int]] = {}
    for _, row in unmapped_non_sonotype.iterrows():
        target = row["primary_label"]
        sci    = str(row["scientific_name"])
        genus  = sci.split()[0] if sci else ""
        if not genus:
            continue
        hits = bc_labels[
            bc_labels["scientific_name"].astype(str).str.match(
                rf"^{re.escape(genus)}\s", na=False
            )
        ]
        if len(hits) > 0:
            proxy_map[target] = hits["bc_index"].astype(int).tolist()

    selected_targets = sorted([
        t for t in proxy_map.keys() if class_name_map.get(t) in PROXY_TAXA
    ])
    proxy_pos_to_bc = {
        label_to_idx[target]: np.array(proxy_map[target], dtype=np.int32)
        for target in selected_targets
    }

    counts = {cls: sum(1 for t in selected_targets if class_name_map.get(t) == cls)
              for cls in PROXY_TAXA}
    print(f"[mapping] competition classes: {n_classes}", flush=True)
    print(f"[mapping] direct-mapped:       {mapped_mask.sum()}", flush=True)
    print(f"[mapping] unmapped:            {(~mapped_mask).sum()}", flush=True)
    print(f"[mapping] proxy targets:       {counts}  (total {len(proxy_pos_to_bc)})", flush=True)

    return n_classes, mapped_pos, mapped_bc_indices, proxy_pos_to_bc


# ── Audio helpers ────────────────────────────────────────────────────────────

def _load_mono_32k(path: str) -> np.ndarray:
    # Partial read: only the first MAX_DURATION_SEC of audio, in the
    # file's native sample rate. This caps peak per-worker RSS at ~76 MB
    # of waveform no matter how long the source recording is.
    with sf.SoundFile(path) as f:
        native_sr  = f.samplerate
        max_frames = int(MAX_DURATION_SEC * native_sr)
        wav = f.read(frames=max_frames, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1) if wav.shape[1] > 1 else wav[:, 0]
    if native_sr != SR:
        import scipy.signal
        num_samples = int(len(wav) * SR / native_sr)
        wav = scipy.signal.resample(wav, num_samples).astype(np.float32)
    return wav


def _windows(wav: np.ndarray):
    """Yield non-overlapping 5-s windows, pad last partial if ≥1.25 s."""
    n = len(wav)
    for start in range(0, n, WINDOW_SAMPLES):
        chunk = wav[start:start + WINDOW_SAMPLES]
        if len(chunk) < MIN_TAIL:
            break
        if len(chunk) < WINDOW_SAMPLES:
            chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))
        yield chunk


# ── Worker state (per-process) ───────────────────────────────────────────────
_session          = None
_mapped_pos       = None
_mapped_bc        = None
_proxy_pos_to_bc  = None
_n_classes        = None


def _init_worker(onnx_path: str, savedmodel_dir: str,
                 mapped_pos: np.ndarray, mapped_bc: np.ndarray,
                 proxy_pos_to_bc: dict, n_classes: int):
    """Initialise ONNX session and cache mapping arrays per worker."""
    global _session, _mapped_pos, _mapped_bc, _proxy_pos_to_bc, _n_classes
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    _mapped_pos      = mapped_pos
    _mapped_bc       = mapped_bc
    _proxy_pos_to_bc = proxy_pos_to_bc
    _n_classes       = n_classes

    if onnx_path and Path(onnx_path).exists():
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        _session = ("onnx", ort.InferenceSession(onnx_path, sess_options=so,
                                                 providers=["CPUExecutionProvider"]))
    else:
        import tensorflow as tf
        model = tf.saved_model.load(savedmodel_dir)
        _session = ("tf", model.signatures["serving_default"])


def _infer_batch(batch_wav: np.ndarray):
    """Run Perch on a batch of (N, 160000) float32 waveforms.
    Returns (logits: (N, 14795), emb: (N, 1536))."""
    kind, sess = _session
    if kind == "onnx":
        out = sess.run(None, {"inputs": batch_wav.astype(np.float32, copy=False)})
        names = [o.name for o in sess.get_outputs()]
        vals  = dict(zip(names, out))
        return vals["label"].astype(np.float32, copy=False), \
               vals["embedding"].astype(np.float32, copy=False)
    else:
        import tensorflow as tf
        out = sess(inputs=tf.convert_to_tensor(batch_wav))
        return out["label"].numpy().astype(np.float32, copy=False), \
               out["embedding"].numpy().astype(np.float32, copy=False)


def _build_scores(logits_batch: np.ndarray) -> np.ndarray:
    """logits_batch: (T, 14795) → scores (T, 234) using MAPPED_POS +
    selected genus proxies (max over aliasing Perch logits).
    Unmapped non-proxy positions stay zero (matches notebook cell 5)."""
    T = logits_batch.shape[0]
    scores = np.zeros((T, _n_classes), dtype=np.float32)
    scores[:, _mapped_pos] = logits_batch[:, _mapped_bc]
    for pos, bc_idx_arr in _proxy_pos_to_bc.items():
        scores[:, pos] = logits_batch[:, bc_idx_arr].max(axis=1)
    return scores


# ── Per-clip worker ──────────────────────────────────────────────────────────

def _extract_clip(args):
    ogg_path, out_path = args
    out_path = Path(out_path)
    if out_path.exists():
        return ("skip", str(ogg_path), 0)
    try:
        wav = _load_mono_32k(str(ogg_path))

        # Build windowed batch (at least one window, zero-padded if needed)
        win_list = list(_windows(wav))
        if not win_list:
            win_list = [np.pad(wav[:WINDOW_SAMPLES],
                               (0, max(0, WINDOW_SAMPLES - len(wav))))]
        batch = np.stack(win_list, axis=0).astype(np.float32, copy=False)  # (T, 160000)

        logits, emb = _infer_batch(batch)  # (T, 14795), (T, 1536)
        scores = _build_scores(logits)     # (T, 234)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(out_path),
            emb=emb.astype(np.float32),
            scores=scores.astype(np.float32),
        )
        return ("ok", str(ogg_path), len(win_list))
    except Exception as e:
        return ("err", f"{ogg_path}: {e}", 0)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Track C1 — train_audio Perch v2 feature extraction")
    p.add_argument("--audio-dir",   type=Path, default=DEFAULT_AUDIO)
    p.add_argument("--onnx",        type=Path, default=DEFAULT_ONNX)
    p.add_argument("--savedmodel",  type=Path, default=DEFAULT_SAVEDMODEL)
    p.add_argument("--labels-csv",  type=Path, default=DEFAULT_LABELS)
    p.add_argument("--taxonomy",    type=Path, default=DEFAULT_TAXONOMY)
    p.add_argument("--sample-sub",  type=Path, default=DEFAULT_SAMPLE_SUB)
    p.add_argument("--output-dir",  type=Path, default=DEFAULT_OUT)
    p.add_argument("--workers",     type=int,  default=8)
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60, flush=True)
    print("Track C1 — train_audio Perch v2 extraction", flush=True)
    print(f"  audio-dir:  {args.audio_dir}", flush=True)
    print(f"  onnx:       {args.onnx}  exists={args.onnx.exists()}", flush=True)
    print(f"  savedmodel: {args.savedmodel}  exists={args.savedmodel.exists()}", flush=True)
    print(f"  output-dir: {args.output_dir}", flush=True)
    print(f"  workers:    {args.workers}", flush=True)

    for p in (args.labels_csv, args.taxonomy, args.sample_sub):
        if not p.exists():
            sys.exit(f"ERROR: missing required file: {p}")
    if not args.onnx.exists() and not (args.savedmodel / "saved_model.pb").exists():
        sys.exit("ERROR: neither ONNX nor TF SavedModel is available")

    print("\n[step 1/3] building 234-class mapping …", flush=True)
    n_classes, mapped_pos, mapped_bc, proxy_pos_to_bc = build_mapping(
        args.labels_csv, args.taxonomy, args.sample_sub,
    )

    print("\n[step 2/3] enumerating train_audio clips …", flush=True)
    ogg_files = sorted(args.audio_dir.rglob("*.ogg"))
    print(f"  {len(ogg_files)} clips found", flush=True)

    tasks = []
    for ogg in ogg_files:
        species = ogg.parent.name
        out     = args.output_dir / species / (ogg.stem + ".npz")
        tasks.append((str(ogg), str(out)))

    n_done_before = sum(1 for _, o in tasks if Path(o).exists())
    print(f"  already extracted: {n_done_before}", flush=True)
    print(f"  to process:        {len(tasks) - n_done_before}", flush=True)

    print("\n[step 3/3] extracting …", flush=True)
    t0 = time.time()
    ok = err = skip = 0
    total_windows = 0

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(
            str(args.onnx) if args.onnx.exists() else "",
            str(args.savedmodel),
            mapped_pos, mapped_bc, proxy_pos_to_bc, n_classes,
        ),
        maxtasksperchild=500,
    ) as pool:
        for status, msg, n_win in pool.imap_unordered(_extract_clip, tasks, chunksize=4):
            if status == "ok":
                ok += 1
                total_windows += n_win
            elif status == "skip":
                skip += 1
            else:
                err += 1
                print(f"  [ERR] {msg}", flush=True)
            done = ok + skip + err
            if done % 500 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate    = (ok + skip) / elapsed if elapsed > 0 else 0
                eta     = (len(tasks) - done) / rate if rate > 0 else float("inf")
                print(
                    f"  [{done}/{len(tasks)}] ok={ok} skip={skip} err={err} "
                    f"windows={total_windows}  {elapsed/60:.1f}m elapsed  "
                    f"ETA {eta/60:.1f}m",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(f"\n[done] ok={ok} skip={skip} err={err} new_windows={total_windows} "
          f"({elapsed/60:.1f} min)", flush=True)

    # Build manifest
    print("\n[manifest] scanning output and writing manifest.csv …", flush=True)
    manifest_path = args.output_dir / "manifest.csv"
    n_rows = 0
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "stem", "n_windows", "path"])
        for species_dir in sorted(args.output_dir.iterdir()):
            if not species_dir.is_dir():
                continue
            for npz in sorted(species_dir.glob("*.npz")):
                try:
                    with np.load(str(npz)) as d:
                        t_ = int(d["emb"].shape[0])
                except Exception as e:
                    print(f"  [WARN] unreadable {npz}: {e}", flush=True)
                    continue
                rel = npz.relative_to(args.output_dir)
                w.writerow([species_dir.name, npz.stem, t_, str(rel)])
                n_rows += 1
    print(f"[manifest] {n_rows} clips in {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
