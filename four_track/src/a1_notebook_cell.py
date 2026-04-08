"""A1 inference + rank-space fusion cell for the LB notebook.

This file is the canonical source for the cell that gets inserted into
`jupyter/protossm-postproc/birdclef2026-protossm-postproc.ipynb` just
after cell 36 (Score Fusion). Keeping the cell source here in version
control means we can iterate on it without manually editing the notebook.

Runtime contract (what must exist in the notebook kernel when this cell runs):
  - `test_paths`       : list[Path]        (defined in cell 35)
  - `meta_test`        : pd.DataFrame with the exact row order produced by
                          `infer_perch_with_embeddings` — one row per
                          (file, window) in `test_paths` order, 12 windows
                          per file at t = 5,10,...,60.
  - `final_test_scores`: np.ndarray (n_rows, N_CLASSES) — the ProtoSSM +
                          MLP-probe + residual-SSM output from cell 36.
                          Cell 37 will consume whatever we leave in this
                          variable.
  - `N_CLASSES`, `SR`, `WINDOW_SAMPLES`, `N_WINDOWS`, `FILE_SAMPLES`,
    `DEVICE`, `BASE` — all from cell 2.
  - `torch`, `np`, `sf`, `pd`, `tqdm`, `Path`, `time` — stdlib / imported
    in cell 2.

Mounted dataset:
  /kaggle/input/birdclef-2026-a1-effb0-ckpts/
    a1_fold0.pt
    a1_fold1.pt
    a1_fold2.pt
    a1_fold4.pt
  (fold 3 intentionally excluded — see new_plan.md "A1 training results").

What this cell does:
  1. Computes A1's 4-fold rank-averaged prediction on test, producing
     `a1_scores` of shape (n_rows, N_CLASSES) in [0, 1].
  2. Fuses with `final_test_scores` in rank space at `A1_WEIGHT = 0.10`,
     preserving ProtoSSM's per-class score distribution so cell 37's
     per-class thresholds retain their semantics.
  3. Overwrites `final_test_scores` with the fused result for cell 37.
"""

CELL_SOURCE = r'''
# Cell 17 — Track A1 SED fusion (4-fold rank-averaged EffNet-B0)
#
# A1 = independent log-mel + PCEN SED branch trained on train_audio focal
# clips. See four_track/new_plan.md §2 "A1 training results (2026-04-07)"
# for why folds = {0,1,2,4} (drop fold 3), why rank-avg, and why w=0.10.

_A1_START = time.time()

# Free Perch test embeddings before A1 starts — they were consumed by cell 36
# (ProtoSSM, prior fusion, MLP probes, residual SSM) and aren't needed past
# this point. emb_test for ~700 files is ~52 MB, but the larger payoff is
# freeing the per-file Perch tensors and giving the kernel headroom for
# A1's mel + JIT working set.
try:
    del emb_test, emb_test_files, emb_test_tensor, logits_test_files, logits_test_tensor
except NameError:
    pass
gc.collect()

A1_CKPT_DIR  = Path("/kaggle/input/birdclef-2026-a1-effb0-ckpts")
A1_FOLDS     = [0, 1, 2, 4]
A1_WEIGHT    = 0.15
A1_BATCH     = 16
A1_N_MELS    = 224
A1_N_FFT     = 4096
A1_HOP       = 1252   # → 512 frames for 20s at 32 kHz
A1_F_MIN     = 0
A1_F_MAX     = 16_000
A1_DURATION  = 20      # A1 was trained on 20s chunks; 5s windows are tiled 4x

# --- Build the mel transform once ---
import torchaudio.transforms as _tT
_a1_mel_transform = _tT.MelSpectrogram(
    sample_rate=SR,
    n_fft=A1_N_FFT,
    hop_length=A1_HOP,
    n_mels=A1_N_MELS,
    f_min=A1_F_MIN,
    f_max=A1_F_MAX,
    power=2.0,
    norm="slaney",
    mel_scale="slaney",
    center=True,
)

# --- PCEN (ported verbatim from BirdCLEF/src/utils.py pcen()) ---
_A1_PCEN_GAIN   = 0.98
_A1_PCEN_SMOOTH = 0.025
_A1_PCEN_BIAS   = 2.0
_A1_PCEN_POWER  = 0.5
_A1_PCEN_EPS    = 1e-6

def _a1_pcen(mel):
    with torch.no_grad():
        E = mel.float()
        T_ = E.shape[2]
        M = E[:, :, 0].clone()
        out = torch.empty_like(E)
        bias_r = _A1_PCEN_BIAS ** _A1_PCEN_POWER
        for t in range(T_):
            M = (1.0 - _A1_PCEN_SMOOTH) * M + _A1_PCEN_SMOOTH * E[:, :, t]
            denom = (M + _A1_PCEN_EPS).pow(_A1_PCEN_GAIN)
            out[:, :, t] = (E[:, :, t] / denom + _A1_PCEN_BIAS).pow(_A1_PCEN_POWER) - bias_r
    return out

def _a1_waveform_to_mel(waveform_1d):
    """5-second waveform (np.float32, len=WINDOW_SAMPLES) → (3, N_MELS, 512) tensor.

    Matches BirdCLEF/src/utils.py waveform_to_mel() exactly: tile the 5s slice
    up to the 20s chunk length A1 was trained on, then mel → PCEN → per-sample
    [0,1] normalise → 3-channel repeat.
    """
    chunk_len = SR * A1_DURATION
    # Tile: 5s slice → 20s (4x repeat) — mirrors pad_or_crop() behavior on short clips
    reps = -(-chunk_len // len(waveform_1d))
    wav  = np.tile(waveform_1d, reps)[:chunk_len]
    wav_t = torch.from_numpy(wav).float().unsqueeze(0)          # (1, T)
    mel   = _a1_mel_transform(wav_t)                             # (1, N_MELS, T')
    out   = _a1_pcen(mel)                                        # (1, N_MELS, T')
    out   = out - out.min()
    peak  = out.max()
    if peak > 0:
        out = out / peak
    return out.repeat(3, 1, 1)                                   # (3, N_MELS, T')

# --- Load the 4 JIT models once ---
print("Loading A1 JIT models …", flush=True)
_a1_models = []
for _f in A1_FOLDS:
    _p = A1_CKPT_DIR / f"a1_fold{_f}.pt"
    if not _p.exists():
        raise FileNotFoundError(f"A1 checkpoint missing: {_p}")
    _m = torch.jit.load(str(_p), map_location=DEVICE).eval()
    _a1_models.append(_m)
    print(f"  loaded fold {_f}: {_p.stat().st_size / 1e6:.1f} MB", flush=True)

# --- Streaming inference: per-file mel + 4-fold forward, then free mels ---
#
# We do NOT pre-compute all mels into a list — at hidden test scale that
# would hold ~12 GB of float32 tensors (8400 segments × 3 × 224 × 512) and
# OOM the kernel. Instead, for each file we compute its 12 mels, run all
# 4 folds on those 12 mels, store the resulting (4, 12, N_CLASSES) sigmoids
# in a pre-allocated numpy array, and immediately free the mel tensors.
# Peak A1 working memory: ~50 MB.
_a1_n_rows = len(meta_test)
_a1_fold_sigmoids = np.zeros((len(A1_FOLDS), _a1_n_rows, N_CLASSES), dtype=np.float32)

print(f"A1 streaming inference: {len(test_paths)} files × {N_WINDOWS} windows "
      f"× {len(A1_FOLDS)} folds …", flush=True)
_a1_t0 = time.time()
_row_offset = 0
for _path in tqdm(test_paths, desc="A1 streaming"):
    _y, _sr = sf.read(str(_path), dtype="float32", always_2d=False)
    if _y.ndim == 2:
        _y = _y.mean(axis=1)
    if _sr != SR:
        raise ValueError(f"Unexpected sample rate {_sr} in {_path.name}")
    if len(_y) < FILE_SAMPLES:
        _y = np.pad(_y, (0, FILE_SAMPLES - len(_y)))
    elif len(_y) > FILE_SAMPLES:
        _y = _y[:FILE_SAMPLES]

    # 12 mels for this file
    _file_mels = []
    for _w in range(N_WINDOWS):
        _seg = _y[_w * WINDOW_SAMPLES : (_w + 1) * WINDOW_SAMPLES]
        _file_mels.append(_a1_waveform_to_mel(_seg))
    _batch_in = torch.stack(_file_mels)                          # (12, 3, 224, 512)
    del _file_mels, _y

    # All 4 folds on this single 12-window batch
    with torch.no_grad():
        for _fi, _m in enumerate(_a1_models):
            _logits = _m(_batch_in)                              # (12, N_CLASSES)
            _a1_fold_sigmoids[_fi, _row_offset:_row_offset + N_WINDOWS] = \
                torch.sigmoid(_logits).float().cpu().numpy()
    del _batch_in
    _row_offset += N_WINDOWS

assert _row_offset == _a1_n_rows, \
    f"A1/meta_test row count mismatch after streaming: {_row_offset} vs {_a1_n_rows}"

del _a1_models
gc.collect()

print(f"  A1 inference complete: {_a1_fold_sigmoids.shape}  "
      f"({time.time() - _a1_t0:.1f}s)", flush=True)

# --- Rank-average across folds, per class ---
# scipy not guaranteed in the kernel; use numpy argsort for ranks.
def _rank01_per_col(mat):
    """Dense percentile rank per column, output in [0, 1]."""
    n = mat.shape[0]
    order = np.argsort(mat, axis=0, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    rows = np.arange(n, dtype=np.float32)
    for c in range(mat.shape[1]):
        ranks[order[:, c], c] = rows
    if n > 1:
        ranks /= (n - 1)
    return ranks

_a1_ranks = np.stack(
    [_rank01_per_col(_a1_fold_sigmoids[i]) for i in range(len(A1_FOLDS))],
    axis=0,
).mean(axis=0).astype(np.float32)                             # (n_rows, N_CLASSES)

# --- Rank-space fusion with ProtoSSM output, preserving score distribution ---
# Round trip: per-class CDF → rank space → linear mix → inverse CDF back
# to ProtoSSM's original score distribution so cell 37's per-class
# thresholds retain their semantics.
print(f"Rank fusion with A1_WEIGHT={A1_WEIGHT:.2f} …", flush=True)

_proto_scores_before_fusion = final_test_scores.astype(np.float32).copy()
_proto_ranks = _rank01_per_col(_proto_scores_before_fusion)   # per-class [0,1] rank
_fused_ranks = (1.0 - A1_WEIGHT) * _proto_ranks + A1_WEIGHT * _a1_ranks

# Per-class inverse CDF: look up the quantile of _fused_ranks in the
# sorted proto scores, so the output has the SAME marginal distribution
# per class as final_test_scores did.
final_test_scores = np.empty_like(_proto_scores_before_fusion)
_n_rows = _proto_scores_before_fusion.shape[0]
_sorted_proto = np.sort(_proto_scores_before_fusion, axis=0)
_idx = np.clip(
    (_fused_ranks * (_n_rows - 1)).round().astype(np.int64),
    0, _n_rows - 1,
)
for _c in range(_proto_scores_before_fusion.shape[1]):
    final_test_scores[:, _c] = _sorted_proto[_idx[:, _c], _c]

# Per-row Δ for logging
_abs_delta = np.abs(final_test_scores - _proto_scores_before_fusion).mean()
print(f"  A1 fusion mean |Δ score| : {_abs_delta:.5f}", flush=True)
print(f"  A1 cell wall time        : {time.time() - _A1_START:.1f}s", flush=True)

LOGS.setdefault("a1_fusion", {})
LOGS["a1_fusion"].update({
    "weight": A1_WEIGHT,
    "folds": A1_FOLDS,
    "mean_abs_delta": float(_abs_delta),
    "wall_time_seconds": float(time.time() - _A1_START),
    "score_range_after": [float(final_test_scores.min()), float(final_test_scores.max())],
})
'''
