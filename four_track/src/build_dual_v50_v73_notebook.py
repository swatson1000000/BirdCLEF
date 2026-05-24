"""Build the Probe-D dual v50+v73 inference notebook from the live
protossm-postproc kernel.

Reads:
    jupyter/protossm-postproc/birdclef2026-protossm-postproc.ipynb

Writes:
    jupyter/dual-v50-v73/birdclef2026-dual-v50-v73.ipynb
    jupyter/dual-v50-v73/kernel-metadata.json

The output notebook is a near-clone of the source with three changes,
all localized to Cell 17 (the A1 fusion cell) and downstream:

    1. Cell 17 modified to save the post-Quantile-Mix fused output as
       `final_test_scores_v73` AND save the pre-A1 ProtoSSM+B1 base as
       `_proto_scores_before_fusion_base`. Existing behavior preserved
       (final_test_scores still bound for downstream).

    2. NEW Cell 17b inserted immediately after Cell 17. Time-guards on
       elapsed wall (skip if > 75 min). Loads v50 ckpts from
       /kaggle/input/birdclef-2026-a1-effb0-v50-ckpts (folds 0,1,2,4),
       runs the same A1 streaming inference pattern, applies the same
       Quantile-Mix into a clone of `_proto_scores_before_fusion_base`,
       saves as `final_test_scores_v50`. On any exception sets
       `final_test_scores_v50 = None` and prints a warning.

    3. NEW Cell 17c inserted immediately after Cell 17b. If
       `final_test_scores_v50 is None`, falls through with v73-only
       output. Otherwise rank-fuses (equal-weight) the two stacks
       per-class, inverse-CDFs back to the ProtoSSM marginal scale so
       Cell 18's hardcoded per-class thresholds remain meaningful, and
       overwrites `final_test_scores`.

    4. The new kernel-metadata.json adds the v50 ckpts dataset slug
       (stevewatson999/birdclef-2026-a1-effb0-v50-ckpts) — that dataset
       must be created+pushed on Kaggle before the kernel can run.

Other risks documented in four_track/new_plan.md §25.12.

Run from project root with the kaggle conda env active:

    source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle
    cd /home/swatson/work/kaggle/BirdCLEF/four_track
    mkdir -p ../jupyter/dual-v50-v73
    python -u src/build_dual_v50_v73_notebook.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]   # /home/swatson/work/kaggle/BirdCLEF
SRC_NB = PROJ_ROOT / "jupyter" / "protossm-postproc" / "birdclef2026-protossm-postproc.ipynb"
DST_DIR = PROJ_ROOT / "jupyter" / "dual-v50-v73"
DST_NB = DST_DIR / "birdclef2026-dual-v50-v73.ipynb"
DST_META = DST_DIR / "kernel-metadata.json"


# ---------------------------------------------------------------------------
# Cell 17 patch: save v73 fusion output + base scores
# ---------------------------------------------------------------------------

# Marker that uniquely identifies Cell 17 in the source notebook.
CELL17_MARKER = "# Cell 17 — Track A1 SED fusion"

# Insert lines RIGHT BEFORE the existing
# `final_test_scores = np.empty_like(_proto_scores_before_fusion)` line.
# These alias state for cell 17b/c WITHOUT changing existing behavior
# (final_test_scores is still bound below).
CELL17_INSERT_BEFORE_FINAL = (
    "final_test_scores = np.empty_like(_proto_scores_before_fusion)\n"
)
CELL17_INSERT_REPLACEMENT = (
    "# DUAL-STACK PATCH: stash pre-A1 base + retain v73 ranks for cell 17b/c.\n"
    "_proto_scores_before_fusion_base = _proto_scores_before_fusion.copy()\n"
    "_a1_v73_ranks = _a1_ranks.copy()\n"
    "final_test_scores = np.empty_like(_proto_scores_before_fusion)\n"
)

# Append at end of cell 17 — copy v73 result for cell 17c.
CELL17_APPEND = (
    "\n"
    "# DUAL-STACK PATCH: snapshot the v73 fused output for cell 17c.\n"
    "final_test_scores_v73 = final_test_scores.copy()\n"
)


# ---------------------------------------------------------------------------
# Cell 17b: v50 stack
# ---------------------------------------------------------------------------

CELL17B_SOURCE = r'''# Cell 17b — DUAL-STACK PATCH: A1 v50 LB-0.931 4-fold stack
#
# Time-guarded re-run of A1 inference using the v50 ckpts (folds 0,1,2,4,
# base EffNet-B0, no iNat pretrain). On success produces
# `final_test_scores_v50`; on any failure sets it to None and the
# downstream cell (17c) falls through to v73-only.
#
# See four_track/new_plan.md §25.12 for design + risks.

_DUAL_T0 = time.time()
_DUAL_ELAPSED_MIN = (_DUAL_T0 - _WALL_START) / 60.0
_DUAL_TIME_GUARD_MIN = 75.0   # skip v50 if we are already >75 min into the kernel

print(f"[dual] elapsed at cell 17b start: {_DUAL_ELAPSED_MIN:.1f} min", flush=True)

final_test_scores_v50 = None

if _DUAL_ELAPSED_MIN > _DUAL_TIME_GUARD_MIN:
    print(f"[dual] elapsed > {_DUAL_TIME_GUARD_MIN:.0f} min — SKIP v50 stack to protect 90-min cap", flush=True)
else:
    try:
        _A1_V50_CKPT_DIR = Path("/kaggle/input/birdclef-2026-a1-effb0-v50-ckpts")
        _A1_V50_FOLDS    = [0, 1, 2, 4]   # v50 production fold list (no fold 3)
        if not _A1_V50_CKPT_DIR.exists():
            raise FileNotFoundError(f"v50 ckpt dir missing: {_A1_V50_CKPT_DIR}")

        # --- Load v50 JIT models ---
        print(f"[dual] loading v50 A1 JIT models from {_A1_V50_CKPT_DIR} …", flush=True)
        _a1_v50_models = []
        for _f in _A1_V50_FOLDS:
            _p = _A1_V50_CKPT_DIR / f"a1_fold{_f}.pt"
            if not _p.exists():
                raise FileNotFoundError(f"v50 ckpt missing: {_p}")
            _m = torch.jit.load(str(_p), map_location=DEVICE).eval()
            _a1_v50_models.append(_m)
            print(f"  loaded v50 fold {_f}: {_p.stat().st_size / 1e6:.1f} MB", flush=True)

        # --- Streaming inference, identical pattern to cell 17 ---
        _a1_v50_n_rows = len(meta_test)
        _a1_v50_fold_sigmoids = np.zeros(
            (len(_A1_V50_FOLDS), _a1_v50_n_rows, N_CLASSES), dtype=np.float32
        )
        print(f"[dual] v50 streaming: {len(test_paths)} files × {N_WINDOWS} windows × "
              f"{len(_A1_V50_FOLDS)} folds …", flush=True)
        _a1_v50_t0 = time.time()
        _row_offset = 0
        for _path in tqdm(test_paths, desc="A1 v50 streaming"):
            _y, _sr = sf.read(str(_path), dtype="float32", always_2d=False)
            if _y.ndim == 2:
                _y = _y.mean(axis=1)
            if _sr != SR:
                raise ValueError(f"Unexpected sample rate {_sr} in {_path.name}")
            if len(_y) < FILE_SAMPLES:
                _y = np.pad(_y, (0, FILE_SAMPLES - len(_y)))
            elif len(_y) > FILE_SAMPLES:
                _y = _y[:FILE_SAMPLES]

            _file_mels = []
            for _w in range(N_WINDOWS):
                _seg = _y[_w * WINDOW_SAMPLES : (_w + 1) * WINDOW_SAMPLES]
                _file_mels.append(_a1_waveform_to_mel(_seg))
            _batch_in = torch.stack(_file_mels)
            del _file_mels, _y

            with torch.no_grad():
                for _fi, _m in enumerate(_a1_v50_models):
                    _logits = _m(_batch_in)
                    _a1_v50_fold_sigmoids[_fi, _row_offset:_row_offset + N_WINDOWS] = \
                        torch.sigmoid(_logits).float().cpu().numpy()
            del _batch_in
            _row_offset += N_WINDOWS

        assert _row_offset == _a1_v50_n_rows

        del _a1_v50_models
        gc.collect()
        print(f"[dual] v50 A1 inference complete: {_a1_v50_fold_sigmoids.shape}  "
              f"({time.time() - _a1_v50_t0:.1f}s)", flush=True)

        # --- Fold reduce (rank each fold, mean across fold-ranks; same as cell 17 baseline) ---
        _a1_v50_ranks = np.stack(
            [_rank01_per_col(_a1_v50_fold_sigmoids[i]) for i in range(len(_A1_V50_FOLDS))],
            axis=0,
        ).mean(axis=0).astype(np.float32)

        # --- Quantile-Mix into a CLONE of the pre-A1 ProtoSSM+B1 base ---
        # Same logic as cell 17 (alpha=0 → pure rank-space cascade).
        _A1_V50_WEIGHT = 0.20
        _A1_V50_QMIX_ALPHA = 0.0
        print(f"[dual] v50 Quantile-Mix: A1_WEIGHT={_A1_V50_WEIGHT:.2f}, α={_A1_V50_QMIX_ALPHA:.2f}",
              flush=True)

        _v50_proto_base = _proto_scores_before_fusion_base.copy()
        _v50_proto_ranks = _rank01_per_col(_v50_proto_base)
        _v50_rank_blend = (1.0 - _A1_V50_WEIGHT) * _v50_proto_ranks + _A1_V50_WEIGHT * _a1_v50_ranks

        _v50_proto_lo = _v50_proto_base.min(axis=0, keepdims=True)
        _v50_proto_hi = _v50_proto_base.max(axis=0, keepdims=True)
        _v50_proto_01 = (_v50_proto_base - _v50_proto_lo) / (_v50_proto_hi - _v50_proto_lo + 1e-9)
        _v50_a1_sig = _a1_v50_fold_sigmoids.mean(axis=0).astype(np.float32)
        _v50_prob_blend = (1.0 - _A1_V50_WEIGHT) * _v50_proto_01 + _A1_V50_WEIGHT * _v50_a1_sig
        _v50_qmix_ranks = _A1_V50_QMIX_ALPHA * _v50_prob_blend + (1.0 - _A1_V50_QMIX_ALPHA) * _v50_rank_blend
        _v50_qmix_ranks = np.clip(_v50_qmix_ranks, 0.0, 1.0)

        # Per-class inverse CDF onto v50 base marginals — same pattern as cell 17.
        final_test_scores_v50 = np.empty_like(_v50_proto_base)
        _n_rows_v50 = _v50_proto_base.shape[0]
        _sorted_proto_v50 = np.sort(_v50_proto_base, axis=0)
        _idx_v50 = np.clip(
            (_v50_qmix_ranks * (_n_rows_v50 - 1)).round().astype(np.int64),
            0, _n_rows_v50 - 1,
        )
        for _c in range(_v50_proto_base.shape[1]):
            final_test_scores_v50[:, _c] = _sorted_proto_v50[_idx_v50[:, _c], _c]

        # Free v50 working arrays before cell 17c.
        del _a1_v50_fold_sigmoids, _v50_proto_base, _v50_proto_ranks
        del _v50_rank_blend, _v50_proto_lo, _v50_proto_hi, _v50_proto_01
        del _v50_a1_sig, _v50_prob_blend, _v50_qmix_ranks, _sorted_proto_v50, _idx_v50
        gc.collect()

        print(f"[dual] v50 stack complete: shape={final_test_scores_v50.shape}, "
              f"range=[{final_test_scores_v50.min():.4f}, {final_test_scores_v50.max():.4f}]",
              flush=True)
        print(f"[dual] cell 17b wall time: {time.time() - _DUAL_T0:.1f}s", flush=True)

        LOGS.setdefault("dual_v50", {})
        LOGS["dual_v50"].update({
            "ckpt_dir": str(_A1_V50_CKPT_DIR),
            "folds": _A1_V50_FOLDS,
            "weight": _A1_V50_WEIGHT,
            "qmix_alpha": _A1_V50_QMIX_ALPHA,
            "wall_time_seconds": float(time.time() - _DUAL_T0),
            "score_range": [float(final_test_scores_v50.min()), float(final_test_scores_v50.max())],
            "elapsed_at_start_min": _DUAL_ELAPSED_MIN,
        })

    except Exception as _exc:
        # Any failure — missing ckpts, OOM, anything — falls through to v73 only.
        import traceback
        print(f"[dual] WARNING: v50 stack FAILED, falling back to v73-only: {_exc}", flush=True)
        traceback.print_exc()
        final_test_scores_v50 = None
        LOGS.setdefault("dual_v50", {})
        LOGS["dual_v50"]["error"] = repr(_exc)
'''

# ---------------------------------------------------------------------------
# Cell 17c: dual rank-fusion
# ---------------------------------------------------------------------------

CELL17C_SOURCE = r'''# Cell 17c — DUAL-STACK PATCH: rank-fuse v73 + v50 outputs (Probe D)
#
# If v50 stack succeeded, equal-weight rank-fuse the two production stacks
# per-class, inverse-CDF back to the v73 marginal scale (so cell 18's
# hardcoded per-class thresholds stay calibrated), and overwrite
# final_test_scores. Otherwise leave final_test_scores at the v73 value.
#
# See four_track/new_plan.md §25.12.

_DUAL_FUSE_T0 = time.time()

if final_test_scores_v50 is None:
    print("[dual] v50 unavailable — using v73-only output (no fusion).", flush=True)
    final_test_scores = final_test_scores_v73
    LOGS.setdefault("dual_fusion", {})
    LOGS["dual_fusion"].update({"mode": "v73_only_fallback"})
else:
    print("[dual] rank-fusing v73 + v50 stacks (equal weight) …", flush=True)
    _v73_ranks_final = _rank01_per_col(final_test_scores_v73.astype(np.float32, copy=False))
    _v50_ranks_final = _rank01_per_col(final_test_scores_v50.astype(np.float32, copy=False))
    _DUAL_W_V73 = 0.5
    _DUAL_W_V50 = 1.0 - _DUAL_W_V73
    _dual_ranks = _DUAL_W_V73 * _v73_ranks_final + _DUAL_W_V50 * _v50_ranks_final
    _dual_ranks = np.clip(_dual_ranks, 0.0, 1.0)

    # Inverse-CDF back to v73 marginal scale so cell 18 thresholds remain meaningful.
    _dual_out = np.empty_like(final_test_scores_v73)
    _n_rows_dual = final_test_scores_v73.shape[0]
    _sorted_v73 = np.sort(final_test_scores_v73, axis=0)
    _idx_dual = np.clip(
        (_dual_ranks * (_n_rows_dual - 1)).round().astype(np.int64),
        0, _n_rows_dual - 1,
    )
    for _c in range(final_test_scores_v73.shape[1]):
        _dual_out[:, _c] = _sorted_v73[_idx_dual[:, _c], _c]

    _abs_delta_dual = float(np.abs(_dual_out - final_test_scores_v73).mean())
    print(f"[dual] dual mean |Δ vs v73| = {_abs_delta_dual:.5f}", flush=True)
    print(f"[dual] cell 17c wall time: {time.time() - _DUAL_FUSE_T0:.1f}s", flush=True)

    final_test_scores = _dual_out

    LOGS.setdefault("dual_fusion", {})
    LOGS["dual_fusion"].update({
        "mode": "v50_v73_dual",
        "weight_v73": _DUAL_W_V73,
        "weight_v50": _DUAL_W_V50,
        "mean_abs_delta_vs_v73": _abs_delta_dual,
        "wall_time_seconds": float(time.time() - _DUAL_FUSE_T0),
    })

    # Free intermediates.
    del _v73_ranks_final, _v50_ranks_final, _dual_ranks, _sorted_v73, _idx_dual, _dual_out
    gc.collect()
'''


def _make_code_cell(source_str: str) -> dict:
    """Construct a minimal Jupyter v4 code cell from a source string."""
    lines = source_str.splitlines(keepends=True)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def _join(src) -> str:
    """nbformat source can be either a string or a list of strings."""
    return src if isinstance(src, str) else "".join(src)


def patch_cell17(cell: dict) -> dict:
    src = _join(cell["source"])
    if CELL17_MARKER not in src:
        raise RuntimeError("Cell 17 marker not found — source notebook layout changed.")

    # 1. Replace the pre-final-init block with the patched version.
    if CELL17_INSERT_BEFORE_FINAL not in src:
        raise RuntimeError(
            "Cell 17: expected 'final_test_scores = np.empty_like(...)' init "
            "block not found. Did the source notebook change?"
        )
    src = src.replace(CELL17_INSERT_BEFORE_FINAL, CELL17_INSERT_REPLACEMENT, 1)

    # 2. Append the v73 snapshot at the end.
    src = src + CELL17_APPEND

    cell = copy.deepcopy(cell)
    cell["source"] = src.splitlines(keepends=True)
    return cell


def main() -> int:
    if not SRC_NB.exists():
        print(f"ERROR: source notebook missing: {SRC_NB}", file=sys.stderr)
        return 1
    if not DST_DIR.exists():
        print(f"ERROR: destination dir missing: {DST_DIR}", file=sys.stderr)
        print(f"       run `mkdir -p {DST_DIR}` first.", file=sys.stderr)
        return 2

    print(f"reading {SRC_NB}", flush=True)
    nb = json.loads(SRC_NB.read_text())
    cells = nb["cells"]

    # Locate cell 17 by marker.
    cell17_idx = None
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        if CELL17_MARKER in _join(c["source"]):
            cell17_idx = i
            break
    if cell17_idx is None:
        print("ERROR: could not locate Cell 17 by marker", file=sys.stderr)
        return 3

    print(f"found Cell 17 at notebook cell index {cell17_idx}", flush=True)

    # Patch cell 17 in place.
    cells[cell17_idx] = patch_cell17(cells[cell17_idx])

    # Insert 17b and 17c immediately after.
    cells.insert(cell17_idx + 1, _make_code_cell(CELL17B_SOURCE))
    cells.insert(cell17_idx + 2, _make_code_cell(CELL17C_SOURCE))

    # Bump kernel description if present.
    if "metadata" in nb:
        nb["metadata"].setdefault("dual_v50_v73", {
            "source_notebook": str(SRC_NB.relative_to(PROJ_ROOT)),
            "design_doc": "four_track/new_plan.md §25.12",
        })

    print(f"writing {DST_NB}", flush=True)
    DST_NB.write_text(json.dumps(nb, indent=1))

    # Kernel metadata.
    kernel_meta = {
        "id": "stevewatson999/birdclef-2026-dual-v50-v73",
        "title": "BirdCLEF 2026 Dual v50+v73",
        "code_file": "birdclef2026-dual-v50-v73.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [
            "jaejohn/perch-meta",
            "brucewu1200/birdclef-2026-cvlb-assets-0911",
            "stevewatson999/birdclef-2026-a1-effb0-ckpts",
            "stevewatson999/birdclef-2026-a1-effb0-ns-ckpts",
            "stevewatson999/birdclef-2026-a1-effb0-v50-ckpts",
            "stevewatson999/birdclef-2026-b2-convnext-ckpts",
        ],
        "competition_sources": ["birdclef-2026"],
        "kernel_sources": ["ashok205/tf-wheels"],
        "model_sources": [
            "google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1"
        ],
    }
    print(f"writing {DST_META}", flush=True)
    DST_META.write_text(json.dumps(kernel_meta, indent=2) + "\n")

    print("DONE", flush=True)
    print(f"  notebook : {DST_NB}", flush=True)
    print(f"  metadata : {DST_META}", flush=True)
    print("", flush=True)
    print("Next steps (see new_plan.md §25.12):", flush=True)
    print("  1. (optional) run local broader-pool gate before pushing", flush=True)
    print("  2. push v50 ckpts dataset:", flush=True)
    print("       cd four_track/kaggle_datasets/a1-effb0-v50-ckpts/", flush=True)
    print("       kaggle datasets create -p .   # first time", flush=True)
    print("  3. push the kernel:", flush=True)
    print(f"       cd {DST_DIR.relative_to(PROJ_ROOT)}/", flush=True)
    print("       kaggle kernels push -p .", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
