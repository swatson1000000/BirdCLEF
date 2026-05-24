"""D2-β OOF dump cell for the LB notebook (Phase 1 of D2-β).

This file is the canonical source for a single cell that gets injected
into `jupyter/protossm-postproc/birdclef2026-protossm-postproc.ipynb`
immediately after cell 31b (Track B1 OOF). Keeping the source here in
version control means we can iterate without hand-editing the notebook.

**What this cell does (train mode only)**:

  1. Asserts the three shared-substrate variables produced by cell 31b
     (`oof_proto_flat`, `oof_b1_flat`, `y_flat`, `file_groups`) exist
     and have the expected shapes.
  2. Loads the A1 5-fold JIT checkpoints and runs streaming inference on
     the **same 59 fully-labeled train_soundscapes files** that B1 and
     ProtoSSM OOF'd on (`full_paths` from cell 16). One-time ~3-5 minute
     cost. Produces `_a1_fold_sigmoids_oof` of shape
     `(5, n_files * N_WINDOWS, N_CLASSES)`.
  3. Rank-averages A1 across its 5 folds per class to mirror what the
     submit-path A1 cell (cell 37) does, producing `a1_ranks_oof` of
     shape `(n_files * N_WINDOWS, N_CLASSES) = (708, 234)` on the
     default 59-file labeled pool.
  4. Dumps `/kaggle/working/d2_beta_oofs.npz` with:
       - `a1_ranks`       (708, 234) float32 — rank-averaged A1 on the OOF pool
       - `b1_oof`         (708, 234) float32 — B1 PerceiverIO GroupKFold OOF sigmoids
       - `proto_oof`      (708, 234) float32 — ProtoSSM GroupKFold OOF sigmoids
       - `y_true`         (708, 234) float32 — per-window labels
       - `file_groups`    (n_files,) int     — GroupKFold group IDs (per-file)
       - `fold_ids`       (708,)     int     — 12-tiled repeat of file_groups
       - `n_windows`      scalar int         — N_WINDOWS (12)
       - `a1_folds`       (5,)       int     — [0,1,2,3,4] (v4 5-fold bundle 2026-05-12)
       - `notebook_sha`   str                — short git sha at dump time (best-effort)

In submit mode this cell is a no-op except for a short log line — it's
safe to leave injected and the LB notebook continues to work normally.

**Runtime contract (what must exist in the kernel when this cell runs):**

  - From cell 1: `MODE` ∈ {"train", "submit"}.
  - From cell 2: `torch`, `np`, `sf`, `Path`, `tqdm`, `time`, `gc`,
    `SR`, `WINDOW_SAMPLES`, `FILE_SAMPLES`, `N_WINDOWS`, `N_CLASSES`, `DEVICE`.
  - From cell 16: `full_paths` (list[Path] of the 59 labeled train_soundscapes files).
  - From cell 31 (ProtoSSM OOF): `oof_proto_flat` shape (n_files * N_WINDOWS, N_CLASSES).
  - From cell 31b (B1 OOF, `b1_perceiver.py:CELL_31B`): `oof_b1_flat` shape
    (n_files * N_WINDOWS, N_CLASSES), `y_flat` shape (n_files * N_WINDOWS, N_CLASSES),
    `file_groups` shape (n_files,).

**Mounted dataset (reused from cell 37):**

  /kaggle/input/birdclef-2026-a1-effb0-ckpts/
    a1_fold0.pt
    a1_fold1.pt
    a1_fold2.pt
    a1_fold3.pt
    a1_fold4.pt

  Already in `kernel-metadata.json:dataset_sources` from the A1 cell
  injection — no new dataset is required.

**Output artifact:**

  /kaggle/working/d2_beta_oofs.npz   (~7 MB for 708 rows × 234 classes × 4 arrays)

  After the kernel run completes in MODE="train", download this file
  from the kernel output and save it as
  `four_track/data/d2_beta_oofs.npz`. Phase 2 (`d2_lgbm_stack.py`) consumes it.

**Side effect**: A1 JIT models are deleted at the end of this cell, so
the later cell 37 (submit-path A1 fusion) will reload them. Memory
footprint matches what cell 37 does in isolation; no net change.
"""

CELL_SOURCE = r'''
# Cell 31c — D2-β OOF dump (train-mode only; no-op in submit mode)
#
# Phase 1 of D2-β. Dumps A1/B1/ProtoSSM OOF sigmoids on the 59-labeled-
# file substrate to /kaggle/working/d2_beta_oofs.npz so a LightGBM
# stacker can be developed locally. See four_track/new_plan.md §D2-β.

_D2B_START = time.time()

if MODE != "train":
    print("[D2-β OOF dump] MODE != 'train' — skipping OOF dump cell.", flush=True)
else:
    print("\n=== D2-β OOF dump (train mode) ===", flush=True)

    # --- 1. Sanity-check the B1/ProtoSSM variables from cell 31 / 31b ---
    for _name in ("oof_proto_flat", "oof_b1_flat", "y_flat", "file_groups", "full_paths"):
        if _name not in globals():
            raise RuntimeError(
                f"[D2-β OOF dump] required variable `{_name}` is missing. "
                f"This cell must run after cell 31b (B1 OOF) in train mode."
            )

    _d2b_n_files = len(full_paths)
    _d2b_n_rows  = _d2b_n_files * N_WINDOWS

    if oof_proto_flat.shape != (_d2b_n_rows, N_CLASSES):
        raise RuntimeError(
            f"[D2-β OOF dump] oof_proto_flat shape {oof_proto_flat.shape} "
            f"!= expected ({_d2b_n_rows}, {N_CLASSES})"
        )
    if oof_b1_flat.shape != (_d2b_n_rows, N_CLASSES):
        raise RuntimeError(
            f"[D2-β OOF dump] oof_b1_flat shape {oof_b1_flat.shape} "
            f"!= expected ({_d2b_n_rows}, {N_CLASSES})"
        )
    if y_flat.shape != (_d2b_n_rows, N_CLASSES):
        raise RuntimeError(
            f"[D2-β OOF dump] y_flat shape {y_flat.shape} "
            f"!= expected ({_d2b_n_rows}, {N_CLASSES})"
        )
    if len(file_groups) != _d2b_n_files:
        raise RuntimeError(
            f"[D2-β OOF dump] file_groups length {len(file_groups)} "
            f"!= expected {_d2b_n_files}"
        )

    print(f"  OOF substrate: {_d2b_n_files} files × {N_WINDOWS} windows = "
          f"{_d2b_n_rows} rows, {N_CLASSES} classes", flush=True)

    # --- 2. A1 streaming inference on full_paths (same mel + PCEN pipeline
    #        as cell 37's A1 fusion — mirrored here so we don't depend on
    #        cell ordering). A1 uses mels, NOT Perch embeddings, so there
    #        is no local-vs-Kaggle embedding mismatch to worry about. ---

    _D2B_A1_CKPT_DIR = Path("/kaggle/input/birdclef-2026-a1-effb0-ckpts")
    _D2B_A1_FOLDS    = [0, 1, 2, 3, 4]
    _D2B_A1_N_MELS   = 224
    _D2B_A1_N_FFT    = 4096
    _D2B_A1_HOP      = 1252
    _D2B_A1_F_MIN    = 0
    _D2B_A1_F_MAX    = 16_000
    _D2B_A1_DURATION = 20

    import torchaudio.transforms as _d2b_tT
    _d2b_mel_transform = _d2b_tT.MelSpectrogram(
        sample_rate=SR,
        n_fft=_D2B_A1_N_FFT,
        hop_length=_D2B_A1_HOP,
        n_mels=_D2B_A1_N_MELS,
        f_min=_D2B_A1_F_MIN,
        f_max=_D2B_A1_F_MAX,
        power=2.0,
        norm="slaney",
        mel_scale="slaney",
        center=True,
    )

    _D2B_PCEN_GAIN   = 0.98
    _D2B_PCEN_SMOOTH = 0.025
    _D2B_PCEN_BIAS   = 2.0
    _D2B_PCEN_POWER  = 0.5
    _D2B_PCEN_EPS    = 1e-6

    def _d2b_pcen(mel):
        with torch.no_grad():
            E = mel.float()
            T_ = E.shape[2]
            M = E[:, :, 0].clone()
            out = torch.empty_like(E)
            bias_r = _D2B_PCEN_BIAS ** _D2B_PCEN_POWER
            for t in range(T_):
                M = (1.0 - _D2B_PCEN_SMOOTH) * M + _D2B_PCEN_SMOOTH * E[:, :, t]
                denom = (M + _D2B_PCEN_EPS).pow(_D2B_PCEN_GAIN)
                out[:, :, t] = (E[:, :, t] / denom + _D2B_PCEN_BIAS).pow(_D2B_PCEN_POWER) - bias_r
        return out

    def _d2b_waveform_to_mel(waveform_1d):
        chunk_len = SR * _D2B_A1_DURATION
        reps = -(-chunk_len // len(waveform_1d))
        wav  = np.tile(waveform_1d, reps)[:chunk_len]
        wav_t = torch.from_numpy(wav).float().unsqueeze(0)
        mel   = _d2b_mel_transform(wav_t)
        out   = _d2b_pcen(mel)
        out   = out - out.min()
        peak  = out.max()
        if peak > 0:
            out = out / peak
        return out.repeat(3, 1, 1)

    print("  Loading A1 JIT models for OOF inference …", flush=True)
    _d2b_a1_models = []
    for _f in _D2B_A1_FOLDS:
        _p = _D2B_A1_CKPT_DIR / f"a1_fold{_f}.pt"
        if not _p.exists():
            raise FileNotFoundError(f"A1 checkpoint missing: {_p}")
        _m = torch.jit.load(str(_p), map_location=DEVICE).eval()
        _d2b_a1_models.append(_m)

    _d2b_a1_fold_sigmoids = np.zeros(
        (len(_D2B_A1_FOLDS), _d2b_n_rows, N_CLASSES), dtype=np.float32
    )

    print(f"  A1 streaming inference: {_d2b_n_files} files × {N_WINDOWS} "
          f"windows × {len(_D2B_A1_FOLDS)} folds …", flush=True)
    _d2b_a1_t0  = time.time()
    _d2b_row_offset = 0
    for _path in tqdm(full_paths, desc="D2-β A1 OOF"):
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
            _file_mels.append(_d2b_waveform_to_mel(_seg))
        _batch_in = torch.stack(_file_mels)
        del _file_mels, _y

        with torch.no_grad():
            for _fi, _m in enumerate(_d2b_a1_models):
                _logits = _m(_batch_in)
                _d2b_a1_fold_sigmoids[_fi,
                    _d2b_row_offset:_d2b_row_offset + N_WINDOWS] = \
                    torch.sigmoid(_logits).float().cpu().numpy()
        del _batch_in
        _d2b_row_offset += N_WINDOWS

    assert _d2b_row_offset == _d2b_n_rows, \
        f"D2-β A1 row count mismatch: {_d2b_row_offset} vs {_d2b_n_rows}"

    del _d2b_a1_models
    gc.collect()

    print(f"  A1 OOF inference done: {_d2b_a1_fold_sigmoids.shape}  "
          f"({time.time() - _d2b_a1_t0:.1f}s)", flush=True)

    # --- 3. Rank-average A1 across its 4 folds per class, mirroring what
    #        cell 37 does on the submit path. Ranks are order-preserving
    #        per class so per-fold temperature/bias drift is nullified. ---
    def _d2b_rank01_per_col(mat):
        n = mat.shape[0]
        order = np.argsort(mat, axis=0, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float32)
        rows = np.arange(n, dtype=np.float32)
        for c in range(mat.shape[1]):
            ranks[order[:, c], c] = rows
        if n > 1:
            ranks /= (n - 1)
        return ranks

    _d2b_a1_ranks_oof = np.stack(
        [_d2b_rank01_per_col(_d2b_a1_fold_sigmoids[i]) for i in range(len(_D2B_A1_FOLDS))],
        axis=0,
    ).mean(axis=0).astype(np.float32)

    # --- 3b. Compute the production "frozen rank fusion" output on the
    #         OOF substrate. Mirrors cells 36b (B1 fusion) → 37 (A1 fusion)
    #         exactly, so the resulting `_d2b_prod_fused` is what the LB
    #         notebook would output on this 708-row substrate at the
    #         frozen weights. This gives D2-β Phase 2 a trustworthy local
    #         reference to compare its stacker against — without this, the
    #         local gate is structurally broken (A1-alone beats rank-mean
    #         blends on this small substrate; see project_b1_weight_sweep
    #         memory and D2-γ null result). ---
    _D2B_B1_WEIGHT_PROD = 0.10   # matches LB 0.933 production (new_plan.md §B1)
    _D2B_A1_WEIGHT_PROD = 0.20   # matches LB 0.933 production (new_plan.md §A1)

    def _d2b_rank_fuse(base_scores, other_ranks, w):
        """Linear rank-mix with inverse-CDF back to base marginals.

        Identical to the fusion in a1_notebook_cell.py:208-229 and
        b1_perceiver.py CELL_36B. Preserves `base_scores`'s per-class
        marginal distribution so downstream per-class thresholds retain
        their semantics.
        """
        base = base_scores.astype(np.float32).copy()
        base_ranks = _d2b_rank01_per_col(base)
        fused_ranks = (1.0 - w) * base_ranks + w * other_ranks
        out = np.empty_like(base)
        n = base.shape[0]
        sorted_base = np.sort(base, axis=0)
        idx = np.clip(
            (fused_ranks * (n - 1)).round().astype(np.int64),
            0, n - 1,
        )
        for c in range(base.shape[1]):
            out[:, c] = sorted_base[idx[:, c], c]
        return out

    # Step 1: ProtoSSM base
    _d2b_proto_base = oof_proto_flat.astype(np.float32).copy()

    # Step 2: B1 rank fusion at frozen production weight.
    # B1's OOF gate FAILS on this substrate (kernel log: "B1 OOF gates
    # FAILED"), so in the notebook's own train-mode run B1_WEIGHT_FROZEN
    # would be 0.0 — but that's NOT what the LB production path does. The
    # submit path uses the frozen-LB weight regardless. We mirror the
    # submit path here so the local reference is apples-to-apples with
    # what actually ships on Kaggle.
    _d2b_b1_ranks_for_fusion = _d2b_rank01_per_col(oof_b1_flat.astype(np.float32))
    _d2b_after_b1 = _d2b_rank_fuse(
        _d2b_proto_base, _d2b_b1_ranks_for_fusion, _D2B_B1_WEIGHT_PROD
    )

    # Step 3: A1 rank fusion on top.
    _d2b_prod_fused = _d2b_rank_fuse(
        _d2b_after_b1, _d2b_a1_ranks_oof, _D2B_A1_WEIGHT_PROD
    )

    # Sanity numbers — compute macro ROC-AUC on present classes for each
    # intermediate. Not strictly needed but extremely useful to catch
    # structural surprises on the kernel log before downloading the npz.
    try:
        from sklearn.metrics import roc_auc_score as _d2b_auc

        def _d2b_macro_auc(y, p):
            present = (y.sum(axis=0) > 0) & (y.sum(axis=0) < y.shape[0])
            if present.sum() == 0:
                return float("nan")
            return float(_d2b_auc(
                y[:, present], p[:, present], average="macro"
            ))

        _d2b_auc_proto   = _d2b_macro_auc(y_flat, _d2b_proto_base)
        _d2b_auc_after_b1 = _d2b_macro_auc(y_flat, _d2b_after_b1)
        _d2b_auc_final   = _d2b_macro_auc(y_flat, _d2b_prod_fused)
        _d2b_auc_a1only  = _d2b_macro_auc(y_flat, _d2b_a1_ranks_oof)
        print(f"  [local AUC] proto-only          = {_d2b_auc_proto:.4f}", flush=True)
        print(f"  [local AUC] proto+B1(w={_D2B_B1_WEIGHT_PROD:.2f})    = {_d2b_auc_after_b1:.4f}", flush=True)
        print(f"  [local AUC] +A1(w={_D2B_A1_WEIGHT_PROD:.2f}) = PROD  = {_d2b_auc_final:.4f}", flush=True)
        print(f"  [local AUC] A1-ranks alone      = {_d2b_auc_a1only:.4f}", flush=True)
    except Exception as _e:
        print(f"  [local AUC] skipped ({_e})", flush=True)
        _d2b_auc_proto = _d2b_auc_after_b1 = _d2b_auc_final = _d2b_auc_a1only = float("nan")

    # --- 4. Dump everything to /kaggle/working/d2_beta_oofs.npz ---
    # file_groups from cell 31 holds site-ID strings (e.g. 'S08') — factorize
    # into contiguous ints so downstream LightGBM GroupKFold can consume them
    # directly. Preserve the original string names as a side array so the
    # mapping is auditable.
    _d2b_fg_names = np.asarray(file_groups)
    _d2b_fg_unique, _d2b_fg_ids = np.unique(_d2b_fg_names, return_inverse=True)
    _d2b_fg_ids = _d2b_fg_ids.astype(np.int64)
    _d2b_fold_ids = np.repeat(_d2b_fg_ids, N_WINDOWS)

    _d2b_out_path = Path("/kaggle/working/d2_beta_oofs.npz")
    np.savez(
        _d2b_out_path,
        a1_ranks         = _d2b_a1_ranks_oof.astype(np.float32),
        b1_oof           = oof_b1_flat.astype(np.float32),
        proto_oof        = oof_proto_flat.astype(np.float32),
        prod_fused       = _d2b_prod_fused.astype(np.float32),
        y_true           = y_flat.astype(np.float32),
        file_groups      = _d2b_fg_ids,
        file_group_names = _d2b_fg_unique.astype(str),
        fold_ids         = _d2b_fold_ids,
        n_windows        = np.int64(N_WINDOWS),
        a1_folds         = np.asarray(_D2B_A1_FOLDS, dtype=np.int64),
        b1_weight_prod   = np.float32(_D2B_B1_WEIGHT_PROD),
        a1_weight_prod   = np.float32(_D2B_A1_WEIGHT_PROD),
    )
    _d2b_size_mb = _d2b_out_path.stat().st_size / 1e6
    print(f"  saved → {_d2b_out_path}  ({_d2b_size_mb:.2f} MB)", flush=True)

    # Log some quick sanity numbers so the kernel run history shows whether
    # the dump is healthy without needing to download it first.
    LOGS.setdefault("d2_beta_oof_dump", {})
    LOGS["d2_beta_oof_dump"].update({
        "n_files":            int(_d2b_n_files),
        "n_rows":             int(_d2b_n_rows),
        "n_classes":          int(N_CLASSES),
        "a1_ranks_mean":      float(_d2b_a1_ranks_oof.mean()),
        "b1_oof_mean":        float(oof_b1_flat.mean()),
        "proto_oof_mean":     float(oof_proto_flat.mean()),
        "y_true_positives":   int(y_flat.sum()),
        "present_classes":    int((y_flat.sum(axis=0) > 0).sum()),
        "b1_weight_prod":     float(_D2B_B1_WEIGHT_PROD),
        "a1_weight_prod":     float(_D2B_A1_WEIGHT_PROD),
        "auc_proto_only":     float(_d2b_auc_proto),
        "auc_after_b1":       float(_d2b_auc_after_b1),
        "auc_prod_fused":     float(_d2b_auc_final),
        "auc_a1_ranks_only":  float(_d2b_auc_a1only),
        "file_size_mb":       float(_d2b_size_mb),
        "wall_time_seconds":  float(time.time() - _D2B_START),
    })
    print(f"  D2-β OOF dump wall time: {time.time() - _D2B_START:.1f}s", flush=True)
    print("  (Download /kaggle/working/d2_beta_oofs.npz after the kernel "
          "finishes → four_track/data/d2_beta_oofs.npz for Phase 2.)",
          flush=True)
'''
