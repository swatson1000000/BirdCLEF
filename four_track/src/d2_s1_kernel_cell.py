"""D2-β S1 stacker — Kaggle notebook cell.

Injected after cell 37 (A1 rank fusion). In submit mode, applies the
fitted per-class logistic stacker `S1` (trained on the 708-row OOF
substrate, see `four_track/src/d2_lgbm_stack.py:fit_final_s1`) to the
A1+B1+ProtoSSM rank features and overwrites `final_test_scores`. In
train mode this cell is a no-op.

**Source variables expected (from earlier cells, all at module scope):**

  - `MODE`                ∈ {"train","submit"}                — cell 1
  - `final_test_scores`   (n_test, N_CLASSES) — set by cells 36/36b/37
  - `_proto_scores_before` (n_test, N_CLASSES) — raw ProtoSSM scores
                          before B1 fusion. Set by cell 36b at line 637.
                          NOT clobbered by cell 37 (which uses a
                          different name `_proto_scores_before_fusion`).
                          If cell 36b was a no-op (B1_WEIGHT_FROZEN<=0),
                          falls back to `_proto_scores_before_fusion`,
                          which in that case is the raw proto scores.
  - `_b1_ranks`           (n_test, N_CLASSES) — B1 test ranks. Set by
                          cell 36b at line 639. Only present if B1
                          fusion ran. We recompute from `_b1_test_flat`
                          (also set in 36b) if needed; ultimate fallback
                          is to set b1 column to 0.5 (rank-neutral).
  - `_a1_ranks`           (n_test, N_CLASSES) — A1 4-fold rank-averaged.
                          Set by cell 37 at line 203.

**External dataset (added to kernel-metadata.json by inject_d2_s1_cell.py):**

  /kaggle/input/birdclef-2026-d2b-s1-coefs/d2_beta_s1_coefs.npz
    — fitted on all 708 OOF rows of the D2-β substrate
    — keys: coefs (n_classes,3), intercepts (n_classes,), fallback (bool,n_classes),
            n_pos (n_classes,), n_classes, n_train_rows
    — fallback==True classes pass A1-rank straight through (no blending)

**Output**: overwrites `final_test_scores` in place. The output is
inverse-CDF-mapped back into ProtoSSM's per-class marginals so cell 38's
per-class thresholds retain their semantics (mirrors A1/B1 cells).
"""

CELL_SOURCE = r'''
# Cell 37b — D2-β S1 stacker (LB probe; submit mode only)
#
# Replaces the rank-fusion output `final_test_scores` with the per-class
# logistic stacker S1 trained on the 708-row D2-β OOF substrate. In train
# mode this is a no-op.

_S1_START = time.time()

if MODE != "submit":
    print("[D2-β S1] MODE != 'submit' — skipping S1 stacker cell.", flush=True)
else:
    print("\n=== D2-β S1 stacker (submit mode) ===", flush=True)

    # --- 1. Load fitted coefficients ---
    _S1_CKPT = Path("/kaggle/input/birdclef-2026-d2b-s1-coefs/d2_beta_s1_coefs.npz")
    if not _S1_CKPT.exists():
        raise FileNotFoundError(f"S1 coefficients missing: {_S1_CKPT}")
    _s1_z = np.load(_S1_CKPT, allow_pickle=False)
    _s1_coefs      = _s1_z["coefs"].astype(np.float32)       # (n_classes, 3)
    _s1_intercepts = _s1_z["intercepts"].astype(np.float32)  # (n_classes,)
    _s1_fallback   = _s1_z["fallback"].astype(bool)          # (n_classes,)
    _s1_n_classes  = int(_s1_z["n_classes"])
    _s1_n_train    = int(_s1_z["n_train_rows"])
    print(f"  loaded S1 coefs: {_s1_coefs.shape}, "
          f"trained on {_s1_n_train} rows × {_s1_n_classes} classes, "
          f"fallback (A1-only) classes: {int(_s1_fallback.sum())}", flush=True)
    if _s1_n_classes != N_CLASSES:
        raise RuntimeError(
            f"S1 coefs n_classes {_s1_n_classes} != notebook N_CLASSES {N_CLASSES}"
        )

    # --- 2. Resolve raw ProtoSSM test scores (before any fusion) ---
    _s1_proto_raw = None
    if "_proto_scores_before" in globals():
        _s1_proto_raw = _proto_scores_before.astype(np.float32)
        _s1_proto_src = "_proto_scores_before (cell 36b, B1 enabled)"
    elif "_proto_scores_before_fusion" in globals():
        # Cell 36b was a no-op so cell 37 captured raw proto here.
        _s1_proto_raw = _proto_scores_before_fusion.astype(np.float32)
        _s1_proto_src = "_proto_scores_before_fusion (cell 37, B1 disabled)"
    else:
        raise RuntimeError(
            "[D2-β S1] cannot locate raw ProtoSSM test scores: "
            "neither `_proto_scores_before` (cell 36b) nor "
            "`_proto_scores_before_fusion` (cell 37) is in scope."
        )
    print(f"  proto raw source: {_s1_proto_src}, shape={_s1_proto_raw.shape}", flush=True)

    # --- 3. Get B1 ranks (cell 36b) and A1 ranks (cell 37) ---
    _s1_n_rows = _s1_proto_raw.shape[0]

    def _s1_rank01_per_col(mat):
        n = mat.shape[0]
        order = np.argsort(mat, axis=0, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float32)
        rows  = np.arange(n, dtype=np.float32)
        for c in range(mat.shape[1]):
            ranks[order[:, c], c] = rows
        if n > 1:
            ranks /= (n - 1)
        return ranks

    _s1_proto_ranks = _s1_rank01_per_col(_s1_proto_raw)

    if "_b1_ranks" in globals() and _b1_ranks.shape == _s1_proto_ranks.shape:
        _s1_b1_ranks = _b1_ranks.astype(np.float32)
        _s1_b1_src = "_b1_ranks (cell 36b)"
    else:
        # B1 fusion was a no-op or shape mismatch; rank-neutral B1 column.
        _s1_b1_ranks = np.full_like(_s1_proto_ranks, 0.5, dtype=np.float32)
        _s1_b1_src = "rank-neutral 0.5 fill (B1 fusion was a no-op)"
    print(f"  b1 ranks source : {_s1_b1_src}", flush=True)

    if "_a1_ranks" not in globals():
        raise RuntimeError(
            "[D2-β S1] `_a1_ranks` not in scope. Cell 37 (A1 rank fusion) "
            "must run before this cell."
        )
    _s1_a1_ranks = _a1_ranks.astype(np.float32)
    if _s1_a1_ranks.shape != _s1_proto_ranks.shape:
        raise RuntimeError(
            f"[D2-β S1] _a1_ranks shape {_s1_a1_ranks.shape} != "
            f"proto shape {_s1_proto_ranks.shape}"
        )
    print(f"  a1 ranks source : _a1_ranks (cell 37)", flush=True)

    # --- 4. Apply per-class logistic: logits = a·a1 + b·b1 + c·proto + intercept
    _s1_logits = (
        _s1_a1_ranks    * _s1_coefs[:, 0]
      + _s1_b1_ranks    * _s1_coefs[:, 1]
      + _s1_proto_ranks * _s1_coefs[:, 2]
      + _s1_intercepts
    ).astype(np.float32)
    print(f"  s1 logits range : [{_s1_logits.min():.4f}, {_s1_logits.max():.4f}]", flush=True)

    # --- 5. Inverse-CDF back to ProtoSSM marginals (mirrors cells 36b/37
    #        so cell 38's per-class thresholds keep their meaning). For
    #        fallback classes (A1-only) the logits == _s1_a1_ranks exactly,
    #        so the inverse-CDF maps them to ProtoSSM's per-class quantiles
    #        same as the A1-only fusion path would. ---
    _s1_logit_ranks = _s1_rank01_per_col(_s1_logits)
    _s1_sorted_proto = np.sort(_s1_proto_raw, axis=0)
    _s1_idx = np.clip(
        (_s1_logit_ranks * (_s1_n_rows - 1)).round().astype(np.int64),
        0, _s1_n_rows - 1,
    )
    _s1_new_scores = np.empty_like(_s1_proto_raw)
    for _s1_c in range(N_CLASSES):
        _s1_new_scores[:, _s1_c] = _s1_sorted_proto[_s1_idx[:, _s1_c], _s1_c]

    _s1_abs_delta = float(np.abs(_s1_new_scores - final_test_scores).mean())
    print(f"  S1 vs prior final_test_scores mean |Δ| = {_s1_abs_delta:.5f}", flush=True)

    final_test_scores = _s1_new_scores

    LOGS.setdefault("d2_beta_s1_stacker", {})
    LOGS["d2_beta_s1_stacker"].update({
        "n_classes":               int(N_CLASSES),
        "n_fallback_classes":      int(_s1_fallback.sum()),
        "n_fitted_classes":        int((~_s1_fallback).sum()),
        "n_train_rows":            int(_s1_n_train),
        "proto_raw_source":        _s1_proto_src,
        "b1_ranks_source":         _s1_b1_src,
        "logit_min":               float(_s1_logits.min()),
        "logit_max":               float(_s1_logits.max()),
        "mean_abs_delta":          _s1_abs_delta,
        "score_range_after":       [float(final_test_scores.min()), float(final_test_scores.max())],
        "wall_time_seconds":       float(time.time() - _S1_START),
    })
    print(f"  D2-β S1 cell wall time: {time.time() - _S1_START:.1f}s", flush=True)
'''
