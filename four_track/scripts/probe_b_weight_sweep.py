"""
Probe B — Re-optimize production stack ensemble weights (A1/B1/ProtoSSM)
on the 708-window d2_beta substrate using broader-pool per-class macro AUC
methodology (active-class subset).

WHY 708-substrate (not 1478):
    The 1478-window broader-pool substrate is files-with-labels rows. B1
    PerceiverIO and ProtoSSM v4 both consume per-file Perch tensors at a
    fixed n_windows=12 and produce (n_files, 12, n_classes) outputs. They
    cannot natively run on the variable-window 1478 substrate. Producing
    1478-aligned B1/Proto OOFs would require:
      (a) extending the local Perch cache from 59 → 66 files (Perch ONNX
          inference on 7 missing audio files), AND
      (b) reproducing ~10 notebook cells of upstream context (taxonomy,
          site mapping, OOF base/prior, MLP probes, B1 + ProtoSSM 5-fold
          training) outside the notebook.
    Estimated effort: 4-8 hours of porting + 1.5-2.5 hours of compute.
    Per the user's gate (`>1-2 hours of compute = STOP and report`),
    that's a blocker.

WHAT THIS SCRIPT DOES INSTEAD:
    - Loads d2_beta_oofs.npz (the 708-window substrate where B1+Proto OOFs
      DO exist, computed at the same time as the production weights were
      set).
    - Runs the rank-fusion sweep over A1 ∈ [0..0.50] × B1 ∈ [0..0.30]
      (Proto = 1 - A1 - B1), with rank-fusion identical to production.
    - Scores via broader-pool per-class macro AUC over active classes
      (matches §25.5 gate methodology) on this substrate.
    - Reports top-5 combos and gate verdict vs production weights.

CAVEAT (read before acting on results):
    Per §25.9, the d2_beta substrate (708 win, site-level folds) is NOT
    LB-correlated at small AUC deltas. §25.9 showed isotonic calibration
    gains of +0.05 on this substrate collapsed to +0.007 on the
    production-relevant 1478 substrate. Any weight-sweep gain measured
    here may not transfer to LB. The right validation substrate would be
    1478, but that requires the full pipeline rebuild above.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path

OUT_PATH = Path("/home/swatson/work/kaggle/BirdCLEF/four_track/data/probe_b_weight_sweep.npz")


def macro_auc_skip_empty(y, p):
    """Per-class macro AUC over rows where 0 < pos < N (active classes)."""
    aucs = []
    for c in range(y.shape[1]):
        yc = y[:, c]
        if yc.sum() == 0 or yc.sum() == len(yc):
            continue
        try:
            aucs.append(roc_auc_score(yc, p[:, c]))
        except Exception:
            pass
    return float(np.mean(aucs)), len(aucs)


def rank01_per_col(mat):
    """Per-column dense rank in [0, 1]."""
    n = mat.shape[0]
    order = np.argsort(mat, axis=0, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    rows = np.arange(n, dtype=np.float32)
    for c in range(mat.shape[1]):
        ranks[order[:, c], c] = rows
    if n > 1:
        ranks /= (n - 1)
    return ranks


def main():
    print("Loading d2_beta_oofs.npz (708 substrate) ...", flush=True)
    d = np.load(
        "/home/swatson/work/kaggle/BirdCLEF/four_track/data/d2_beta_oofs.npz",
        allow_pickle=True,
    )
    a1_ranks = d["a1_ranks"].astype(np.float32)        # already rank-fused over folds
    b1_oof = d["b1_oof"].astype(np.float32)
    proto_oof = d["proto_oof"].astype(np.float32)
    prod_fused = d["prod_fused"].astype(np.float32)
    y = d["y_true"].astype(np.float32)
    a1_w_prod = float(d["a1_weight_prod"])
    b1_w_prod = float(d["b1_weight_prod"])

    print(f"Substrate: {y.shape[0]} windows × {y.shape[1]} classes", flush=True)
    print(f"Production weights (per d2_beta): A1={a1_w_prod:.2f} B1={b1_w_prod:.2f} "
          f"Proto={1-a1_w_prod-b1_w_prod:.2f}", flush=True)

    # Convert each component to rank-space [0,1] per class so the linear
    # combo is in a unified scale (matches production cells 36b + 37 ordering).
    a1_r = a1_ranks  # already a per-fold-mean rank
    b1_r = rank01_per_col(b1_oof)
    proto_r = rank01_per_col(proto_oof)

    # Standalone sanity checks
    auc_a1, n_a = macro_auc_skip_empty(y, a1_r)
    auc_b1, n_b = macro_auc_skip_empty(y, b1_r)
    auc_proto, n_p = macro_auc_skip_empty(y, proto_r)
    auc_prod, n_pf = macro_auc_skip_empty(y, prod_fused)
    print(f"\nStandalone broader-pool macro AUC (active-class subset):", flush=True)
    print(f"  A1 (rank)        : {auc_a1:.4f}   (n_active={n_a})", flush=True)
    print(f"  B1 (rank)        : {auc_b1:.4f}   (n_active={n_b})", flush=True)
    print(f"  ProtoSSM (rank)  : {auc_proto:.4f}   (n_active={n_p})", flush=True)
    print(f"  prod_fused (raw) : {auc_prod:.4f}   (n_active={n_pf})", flush=True)

    # Production-formula baseline: linear rank-fusion at (a1_w_prod, b1_w_prod)
    proto_w_prod = 1.0 - a1_w_prod - b1_w_prod
    fused_prod = (
        a1_w_prod * a1_r + b1_w_prod * b1_r + proto_w_prod * proto_r
    )
    auc_baseline, _ = macro_auc_skip_empty(y, fused_prod)
    print(f"\nProduction-weight rank-fused AUC (linear): {auc_baseline:.4f}", flush=True)

    # ---- Sweep ----
    a1_grid = np.arange(0.0, 0.51, 0.05)
    b1_grid = np.arange(0.0, 0.31, 0.05)
    print(f"\nSweep: A1 ∈ {a1_grid.tolist()}", flush=True)
    print(f"       B1 ∈ {b1_grid.tolist()}", flush=True)
    print(f"       Proto = 1 - A1 - B1 (require ≥ 0)", flush=True)

    sweep_grid = []
    sweep_aucs = []
    for a1_w in a1_grid:
        for b1_w in b1_grid:
            proto_w = 1.0 - a1_w - b1_w
            if proto_w < -1e-9:
                continue
            fused = a1_w * a1_r + b1_w * b1_r + proto_w * proto_r
            auc, _ = macro_auc_skip_empty(y, fused)
            sweep_grid.append((float(a1_w), float(b1_w), float(proto_w)))
            sweep_aucs.append(float(auc))

    sweep_grid_arr = np.array(sweep_grid, dtype=np.float32)
    sweep_aucs_arr = np.array(sweep_aucs, dtype=np.float32)

    # Top 5
    order = np.argsort(-sweep_aucs_arr)
    print(f"\nTop 5 weight combos (sorted by broader-pool macro AUC):", flush=True)
    print(f"{'A1':>6}  {'B1':>6}  {'Proto':>6}  {'AUC':>8}  {'Δ vs baseline':>14}", flush=True)
    for i in order[:5]:
        a, b, p = sweep_grid_arr[i]
        auc = sweep_aucs_arr[i]
        print(f"{a:6.2f}  {b:6.2f}  {p:6.2f}  {auc:8.4f}  {auc - auc_baseline:+14.4f}", flush=True)

    best_idx = int(order[0])
    best_w = sweep_grid_arr[best_idx]
    best_auc = float(sweep_aucs_arr[best_idx])
    print(f"\nBest:     A1={best_w[0]:.2f} B1={best_w[1]:.2f} Proto={best_w[2]:.2f} → {best_auc:.4f}", flush=True)
    print(f"Baseline: A1={a1_w_prod:.2f} B1={b1_w_prod:.2f} Proto={1-a1_w_prod-b1_w_prod:.2f} → {auc_baseline:.4f}", flush=True)
    delta = best_auc - auc_baseline
    print(f"Delta:    {delta:+.4f}   (gate: ≥ +0.005)", flush=True)
    gate_pass = delta >= 0.005
    print(f"Gate:     {'PASS' if gate_pass else 'FAIL'}", flush=True)

    np.savez(
        OUT_PATH,
        best_weights=best_w,
        best_auc=best_auc,
        baseline_auc=auc_baseline,
        sweep_grid=sweep_grid_arr,
        sweep_aucs=sweep_aucs_arr,
        a1_w_prod=a1_w_prod,
        b1_w_prod=b1_w_prod,
        substrate_n_windows=int(y.shape[0]),
        substrate_n_classes=int(y.shape[1]),
        n_active_classes=int(n_a),
        gate_pass=int(gate_pass),
        gate_threshold=0.005,
        # Standalone components for downstream reference
        auc_a1_standalone=auc_a1,
        auc_b1_standalone=auc_b1,
        auc_proto_standalone=auc_proto,
        auc_prod_fused_raw=auc_prod,
    )
    print(f"\nSaved: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
