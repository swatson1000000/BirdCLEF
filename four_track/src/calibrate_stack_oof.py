"""Per-class isotonic calibration on the d2_beta stack OOF.

Leave-one-fold-out: for each component (A1, B1, ProtoSSM) and each class,
fit IsotonicRegression on (n-1) folds of the OOF, apply to the held-out fold.
Concatenate to get clean calibrated OOF for each component, then re-rank-fuse
with the production weights and measure broader-pool macro AUC vs the
uncalibrated `prod_fused` baseline (0.6699).

Gate: +0.005 macro AUC delta to recommend a Kaggle slot for the calibrated
stack.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

ART = Path('/home/swatson/work/kaggle/BirdCLEF/four_track/data')
OOF = ART / 'd2_beta_oofs.npz'


def macro_auc(y, p):
    n_pos = y.sum(axis=0).astype(int)
    n_total = y.shape[0]
    active = np.where((n_pos > 0) & (n_pos < n_total))[0]
    aucs = [roc_auc_score(y[:, c], p[:, c]) for c in active]
    return float(np.mean(aucs)), len(active), aucs


def lofo_isotonic(scores: np.ndarray, y: np.ndarray, fold_ids: np.ndarray) -> np.ndarray:
    """Per-class leave-one-fold-out isotonic calibration.

    For each class c and fold k, fit IsotonicRegression on (scores, y) for
    fold_ids != k, predict on fold_ids == k. Skip (passthrough) classes
    where the train side has all-zero or all-one labels (isotonic undefined).
    """
    n, n_cls = scores.shape
    out = np.zeros_like(scores, dtype=np.float64)
    folds = np.unique(fold_ids)
    for c in range(n_cls):
        for k in folds:
            mask_tr = fold_ids != k
            mask_va = fold_ids == k
            y_tr = y[mask_tr, c]
            s_tr = scores[mask_tr, c]
            s_va = scores[mask_va, c]
            n_pos_tr = int(y_tr.sum())
            if n_pos_tr == 0 or n_pos_tr == len(y_tr):
                out[mask_va, c] = s_va  # passthrough
                continue
            try:
                ir = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
                ir.fit(s_tr, y_tr)
                out[mask_va, c] = ir.predict(s_va)
            except Exception:
                out[mask_va, c] = s_va
    return out.astype(np.float32)


def main():
    d = np.load(OOF, allow_pickle=True)
    y = d['y_true']
    fold_ids = d['fold_ids']
    print(f'Loaded OOF: {y.shape}  folds: {sorted(set(fold_ids.tolist()))}')

    # Baselines: uncalibrated component AUCs and prod_fused
    print('\n=== Uncalibrated baselines ===')
    components = {'a1_ranks': d['a1_ranks'], 'b1_oof': d['b1_oof'], 'proto_oof': d['proto_oof']}
    base_aucs = {}
    for name, p in components.items():
        auc, n, _ = macro_auc(y, p)
        base_aucs[name] = auc
        print(f'  {name:<12} {auc:.4f}  (active={n})')
    prod_fused_auc, _, _ = macro_auc(y, d['prod_fused'])
    print(f'  prod_fused   {prod_fused_auc:.4f}  (uncalibrated production stack)')

    # Per-class isotonic per-component
    print('\n=== Isotonic calibration (per-class, leave-one-fold-out) ===')
    cal = {}
    for name, p in components.items():
        cp = lofo_isotonic(p, y, fold_ids)
        auc, _, _ = macro_auc(y, cp)
        cal[name] = cp
        print(f'  cal_{name:<12} {auc:.4f}  (Δ vs uncal: {auc - base_aucs[name]:+.4f})')

    # Rank-fuse the calibrated components.
    # Per d2_beta_oofs metadata: a1_weight_prod=0.2, b1_weight_prod=0.1.
    # The remaining weight is on ProtoSSM (1.0 - 0.2 - 0.1 = 0.7).
    a1_w, b1_w, proto_w = 0.2, 0.1, 0.7

    def weighted_rank_fuse(arrs, weights):
        ranks = [rankdata(a, axis=0) for a in arrs]
        return sum(w * r for w, r in zip(weights, ranks)) / sum(weights)

    # Calibrated, prod weights
    cal_fused_prodw = weighted_rank_fuse(
        [cal['a1_ranks'], cal['b1_oof'], cal['proto_oof']],
        [a1_w, b1_w, proto_w],
    )
    auc_cal_prod, _, _ = macro_auc(y, cal_fused_prodw)
    print(f'\n  cal_fused (prod weights {a1_w},{b1_w},{proto_w}):  {auc_cal_prod:.4f}  '
          f'(Δ vs prod_fused: {auc_cal_prod - prod_fused_auc:+.4f})')

    # Equal weights, calibrated
    cal_fused_eq = weighted_rank_fuse(
        [cal['a1_ranks'], cal['b1_oof'], cal['proto_oof']],
        [1, 1, 1],
    )
    auc_cal_eq, _, _ = macro_auc(y, cal_fused_eq)
    print(f'  cal_fused (equal weights):                 {auc_cal_eq:.4f}  '
          f'(Δ vs prod_fused: {auc_cal_eq - prod_fused_auc:+.4f})')

    # Drop B1 entirely, calibrated A1 + Proto
    cal_fused_no_b1 = weighted_rank_fuse(
        [cal['a1_ranks'], cal['proto_oof']],
        [1, 1],
    )
    auc_cal_no_b1, _, _ = macro_auc(y, cal_fused_no_b1)
    print(f'  cal_fused (A1+Proto, drop B1):              {auc_cal_no_b1:.4f}  '
          f'(Δ vs prod_fused: {auc_cal_no_b1 - prod_fused_auc:+.4f})')

    # Best single calibrated component
    print()
    print('--- Best single calibrated component ---')
    for name in components:
        auc, _, _ = macro_auc(y, cal[name])
        print(f'  cal_{name}:  {auc:.4f}')

    # Sweep blend weights between cal_a1 and cal_proto (drop B1)
    print('\n--- Blend sweep (cal_a1, cal_proto), B1 dropped ---')
    best_w, best_auc = None, 0
    for w in np.linspace(0, 1, 11):
        rf = weighted_rank_fuse(
            [cal['a1_ranks'], cal['proto_oof']],
            [w, 1 - w],
        )
        auc, _, _ = macro_auc(y, rf)
        marker = ' ★' if auc > best_auc else ''
        if auc > best_auc:
            best_auc, best_w = auc, w
        print(f'  w_a1={w:.1f}, w_proto={1-w:.1f}:  {auc:.4f}{marker}')

    print(f'\n=== BEST: w_a1={best_w:.2f}, w_proto={1-best_w:.2f} → {best_auc:.4f}  '
          f'(Δ vs prod_fused: {best_auc - prod_fused_auc:+.4f}) ===')

    # Save calibrators output for posterity
    out = ART / 'd2_beta_isotonic_oof.npz'
    np.savez_compressed(
        out,
        cal_a1=cal['a1_ranks'],
        cal_b1=cal['b1_oof'],
        cal_proto=cal['proto_oof'],
        cal_fused_best=weighted_rank_fuse(
            [cal['a1_ranks'], cal['proto_oof']], [best_w, 1 - best_w]
        ),
        y_true=y,
        fold_ids=fold_ids,
        prod_fused_auc=prod_fused_auc,
        best_auc=best_auc,
        best_w_a1=best_w,
    )
    print(f'\nSaved → {out}')


if __name__ == '__main__':
    main()
