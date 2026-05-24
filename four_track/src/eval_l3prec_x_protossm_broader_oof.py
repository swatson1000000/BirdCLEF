"""L4-v2 teacher smoke gate: L3-precursor 5-fold sig-mean ⊕ ProtoSSM OOF.

Loads:
- L3-prec broader-pool OOF (1478 rows, 234 classes) — already cached at
  data/l3_precursor_broader_oof.npz.
- ProtoSSM OOF (S seeds, F=59 files, T=12 windows, C=234 classes) — produced
  by train_protossm_local.py --save-oof-path. Logits.

Maps each broader-pool val row → ProtoSSM (file_idx, window_idx). The 7 files
with partial labels (62 of 1478 rows) are not in ProtoSSM's training/OOF, so
the gate is computed on the 1416-row subset where coverage exists.

Fusion modes:
- sig-mean weighted blend: w * L3 + (1-w) * ProtoSSM   (both probs)
- rank-mean weighted blend: w * rank(L3) + (1-w) * rank(ProtoSSM)

Gate (plan §36.2 step 3):
  combined < L3-prec subset → ABORT L4-v2 (write closeout)
  combined < L3-prec subset + 0.010 → L4-v2 hypothesis weak, fold-0 only
  combined ≥ L3-prec subset + 0.010 → DISPATCH L4-v2 pseudo + fold-0 smoke

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle-arch
    python -u src/eval_l3prec_x_protossm_broader_oof.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

import argparse

FT_ROOT = Path(__file__).resolve().parents[1]
L3_OOF = FT_ROOT / "data" / "l3_precursor_broader_oof.npz"
PROTO_OOF_DEFAULT = FT_ROOT / "data" / "protossm_oof.npz"

L3_ANCHOR_FULL = 0.8700  # L3-prec 5-fold sig-mean on full 1478-row pool


def _macro_auc_present(probs: np.ndarray, y_true: np.ndarray) -> float:
    present = y_true.sum(axis=0) > 0
    return float(
        roc_auc_score(y_true[:, present], probs[:, present], average="macro")
    )


def _to_ranks(probs: np.ndarray) -> np.ndarray:
    """Per-class rank-normalize to [0, 1]."""
    n = probs.shape[0]
    out = np.empty_like(probs, dtype=np.float32)
    for c in range(probs.shape[1]):
        out[:, c] = (rankdata(probs[:, c], method="average") - 0.5) / n
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto-oof", type=str, default=str(PROTO_OOF_DEFAULT),
                    help="Path to ProtoSSM OOF npz (default: data/protossm_oof.npz)")
    args = ap.parse_args()
    proto_oof_path = Path(args.proto_oof)
    if not proto_oof_path.exists():
        print(f"[err] missing ProtoSSM OOF at {proto_oof_path}", file=sys.stderr)
        print("       run: python -u src/train_protossm_local.py --save-oof-path <path>",
              file=sys.stderr)
        return 2

    l3 = np.load(L3_OOF, allow_pickle=True)
    l3_probs = l3["probs_mean"].astype(np.float32)          # (1478, 234)
    y_true = l3["y_true"].astype(np.float32)                # (1478, 234)
    filenames = np.array([str(f) for f in l3["filenames"]])  # (1478,)
    start_sec = l3["start_sec"].astype(np.int32)             # (1478,)
    print(f"[L3] probs={l3_probs.shape}  full_AUC={float(l3['ensemble_auc']):.4f}", flush=True)

    p = np.load(proto_oof_path, allow_pickle=True)
    print(f"[Proto] loaded {proto_oof_path}", flush=True)
    oof_per_seed = p["oof_per_seed"]                         # (S, F, T, C) logits
    file_list = np.array([str(f) for f in p["file_list"]])   # (F,)
    per_seed_auc = p["per_seed_auc"]
    print(f"[Proto] oof_per_seed={oof_per_seed.shape}  per_seed_auc={per_seed_auc.tolist()}", flush=True)

    # Convert ProtoSSM logits → probs (mean of sigmoid of mean-logits across seeds)
    oof_mean_logits = oof_per_seed.mean(axis=0)                # (F, T, C)
    oof_mean_probs = 1.0 / (1.0 + np.exp(-oof_mean_logits))    # (F, T, C)
    F_, T_, C_ = oof_mean_probs.shape
    print(f"[Proto] mean probs shape={oof_mean_probs.shape}", flush=True)

    # Map each broader-pool row → ProtoSSM (file_idx, window_idx). window_idx
    # = start_sec / 5. Rows whose file isn't in ProtoSSM training are unmapped.
    file_to_idx = {f: i for i, f in enumerate(file_list)}
    n_rows = filenames.shape[0]
    proto_probs_full = np.full((n_rows, C_), np.nan, dtype=np.float32)
    covered = np.zeros(n_rows, dtype=bool)
    for i in range(n_rows):
        fname = filenames[i]
        if fname not in file_to_idx:
            continue
        widx = int(start_sec[i] // 5)
        if not (0 <= widx < T_):
            continue
        f_idx = file_to_idx[fname]
        proto_probs_full[i] = oof_mean_probs[f_idx, widx]
        covered[i] = True

    n_cov = int(covered.sum())
    n_files_cov = int(np.unique(filenames[covered]).shape[0])
    print(f"[map] covered rows = {n_cov} of {n_rows}  files={n_files_cov}", flush=True)

    # Subset to covered rows for apples-to-apples comparison
    l3_sub = l3_probs[covered]                                  # (1416-ish, 234)
    proto_sub = proto_probs_full[covered]
    y_sub = y_true[covered]
    n_present_sub = int((y_sub.sum(axis=0) > 0).sum())
    print(f"[subset] {l3_sub.shape[0]} rows, {n_present_sub} classes present", flush=True)

    # Standalone AUCs on the subset
    l3_auc = _macro_auc_present(l3_sub, y_sub)
    proto_auc = _macro_auc_present(proto_sub, y_sub)
    print("", flush=True)
    print("=" * 70, flush=True)
    print("Standalones on covered subset:", flush=True)
    print(f"  L3-prec      = {l3_auc:.4f}", flush=True)
    print(f"  ProtoSSM     = {proto_auc:.4f}", flush=True)
    print(f"  L3-prec full = {L3_ANCHOR_FULL:.4f} (1478-row baseline, for reference)", flush=True)
    print("=" * 70, flush=True)

    # Sig-mean fusion weight sweep
    print("\nSig-mean fusion (w*L3 + (1-w)*Proto):", flush=True)
    sig_results = []
    for w in [0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        fused = w * l3_sub + (1 - w) * proto_sub
        auc = _macro_auc_present(fused, y_sub)
        delta = auc - l3_auc
        sig_results.append((w, auc, delta))
        print(f"  w={w:.2f}  AUC={auc:.4f}  Δ vs L3={delta:+.4f}", flush=True)

    # Rank-mean fusion weight sweep
    print("\nRank-mean fusion (w*rank(L3) + (1-w)*rank(Proto)):", flush=True)
    l3_rank = _to_ranks(l3_sub)
    proto_rank = _to_ranks(proto_sub)
    rank_results = []
    for w in [0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        fused = w * l3_rank + (1 - w) * proto_rank
        auc = _macro_auc_present(fused, y_sub)
        delta = auc - l3_auc
        rank_results.append((w, auc, delta))
        print(f"  w={w:.2f}  AUC={auc:.4f}  Δ vs L3={delta:+.4f}", flush=True)

    # Pick best
    best_sig = max(sig_results, key=lambda r: r[1])
    best_rank = max(rank_results, key=lambda r: r[1])
    best_overall = max(best_sig, best_rank, key=lambda r: r[1])
    best_mode = "sig-mean" if best_overall is best_sig else "rank-mean"

    print("", flush=True)
    print("=" * 70, flush=True)
    print("Best fusion:", flush=True)
    print(f"  best sig-mean:   w={best_sig[0]:.2f}  AUC={best_sig[1]:.4f}  Δ={best_sig[2]:+.4f}", flush=True)
    print(f"  best rank-mean:  w={best_rank[0]:.2f}  AUC={best_rank[1]:.4f}  Δ={best_rank[2]:+.4f}", flush=True)
    print(f"  overall winner:  {best_mode}  w={best_overall[0]:.2f}  AUC={best_overall[1]:.4f}  Δ={best_overall[2]:+.4f}", flush=True)
    print("=" * 70, flush=True)

    # Gate verdict (per plan §36.2 step 3)
    print("\nL4-v2 teacher gate (plan §36.2):", flush=True)
    if best_overall[1] < l3_auc:
        verdict = "ABORT L4-v2 — fusion regresses vs L3-prec standalone"
    elif best_overall[1] < l3_auc + 0.010:
        verdict = "WEAK — L4-v2 fold-0 smoke only (information value)"
    else:
        verdict = f"DISPATCH — fusion +{best_overall[2]:.4f} over L3-prec subset"
    print(f"  L3-prec subset      = {l3_auc:.4f}", flush=True)
    print(f"  best fusion         = {best_overall[1]:.4f}", flush=True)
    print(f"  fusion delta        = {best_overall[2]:+.4f}", flush=True)
    print(f"  verdict             = {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
