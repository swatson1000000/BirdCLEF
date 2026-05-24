"""§33 Phase D eval — Cerrado-pretrained B0 5-fold broader-pool + A2 fusion.

Mirror of eval_a2_broader_oof.py but for the prodft ckpts produced by Phase D
(`train_a1.py --init-from <cerrado pretrain> --ft-recipe production`).
Filename pattern is `a1_tf_efficientnet_b0.ns_jft_in1k_fold{f}_seed42_hybrid_prodft.pt`.

Reports:
  1. Per-fold broader-pool AUC + 5-fold sig-mean ensemble AUC
  2. Fusion with A2 ensemble (sig-mean AND rank-mean) — same-arch fusion
     normally only gains marginally (§29 within-arch finding: +0.0015), but
     the cross-pretrain-lineage (Cerrado vs A2's ImageNet+pseudo) might
     produce slightly more diversity than vanilla within-arch fusion.
  3. Gate check: broader-pool ≥ 0.8902 (A2 anchor 0.8402 + 0.05 per
     feedback_min_oof_delta_to_burn_slot)

This script is purely informational given Phase C gate-failed (val 0.7267
< 0.7414). Used for the §33 closeout's empirical record per user request.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

FT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FT_ROOT.parent / "src"))
sys.path.insert(0, str(FT_ROOT / "src"))

import config  # noqa: E402
from config import RAW, get_species_index  # noqa: E402
from model_a1 import BirdSEDModelA1  # noqa: E402
from train_a1 import build_soundscape_val  # noqa: E402

CKPT_DIR = FT_ROOT / "models" / "a1"
CKPT_NAME_FMT = (
    "a1_tf_efficientnet_b0.ns_jft_in1k_fold{f}_seed42_hybrid_prodft.pt"
)
A2_OOF_PATH = FT_ROOT / "data" / "a2_a1_5fold_broader_oof.npz"
OUT_PATH = FT_ROOT / "data" / "xc_cerrado_5fold_broader_oof.npz"

BATCH_SIZE = 32
N_FOLDS = 5

V4_ANCHOR_AUC = 0.7775   # the v4 5-fold (no pseudo, no cerrado pretrain)
A2_ANCHOR_AUC = 0.8402   # the v75 production stack (A2 ensemble)
GATE_AUC = A2_ANCHOR_AUC + 0.05  # 0.8902


def _macro_auc_present(probs: np.ndarray, y_true: np.ndarray) -> float:
    present = y_true.sum(axis=0) > 0
    return float(
        roc_auc_score(y_true[:, present], probs[:, present], average="macro")
    )


@torch.no_grad()
def _infer_probs(
    model: torch.nn.Module,
    val_mels: list,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    model.eval()
    out_chunks = []
    for i in range(0, len(val_mels), batch_size):
        batch = torch.stack(val_mels[i: i + batch_size]).to(device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            out = model(batch)
        out_chunks.append(torch.sigmoid(out["clip_logits"]).float().cpu().numpy())
    return np.concatenate(out_chunks, axis=0)


def rank01_per_col(mat: np.ndarray) -> np.ndarray:
    """Per-column rank normalized to [0,1] (matches probe_b_weight_sweep_v2)."""
    n = mat.shape[0]
    order = np.argsort(mat, axis=0, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    rows = np.arange(n, dtype=np.float32)
    for c in range(mat.shape[1]):
        ranks[order[:, c], c] = rows
    if n > 1:
        ranks /= (n - 1)
    return ranks


def _read_filenames_starts() -> tuple:
    df = pd.read_csv(RAW / "train_soundscapes_labels.csv")

    def _parse_time(s: str) -> int:
        h, m, sec = str(s).split(":")
        return int(h) * 3600 + int(m) * 60 + int(sec)

    filenames = df["filename"].astype(str).to_numpy()
    starts = df["start"].apply(_parse_time).astype(np.int32).to_numpy()
    return filenames, starts


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    # Verify all 5 ckpts present.
    missing = []
    for f in range(N_FOLDS):
        p = CKPT_DIR / CKPT_NAME_FMT.format(f=f)
        if not p.exists():
            missing.append(str(p))
    if missing:
        for p in missing:
            print(f"  [missing] {p}", flush=True)
        sys.exit(f"missing {len(missing)} ckpt(s)")

    print("[val] building soundscape val …", flush=True)
    t0 = time.time()
    sp2idx = get_species_index()
    val_mels, val_labels = build_soundscape_val(sp2idx)
    filenames, starts = _read_filenames_starts()
    n_segments = len(val_mels)
    n_present = int((val_labels.sum(axis=0) > 0).sum())
    print(f"  {n_segments} segments, {n_present} species present  "
          f"({time.time() - t0:.1f}s)", flush=True)

    n_classes = val_labels.shape[1]
    probs_per_fold = np.zeros((N_FOLDS, n_segments, n_classes), dtype=np.float32)
    per_fold_auc = np.zeros(N_FOLDS, dtype=np.float32)

    for f in range(N_FOLDS):
        ckpt_path = CKPT_DIR / CKPT_NAME_FMT.format(f=f)
        print(f"[fold {f}] {ckpt_path.name}", flush=True)
        model = BirdSEDModelA1(
            backbone_name=config.BACKBONE, mixstyle_p=0.0,
        ).to(device).eval()
        sd = torch.load(ckpt_path, map_location=device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  [warn] missing keys: {len(missing)}", flush=True)
        if unexpected:
            print(f"  [warn] unexpected keys: {len(unexpected)}", flush=True)

        t1 = time.time()
        probs = _infer_probs(model, val_mels, device)
        auc = _macro_auc_present(probs, val_labels)
        per_fold_auc[f] = auc
        probs_per_fold[f] = probs
        print(f"  AUC = {auc:.4f}  ({time.time() - t1:.1f}s)", flush=True)

        del model
        torch.cuda.empty_cache()

    # Standalone ensemble.
    probs_mean = probs_per_fold.mean(axis=0)
    ensemble_auc = _macro_auc_present(probs_mean, val_labels)

    print("", flush=True)
    print("=" * 60, flush=True)
    print("Cerrado-pretrained B0 5-fold (standalone)", flush=True)
    print("=" * 60, flush=True)
    for f in range(N_FOLDS):
        print(f"  fold {f}: {per_fold_auc[f]:.4f}", flush=True)
    print(f"  per-fold mean:       {per_fold_auc.mean():.4f}", flush=True)
    print(f"  ensemble (sig-mean): {ensemble_auc:.4f}", flush=True)
    print("", flush=True)
    print(f"  v4 anchor (5-fold no-pseudo):  {V4_ANCHOR_AUC:.4f}  "
          f"Δ={ensemble_auc - V4_ANCHOR_AUC:+.4f}", flush=True)
    print(f"  A2 anchor (5-fold w/ pseudo):  {A2_ANCHOR_AUC:.4f}  "
          f"Δ={ensemble_auc - A2_ANCHOR_AUC:+.4f}", flush=True)
    print(f"  slot-burn gate (A2+0.05):       {GATE_AUC:.4f}  "
          f"Δ={ensemble_auc - GATE_AUC:+.4f}", flush=True)

    # Fusion with A2.
    if A2_OOF_PATH.exists():
        print("", flush=True)
        print("=" * 60, flush=True)
        print("Fusion with A2 ensemble", flush=True)
        print("=" * 60, flush=True)
        a2 = np.load(A2_OOF_PATH, allow_pickle=True)
        a2_probs_mean = a2["probs_mean"]
        a2_auc_re = _macro_auc_present(a2_probs_mean, val_labels)
        print(f"  A2 ensemble (recomputed): {a2_auc_re:.4f}", flush=True)

        # Rank-normalize for rank-mean fusion.
        a2_rank = rank01_per_col(a2_probs_mean)
        cer_rank = rank01_per_col(probs_mean)

        weights = np.round(np.arange(0.05, 1.00, 0.05), 2)
        sig_rows, rank_rows = [], []
        for w in weights:
            sig_fused = w * probs_mean + (1.0 - w) * a2_probs_mean
            rank_fused = w * cer_rank + (1.0 - w) * a2_rank
            sig_rows.append((float(w), _macro_auc_present(sig_fused, val_labels)))
            rank_rows.append((float(w), _macro_auc_present(rank_fused, val_labels)))

        print("", flush=True)
        print(f"  {'cer_w':>6}  {'sig_mean_AUC':>13}  {'rank_mean_AUC':>13}", flush=True)
        for (w, sa), (_, ra) in zip(sig_rows, rank_rows):
            marker = "  ★ GATE PASS" if max(sa, ra) >= GATE_AUC else ""
            print(f"  {w:>6.2f}  {sa:>13.4f}  {ra:>13.4f}{marker}", flush=True)

        best_sig = max(sig_rows, key=lambda x: x[1])
        best_rank = max(rank_rows, key=lambda x: x[1])
        print("", flush=True)
        print(f"  best sig-mean fusion : w={best_sig[0]:.2f}  AUC={best_sig[1]:.4f}  "
              f"Δ vs A2 anchor={best_sig[1] - A2_ANCHOR_AUC:+.4f}", flush=True)
        print(f"  best rank-mean fusion: w={best_rank[0]:.2f}  AUC={best_rank[1]:.4f}  "
              f"Δ vs A2 anchor={best_rank[1] - A2_ANCHOR_AUC:+.4f}", flush=True)
        print(f"  gate (A2+0.05):        {GATE_AUC:.4f}", flush=True)
        best_overall_auc = max(best_sig[1], best_rank[1])
        if best_overall_auc >= GATE_AUC:
            verdict = (
                f"GATE PASS  ({best_overall_auc:.4f} >= {GATE_AUC:.4f})  → "
                f"v77 LB push justified"
            )
        else:
            verdict = (
                f"GATE FAIL  ({best_overall_auc:.4f} < {GATE_AUC:.4f}, "
                f"gap {GATE_AUC - best_overall_auc:.4f})  → "
                f"§33 fully closed; lock v75 LB 0.933"
            )
        print(f"  verdict: {verdict}", flush=True)
        print("=" * 60, flush=True)

    # Save standalone probs (and fusion analyses don't need to be saved — easy to recompute).
    np.savez_compressed(
        OUT_PATH,
        probs_per_fold=probs_per_fold,
        probs_mean=probs_mean.astype(np.float32),
        y_true=val_labels.astype(np.float32),
        filenames=filenames,
        start_sec=starts,
        fold_ids=np.arange(N_FOLDS, dtype=np.int64),
        per_fold_auc=per_fold_auc,
        ensemble_auc=np.float32(ensemble_auc),
    )
    print(f"[save] {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
