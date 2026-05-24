"""Generate AST (Track A4) fold-0 broader-pool predictions + correlate with A2.

Runs the AST fold-0 ckpt on the 1478-segment labeled soundscape val pool
(built via `train_a4_ast.build_ast_soundscape_val`, the AST-feature mirror
of `train_a1.build_soundscape_val`). Confirms broader-pool AUC against the
0.7991 value from training, then computes per-class probability correlation
(Pearson) against A2 fold-0 probs (already saved in
`data/a2_a1_5fold_broader_oof.npz`).

Output: four_track/data/a4_ast_fold0_broader_oof.npz with keys
  probs        (1478, 234)   — sigmoid probs from AST fold-0, float32
  y_true       (1478, 234)   — multi-hot labels (matches A2's y_true)
  ast_auc      scalar        — broader-pool macro AUC on present classes
  corr_per_class (n_present,) — Pearson corr per present class vs A2 fold-0
  classes_present (n_present,) — class indices used for the correlation
  mean_corr    scalar        — mean of corr_per_class
  median_corr  scalar        — median of corr_per_class

Decision rule: mean_corr < 0.7 => AST adds genuine diversity, proceed to
fusion LB sub. mean_corr >= 0.7 => AST is redundant with A2, kill the path.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

FT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FT_ROOT.parent / "src"))
sys.path.insert(0, str(FT_ROOT / "src"))

import config  # noqa: E402
from config import get_species_index  # noqa: E402
from train_a4_ast import (  # noqa: E402
    build_ast_soundscape_val,
    make_ast_model,
)

CKPT_PATH = FT_ROOT / "models" / "a4" / "a4_ast_fold0_seed42_asl.pt"
A2_OOF_PATH = FT_ROOT / "data" / "a2_a1_5fold_broader_oof.npz"
OUT_PATH = FT_ROOT / "data" / "a4_ast_fold0_broader_oof.npz"

BATCH_SIZE = 32
TRAIN_LOG_AUC = 0.7991  # best from training, for sanity match


def _macro_auc_present(probs: np.ndarray, y_true: np.ndarray) -> float:
    present = y_true.sum(axis=0) > 0
    return float(
        roc_auc_score(y_true[:, present], probs[:, present], average="macro")
    )


@torch.no_grad()
def _infer_probs(
    model: torch.nn.Module,
    val_feats: list,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    model.eval()
    out_chunks = []
    for i in range(0, len(val_feats), batch_size):
        batch = torch.stack(val_feats[i: i + batch_size]).to(device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            out = model(input_values=batch)
        out_chunks.append(torch.sigmoid(out.logits).float().cpu().numpy())
    return np.concatenate(out_chunks, axis=0)


def _pearson_per_class(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-column Pearson correlation. a, b shape (N, K). Returns (K,)."""
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    num = (a * b).sum(axis=0)
    den = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    if not CKPT_PATH.exists():
        sys.exit(f"missing ckpt: {CKPT_PATH}")
    if not A2_OOF_PATH.exists():
        sys.exit(f"missing A2 OOF: {A2_OOF_PATH}")

    # Build AST val pool.
    print("[val] building AST soundscape val (1478 windows) …", flush=True)
    t0 = time.time()
    sp2idx = get_species_index()
    val_feats, val_labels = build_ast_soundscape_val(sp2idx)
    n_segments = len(val_feats)
    n_present = int((val_labels.sum(axis=0) > 0).sum())
    print(f"  {n_segments} segments, {n_present} species present  "
          f"({time.time()-t0:.1f}s)", flush=True)

    # Load AST fold-0 ckpt.
    print(f"[ckpt] {CKPT_PATH.name}", flush=True)
    model = make_ast_model(num_classes=config.N_CLASSES).to(device).eval()
    sd = torch.load(CKPT_PATH, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)}", flush=True)
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)}", flush=True)

    # Inference.
    t1 = time.time()
    probs = _infer_probs(model, val_feats, device)
    print(f"[infer] {probs.shape}  ({time.time()-t1:.1f}s)", flush=True)
    ast_auc = _macro_auc_present(probs, val_labels)
    print(f"[AUC] AST fold-0 broader-pool: {ast_auc:.4f}  "
          f"(training log: {TRAIN_LOG_AUC:.4f})", flush=True)
    if abs(ast_auc - TRAIN_LOG_AUC) > 0.005:
        print(f"  [WARN] eval AUC differs from training log by "
              f"{abs(ast_auc - TRAIN_LOG_AUC):.4f}  — pipeline mismatch?",
              flush=True)

    del model
    torch.cuda.empty_cache()

    # Load A2 fold-0 probs + verify same y_true.
    print(f"[A2] loading {A2_OOF_PATH.name}", flush=True)
    a2 = np.load(A2_OOF_PATH)
    a2_probs_f0 = a2["probs_per_fold"][0]  # (1478, 234)
    a2_y_true = a2["y_true"]                # (1478, 234)
    assert a2_probs_f0.shape == probs.shape, \
        f"shape mismatch: A2 {a2_probs_f0.shape} vs AST {probs.shape}"
    if not np.allclose(a2_y_true, val_labels):
        n_diff = int(np.abs(a2_y_true - val_labels).sum())
        print(f"  [WARN] y_true differs from A2 by {n_diff} entries  "
              f"(should be 0)", flush=True)

    # Per-class correlation on present classes only.
    present_mask = val_labels.sum(axis=0) > 0
    classes_present = np.where(present_mask)[0].astype(np.int64)
    corr_all = _pearson_per_class(
        probs[:, present_mask], a2_probs_f0[:, present_mask]
    )
    # Drop NaN (constant-prob columns).
    valid = ~np.isnan(corr_all)
    corr_per_class = corr_all[valid].astype(np.float32)
    classes_present = classes_present[valid]
    mean_corr = float(corr_per_class.mean())
    median_corr = float(np.median(corr_per_class))
    p05 = float(np.percentile(corr_per_class, 5))
    p95 = float(np.percentile(corr_per_class, 95))

    print("", flush=True)
    print("=" * 60, flush=True)
    print("AST fold-0 ↔ A2 fold-0 correlation (broader-pool, present classes)",
          flush=True)
    print("=" * 60, flush=True)
    print(f"  classes evaluated: {len(corr_per_class)} of {n_present} present",
          flush=True)
    print(f"  mean corr:    {mean_corr:.4f}", flush=True)
    print(f"  median corr:  {median_corr:.4f}", flush=True)
    print(f"  5th pctile:   {p05:.4f}", flush=True)
    print(f"  95th pctile:  {p95:.4f}", flush=True)
    print("", flush=True)
    print(f"  AST AUC:  {ast_auc:.4f}", flush=True)
    print(f"  A2 fold-0 AUC: {float(a2['per_fold_auc'][0]):.4f}", flush=True)
    print(f"  A2 ensemble AUC: {float(a2['ensemble_auc']):.4f}", flush=True)
    print("", flush=True)
    if mean_corr < 0.7:
        verdict = (
            f"DIVERSITY-PASS  (mean_corr={mean_corr:.4f} < 0.7)  → "
            f"proceed to AST+A2 fusion LB submission"
        )
    else:
        verdict = (
            f"DIVERSITY-FAIL (mean_corr={mean_corr:.4f} >= 0.7)  → "
            f"AST is redundant with A2; kill the path"
        )
    print(f"  verdict: {verdict}", flush=True)
    print("=" * 60, flush=True)

    # Save.
    np.savez_compressed(
        OUT_PATH,
        probs=probs.astype(np.float32),
        y_true=val_labels.astype(np.float32),
        ast_auc=np.float32(ast_auc),
        corr_per_class=corr_per_class,
        classes_present=classes_present,
        mean_corr=np.float32(mean_corr),
        median_corr=np.float32(median_corr),
    )
    print(f"[save] {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
