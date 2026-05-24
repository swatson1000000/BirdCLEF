"""Eval an a5_xarch fold-0 ckpt on the broader-pool 1478-window soundscape val
pool. Mirror of eval_a4_broader_oof.py but for ckpts that go through the
train_a1 mel pipeline (BirdSEDModelA1 with a non-EfficientNet backbone, e.g.
ConvNeXt-Pico or MobileViT-S — relies on the model_a1.py negative-index patch
shipped in §31).

Outputs data/a5_<short_backbone>_fold0_broader_oof.npz with the same keys as
the AST version (probs, y_true, AUC, corr_per_class, classes_present, mean/
median corr) so downstream fusion code can treat both interchangeably.

Decision rule: mean_corr < 0.7 -> diversity pass.
"""

from __future__ import annotations

import argparse
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
from model_a1 import BirdSEDModelA1  # noqa: E402
from train_a1 import build_soundscape_val  # noqa: E402

A2_OOF_PATH = FT_ROOT / "data" / "a2_a1_5fold_broader_oof.npz"
BATCH_SIZE = 32


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
        out_chunks.append(
            torch.sigmoid(out["clip_logits"]).float().cpu().numpy()
        )
    return np.concatenate(out_chunks, axis=0)


def _pearson_per_class(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    num = (a * b).sum(axis=0)
    den = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _backbone_from_ckpt_name(ckpt_path: Path) -> str:
    """Parse the timm backbone string out of a2_<backbone>_foldN_seed... .pt.
    Backbone may contain dots (convnext_pico.d1_in1k) but no fold tokens.
    """
    stem = ckpt_path.stem
    if not stem.startswith("a2_"):
        raise ValueError(f"unexpected ckpt name (no a2_ prefix): {stem}")
    after = stem[len("a2_"):]
    # Backbone ends at "_foldN" — find that.
    idx = after.find("_fold")
    if idx < 0:
        raise ValueError(f"can't locate _fold token in {stem}")
    return after[:idx]


def _short_backbone(backbone: str) -> str:
    """convnext_pico.d1_in1k -> convnext_pico; mobilevit_s.cvnets_in1k -> mobilevit_s."""
    return backbone.split(".")[0]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="timm backbone name; inferred from ckpt filename if omitted",
    )
    p.add_argument(
        "--train-log-auc",
        type=float,
        default=None,
        help="Optional sanity reference (best val_roc_auc from training).",
    )
    args = p.parse_args()

    if not args.ckpt.exists():
        sys.exit(f"missing ckpt: {args.ckpt}")
    if not A2_OOF_PATH.exists():
        sys.exit(f"missing A2 OOF: {A2_OOF_PATH}")

    backbone = args.backbone or _backbone_from_ckpt_name(args.ckpt)
    short = _short_backbone(backbone)
    out_path = FT_ROOT / "data" / f"a5_{short}_fold0_broader_oof.npz"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)
    print(f"[ckpt]   {args.ckpt}", flush=True)
    print(f"[backbone] {backbone}", flush=True)
    print(f"[out]    {out_path}", flush=True)

    print("[val] building soundscape val (1478 windows) …", flush=True)
    t0 = time.time()
    sp2idx = get_species_index()
    val_mels, val_labels = build_soundscape_val(sp2idx)
    n_segments = len(val_mels)
    n_present = int((val_labels.sum(axis=0) > 0).sum())
    print(
        f"  {n_segments} segments, {n_present} species present "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )

    print("[model] building BirdSEDModelA1 …", flush=True)
    model = BirdSEDModelA1(
        backbone_name=backbone,
        mixstyle_p=0.0,
    ).to(device).eval()

    sd = torch.load(args.ckpt, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)} (first 5: "
              f"{missing[:5]})", flush=True)
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)} (first 5: "
              f"{unexpected[:5]})", flush=True)

    t1 = time.time()
    probs = _infer_probs(model, val_mels, device)
    print(f"[infer] {probs.shape}  ({time.time() - t1:.1f}s)", flush=True)
    arch_auc = _macro_auc_present(probs, val_labels)
    print(f"[AUC] {short} fold-0 broader-pool: {arch_auc:.4f}", flush=True)
    if args.train_log_auc is not None:
        if abs(arch_auc - args.train_log_auc) > 0.005:
            print(
                f"  [WARN] eval AUC differs from training log "
                f"({args.train_log_auc:.4f}) by "
                f"{abs(arch_auc - args.train_log_auc):.4f} — pipeline mismatch?",
                flush=True,
            )

    del model
    torch.cuda.empty_cache()

    print(f"[A2] loading {A2_OOF_PATH.name}", flush=True)
    a2 = np.load(A2_OOF_PATH)
    a2_probs_f0 = a2["probs_per_fold"][0]
    a2_y_true = a2["y_true"]
    assert a2_probs_f0.shape == probs.shape, (
        f"shape mismatch: A2 {a2_probs_f0.shape} vs arch {probs.shape}"
    )
    if not np.allclose(a2_y_true, val_labels):
        n_diff = int(np.abs(a2_y_true - val_labels).sum())
        print(
            f"  [WARN] y_true differs from A2 by {n_diff} entries (should be 0)",
            flush=True,
        )

    present_mask = val_labels.sum(axis=0) > 0
    classes_present = np.where(present_mask)[0].astype(np.int64)
    corr_all = _pearson_per_class(
        probs[:, present_mask], a2_probs_f0[:, present_mask]
    )
    valid = ~np.isnan(corr_all)
    corr_per_class = corr_all[valid].astype(np.float32)
    classes_present = classes_present[valid]
    mean_corr = float(corr_per_class.mean())
    median_corr = float(np.median(corr_per_class))
    p05 = float(np.percentile(corr_per_class, 5))
    p95 = float(np.percentile(corr_per_class, 95))

    print("", flush=True)
    print("=" * 60, flush=True)
    print(
        f"{short} fold-0 <-> A2 fold-0 correlation "
        f"(broader-pool, present classes)",
        flush=True,
    )
    print("=" * 60, flush=True)
    print(
        f"  classes evaluated: {len(corr_per_class)} of {n_present} present",
        flush=True,
    )
    print(f"  mean corr:    {mean_corr:.4f}", flush=True)
    print(f"  median corr:  {median_corr:.4f}", flush=True)
    print(f"  5th pctile:   {p05:.4f}", flush=True)
    print(f"  95th pctile:  {p95:.4f}", flush=True)
    print("", flush=True)
    print(f"  {short} AUC:        {arch_auc:.4f}", flush=True)
    print(f"  A2 fold-0 AUC:      {float(a2['per_fold_auc'][0]):.4f}", flush=True)
    print(f"  A2 ensemble AUC:    {float(a2['ensemble_auc']):.4f}", flush=True)
    print("", flush=True)
    if mean_corr < 0.7:
        verdict = (
            f"DIVERSITY-PASS  (mean_corr={mean_corr:.4f} < 0.7)  -> "
            f"proceed to {short}+A2 rank-mean fusion check"
        )
    else:
        verdict = (
            f"DIVERSITY-FAIL (mean_corr={mean_corr:.4f} >= 0.7)  -> "
            f"{short} is redundant with A2; kill the path"
        )
    print(f"  verdict: {verdict}", flush=True)
    print("=" * 60, flush=True)

    np.savez_compressed(
        out_path,
        probs=probs.astype(np.float32),
        y_true=val_labels.astype(np.float32),
        arch_auc=np.float32(arch_auc),
        corr_per_class=corr_per_class,
        classes_present=classes_present,
        mean_corr=np.float32(mean_corr),
        median_corr=np.float32(median_corr),
        backbone=backbone,
    )
    print(f"[save] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
