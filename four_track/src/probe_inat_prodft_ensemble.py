"""Probe: does iNat-prodft fold 0 add ensemble diversity to the existing A1 4-fold?

Question: the iNat-init + production-recipe finetune landed at val_v2=0.7130
on fold 0 — below the production ImageNet-init baseline of 0.7414. But maybe
its errors are *uncorrelated* with the ImageNet folds, so adding it to the
soft-vote could still lift the ensemble.

Pipeline:
  1. Build soundscape val (1478 segments × 234 classes) — same path as
     train_a1.build_soundscape_val so segment ordering matches v56_oof.
  2. Load iNat-prodft fold 0 ckpt, run inference → probs_inat (1478, 234).
  3. Load v56_soundscape_oof.npz → probs_per_fold (4, 1478, 234).
  4. Report:
     - iNat-prodft standalone macro AUC (sanity vs training-log 0.7130)
     - Mean per-class probability correlation: iNat-prodft vs each of 4 folds
     - 4-fold soft-vote AUC (baseline, 0.7290)
     - 5-input soft-vote AUC (4 folds + iNat-prodft)
     - Δ AUC
  5. Save iNat-prodft predictions to data/inat_prodft_fold0_val_v2_probs.npz.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

# Repo paths
FT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FT_ROOT.parent / "src"))
sys.path.insert(0, str(FT_ROOT / "src"))

import config  # noqa: E402
from config import get_species_index  # noqa: E402
from model_a1 import BirdSEDModelA1  # noqa: E402
from train_a1 import build_soundscape_val  # noqa: E402


CKPT_PATH = (
    FT_ROOT / "models" / "a1"
    / "a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_hybrid_prodft.pt"
)
V56_OOF_PATH = FT_ROOT / "data" / "v56_soundscape_oof.npz"
OUT_PROBS_PATH = FT_ROOT / "data" / "inat_prodft_fold0_val_v2_probs.npz"

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
    """Run model on all val mels, return probs (N, 234)."""
    model.eval()
    out_chunks = []
    for i in range(0, len(val_mels), batch_size):
        batch = torch.stack(val_mels[i: i + batch_size]).to(device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            out = model(batch)
        out_chunks.append(torch.sigmoid(out["clip_logits"]).float().cpu().numpy())
    return np.concatenate(out_chunks, axis=0)


def main() -> int:
    if not CKPT_PATH.exists():
        sys.exit(f"missing ckpt: {CKPT_PATH}")
    if not V56_OOF_PATH.exists():
        sys.exit(f"missing OOF: {V56_OOF_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    # ── Build val (same path as training-time val) ───────────────────────────
    print("[val] building soundscape val …", flush=True)
    sp2idx = get_species_index()
    val_mels, val_labels = build_soundscape_val(sp2idx)
    n_present = int((val_labels.sum(axis=0) > 0).sum())
    print(f"  {len(val_mels)} segments, {n_present} species present", flush=True)

    # ── Load iNat-prodft ckpt + model ────────────────────────────────────────
    print(f"[load] {CKPT_PATH.name}", flush=True)
    model = BirdSEDModelA1(
        backbone_name=config.BACKBONE, mixstyle_p=0.0,
    ).to(device).eval()
    sd = torch.load(CKPT_PATH, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)} (first 5): {missing[:5]}",
              flush=True)
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)} (first 5): {unexpected[:5]}",
              flush=True)

    # ── Inference ────────────────────────────────────────────────────────────
    print("[infer] running iNat-prodft fold 0 on val …", flush=True)
    probs_inat = _infer_probs(model, val_mels, device)
    print(f"  probs shape: {probs_inat.shape}", flush=True)

    inat_auc = _macro_auc_present(probs_inat, val_labels)
    print(f"  iNat-prodft standalone AUC: {inat_auc:.4f}  "
          f"(training-log best was 0.7130)", flush=True)

    # ── Save predictions for future use ──────────────────────────────────────
    np.savez_compressed(
        OUT_PROBS_PATH,
        probs=probs_inat.astype(np.float32),
        labels=val_labels.astype(np.float32),
        ckpt_path=str(CKPT_PATH),
        standalone_auc=np.float32(inat_auc),
    )
    print(f"[save] {OUT_PROBS_PATH}", flush=True)

    # ── Load existing 4-fold OOF ─────────────────────────────────────────────
    d = np.load(V56_OOF_PATH, allow_pickle=True)
    probs_per_fold = d["probs_per_fold"]  # (4, N, 234)
    probs_mean_4 = d["probs_mean"]        # (N, 234) — uncalibrated soft-vote
    fold_ids = d["fold_ids"]
    F, N_oof, C = probs_per_fold.shape
    if N_oof != probs_inat.shape[0]:
        print(f"  [warn] OOF N={N_oof} but inference N={probs_inat.shape[0]} — "
              f"alignment may be off if segments don't match by index",
              flush=True)

    # Also confirm v56 OOF labels match what build_soundscape_val produced
    oof_labels = d["y_true"]
    if oof_labels.shape == val_labels.shape:
        if not np.allclose(oof_labels, val_labels):
            print("  [warn] OOF labels != val_labels — segment order differs!",
                  flush=True)
        else:
            print("  [check] OOF labels match val_labels — segments aligned ✓",
                  flush=True)

    # ── Correlation: iNat-prodft vs each existing fold ───────────────────────
    print("\n[correlation] iNat-prodft vs each ImageNet-init fold "
          "(mean Pearson r per class):", flush=True)
    flat_inat = probs_inat.reshape(-1)
    for i, f in enumerate(fold_ids):
        flat_f = probs_per_fold[i].reshape(-1)
        r = float(np.corrcoef(flat_inat, flat_f)[0, 1])
        print(f"  vs fold {f}: r = {r:.4f}", flush=True)

    # ── Ensemble: 4-fold baseline vs 5-input including iNat-prodft ───────────
    print("\n[ensemble]", flush=True)
    auc_4 = _macro_auc_present(probs_mean_4, val_labels)
    print(f"  4-fold soft-vote (baseline):  AUC = {auc_4:.4f}", flush=True)

    probs_mean_5 = (probs_per_fold.sum(axis=0) + probs_inat) / 5.0
    auc_5 = _macro_auc_present(probs_mean_5, val_labels)
    print(f"  5-input soft-vote (+iNat-prodft):  AUC = {auc_5:.4f}",
          flush=True)

    # Also try weighted: down-weight iNat-prodft since it's individually weaker
    for w_inat in (0.25, 0.5, 0.75):
        w_each_existing = (1.0 - w_inat) / 4.0
        probs_w = w_each_existing * probs_per_fold.sum(axis=0) + w_inat * probs_inat
        auc_w = _macro_auc_present(probs_w, val_labels)
        print(f"  weighted (w_iNat={w_inat:.2f}, w_each_existing={w_each_existing:.4f}): "
              f"AUC = {auc_w:.4f}", flush=True)

    delta = auc_5 - auc_4
    print(f"\n[summary]", flush=True)
    print(f"  4-fold baseline AUC:  {auc_4:.4f}", flush=True)
    print(f"  5-input AUC:          {auc_5:.4f}", flush=True)
    print(f"  Δ (5-input − 4-fold): {delta:+.4f}", flush=True)
    print(f"  iNat-prodft standalone: {inat_auc:.4f}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
