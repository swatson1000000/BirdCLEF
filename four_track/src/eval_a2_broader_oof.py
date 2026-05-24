"""Generate A2 (A1-as-teacher self-trained) 5-fold broader-pool OOF.

Runs each of the 5 A2 fold ckpts on the 1478-segment labeled soundscape
val pool (built via `train_a1.build_soundscape_val`, identical to v4 OOF
window selection so the comparison is apples-to-apples). Stacks per-fold
probs, takes sig-mean across folds, and reports macro AUC on present
classes vs the v4 anchor (0.7775).

Output: four_track/data/a2_a1_5fold_broader_oof.npz with keys
  probs_per_fold (5, 1478, 234) — float32
  probs_mean     (1478, 234)   — sig-mean across folds, float32
  y_true         (1478, 234)   — multi-hot labels from train_soundscapes_labels.csv
  filenames      (1478,)       — soundscape file names
  start_sec      (1478,)       — window start in seconds
  fold_ids       (5,)
  per_fold_auc   (5,)          — standalone per-fold broader-pool AUC
  ensemble_auc   scalar        — sig-mean ensemble AUC (the gate value)
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
CKPT_NAME_FMT = "a1_tf_efficientnet_b0.ns_jft_in1k_fold{f}_seed42_asl.pt"
OUT_PATH = FT_ROOT / "data" / "a2_a1_5fold_broader_oof.npz"

BATCH_SIZE = 32
N_FOLDS = 5

V4_ANCHOR_AUC = 0.7775
GATE_DELTA = 0.05  # broader-pool OOF must beat v4 by ≥+0.05 to justify slot
GATE_AUC = V4_ANCHOR_AUC + GATE_DELTA  # 0.8275


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

    # Verify all 5 ckpts present before building val (val build is slow).
    missing_ckpts = []
    for f in range(N_FOLDS):
        p = CKPT_DIR / CKPT_NAME_FMT.format(f=f)
        if not p.exists():
            missing_ckpts.append(str(p))
    if missing_ckpts:
        for p in missing_ckpts:
            print(f"  [missing] {p}", flush=True)
        sys.exit(f"missing {len(missing_ckpts)} ckpt(s)")

    # Build the 1478-segment val pool.
    print("[val] building soundscape val …", flush=True)
    t0 = time.time()
    sp2idx = get_species_index()
    val_mels, val_labels = build_soundscape_val(sp2idx)
    filenames, starts = _read_filenames_starts()
    n_segments = len(val_mels)
    n_present = int((val_labels.sum(axis=0) > 0).sum())
    print(f"  {n_segments} segments, {n_present} species present  "
          f"({time.time()-t0:.1f}s)", flush=True)
    assert n_segments == filenames.shape[0] == starts.shape[0]

    # Run each fold ckpt.
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
        print(f"  AUC = {auc:.4f}  ({time.time()-t1:.1f}s)", flush=True)

        del model
        torch.cuda.empty_cache()

    # Ensemble (sig-mean across folds).
    probs_mean = probs_per_fold.mean(axis=0)
    ensemble_auc = _macro_auc_present(probs_mean, val_labels)

    # Report.
    print("", flush=True)
    print("=" * 60, flush=True)
    print("A2 broader-pool 5-fold OOF results", flush=True)
    print("=" * 60, flush=True)
    for f in range(N_FOLDS):
        print(f"  fold {f}: {per_fold_auc[f]:.4f}", flush=True)
    print(f"  per-fold mean:   {per_fold_auc.mean():.4f}", flush=True)
    print(f"  ensemble (sig-mean): {ensemble_auc:.4f}", flush=True)
    print("", flush=True)
    print(f"  v4 anchor:    {V4_ANCHOR_AUC:.4f}", flush=True)
    print(f"  gate (anchor + {GATE_DELTA:+.2f}): {GATE_AUC:.4f}", flush=True)
    delta = ensemble_auc - V4_ANCHOR_AUC
    print(f"  delta vs anchor: {delta:+.4f}", flush=True)
    if ensemble_auc < V4_ANCHOR_AUC:
        verdict = "REGRESSION — pseudo-labels actively hurt"
    elif ensemble_auc < GATE_AUC:
        verdict = "GATE-FAIL — improvement below LB SE noise floor (±0.005)"
    else:
        verdict = "GATE-PASS — proceed to Kaggle slot"
    print(f"  verdict: {verdict}", flush=True)
    print("=" * 60, flush=True)

    # Save.
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
