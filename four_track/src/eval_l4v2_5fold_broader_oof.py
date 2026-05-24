"""L4-v2 (CE+seed123+no-mixstyle, combined L3-prec+ProtoSSM pseudo) 5-fold broader-pool OOF.

Near-clone of eval_l3_precursor_broader_oof.py — only the ckpt name format and
the anchors change. Reports per-fold AUC + sig-mean ensemble AUC vs the two
in-family references:

  - A2 anchor:       0.8402 (5-fold sig-mean, original baseline)
  - L3-prec anchor:  0.8700 (5-fold sig-mean, stronger same-arch baseline)

Output: four_track/data/l4v2_5fold_broader_oof.npz with keys
  probs_per_fold (5, 1478, 234) — float32
  probs_mean     (1478, 234)    — sig-mean across folds, float32
  y_true         (1478, 234)
  filenames      (1478,)
  start_sec      (1478,)
  fold_ids       (5,)
  per_fold_auc   (5,)
  ensemble_auc   scalar
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
CKPT_NAME_FMT = "a1_tf_efficientnet_b0.ns_jft_in1k_fold{f}_seed123_ce_l4v2.pt"
OUT_PATH = FT_ROOT / "data" / "l4v2_5fold_broader_oof.npz"

BATCH_SIZE = 32
N_FOLDS = 5

A2_ANCHOR_AUC = 0.8402
L3_PREC_ANCHOR_AUC = 0.8700
GATE_DELTA = 0.05
GATE_AUC = A2_ANCHOR_AUC + GATE_DELTA  # 0.8902 — the +0.05 slot rule vs A2


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

    missing_ckpts = []
    for f in range(N_FOLDS):
        p = CKPT_DIR / CKPT_NAME_FMT.format(f=f)
        if not p.exists():
            missing_ckpts.append(str(p))
    if missing_ckpts:
        for p in missing_ckpts:
            print(f"  [missing] {p}", flush=True)
        sys.exit(f"missing {len(missing_ckpts)} ckpt(s)")

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

    probs_mean = probs_per_fold.mean(axis=0)
    ensemble_auc = _macro_auc_present(probs_mean, val_labels)

    print("", flush=True)
    print("=" * 60, flush=True)
    print("L4-v2 (CE+seed123+no-mixstyle, L3-prec⊕ProtoSSM pseudo) 5-fold broader-pool OOF", flush=True)
    print("=" * 60, flush=True)
    for f in range(N_FOLDS):
        print(f"  fold {f}: {per_fold_auc[f]:.4f}", flush=True)
    print(f"  per-fold mean:   {per_fold_auc.mean():.4f}", flush=True)
    print(f"  ensemble (sig-mean): {ensemble_auc:.4f}", flush=True)
    print("", flush=True)
    print(f"  A2 anchor:        {A2_ANCHOR_AUC:.4f}", flush=True)
    print(f"  L3-prec anchor:   {L3_PREC_ANCHOR_AUC:.4f}", flush=True)
    print(f"  +0.05 slot gate:  {GATE_AUC:.4f}", flush=True)
    delta_a2 = ensemble_auc - A2_ANCHOR_AUC
    delta_l3 = ensemble_auc - L3_PREC_ANCHOR_AUC
    print(f"  delta vs A2:      {delta_a2:+.4f}", flush=True)
    print(f"  delta vs L3-prec: {delta_l3:+.4f}", flush=True)
    if ensemble_auc >= GATE_AUC:
        verdict = f"GATE-PASS — clears +{GATE_DELTA:.2f} broader-pool over A2; push v77 LB probe"
    elif ensemble_auc > L3_PREC_ANCHOR_AUC:
        verdict = "ABOVE L3-prec but below +0.05 gate — borderline; check transfer ratio before pushing"
    else:
        verdict = "BELOW L3-prec anchor — main run regressed"
    print(f"  verdict: {verdict}", flush=True)
    print("=" * 60, flush=True)

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
