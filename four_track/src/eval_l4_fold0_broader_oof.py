"""L4 / L4-v2 fold-0 single-ckpt broader-pool eval.

Clone of eval_l3_precursor_broader_oof.py, but runs only fold 0 (not the full
5-fold sigmean ensemble) and accepts a --ckpt-suffix arg so the same script
works for L4-v1 (_l4) and L4-v2 (_l4v2).

Why fold-0 only: the L4 smoke gate is evaluated on a single fold-0 ckpt
before committing to full 5-fold dispatch (per plan §36.1 / §36.2 gate matrix).
Per `feedback_per_fold_val_misleads_ensemble`, single-fold AUC is noisy but the
gate threshold (0.8596) is the per-fold mean of L3-prec, not its 5-fold
ensemble (0.8700) — so this IS the right comparator.

Output: data/l4{,v2}_fold0_broader_oof.npz with keys
  probs_fold0   (1478, 234) — fold-0 standalone probs
  y_true        (1478, 234)
  filenames     (1478,)
  start_sec     (1478,)
  fold0_auc     scalar

Usage:
    python -u src/eval_l4_fold0_broader_oof.py --ckpt-suffix _l4
    python -u src/eval_l4_fold0_broader_oof.py --ckpt-suffix _l4v2
"""
from __future__ import annotations

import argparse
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
CKPT_FMT = "a1_tf_efficientnet_b0.ns_jft_in1k_fold{f}_seed{seed}_ce{suffix}.pt"

BATCH_SIZE = 32

# L3-prec fold-0 broader-pool AUC = 0.8596 (the per-fold reference)
L3_FOLD0_AUC = 0.8596
GATE_DELTA = 0.010
GATE_AUC = L3_FOLD0_AUC + GATE_DELTA  # 0.8696


def _macro_auc_present(probs: np.ndarray, y_true: np.ndarray) -> float:
    present = y_true.sum(axis=0) > 0
    return float(
        roc_auc_score(y_true[:, present], probs[:, present], average="macro")
    )


@torch.no_grad()
def _infer_probs(model, val_mels, device, batch_size: int = BATCH_SIZE) -> np.ndarray:
    model.eval()
    chunks = []
    for i in range(0, len(val_mels), batch_size):
        batch = torch.stack(val_mels[i: i + batch_size]).to(device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            out = model(batch)
        chunks.append(torch.sigmoid(out["clip_logits"]).float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _read_filenames_starts() -> tuple:
    df = pd.read_csv(RAW / "train_soundscapes_labels.csv")

    def _parse_time(s: str) -> int:
        h, m, sec = str(s).split(":")
        return int(h) * 3600 + int(m) * 60 + int(sec)

    filenames = df["filename"].astype(str).to_numpy()
    starts = df["start"].apply(_parse_time).astype(np.int32).to_numpy()
    return filenames, starts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-suffix", required=True,
                    help="Suffix appended to ckpt name, e.g. '_l4' or '_l4v2'")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=123,
                    help="Training seed encoded in ckpt name (default 123)")
    ap.add_argument("--out", type=str, default=None,
                    help="Path to NPZ output. Default: data/l4{suffix}_seed{S}_fold{F}_broader_oof.npz")
    args = ap.parse_args()

    seed_tag = f"_seed{args.seed}" if args.seed != 123 else ""
    out_path = (FT_ROOT / "data" /
                f"l4{args.ckpt_suffix.lstrip('_')}{seed_tag}_fold{args.fold}_broader_oof.npz") \
        if args.out is None else Path(args.out)

    ckpt = CKPT_DIR / CKPT_FMT.format(f=args.fold, seed=args.seed, suffix=args.ckpt_suffix)
    if not ckpt.exists():
        print(f"[err] missing ckpt: {ckpt}", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)
    print(f"[ckpt] {ckpt}", flush=True)

    print("[val] building soundscape val …", flush=True)
    t0 = time.time()
    sp2idx = get_species_index()
    val_mels, val_labels = build_soundscape_val(sp2idx)
    filenames, starts = _read_filenames_starts()
    n_seg = len(val_mels)
    n_present = int((val_labels.sum(axis=0) > 0).sum())
    print(f"  {n_seg} segments, {n_present} species present  ({time.time()-t0:.1f}s)", flush=True)
    assert n_seg == filenames.shape[0] == starts.shape[0]

    print(f"[infer] fold {args.fold}", flush=True)
    t1 = time.time()
    model = BirdSEDModelA1(backbone_name=config.BACKBONE, mixstyle_p=0.0).to(device).eval()
    sd = torch.load(ckpt, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)}", flush=True)
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)}", flush=True)

    probs = _infer_probs(model, val_mels, device)
    auc = _macro_auc_present(probs, val_labels)
    print(f"  fold-{args.fold} AUC = {auc:.4f}  ({time.time()-t1:.1f}s)", flush=True)

    print("", flush=True)
    print("=" * 60, flush=True)
    print(f"L4{args.ckpt_suffix} fold-{args.fold} broader-pool single-fold OOF", flush=True)
    print("=" * 60, flush=True)
    print(f"  ckpt suffix: {args.ckpt_suffix}", flush=True)
    print(f"  fold-{args.fold} AUC: {auc:.4f}", flush=True)
    print(f"  L3-prec fold-0:   {L3_FOLD0_AUC:.4f}  (per-fold reference)", flush=True)
    print(f"  gate (L3-prec + {GATE_DELTA:+.3f}): {GATE_AUC:.4f}", flush=True)
    delta = auc - L3_FOLD0_AUC
    print(f"  delta vs L3-prec: {delta:+.4f}", flush=True)
    if auc < L3_FOLD0_AUC:
        verdict = "ABORT — fold-0 below L3-prec single-fold reference"
    elif auc < GATE_AUC:
        verdict = "MARGINAL — fold-0 above reference but below +0.010 gate; defer 5-fold"
    else:
        verdict = "GATE-PASS — fold-0 ≥ L3-prec + 0.010; commit to 5-fold"
    print(f"  verdict: {verdict}", flush=True)
    print("=" * 60, flush=True)

    np.savez_compressed(
        out_path,
        probs_fold0=probs.astype(np.float32),
        y_true=val_labels.astype(np.float32),
        filenames=filenames,
        start_sec=starts,
        fold0_auc=np.float32(auc),
    )
    print(f"[save] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
