"""Evaluate the A1 5-fold ensemble on the soundscape validation set.

Loads each fold checkpoint, runs inference on the same shared
`train_soundscapes_labels.csv` segments used during training, and reports:
  - per-fold macro ROC-AUC (sanity vs the training log)
  - mean of per-fold AUCs
  - 5-fold ensemble AUC (mean of sigmoids)

Also dumps predictions + labels to `four_track/data/a1_soundscape_preds.npz`
so downstream stacking with ProtoSSM can be scripted without redoing inference.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE    = Path(__file__).resolve().parent          # four_track/src/
FT_ROOT = HERE.parent                              # four_track/
ROOT    = FT_ROOT.parent                           # BirdCLEF/

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config
from config import get_species_index

from train_a1 import build_soundscape_val
from model_a1 import BirdSEDModelA1


CHECKPOINTS = [
    FT_ROOT / "models" / "a1" / f"a1_{config.BACKBONE}_fold{f}_seed42_hybrid.pt"
    for f in range(5)
]

OUT_DIR = FT_ROOT / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "a1_soundscape_preds.npz"


@torch.no_grad()
def predict(model, val_mels, device, batch_size=32):
    model.eval()
    chunks = []
    for i in range(0, len(val_mels), batch_size):
        batch = torch.stack(val_mels[i: i + batch_size]).to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(batch)
        chunks.append(torch.sigmoid(out["clip_logits"]).float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def macro_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    present = y_true.sum(axis=0) > 0
    return float(
        roc_auc_score(y_true[:, present], y_prob[:, present], average="macro")
    )


def main() -> None:
    device = torch.device("cuda")
    sp2idx = get_species_index()

    print("Building soundscape validation set …", flush=True)
    val_mels, val_labels = build_soundscape_val(sp2idx)
    n_present = int((val_labels.sum(axis=0) > 0).sum())
    print(f"  {len(val_mels)} segments, {n_present}/{val_labels.shape[1]} species present",
          flush=True)

    fold_probs = []
    fold_aucs  = []
    for f, ckpt in enumerate(CHECKPOINTS):
        if not ckpt.exists():
            sys.exit(f"missing checkpoint: {ckpt}")
        model = BirdSEDModelA1(backbone_name=config.BACKBONE, mixstyle_p=0.0).to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        probs = predict(model, val_mels, device)
        auc   = macro_auc(val_labels, probs)
        print(f"  fold {f}: val_roc_auc = {auc:.4f}", flush=True)
        fold_probs.append(probs)
        fold_aucs.append(auc)
        del model
        torch.cuda.empty_cache()

    fold_probs = np.stack(fold_probs, axis=0)        # (5, N, C)
    ens_probs  = fold_probs.mean(axis=0)             # mean of sigmoids
    ens_auc    = macro_auc(val_labels, ens_probs)

    print()
    print(f"  mean per-fold AUC : {np.mean(fold_aucs):.4f}")
    print(f"  ensemble  AUC     : {ens_auc:.4f}")
    print(f"  bagging lift      : {ens_auc - float(np.mean(fold_aucs)):+.4f}")

    np.savez(
        OUT_PATH,
        fold_probs=fold_probs,
        ens_probs=ens_probs,
        y_true=val_labels,
    )
    print(f"\n  saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
