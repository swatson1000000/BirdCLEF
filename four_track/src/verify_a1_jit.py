"""Sanity-check the JIT-traced A1 fold checkpoints against eager mode.

For each kept fold (0,1,2,4), runs both the eager `BirdSEDModelA1` and the
traced `torch.jit.load(...)` model on the soundscape validation set, and
asserts they produce identical sigmoid outputs (within fp32 tolerance).
Then re-computes the 4-fold rank-averaged ensemble AUC and asserts it
matches the previously-saved value of 0.7431.

If this script fails, the LB integration cell would silently use a
different model than the one we just analyzed.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE    = Path(__file__).resolve().parent
FT_ROOT = HERE.parent
ROOT    = FT_ROOT.parent

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config
from config import get_species_index
from train_a1 import build_soundscape_val
from model_a1 import BirdSEDModelA1


KEEP_FOLDS = [0, 1, 2, 4]
EAGER_DIR  = FT_ROOT / "models" / "a1"
JIT_DIR    = FT_ROOT / "kaggle_datasets" / "a1-effb0-ckpts"
EXPECTED_RANK_AVG_DROP3_AUC = 0.7431

device = torch.device("cpu")  # match the Kaggle inference environment


@torch.no_grad()
def predict(model: torch.nn.Module, val_mels: list, batch_size: int = 16) -> np.ndarray:
    chunks = []
    for i in range(0, len(val_mels), batch_size):
        batch = torch.stack(val_mels[i: i + batch_size]).to(device)
        out = model(batch)
        if isinstance(out, dict):
            out = out["clip_logits"]
        chunks.append(torch.sigmoid(out).float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def macro_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    present = y_true.sum(axis=0) > 0
    return float(roc_auc_score(y_true[:, present], y_prob[:, present], average="macro"))


def rank_per_col(p: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(rankdata, 0, p)


def main() -> None:
    sp2idx = get_species_index()
    print("Building soundscape val …", flush=True)
    val_mels, val_labels = build_soundscape_val(sp2idx)
    print(f"  {len(val_mels)} segments", flush=True)

    eager_probs = []
    jit_probs   = []

    for f in KEEP_FOLDS:
        print(f"\nFold {f}:", flush=True)

        # Eager
        eager_model = BirdSEDModelA1(
            backbone_name=config.BACKBONE,
            mixstyle_p=0.0,
        ).to(device).eval()
        eager_model.mixstyle.active = False
        eager_model.load_state_dict(
            torch.load(EAGER_DIR / f"a1_{config.BACKBONE}_fold{f}_seed42_hybrid.pt",
                       map_location=device)
        )
        ep = predict(eager_model, val_mels)
        del eager_model

        # JIT
        jit_model = torch.jit.load(str(JIT_DIR / f"a1_fold{f}.pt"), map_location=device).eval()
        jp = predict(jit_model, val_mels)
        del jit_model

        max_abs = float(np.max(np.abs(ep - jp)))
        print(f"  eager AUC : {macro_auc(val_labels, ep):.4f}")
        print(f"  jit   AUC : {macro_auc(val_labels, jp):.4f}")
        print(f"  max |Δ|   : {max_abs:.2e}")
        assert max_abs < 1e-4, f"fold {f} eager vs jit mismatch: max |Δ| = {max_abs}"

        eager_probs.append(ep)
        jit_probs.append(jp)

    eager_probs = np.stack(eager_probs, 0)
    jit_probs   = np.stack(jit_probs, 0)

    # Rank-avg ensemble
    e_rank = np.stack([rank_per_col(eager_probs[i]) for i in range(len(KEEP_FOLDS))], 0).mean(0)
    j_rank = np.stack([rank_per_col(jit_probs[i])   for i in range(len(KEEP_FOLDS))], 0).mean(0)

    e_auc = macro_auc(val_labels, e_rank)
    j_auc = macro_auc(val_labels, j_rank)

    print()
    print(f"  eager 4-fold rank-avg AUC : {e_auc:.4f}")
    print(f"  jit   4-fold rank-avg AUC : {j_auc:.4f}")
    print(f"  expected (from earlier)   : {EXPECTED_RANK_AVG_DROP3_AUC:.4f}")
    assert abs(e_auc - EXPECTED_RANK_AVG_DROP3_AUC) < 5e-4, "eager regressed"
    assert abs(j_auc - EXPECTED_RANK_AVG_DROP3_AUC) < 5e-4, "jit  regressed"
    print("\n  ✓ JIT outputs match eager outputs and reproduce the 0.7431 ensemble AUC")


if __name__ == "__main__":
    main()
