"""Phase 3 of val-v2 rebuild — rescore A1 v56 fold-0 on val_v2.

Loads the production JIT checkpoint `kaggle_datasets/a1-effb0-ckpts/a1_fold0.pt`
(LB 0.931) and reports val_roc_auc on both channels of val_v2. Establishes
the NEW gate baseline for any fold-0 retrain experiment.

Kaggle A1 ckpts are `torch.jit.load`-able script modules with an `inner.`
prefix on the state_dict keys (see feedback_kaggle_ckpt_is_jit memory).

Usage:
    python -u src/rescore_baseline_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE       = Path(__file__).resolve().parent
FT_ROOT    = HERE.parent
ROOT       = FT_ROOT.parent
PARENT_SRC = ROOT / "src"

if str(PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(PARENT_SRC))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config  # noqa: E402
from model_a1 import BirdSEDModelA1  # noqa: E402
from train_a1 import validate  # noqa: E402

V56_CKPT     = FT_ROOT / "kaggle_datasets" / "a1-effb0-ckpts" / "a1_fold0.pt"
VAL_V2_SCAPE = FT_ROOT / "data" / "processed" / "val_v2" / "val_v2_soundscape.npz"
VAL_V2_FOCAL = FT_ROOT / "data" / "processed" / "val_v2" / "val_v2_focal.npz"


def load_jit_state_dict(ckpt_path: Path) -> dict[str, torch.Tensor]:
    mod = torch.jit.load(str(ckpt_path), map_location="cpu")
    sd = mod.state_dict()
    # Strip `inner.` prefix — BirdSEDModelA1 was JIT-scripted by wrapping
    # in a dummy module with `self.inner = model`.
    stripped: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("inner."):
            stripped[k[len("inner."):]] = v
        else:
            stripped[k] = v
    return stripped


def rescore(label: str, npz_path: Path, model: torch.nn.Module,
            device: torch.device) -> float:
    if not npz_path.exists():
        print(f"[{label}] missing {npz_path} — skipping", flush=True)
        return float("nan")

    data = np.load(npz_path, allow_pickle=True)
    mels   = data["mels"]      # (N, 3, N_MELS, 512) fp16
    labels = data["labels"].astype(np.float32)
    n = len(mels)
    n_species = int((labels.sum(axis=0) > 0).sum())
    print(f"[{label}] {n} segments,  {n_species} species present,  "
          f"mels {mels.nbytes/1e9:.2f} GB",
          flush=True)

    # Cast to fp32 tensor list (validate() expects that shape).
    val_mels = [torch.from_numpy(mels[i].astype(np.float32)) for i in range(n)]
    auc = validate(model, val_mels, labels, device)
    print(f"[{label}] val_roc_auc = {auc:.4f}", flush=True)
    return auc


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    print(f"[load] building BirdSEDModelA1 ({config.BACKBONE})", flush=True)
    model = BirdSEDModelA1(backbone_name=config.BACKBONE, mixstyle_p=0.0).to(device).eval()

    print(f"[load] {V56_CKPT.name}", flush=True)
    sd = load_jit_state_dict(V56_CKPT)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)} (first 5): {missing[:5]}",
              flush=True)
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)} (first 5): {unexpected[:5]}",
              flush=True)

    print("\n" + "=" * 60, flush=True)
    print("Channel A — soundscape val (primary gate)", flush=True)
    print("=" * 60, flush=True)
    auc_a = rescore("channel-a", VAL_V2_SCAPE, model, device)

    print("\n" + "=" * 60, flush=True)
    print("Channel B — focal holdout (diagnostic)", flush=True)
    print("=" * 60, flush=True)
    auc_b = rescore("channel-b", VAL_V2_FOCAL, model, device)

    print("\n" + "=" * 60, flush=True)
    print("Summary — v56 rescored on val_v2", flush=True)
    print("=" * 60, flush=True)
    print(f"  Channel A (soundscape, NEW gate):  {auc_a:.4f}", flush=True)
    print(f"  Channel B (focal diagnostic):      {auc_b:.4f}", flush=True)
    print(f"  (Prior val_v1 gate: 0.7414)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
