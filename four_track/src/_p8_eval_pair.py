"""One-shot diagnostic: score preserved P8 25-ep fold-0 ckpt AND fresh
5-fold fold-0 ckpt against the current val build. Used to confirm
val pipeline stability — delta between the two should match the
delta between their claimed BEST vals (0.7298 vs 0.6983) if the val
build is stable, proving the gap is training-trajectory noise.
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train_a1 import build_soundscape_val, validate
from model_a1 import BirdSEDModelA1
from config import get_species_index

CKPT_DIR = HERE.parent / "models" / "a1"
PAIRS = [
    ("P8_25ep (claimed 0.7298)",
     CKPT_DIR / "a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_hybrid_P8_25ep.pt"),
    ("P8_5fold_fold0 (claimed 0.6983)",
     CKPT_DIR / "a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_hybrid.pt"),
]

sp2idx = get_species_index()
print("Building val set …", flush=True)
val_mels, val_labels = build_soundscape_val(sp2idx)
print(f"  {len(val_mels)} segments, {(val_labels.sum(0) > 0).sum()} present", flush=True)

device = torch.device("cuda")
for name, path in PAIRS:
    print(f"\n→ {name}")
    print(f"   {path}")
    model = BirdSEDModelA1(
        backbone_name="tf_efficientnet_b0.ns_jft_in1k",
        mixstyle_p=0.0,
    ).to(device)
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval()
    auc = validate(model, val_mels, val_labels, device)
    print(f"   val_roc_auc = {auc:.4f}")
