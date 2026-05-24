"""Probe: split baseline A1 fold-0 val_roc_auc by taxonomy class_name.

Answers: is the 0.7414 macro AUC dominated by birds, or does the model already
handle non-bird classes (amphibians, insects, mammals, reptiles)? If non-Aves
AUC ≈ 0.5, the 28+ non-bird species are the real bottleneck and L5b (external
herp/insect audio) is high-EV. If non-Aves AUC is high, the bird corpus is the
frontier and L5 is low-EV either way.

Run:
    python -u src/probe_class_name_auc.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

HERE       = Path(__file__).resolve().parent
FT_ROOT    = HERE.parent
ROOT       = FT_ROOT.parent
PARENT_SRC = ROOT / "src"

for p in (str(PARENT_SRC), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config                                                       # noqa: E402
from config import RAW, get_species_index                           # noqa: E402
from train_a1 import build_soundscape_val                           # noqa: E402
from model_a1 import BirdSEDModelA1                                 # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_asl.pt")
    parser.add_argument("--out", default="probe_class_name_auc.csv")
    parser.add_argument("--ckpt-path", default=None,
                        help="absolute path to ckpt; overrides --ckpt (which is a name under models/a1/)")
    parser.add_argument("--jit", action="store_true",
                        help="load as torch.jit ScriptModule and strip 'inner.' prefix from state_dict")
    args = parser.parse_args()

    ckpt = Path(args.ckpt_path) if args.ckpt_path else (FT_ROOT / "models" / "a1" / args.ckpt)
    assert ckpt.exists(), f"missing {ckpt}"

    device = torch.device("cuda")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cudnn.benchmark = True

    sp2idx = get_species_index()
    idx2sp = {v: k for k, v in sp2idx.items()}
    print(f"[probe] n_classes = {len(sp2idx)}", flush=True)

    tax = pd.read_csv(RAW / "taxonomy.csv")
    tax["primary_label"] = tax["primary_label"].astype(str)
    sp2class = dict(zip(tax["primary_label"], tax["class_name"]))

    print("[probe] Building soundscape val set …", flush=True)
    val_mels, val_labels = build_soundscape_val(sp2idx)
    print(f"  {len(val_mels)} segments, labels shape={val_labels.shape}", flush=True)

    model = BirdSEDModelA1().to(device)
    if args.jit:
        jit_mod = torch.jit.load(str(ckpt), map_location="cpu")
        sd = {k.removeprefix("inner."): v for k, v in jit_mod.state_dict().items()}
    else:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
    model.load_state_dict(sd)
    model.eval()
    print(f"[probe] loaded {ckpt.name} (jit={args.jit})", flush=True)

    bsz = 32
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(val_mels), bsz):
            batch = torch.stack(val_mels[i: i + bsz]).to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(batch)
            all_probs.append(torch.sigmoid(out["clip_logits"]).float().cpu().numpy())
    probs = np.concatenate(all_probs, axis=0)
    print(f"[probe] inference done, probs shape={probs.shape}", flush=True)

    present = val_labels.sum(axis=0) > 0
    present_idx = np.where(present)[0]
    print(f"[probe] {present.sum()} species present in val", flush=True)

    overall_auc = float(roc_auc_score(
        val_labels[:, present], probs[:, present], average="macro"))
    print(f"\n[probe] OVERALL macro AUC = {overall_auc:.4f}\n", flush=True)

    rows = []
    for c in present_idx:
        sp    = idx2sp[c]
        cname = sp2class.get(sp, "unknown")
        try:
            auc = float(roc_auc_score(val_labels[:, c], probs[:, c]))
        except ValueError:
            auc = float("nan")
        n_pos = int(val_labels[:, c].sum())
        rows.append({"primary_label": sp, "class_name": cname,
                     "n_pos": n_pos, "auc": auc})
    per_class = pd.DataFrame(rows)

    print("By class_name:")
    print(f"  {'class':10s} {'n_sp':>5s} {'n_pos_total':>12s} {'mean_auc':>9s} {'median_auc':>11s} {'<0.55':>6s}")
    for cn, sub in per_class.groupby("class_name"):
        n_below = int((sub["auc"] < 0.55).sum())
        print(f"  {cn:10s} {len(sub):5d} {int(sub['n_pos'].sum()):12d} "
              f"{sub['auc'].mean():9.4f} {sub['auc'].median():11.4f} {n_below:6d}")

    print("\nPer-species (all, sorted by AUC):")
    print(per_class.sort_values("auc").to_string(index=False))

    out_csv = FT_ROOT / "data" / args.out
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(out_csv, index=False)
    print(f"\n[probe] Saved → {out_csv}")


if __name__ == "__main__":
    main()
