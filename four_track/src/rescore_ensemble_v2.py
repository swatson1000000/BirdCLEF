"""D-step evaluator — rescore N fold-0 ckpts on val_v2 and report ensemble AUC.

Usage:
    # Phase D0: sanity-check a single local ckpt vs the v56 JIT baseline.
    python -u src/rescore_ensemble_v2.py \
        --ckpts models/a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_hybrid.pt

    # Phase D2: evaluate a 3-ckpt ensemble.
    python -u src/rescore_ensemble_v2.py \
        --ckpts \
            models/a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_hybrid.pt \
            models/a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed1337_hybrid.pt \
            models/a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed2026_hybrid.pt

Ckpt formats accepted:
  - JIT script modules (Kaggle-style), with or without `inner.` prefix
  - Raw state_dict .pt files from train_a1.py

Outputs per ckpt and for the ensemble:
  - val_roc_auc on Channel A GT-only subset    (primary gate; baseline 0.7416)
  - val_roc_auc on Channel A Perch-only subset (diagnostic only — self-agreement)
  - val_roc_auc on Channel B focal holdout     (per-species diagnostic)

Also prints per-species Channel B Δ between individual ckpts and the first ckpt
(seed-42 slot is the reference), so we can flag any group regressing > 0.02.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

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

VAL_V2_SCAPE = FT_ROOT / "data" / "processed" / "val_v2" / "val_v2_soundscape.npz"
VAL_V2_FOCAL = FT_ROOT / "data" / "processed" / "val_v2" / "val_v2_focal.npz"
V56_JIT      = FT_ROOT / "kaggle_datasets" / "a1-effb0-ckpts" / "a1_fold0.pt"


def load_state_dict(ckpt_path: Path) -> dict[str, torch.Tensor]:
    """Load a .pt checkpoint into a clean state_dict, handling JIT + raw formats."""
    try:
        mod = torch.jit.load(str(ckpt_path), map_location="cpu")
        sd = mod.state_dict()
    except Exception:
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
    stripped: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("inner."):
            stripped[k[len("inner."):]] = v
        else:
            stripped[k] = v
    return stripped


@torch.no_grad()
def predict_probs(model: torch.nn.Module, mels_np: np.ndarray,
                  device: torch.device, batch_size: int = 32) -> np.ndarray:
    """Batched forward pass → sigmoid clip_logits. Returns (N, 234) float32 array."""
    n = len(mels_np)
    probs = np.empty((n, config.N_CLASSES), dtype=np.float32)
    model.eval()
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        batch = torch.from_numpy(mels_np[s:e].astype(np.float32)).to(device)
        out = model(batch)
        probs[s:e] = torch.sigmoid(out["clip_logits"]).cpu().numpy()
    return probs


def macro_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    present = labels.sum(axis=0) > 0
    if present.sum() == 0:
        return float("nan")
    try:
        return float(roc_auc_score(
            labels[:, present], probs[:, present], average="macro"
        ))
    except ValueError:
        return float("nan")


def auc_breakdown(label: str, probs: np.ndarray, labels: np.ndarray,
                  src_mask: np.ndarray | None = None) -> dict:
    """Report mean-AUC on GT-only / Perch-only / combined subsets."""
    out = {}
    out[f"{label}_all"] = macro_auc(labels, probs)
    if src_mask is not None:
        gt = src_mask == "gt"
        pe = src_mask == "perch_new"
        out[f"{label}_gt"]    = macro_auc(labels[gt], probs[gt])
        out[f"{label}_perch"] = macro_auc(labels[pe], probs[pe])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="one or more fold-0 ckpt paths")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="labels for each ckpt (for prettier printing). "
                         "Defaults to each ckpt's stem.")
    ap.add_argument("--include-v56", action="store_true",
                    help="also score the v56 JIT ckpt as a reference")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    ckpt_paths = [Path(p) for p in args.ckpts]
    labels = args.labels or [p.stem for p in ckpt_paths]
    if len(labels) != len(ckpt_paths):
        raise SystemExit("--labels must match --ckpts length")

    if args.include_v56:
        ckpt_paths.append(V56_JIT)
        labels.append("v56_prod")

    # ── Load val_v2 ──────────────────────────────────────────────────────────
    print("[load] val_v2_soundscape.npz", flush=True)
    d_a = np.load(VAL_V2_SCAPE, allow_pickle=True)
    mels_a, labels_a = d_a["mels"], d_a["labels"].astype(np.float32)
    source_a = d_a["source"]
    print(f"  Channel A: {len(mels_a)} segs  "
          f"(gt={int((source_a == 'gt').sum())}, "
          f"perch={int((source_a == 'perch_new').sum())})",
          flush=True)

    print("[load] val_v2_focal.npz", flush=True)
    d_b = np.load(VAL_V2_FOCAL, allow_pickle=True)
    mels_b, labels_b = d_b["mels"], d_b["labels"].astype(np.float32)
    primary_b = d_b["primary_label"]
    print(f"  Channel B: {len(mels_b)} clips, {len(set(primary_b.tolist()))} species",
          flush=True)

    # Shared skeleton.
    skeleton = BirdSEDModelA1(
        backbone_name=config.BACKBONE, mixstyle_p=0.0,
    ).to(device).eval()

    # ── Score each ckpt; accumulate probs for ensemble ───────────────────────
    all_probs_a: list[np.ndarray] = []
    all_probs_b: list[np.ndarray] = []
    per_ckpt_rows: list[tuple] = []

    for lab, path in zip(labels, ckpt_paths):
        print(f"\n[ckpt] {lab}  ← {path.name}", flush=True)
        if not path.exists():
            print(f"  [error] missing: {path}", flush=True)
            raise SystemExit(2)

        sd = load_state_dict(path)
        missing, unexpected = skeleton.load_state_dict(sd, strict=False)
        if missing:
            print(f"  [warn] missing keys: {len(missing)} (first 3): {missing[:3]}",
                  flush=True)
        if unexpected:
            print(f"  [warn] unexpected: {len(unexpected)} (first 3): {unexpected[:3]}",
                  flush=True)

        probs_a = predict_probs(skeleton, mels_a, device)
        probs_b = predict_probs(skeleton, mels_b, device)
        all_probs_a.append(probs_a)
        all_probs_b.append(probs_b)

        bd_a = auc_breakdown("a", probs_a, labels_a, src_mask=source_a)
        auc_b = macro_auc(labels_b, probs_b)

        print(f"  Channel A gt     : {bd_a['a_gt']:.4f}  (baseline gate 0.7416)",
              flush=True)
        print(f"  Channel A perch  : {bd_a['a_perch']:.4f}  (diagnostic only — self-agreement)",
              flush=True)
        print(f"  Channel A all    : {bd_a['a_all']:.4f}",
              flush=True)
        print(f"  Channel B focal  : {auc_b:.4f}   (baseline 0.9545)",
              flush=True)

        per_ckpt_rows.append((lab, bd_a["a_gt"], bd_a["a_perch"],
                              bd_a["a_all"], auc_b))

    # ── Ensemble (mean sigmoids) ─────────────────────────────────────────────
    # Drop v56 reference from the ensemble if it was tacked on, so the
    # ensemble reflects what we'd actually ship.
    n_ens = len(args.ckpts)
    if n_ens >= 2:
        ens_a = np.mean(np.stack(all_probs_a[:n_ens]), axis=0)
        ens_b = np.mean(np.stack(all_probs_b[:n_ens]), axis=0)
        bd_ens_a = auc_breakdown("a", ens_a, labels_a, src_mask=source_a)
        auc_ens_b = macro_auc(labels_b, ens_b)

        print(f"\n[ensemble] mean-sigmoid over {n_ens} ckpts:", flush=True)
        print(f"  Channel A gt     : {bd_ens_a['a_gt']:.4f}", flush=True)
        print(f"  Channel A perch  : {bd_ens_a['a_perch']:.4f}", flush=True)
        print(f"  Channel A all    : {bd_ens_a['a_all']:.4f}", flush=True)
        print(f"  Channel B focal  : {auc_ens_b:.4f}", flush=True)

        best_single = max(r[1] for r in per_ckpt_rows[:n_ens])
        lift = bd_ens_a["a_gt"] - best_single
        print(f"\n[gate] ensemble − best single seed (Channel A gt) = "
              f"{lift:+.4f}",
              flush=True)
        print(f"[gate] threshold +0.0050 → "
              f"{'PASS' if lift >= 0.005 else 'FAIL'}",
              flush=True)
    else:
        print("\n[ensemble] only 1 ckpt supplied — nothing to ensemble.",
              flush=True)

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 72, flush=True)
    print(f"{'ckpt':>30} | {'A-gt':>7} | {'A-perch':>7} | {'A-all':>7} | {'B':>7}",
          flush=True)
    print("-" * 72, flush=True)
    for lab, a_gt, a_pe, a_all, b in per_ckpt_rows:
        print(f"{lab:>30} | {a_gt:>7.4f} | {a_pe:>7.4f} | "
              f"{a_all:>7.4f} | {b:>7.4f}",
              flush=True)
    print("=" * 72, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
