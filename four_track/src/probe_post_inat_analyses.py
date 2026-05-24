"""Post-iNat-probe analyses — addresses the open questions from the 2026-05-10
iNat-prodft ensemble probe (+0.0037 lift, correlations 0.54-0.61).

Three analyses, in order:

  (1) ImageNet fold-vs-fold correlation baseline.
      Are correlations of 0.54-0.61 (iNat vs ImageNet folds) actually
      "diverse," or are they at the val-set noise floor? Compare to ImageNet
      fold-vs-fold correlation (same backbone, different fold split). If those
      correlate at 0.85+, iNat IS diverse. If at 0.65, our diversity finding
      was noise.

  (2) Per-class diversity breakdown for iNat-prodft.
      The mean correlation (0.55) averages over 234 classes. Where does iNat
      agree with ImageNet, and where does it diverge? Distribution +
      top-10 most/least-divergent classes. Cross-reference taxonomy if
      possible (Aves vs non-Aves was the original lever-targeting hypothesis).

  (3) V2-S cross-arch ensemble.
      Cross-backbone diversity is generally larger than within-backbone seed
      variation. We have V2-S fold-0/1/2/4 ckpts from May 3. Run inference,
      check standalone AUC, cross-correlation with B0 ImageNet folds, and
      cross-arch ensemble lift.

Output: stdout log (caller redirects to log/) + summary saved as
  data/post_inat_analyses_results.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

FT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FT_ROOT.parent / "src"))
sys.path.insert(0, str(FT_ROOT / "src"))

import config  # noqa: E402
from config import get_species_index  # noqa: E402
from model_a1 import BirdSEDModelA1  # noqa: E402
from train_a1 import build_soundscape_val  # noqa: E402

V56_OOF_PATH = FT_ROOT / "data" / "v56_soundscape_oof.npz"
INAT_PROBS_PATH = FT_ROOT / "data" / "inat_prodft_fold0_val_v2_probs.npz"
V2S_CKPT_GLOB = "a1_tf_efficientnetv2_s.in21k_ft_in1k_fold*_seed42_hybrid.pt"
V2S_CKPT_DIR = FT_ROOT / "models" / "a1"
RESULTS_PATH = FT_ROOT / "data" / "post_inat_analyses_results.npz"

BATCH_SIZE = 32


def _macro_auc_present(probs: np.ndarray, y_true: np.ndarray) -> float:
    present = y_true.sum(axis=0) > 0
    return float(
        roc_auc_score(y_true[:, present], probs[:, present], average="macro")
    )


@torch.no_grad()
def _infer_probs(model, val_mels, device, batch_size=BATCH_SIZE):
    model.eval()
    out_chunks = []
    for i in range(0, len(val_mels), batch_size):
        batch = torch.stack(val_mels[i: i + batch_size]).to(device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            out = model(batch)
        out_chunks.append(torch.sigmoid(out["clip_logits"]).float().cpu().numpy())
    return np.concatenate(out_chunks, axis=0)


def _flat_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])


def _per_class_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r per class. Returns shape (C,)."""
    # a, b: (N, C). For each class c: corr over N samples.
    a_centered = a - a.mean(axis=0, keepdims=True)
    b_centered = b - b.mean(axis=0, keepdims=True)
    num = (a_centered * b_centered).sum(axis=0)
    den = np.sqrt((a_centered**2).sum(axis=0) * (b_centered**2).sum(axis=0))
    den = np.where(den < 1e-12, 1e-12, den)
    return num / den


# ── (1) ImageNet fold-vs-fold correlation baseline ──────────────────────────
def analysis_1_imagenet_fold_correlation(oof: dict) -> dict:
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 1 — ImageNet fold-vs-fold correlation baseline", flush=True)
    print("=" * 70, flush=True)
    print("Q: Is iNat-prodft's 0.54-0.61 correlation 'diverse', or noise?\n",
          flush=True)

    probs = oof["probs_per_fold"]   # (4, N, C)
    fold_ids = oof["fold_ids"]
    F = probs.shape[0]

    pair_corrs = []
    print("Pairwise flattened Pearson r between ImageNet folds:", flush=True)
    for i in range(F):
        for j in range(i + 1, F):
            r = _flat_corr(probs[i], probs[j])
            pair_corrs.append(r)
            print(f"  fold {fold_ids[i]} vs fold {fold_ids[j]}: r = {r:.4f}",
                  flush=True)

    mean_imagenet_fold_corr = float(np.mean(pair_corrs))
    print(f"\n  mean ImageNet fold-vs-fold r: {mean_imagenet_fold_corr:.4f}",
          flush=True)
    print(f"  (compare to iNat-prodft vs ImageNet folds: 0.54-0.61)",
          flush=True)

    if mean_imagenet_fold_corr > 0.80:
        verdict = ("iNat-prodft IS DIVERSE — ImageNet folds correlate strongly "
                   "with each other but iNat-prodft is genuinely different")
    elif mean_imagenet_fold_corr > 0.65:
        verdict = ("iNat-prodft is MODERATELY DIVERSE — ImageNet folds are "
                   "already somewhat decorrelated, so iNat's gap is partly noise")
    else:
        verdict = ("iNat-prodft DIVERSITY IS NOISE — even ImageNet folds with "
                   "same backbone correlate at val-set noise floor")
    print(f"\n  verdict: {verdict}", flush=True)

    return {
        "imagenet_pair_corrs": np.array(pair_corrs, dtype=np.float32),
        "imagenet_pair_corr_mean": np.float32(mean_imagenet_fold_corr),
    }


# ── (2) Per-class diversity breakdown for iNat-prodft ───────────────────────
def analysis_2_per_class_diversity(oof: dict, inat_data: dict,
                                    sp2idx: dict) -> dict:
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 2 — Per-class diversity breakdown (iNat-prodft vs ImageNet "
          "soft-vote)", flush=True)
    print("=" * 70, flush=True)
    print("Q: Where does iNat agree with ImageNet, and where does it diverge?\n",
          flush=True)

    inat_probs = inat_data["probs"]
    imagenet_softvote = oof["probs_mean"]   # (N, C)
    y_true = oof["y_true"]

    per_class_r = _per_class_corr(inat_probs, imagenet_softvote)
    present = y_true.sum(axis=0) > 0
    present_r = per_class_r[present]
    n_present = int(present.sum())

    print(f"Per-class Pearson r distribution ({n_present} present classes):",
          flush=True)
    print(f"  min      : {present_r.min():.4f}", flush=True)
    print(f"  25 pctl  : {np.percentile(present_r, 25):.4f}", flush=True)
    print(f"  median   : {np.percentile(present_r, 50):.4f}", flush=True)
    print(f"  75 pctl  : {np.percentile(present_r, 75):.4f}", flush=True)
    print(f"  max      : {present_r.max():.4f}", flush=True)
    print(f"  mean     : {present_r.mean():.4f}", flush=True)
    print(f"  n_high (r > 0.80) : {int((present_r > 0.80).sum())}", flush=True)
    print(f"  n_med  (0.40-0.80): {int(((present_r > 0.40) & (present_r <= 0.80)).sum())}",
          flush=True)
    print(f"  n_low  (r < 0.40) : {int((present_r < 0.40).sum())}", flush=True)

    # Identify species names for present classes
    idx2sp = {v: k for k, v in sp2idx.items()}
    present_idx = np.where(present)[0]
    present_species = [idx2sp.get(i, f"<idx_{i}>") for i in present_idx]

    # Top 10 least-correlated (iNat agrees least with ImageNet)
    order_by_corr = np.argsort(present_r)
    print(f"\nTop 10 least-correlated species (iNat most different):", flush=True)
    for k in range(min(10, len(order_by_corr))):
        idx_in_present = order_by_corr[k]
        sp = present_species[idx_in_present]
        r = present_r[idx_in_present]
        print(f"  {r:+.4f}  {sp}", flush=True)

    print(f"\nTop 10 most-correlated species (iNat agrees with ImageNet):",
          flush=True)
    for k in range(min(10, len(order_by_corr))):
        idx_in_present = order_by_corr[-(k + 1)]
        sp = present_species[idx_in_present]
        r = present_r[idx_in_present]
        print(f"  {r:+.4f}  {sp}", flush=True)

    return {
        "per_class_corr": per_class_r.astype(np.float32),
        "per_class_corr_present": present_r.astype(np.float32),
        "n_present": n_present,
    }


# ── (3) V2-S cross-arch ensemble ────────────────────────────────────────────
def analysis_3_v2s_cross_arch(oof: dict, inat_data: dict,
                               val_mels, val_labels, device) -> dict:
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 3 — V2-S cross-arch ensemble check", flush=True)
    print("=" * 70, flush=True)
    print("Q: Does cross-backbone (B0 + V2-S) ensembling lift more than "
          "within-backbone?\n", flush=True)

    v2s_ckpts = sorted(V2S_CKPT_DIR.glob(V2S_CKPT_GLOB))
    print(f"Found {len(v2s_ckpts)} V2-S ckpts:", flush=True)
    for p in v2s_ckpts:
        print(f"  {p.name}", flush=True)

    v2s_backbone = "tf_efficientnetv2_s.in21k_ft_in1k"
    print(f"\nBuilding V2-S model ({v2s_backbone}) …", flush=True)

    v2s_probs_per_fold = []
    v2s_fold_ids = []
    for ckpt_path in v2s_ckpts:
        # Parse fold from filename: a1_..._foldN_...
        name = ckpt_path.name
        fold_idx = int(name.split("_fold")[1].split("_")[0])
        v2s_fold_ids.append(fold_idx)

        model = BirdSEDModelA1(
            backbone_name=v2s_backbone, mixstyle_p=0.0,
        ).to(device).eval()
        sd = torch.load(ckpt_path, map_location=device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  [warn] {ckpt_path.name}: missing {len(missing)} keys",
                  flush=True)

        print(f"  inferring fold {fold_idx} … ", end="", flush=True)
        probs = _infer_probs(model, val_mels, device)
        auc = _macro_auc_present(probs, val_labels)
        print(f"AUC = {auc:.4f}", flush=True)
        v2s_probs_per_fold.append(probs)

        # Free GPU memory between folds
        del model, sd
        torch.cuda.empty_cache()

    v2s_probs_per_fold = np.stack(v2s_probs_per_fold, axis=0)  # (Fv, N, C)

    # Cross-correlation B0 vs V2-S, fold-pair
    print(f"\nCross-arch correlations (B0 vs V2-S, fold-pair):", flush=True)
    b0_probs = oof["probs_per_fold"]
    b0_fold_ids = oof["fold_ids"]
    for i, b0_f in enumerate(b0_fold_ids):
        for j, v2_f in enumerate(v2s_fold_ids):
            r = _flat_corr(b0_probs[i], v2s_probs_per_fold[j])
            print(f"  B0 fold {b0_f} vs V2-S fold {v2_f}: r = {r:.4f}",
                  flush=True)

    # Per-fold V2-S vs corresponding B0 fold (same training partition)
    print(f"\nWithin-fold cross-arch r (B0 fold N vs V2-S fold N):",
          flush=True)
    same_fold_corrs = []
    for j, v2_f in enumerate(v2s_fold_ids):
        if v2_f in list(b0_fold_ids):
            i = list(b0_fold_ids).index(v2_f)
            r = _flat_corr(b0_probs[i], v2s_probs_per_fold[j])
            same_fold_corrs.append(r)
            print(f"  fold {v2_f}: r = {r:.4f}", flush=True)

    # Ensembles
    print(f"\nEnsemble comparisons:", flush=True)
    b0_softvote = b0_probs.mean(axis=0)
    auc_b0 = _macro_auc_present(b0_softvote, val_labels)
    print(f"  B0 4-fold:                       AUC = {auc_b0:.4f}",
          flush=True)

    v2s_softvote = v2s_probs_per_fold.mean(axis=0)
    auc_v2s = _macro_auc_present(v2s_softvote, val_labels)
    print(f"  V2-S 4-fold:                     AUC = {auc_v2s:.4f}",
          flush=True)

    combined_8 = np.concatenate([b0_probs, v2s_probs_per_fold], axis=0).mean(axis=0)
    auc_combined_8 = _macro_auc_present(combined_8, val_labels)
    delta_8 = auc_combined_8 - auc_b0
    print(f"  8-input (B0 4 + V2-S 4):         AUC = {auc_combined_8:.4f}   "
          f"(Δ vs B0: {delta_8:+.4f})", flush=True)

    combined_9 = np.concatenate(
        [b0_probs, v2s_probs_per_fold, inat_data["probs"][None]], axis=0,
    ).mean(axis=0)
    auc_combined_9 = _macro_auc_present(combined_9, val_labels)
    delta_9 = auc_combined_9 - auc_b0
    print(f"  9-input (B0 4 + V2-S 4 + iNat):  AUC = {auc_combined_9:.4f}   "
          f"(Δ vs B0: {delta_9:+.4f})", flush=True)

    # Weighted: V2-S underweighted if individually weaker
    print(f"\nWeighted (split equally within arch, sweep arch weights):",
          flush=True)
    for w_v2s in (0.25, 0.4, 0.5, 0.6, 0.75):
        w_b0 = 1.0 - w_v2s
        probs_w = w_b0 * b0_softvote + w_v2s * v2s_softvote
        auc_w = _macro_auc_present(probs_w, val_labels)
        print(f"  w_v2s={w_v2s:.2f}, w_b0={w_b0:.2f}: AUC = {auc_w:.4f}",
              flush=True)

    return {
        "v2s_probs_per_fold": v2s_probs_per_fold.astype(np.float32),
        "v2s_fold_ids": np.array(v2s_fold_ids, dtype=np.int64),
        "auc_b0_4fold": np.float32(auc_b0),
        "auc_v2s_4fold": np.float32(auc_v2s),
        "auc_combined_8": np.float32(auc_combined_8),
        "auc_combined_9_with_inat": np.float32(auc_combined_9),
        "within_fold_b0_v2s_corrs": np.array(same_fold_corrs, dtype=np.float32),
    }


# ── Main orchestrator ───────────────────────────────────────────────────────
def main() -> int:
    if not V56_OOF_PATH.exists():
        sys.exit(f"missing OOF: {V56_OOF_PATH}")
    if not INAT_PROBS_PATH.exists():
        sys.exit(f"missing iNat probs (run probe_inat_prodft_ensemble.py first): "
                 f"{INAT_PROBS_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    print("[load] OOF + iNat-prodft predictions", flush=True)
    oof = dict(np.load(V56_OOF_PATH, allow_pickle=True))
    inat_data = dict(np.load(INAT_PROBS_PATH, allow_pickle=True))

    out = {}

    # ── (1) ──────────────────────────────────────────────────────────────
    out.update(analysis_1_imagenet_fold_correlation(oof))

    # ── (2) ──────────────────────────────────────────────────────────────
    print("\n[species index]", flush=True)
    sp2idx = get_species_index()
    out.update(analysis_2_per_class_diversity(oof, inat_data, sp2idx))

    # ── (3) ──────────────────────────────────────────────────────────────
    print("\n[val] building soundscape val for V2-S inference …", flush=True)
    val_mels, val_labels = build_soundscape_val(sp2idx)
    n_present = int((val_labels.sum(axis=0) > 0).sum())
    print(f"  {len(val_mels)} segments, {n_present} species present",
          flush=True)
    out.update(analysis_3_v2s_cross_arch(oof, inat_data, val_mels, val_labels,
                                         device))

    # ── Save summary ─────────────────────────────────────────────────────
    np.savez_compressed(RESULTS_PATH, **out)
    print(f"\n[save] {RESULTS_PATH}", flush=True)
    print("\n[done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
