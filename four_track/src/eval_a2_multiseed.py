"""§34 Tier 2 — multi-seed (42 + 43 + 44) A2 bagged broader-pool OOF.

Reuses cached seed-42 probs from `data/a2_a1_5fold_broader_oof.npz` and
infers the 10 new ckpts (seeds 43, 44 × folds 0..4). Computes:

- Per-(seed, fold) AUC: 15 values
- Per-seed 5-fold sig-mean: 3 AUCs (seed 42 should reproduce 0.8402)
- 15-ckpt sig-mean (canonical bagged ensemble)
- 15-ckpt rank-mean (within-arch rank-mean has historically added ~+0.001;
  §29.4 / v76 finding — verify it doesn't suddenly help here)
- Gate verdict vs 0.8902 (A2 anchor + 0.05 per
  feedback_min_oof_delta_to_burn_slot)
"""

from __future__ import annotations

import json
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
CKPT_NAME_FMT = "a1_tf_efficientnet_b0.ns_jft_in1k_fold{f}_seed{s}_asl.pt"
S42_CACHE = FT_ROOT / "data" / "a2_a1_5fold_broader_oof.npz"
OUT_NPZ = FT_ROOT / "data" / "a2_multiseed_broader_oof.npz"
OUT_JSON = FT_ROOT / "data" / "a2_multiseed_results.json"

SEEDS = (42, 43, 44)
FOLDS = (0, 1, 2, 3, 4)

BATCH_SIZE = 32

A2_ANCHOR_AUC = 0.8402
GATE_DELTA = 0.05
GATE_AUC = A2_ANCHOR_AUC + GATE_DELTA  # 0.8902


def macro_auc_present(probs: np.ndarray, y_true: np.ndarray) -> float:
    present = y_true.sum(axis=0) > 0
    return float(
        roc_auc_score(y_true[:, present], probs[:, present], average="macro")
    )


def rank01_per_col(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    order = np.argsort(mat, axis=0, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    rows = np.arange(n, dtype=np.float32)
    for c in range(mat.shape[1]):
        ranks[order[:, c], c] = rows
    if n > 1:
        ranks /= (n - 1)
    return ranks


@torch.no_grad()
def infer_probs(model, val_mels, device, batch_size=BATCH_SIZE):
    model.eval()
    out = []
    for i in range(0, len(val_mels), batch_size):
        batch = torch.stack(val_mels[i: i + batch_size]).to(device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            o = model(batch)
        out.append(torch.sigmoid(o["clip_logits"]).float().cpu().numpy())
    return np.concatenate(out, axis=0)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    # Verify all 15 ckpts present.
    missing = []
    for s in SEEDS:
        for f in FOLDS:
            p = CKPT_DIR / CKPT_NAME_FMT.format(f=f, s=s)
            if not p.exists():
                missing.append(str(p))
    if missing:
        for p in missing:
            print(f"  [missing] {p}", flush=True)
        sys.exit(f"missing {len(missing)} ckpt(s)")

    # Load cached seed-42 probs to skip re-inference.
    if not S42_CACHE.exists():
        sys.exit(f"missing seed-42 cache {S42_CACHE}")
    cache = np.load(S42_CACHE, allow_pickle=True)
    probs_s42 = cache["probs_per_fold"]      # (5, 1478, 234) float32
    y_true = cache["y_true"]                  # (1478, 234) float32
    filenames = cache["filenames"]
    starts = cache["start_sec"]
    cached_per_fold_auc = cache["per_fold_auc"]
    print(f"[cache] seed-42 reused: probs shape {probs_s42.shape}", flush=True)
    print(f"  seed-42 per-fold AUC: {[round(float(a), 4) for a in cached_per_fold_auc]}",
          flush=True)
    print(f"  seed-42 ensemble:     {float(cache['ensemble_auc']):.4f}",
          flush=True)

    # Build val pool once.
    print("[val] building soundscape val …", flush=True)
    t0 = time.time()
    sp2idx = get_species_index()
    val_mels, val_labels = build_soundscape_val(sp2idx)
    print(f"  {len(val_mels)} segments, "
          f"{int((val_labels.sum(axis=0)>0).sum())} species present  "
          f"({time.time()-t0:.1f}s)", flush=True)
    assert val_labels.shape == y_true.shape
    if not np.allclose(val_labels, y_true):
        diff = int(np.abs(val_labels - y_true).sum())
        print(f"  [WARN] val_labels differs from cached y_true by {diff} entries",
              flush=True)

    n_seeds, n_folds, n_seg, n_cls = (
        len(SEEDS), len(FOLDS), val_labels.shape[0], val_labels.shape[1]
    )
    probs_all = np.zeros((n_seeds, n_folds, n_seg, n_cls), dtype=np.float32)
    per_run_auc = np.zeros((n_seeds, n_folds), dtype=np.float32)
    probs_all[0] = probs_s42  # seed 42

    # Reproduce seed-42 AUCs from the cache (sanity).
    for fi, f in enumerate(FOLDS):
        per_run_auc[0, fi] = macro_auc_present(probs_s42[fi], val_labels)
        delta = per_run_auc[0, fi] - cached_per_fold_auc[fi]
        if abs(delta) > 1e-3:
            print(f"  [WARN] seed42/fold{f} AUC recomputed differs by {delta:+.4f}",
                  flush=True)

    # Infer seeds 43, 44.
    for si, s in enumerate(SEEDS[1:], start=1):
        for fi, f in enumerate(FOLDS):
            ckpt = CKPT_DIR / CKPT_NAME_FMT.format(f=f, s=s)
            print(f"[seed {s} fold {f}] {ckpt.name}", flush=True)
            model = BirdSEDModelA1(
                backbone_name=config.BACKBONE, mixstyle_p=0.0,
            ).to(device).eval()
            sd = torch.load(ckpt, map_location=device)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            mk, uk = model.load_state_dict(sd, strict=False)
            if mk:
                print(f"  [warn] missing keys: {len(mk)}", flush=True)
            if uk:
                print(f"  [warn] unexpected keys: {len(uk)}", flush=True)
            t1 = time.time()
            probs = infer_probs(model, val_mels, device)
            auc = macro_auc_present(probs, val_labels)
            probs_all[si, fi] = probs
            per_run_auc[si, fi] = auc
            print(f"  AUC = {auc:.4f}  ({time.time()-t1:.1f}s)", flush=True)
            del model
            torch.cuda.empty_cache()

    # Ensembles.
    per_seed_sigmean_probs = probs_all.mean(axis=1)  # (3, 1478, 234)
    per_seed_sigmean_auc = np.array(
        [macro_auc_present(per_seed_sigmean_probs[i], val_labels)
         for i in range(n_seeds)],
        dtype=np.float32,
    )

    all15_sigmean = probs_all.reshape(n_seeds * n_folds, n_seg, n_cls).mean(axis=0)
    all15_sigmean_auc = macro_auc_present(all15_sigmean, val_labels)

    # Rank-mean (across all 15 ckpts, per-column).
    ranks_all15 = np.zeros((n_seeds * n_folds, n_seg, n_cls), dtype=np.float32)
    flat = probs_all.reshape(n_seeds * n_folds, n_seg, n_cls)
    for k in range(n_seeds * n_folds):
        ranks_all15[k] = rank01_per_col(flat[k])
    all15_rankmean = ranks_all15.mean(axis=0)
    all15_rankmean_auc = macro_auc_present(all15_rankmean, val_labels)

    # Report.
    print("", flush=True)
    print("=" * 70, flush=True)
    print("§34 Tier 2 — multi-seed A2 bagging broader-pool OOF", flush=True)
    print("=" * 70, flush=True)
    print("Per-run AUC (seeds × folds):", flush=True)
    print(f"  {'':>8} {'fold0':>8} {'fold1':>8} {'fold2':>8} {'fold3':>8} {'fold4':>8} {'mean':>8}",
          flush=True)
    for si, s in enumerate(SEEDS):
        row = " ".join(f"{per_run_auc[si, fi]:>8.4f}" for fi in range(n_folds))
        print(f"  seed{s:>4}  {row}  {per_run_auc[si].mean():>8.4f}",
              flush=True)
    print("", flush=True)
    print("Per-seed 5-fold sig-mean ensembles:", flush=True)
    for si, s in enumerate(SEEDS):
        marker = "  (A2 anchor)" if s == 42 else ""
        print(f"  seed {s}: {per_seed_sigmean_auc[si]:.4f}{marker}", flush=True)
    print("", flush=True)
    print(f"  15-ckpt sig-mean: {all15_sigmean_auc:.4f}", flush=True)
    print(f"  15-ckpt rank-mean: {all15_rankmean_auc:.4f}", flush=True)
    print("", flush=True)
    print(f"  A2 anchor:    {A2_ANCHOR_AUC:.4f}", flush=True)
    print(f"  gate (anchor + {GATE_DELTA:+.2f}): {GATE_AUC:.4f}", flush=True)
    best_auc = max(all15_sigmean_auc, all15_rankmean_auc)
    best_kind = "sig-mean" if all15_sigmean_auc >= all15_rankmean_auc else "rank-mean"
    delta = best_auc - A2_ANCHOR_AUC
    print(f"  best 15-ckpt ({best_kind}): {best_auc:.4f}  Δ vs anchor: {delta:+.4f}",
          flush=True)
    if best_auc >= GATE_AUC:
        verdict = "GATE PASS — push v77 (15-ckpt sig-mean) to LB"
    elif best_auc >= 0.85:
        verdict = ("BORDERLINE (≥0.85 but <0.8902) — don't push v77 per "
                   "feedback_min_oof_delta_to_burn_slot; closeout")
    else:
        verdict = "GATE FAIL — closeout, lock v75 LB 0.933"
    print(f"  verdict: {verdict}", flush=True)
    print("=" * 70, flush=True)

    # Persist.
    np.savez_compressed(
        OUT_NPZ,
        probs_all=probs_all,             # (3, 5, 1478, 234)
        per_run_auc=per_run_auc,         # (3, 5)
        per_seed_sigmean_probs=per_seed_sigmean_probs,  # (3, 1478, 234)
        per_seed_sigmean_auc=per_seed_sigmean_auc,      # (3,)
        all15_sigmean=all15_sigmean,     # (1478, 234)
        all15_rankmean=all15_rankmean,   # (1478, 234)
        y_true=val_labels.astype(np.float32),
        filenames=filenames,
        start_sec=starts,
        seeds=np.array(SEEDS, dtype=np.int64),
        folds=np.array(FOLDS, dtype=np.int64),
    )
    print(f"[save] {OUT_NPZ}", flush=True)

    results = {
        "per_run_auc": {
            f"seed{s}_fold{f}": float(per_run_auc[si, fi])
            for si, s in enumerate(SEEDS)
            for fi, f in enumerate(FOLDS)
        },
        "per_seed_sigmean_auc": {
            f"seed{s}": float(per_seed_sigmean_auc[si])
            for si, s in enumerate(SEEDS)
        },
        "all15_sigmean_auc": float(all15_sigmean_auc),
        "all15_rankmean_auc": float(all15_rankmean_auc),
        "a2_anchor_auc": A2_ANCHOR_AUC,
        "gate_auc": GATE_AUC,
        "best_auc": float(best_auc),
        "best_kind": best_kind,
        "delta_vs_anchor": float(delta),
        "verdict": verdict,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[save] {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
