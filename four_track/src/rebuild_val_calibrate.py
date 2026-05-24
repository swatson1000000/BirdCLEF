"""Phase 1 of val-v2 rebuild — calibrate the Perch score threshold τ.

Uses the 59-overlap Perch cache (708 segments, 234 classes) and the
segment-level ground truth in train_soundscapes_labels.csv to answer:

  "At what Perch score threshold τ can we emit pseudo-labels on
   *unlabeled* soundscape files with ≥0.9 segment-level precision?"

Output: console table of (τ, precision, recall, #labels emitted, #species),
plus a recommended τ*. Nothing is written to disk. No Perch inference is
run — consumes only the cached scores. CPU, <5 s.

Usage:
    python -u src/rebuild_val_calibrate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE       = Path(__file__).resolve().parent
FT_ROOT    = HERE.parent
ROOT       = FT_ROOT.parent
PARENT_SRC = ROOT / "src"

if str(PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(PARENT_SRC))

import config  # noqa: E402
from config import get_species_index  # noqa: E402


PERCH_DIR = FT_ROOT / "data" / "kaggle_perch_cache"
GT_CSV    = config.RAW / "train_soundscapes_labels.csv"


def load_perch_cache() -> tuple[np.ndarray, pd.DataFrame]:
    arr = np.load(PERCH_DIR / "full_perch_arrays.npz")
    scores = arr["scores_full_raw"].astype(np.float32)  # (708, 234)
    meta = pd.read_parquet(PERCH_DIR / "full_perch_meta.parquet")
    parts = meta["row_id"].str.extract(r"^(?P<stem>.+)_(?P<sec>\d+)$")
    meta = meta.assign(
        stem=parts["stem"].values,
        end_sec=parts["sec"].astype(int).values,
    )
    # Perch sec is the *end* of its 5-s window; the labels CSV keys on start.
    meta["start_sec"] = meta["end_sec"] - 5
    meta["filename"]  = meta["stem"] + ".ogg"
    return scores, meta


def load_segment_gt(sp2idx: dict) -> dict[tuple[str, int], np.ndarray]:
    df = pd.read_csv(GT_CSV)

    def _parse(s: str) -> int:
        h, m, sec = str(s).split(":")
        return int(h) * 3600 + int(m) * 60 + int(sec)

    df["start_sec"] = df["start"].apply(_parse)
    # CSV is duplicated 2× per (filename, start_sec); drop.
    df = df.drop_duplicates(["filename", "start_sec"]).reset_index(drop=True)

    n_classes = len(sp2idx)
    gt: dict[tuple[str, int], np.ndarray] = {}
    for _, r in df.iterrows():
        vec = np.zeros(n_classes, dtype=np.float32)
        for sp in str(r["primary_label"]).split(";"):
            sp = sp.strip()
            if sp in sp2idx:
                vec[sp2idx[sp]] = 1.0
        gt[(r["filename"], int(r["start_sec"]))] = vec
    return gt


def main() -> int:
    sp2idx = get_species_index()
    scores, meta = load_perch_cache()
    print(f"[calibrate] Perch cache: {len(meta)} segments, {scores.shape[1]} classes",
          flush=True)

    gt = load_segment_gt(sp2idx)
    print(f"[calibrate] GT segments in train_soundscapes_labels.csv: {len(gt)}",
          flush=True)

    # Match Perch segments against GT.
    matched_mask = np.array(
        [(r.filename, int(r.start_sec)) in gt for r in meta.itertuples()],
        dtype=bool,
    )
    n_matched = int(matched_mask.sum())
    print(f"[calibrate] Perch segs with GT: {n_matched}/{len(meta)}",
          flush=True)

    if n_matched == 0:
        print("[error] no Perch segments match GT — check filename/sec schema",
              flush=True)
        return 1

    y_true = np.vstack([
        gt[(r.filename, int(r.start_sec))]
        for r in meta[matched_mask].itertuples()
    ])  # (N, 234)
    y_score = scores[matched_mask]  # (N, 234)

    # Species present in GT among the matched rows.
    species_mask = (y_true.sum(axis=0) > 0)  # (234,) bool
    species_in_gt = int(species_mask.sum())
    print(f"[calibrate] species with ≥1 positive in matched GT: "
          f"{species_in_gt}/234",
          flush=True)
    print(f"[calibrate] precision is RESTRICTED to these {species_in_gt} "
          f"species — Perch emissions on the other "
          f"{234 - species_in_gt} species count as neither TP nor FP "
          f"(GT is silent there, not zero).",
          flush=True)

    # ── τ sweep (logit space — scores are raw Perch logits, not sigmoids) ────
    taus = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    hdr = (f"\n{'τ':>6} | {'prec':>8} | {'recall':>8} | "
           f"{'tp':>6} {'fp':>6} {'fn':>6} | "
           f"{'labels':>7} | {'species≥1':>10}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    rows = []
    for tau in taus:
        pred_full = (y_score > tau).astype(np.float32)  # all 234 classes
        # Restrict tp/fp/fn to species with GT support; keep "labels emitted"
        # on the full 234 so the user sees pseudo-label volume.
        pred_r = pred_full[:, species_mask]
        true_r = y_true[:,  species_mask]
        tp = float((pred_r * true_r).sum())
        fp = float((pred_r * (1.0 - true_r)).sum())
        fn = float(((1.0 - pred_r) * true_r).sum())
        prec   = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        n_lab  = int(pred_full.sum())
        n_sp   = int((pred_full.sum(axis=0) > 0).sum())
        print(f"  {tau:>5.1f} | {prec:.4f}  | {recall:.4f}  | "
              f"{int(tp):>6} {int(fp):>6} {int(fn):>6} | "
              f"{n_lab:>7} | {n_sp:>10}",
              flush=True)
        rows.append((tau, prec, recall, n_lab, n_sp))

    # Recommend τ*.
    ok = [r for r in rows if r[1] >= 0.90]
    if ok:
        tau_star = min(ok, key=lambda r: r[0])[0]
        why = "smallest τ with prec ≥ 0.90"
    else:
        tau_star = max(rows, key=lambda r: r[1])[0]
        why = "no τ hits 0.90 — falling back to highest-precision τ"
    star_row = next(r for r in rows if r[0] == tau_star)
    print(f"\n[calibrate] τ* = {tau_star:.2f} (logit)  ({why})", flush=True)
    print(f"[calibrate]   prec={star_row[1]:.4f}  recall={star_row[2]:.4f}  "
          f"labels={star_row[3]}  species≥1={star_row[4]}",
          flush=True)

    # ── Per-species precision @ τ* for well-supported species ────────────────
    pred_star = (y_score > tau_star).astype(np.float32)
    tp_k = (pred_star * y_true).sum(axis=0)
    fp_k = (pred_star * (1.0 - y_true)).sum(axis=0)
    support_k = y_true.sum(axis=0)
    precision_k = np.where(
        (tp_k + fp_k) > 0, tp_k / np.maximum(1.0, tp_k + fp_k), np.nan,
    )

    # Invert sp2idx for reporting.
    idx2sp = {v: k for k, v in sp2idx.items()}
    well = np.where(support_k >= 5)[0]
    if len(well) > 0:
        print(f"\n[calibrate] per-species precision @ τ={tau_star:.2f} (logit) "
              f"(support ≥ 5, {len(well)} species):",
              flush=True)
        order = well[np.argsort(precision_k[well])]
        print(f"  {'sp_code':>12} | {'support':>7} | {'pred':>6} | {'prec':>6}",
              flush=True)
        for k in order[:10]:
            print(f"  {idx2sp[int(k)]:>12} | {int(support_k[k]):>7} | "
                  f"{int((tp_k[k] + fp_k[k])):>6} | {precision_k[k]:.4f}",
                  flush=True)
        print(f"  … (worst 10 shown; median per-species prec = "
              f"{np.nanmedian(precision_k[well]):.4f})",
              flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
