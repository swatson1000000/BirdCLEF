"""src/a2_calibrate_pseudo.py — Track A2 step 2: pseudo-label quality calibration.

Validates the OOF pseudo-label NPZ from step 1 against the 66 expert-labeled
train_soundscapes files. Reports precision / recall / F1 at multiple
thresholds plus per-class macro AUC on the 75 species covered by GT.

**Hard gate per new_plan.md §14.17.5:** precision @ 0.5 must be >= 0.70 over
the 75-species covered set. Below that, the threshold is too lax and Track A2
should abort or sharpen before retraining.

Two readings are reported:
  - "covered"  : metrics over the 75 species that appear in GT (fair measure)
  - "all 234"  : metrics over every class (catches over-firing on absent species)

Inputs:
  data/processed/a2_train_ss_oof_probs.npz  — (probs, filenames, start_sec, oof_bucket)
  data/raw/train_soundscapes_labels.csv     — GT for 66 labeled files

Outputs:
  Stdout summary table
  data/processed/a2_calibration_report.csv  — per-class P/R/F1/AUC table
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
from config import RAW, get_species_index  # noqa: E402

NPZ_PATH    = FT_ROOT / "data" / "processed" / "a2_train_ss_oof_probs.npz"
GT_CSV      = RAW / "train_soundscapes_labels.csv"
OUT_CSV     = FT_ROOT / "data" / "processed" / "a2_calibration_report.csv"
THRESHOLDS  = [0.30, 0.40, 0.50, 0.60, 0.70]
GATE_THRESH = 0.50
GATE_MIN_P  = 0.70


def _parse_time(s: str) -> int:
    h, m, sec = str(s).split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec)


def _prf(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float, int, int, int]:
    """Return (precision, recall, F1, TP, FP, FN) over a binary array pair."""
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return p, r, f1, tp, fp, fn


def main() -> None:
    print(f"loading NPZ: {NPZ_PATH}", flush=True)
    assert NPZ_PATH.exists(), f"NPZ missing: {NPZ_PATH} — run a2_emit_oof_pseudo.py first"
    z = np.load(NPZ_PATH, allow_pickle=False)
    probs     = z["probs"]                      # (N_chunks, 234)
    filenames = z["filenames"].astype(str)      # (N_chunks,)
    starts    = z["start_sec"]                  # (N_chunks,)
    print(f"  N_chunks={len(probs)}  N_classes={probs.shape[1]}", flush=True)

    sp2idx = get_species_index()
    n_classes = config.N_CLASSES
    assert probs.shape[1] == n_classes

    print(f"loading GT: {GT_CSV}", flush=True)
    gt_df = pd.read_csv(GT_CSV)
    print(f"  raw rows: {len(gt_df)}  files: {gt_df['filename'].nunique()}", flush=True)
    gt_df = gt_df.drop_duplicates(subset=["filename", "start", "end"]).reset_index(drop=True)
    print(f"  after dedup: {len(gt_df)} chunks", flush=True)

    # Build NPZ lookup: (filename, start_sec) -> row index
    key_to_row = {(f, int(s)): i for i, (f, s) in enumerate(zip(filenames, starts))}

    # For each GT chunk, find the matching NPZ row and build label vec
    gt_probs   = np.zeros((len(gt_df), n_classes), dtype=np.float32)
    gt_labels  = np.zeros((len(gt_df), n_classes), dtype=np.float32)
    gt_files   = []
    matched    = 0
    missing    = 0
    species_seen: set[str] = set()
    for i, row in gt_df.iterrows():
        key = (str(row["filename"]), _parse_time(row["start"]))
        if key in key_to_row:
            gt_probs[i] = probs[key_to_row[key]]
            matched += 1
        else:
            missing += 1
            continue
        gt_files.append(key[0])
        for sp in str(row["primary_label"]).split(";"):
            sp = sp.strip()
            if sp in sp2idx:
                gt_labels[i, sp2idx[sp]] = 1.0
                species_seen.add(sp)

    print(f"  matched={matched}  missing_in_npz={missing}", flush=True)
    if missing > 0:
        print(f"  [warn] {missing} GT chunks have no NPZ prediction — "
              f"check OOF emit covered all SS files", flush=True)
    print(f"  GT-covered species: {len(species_seen)} of {n_classes}", flush=True)

    # Mask classes covered by GT (i.e. >=1 positive in val set)
    covered = gt_labels.sum(axis=0) > 0
    n_covered = int(covered.sum())
    print(f"  classes with >=1 positive in val: {n_covered}", flush=True)

    n_pos_total      = int(gt_labels.sum())
    n_pos_covered    = int(gt_labels[:, covered].sum())
    n_chunks_used    = matched
    print(f"  total GT positives: {n_pos_total}  (over covered: {n_pos_covered})", flush=True)
    print()

    # ── Threshold sweep table ────────────────────────────────────────────────
    print("=" * 78, flush=True)
    print("THRESHOLD SWEEP — chunk × class binary classification", flush=True)
    print("=" * 78, flush=True)
    print(f"{'thresh':>7}  {'scope':>9}  {'P':>6}  {'R':>6}  {'F1':>6}  "
          f"{'TP':>7}  {'FP':>7}  {'FN':>7}  {'kept_chunks':>11}", flush=True)
    print("-" * 78, flush=True)

    sweep_rows = []
    for t in THRESHOLDS:
        chunk_kept = (gt_probs.max(axis=1) >= t)
        kept_n = int(chunk_kept.sum())

        # 75-species covered scope
        pred_c = ((gt_probs[:, covered] >= t) & chunk_kept[:, None]).astype(int)
        gt_c   = gt_labels[:, covered].astype(int)
        p_c, r_c, f1_c, tp_c, fp_c, fn_c = _prf(pred_c, gt_c)
        print(f"{t:>7.2f}  {'covered':>9}  {p_c:>6.3f}  {r_c:>6.3f}  {f1_c:>6.3f}  "
              f"{tp_c:>7d}  {fp_c:>7d}  {fn_c:>7d}  {kept_n:>11d}", flush=True)

        # Strict all-234 scope (catches over-firing on uncovered species)
        pred_all = ((gt_probs >= t) & chunk_kept[:, None]).astype(int)
        gt_all   = gt_labels.astype(int)
        p_a, r_a, f1_a, tp_a, fp_a, fn_a = _prf(pred_all, gt_all)
        print(f"{t:>7.2f}  {'all 234':>9}  {p_a:>6.3f}  {r_a:>6.3f}  {f1_a:>6.3f}  "
              f"{tp_a:>7d}  {fp_a:>7d}  {fn_a:>7d}  {kept_n:>11d}", flush=True)
        print("-" * 78, flush=True)

        sweep_rows.append({
            "threshold": t,
            "covered_P": p_c, "covered_R": r_c, "covered_F1": f1_c,
            "all234_P":  p_a, "all234_R": r_a, "all234_F1": f1_a,
            "kept_chunks": kept_n,
        })

    # ── Per-class macro AUC on the 75 covered species ────────────────────────
    print(flush=True)
    print("=" * 78, flush=True)
    print("MACRO AUC — covered species only", flush=True)
    print("=" * 78, flush=True)
    try:
        macro_auc = float(roc_auc_score(
            gt_labels[:, covered], gt_probs[:, covered], average="macro"
        ))
        print(f"  macro AUC over {n_covered} covered species: {macro_auc:.4f}", flush=True)
    except ValueError as ex:
        print(f"  [warn] macro AUC failed: {ex}", flush=True)
        macro_auc = float("nan")

    # ── Per-class breakdown ──────────────────────────────────────────────────
    idx2sp = {v: k for k, v in sp2idx.items()}
    rows = []
    chunk_kept_50 = (gt_probs.max(axis=1) >= GATE_THRESH)
    for c in range(n_classes):
        n_pos = int(gt_labels[:, c].sum())
        if n_pos == 0:
            # Track FP rate only for absent species (over-firing diagnosis)
            fp_at_50 = int(((gt_probs[:, c] >= GATE_THRESH) & chunk_kept_50).sum())
            rows.append({
                "class_idx": c, "species": idx2sp.get(c, f"class_{c}"),
                "n_pos": 0, "covered": False,
                "auc": np.nan, "P_at_0.50": np.nan, "R_at_0.50": np.nan,
                "F1_at_0.50": np.nan, "TP_0.50": 0, "FP_0.50": fp_at_50, "FN_0.50": 0,
            })
            continue
        try:
            auc = float(roc_auc_score(gt_labels[:, c], gt_probs[:, c]))
        except ValueError:
            auc = float("nan")
        pred = ((gt_probs[:, c] >= GATE_THRESH) & chunk_kept_50).astype(int)
        gt   = gt_labels[:, c].astype(int)
        p, r, f1, tp, fp, fn = _prf(pred, gt)
        rows.append({
            "class_idx": c, "species": idx2sp.get(c, f"class_{c}"),
            "n_pos": n_pos, "covered": True,
            "auc": auc, "P_at_0.50": p, "R_at_0.50": r, "F1_at_0.50": f1,
            "TP_0.50": tp, "FP_0.50": fp, "FN_0.50": fn,
        })
    out_df = pd.DataFrame(rows).sort_values("n_pos", ascending=False)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nper-class report → {OUT_CSV}", flush=True)

    # Summary stats over uncovered FPs
    uncovered_fp = int(out_df.loc[~out_df["covered"], "FP_0.50"].sum())
    uncovered_w_any_fp = int((out_df.loc[~out_df["covered"], "FP_0.50"] > 0).sum())
    n_uncovered = int((~out_df["covered"]).sum())
    print(f"\nUncovered-species over-firing diagnosis:", flush=True)
    print(f"  {n_uncovered} species absent from val", flush=True)
    print(f"  {uncovered_w_any_fp} of those fired >=1 FP at score>=0.50", flush=True)
    print(f"  total FP chunks across uncovered species: {uncovered_fp}", flush=True)

    # ── Hard gate ────────────────────────────────────────────────────────────
    gate_p = next(r for r in sweep_rows if r["threshold"] == GATE_THRESH)["covered_P"]
    print(flush=True)
    print("=" * 78, flush=True)
    print(f"GATE: precision @ {GATE_THRESH:.2f} on covered species = {gate_p:.3f}",
          flush=True)
    print(f"      required >= {GATE_MIN_P:.2f}", flush=True)
    if gate_p >= GATE_MIN_P:
        print(f"      RESULT: PASS — proceed to step 3 (a2_filter_pseudo.py)",
              flush=True)
    else:
        print(f"      RESULT: FAIL — abort A2 OR sharpen threshold OR re-emit",
              flush=True)
        print(f"      (BC2025 winners hit P>=0.7 with this exact recipe; if we",
              flush=True)
        print(f"       miss it, recipe is broken — investigate before retrying)",
              flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
