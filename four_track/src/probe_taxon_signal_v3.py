"""Probe v3: GroupKFold by source filename.

The val OOF substrate has 1478 5-sec segments from ~66 unique source files.
StratifiedKFold mixes segments from the same file across train/test, letting
LogReg memorize file-level logit signatures.

GroupKFold by filename forces disjoint files per fold, killing within-file leakage.
If rec_full stays high under GroupKFold, the encoder genuinely separates non-Aves
classes via cross-channel signal => head/calibration fix is the lever.
If rec_full collapses toward rec_single (threshold-only), the prior lift was
file-level leakage and pretrain corpus IS the right lever.
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.special import logit

ROOT = Path("/home/swatson/work/kaggle/BirdCLEF/four_track")
TAX = pd.read_csv("/home/swatson/work/kaggle/BirdCLEF/data/raw/taxonomy.csv")
d = np.load(ROOT / "data/v56_soundscape_oof.npz", allow_pickle=True)
probs = d["probs_mean"].astype(np.float64)
y     = d["y_true"].astype(np.int64)
filenames = d["filenames"]
print(f"OOF: {len(filenames)} segments, {len(np.unique(filenames))} unique files")

EPS = 1e-6
logits_all = logit(np.clip(probs, EPS, 1 - EPS))
classes = sorted(TAX["primary_label"].astype(str).tolist())
TAX["primary_label"] = TAX["primary_label"].astype(str)
class_to_taxon = dict(zip(TAX["primary_label"], TAX["class_name"]))
groups_tax = np.array([class_to_taxon[c] for c in classes])
n_pos = y.sum(axis=0)

# encode filenames -> group id
file_groups = pd.factorize(filenames)[0]
n_groups = len(np.unique(file_groups))
n_splits = min(5, n_groups)

def cv_logreg_grouped(X, yc, file_groups, C=0.5, splits=5):
    # GroupKFold doesn't shuffle, but we permute group ids for variance
    rng = np.random.default_rng(11)
    perm = rng.permutation(np.arange(file_groups.max() + 1))
    g_perm = perm[file_groups]

    gkf = GroupKFold(n_splits=splits)
    oof = np.zeros(len(yc))
    valid_mask = np.zeros(len(yc), dtype=bool)
    for tr, te in gkf.split(X, yc, groups=g_perm):
        # Skip folds where the test fold has no positives or no negatives
        if yc[te].sum() == 0 or yc[te].sum() == len(yc[te]):
            continue
        if yc[tr].sum() < 2:
            continue
        lr = LogisticRegression(
            C=C, class_weight="balanced",
            solver="liblinear", max_iter=2000,
        )
        lr.fit(X[tr], yc[tr])
        oof[te] = lr.decision_function(X[te])
        valid_mask[te] = True
    if valid_mask.sum() == 0 or yc[valid_mask].sum() == 0:
        return np.nan
    if yc[valid_mask].sum() == valid_mask.sum():
        return np.nan
    return roc_auc_score(yc[valid_mask], oof[valid_mask])

results = []
for c in range(234):
    if not (5 <= n_pos[c] < len(y) - 5):
        continue
    yc = y[:, c]
    native = roc_auc_score(yc, probs[:, c])
    rec_full   = cv_logreg_grouped(logits_all,         yc, file_groups, C=0.5, splits=n_splits)
    rec_single = cv_logreg_grouped(logits_all[:, [c]], yc, file_groups, C=0.5, splits=n_splits)

    results.append({
        "class_id":   classes[c],
        "taxon":      groups_tax[c],
        "n_pos":      int(n_pos[c]),
        "native":     native,
        "rec_full_grouped":   rec_full,
        "rec_single_grouped": rec_single,
    })

df = pd.DataFrame(results)
print("=" * 80)
print(f"GroupKFold-by-filename, {n_splits} splits ({n_groups} groups)")
print("=" * 80)
agg = df.groupby("taxon").agg(
    n_classes=("class_id", "count"),
    native=("native", "mean"),
    rec_single_grouped=("rec_single_grouped", "mean"),
    rec_full_grouped=("rec_full_grouped", "mean"),
).round(4)
print(agg.to_string())

print("\nDiagnostic guide (compare to v2 output):")
print("  If rec_full_grouped collapses to ~rec_single_grouped => prior recovery was file leakage.")
print("  If rec_full_grouped stays >> rec_single_grouped     => genuine cross-channel encoder signal.")

print("\nNon-Aves detail (sorted by full-grouped recovery):")
print("-" * 80)
nonaves = df[df["taxon"] != "Aves"].sort_values("rec_full_grouped", ascending=False, na_position="last")
print(nonaves[["taxon","class_id","n_pos","native","rec_single_grouped","rec_full_grouped"]].to_string(index=False))

out = ROOT / "data/probe_taxon_signal_v3_results.csv"
df.to_csv(out, index=False)
print(f"\nWrote {out}")
