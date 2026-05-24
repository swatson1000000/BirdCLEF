"""Probe stage A v2: same as v1 but with sanity controls.

For each non-Aves class with >=5 positives, fit three 5-fold CV LogRegs:
  (1) full 234 logit features (original)
  (2) single-feature: only the diagonal logit (tests pure threshold/scale recovery)
  (3) permutation: shuffle y labels then refit with full 234 features
       — if (3) still gets AUC ~ 1.0, the recovery is overfitting, not signal.

A clean recovery => (1) >> native, (2) ~ native, (3) ~ 0.5.
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.special import logit

ROOT = Path("/home/swatson/work/kaggle/BirdCLEF/four_track")
TAX = pd.read_csv("/home/swatson/work/kaggle/BirdCLEF/data/raw/taxonomy.csv")
d = np.load(ROOT / "data/v56_soundscape_oof.npz", allow_pickle=True)
probs = d["probs_mean"].astype(np.float64)
y     = d["y_true"].astype(np.int64)

EPS = 1e-6
logits_all = logit(np.clip(probs, EPS, 1 - EPS))
classes = sorted(TAX["primary_label"].astype(str).tolist())
TAX["primary_label"] = TAX["primary_label"].astype(str)
class_to_taxon = dict(zip(TAX["primary_label"], TAX["class_name"]))
groups = np.array([class_to_taxon[c] for c in classes])
n_pos = y.sum(axis=0)

rng = np.random.default_rng(7)

def cv_logreg(X, yc, C=0.5):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(yc))
    for tr, te in skf.split(X, yc):
        lr = LogisticRegression(
            C=C, class_weight="balanced",
            solver="liblinear", max_iter=2000,
        )
        lr.fit(X[tr], yc[tr])
        if X.shape[1] == 1:
            oof[te] = lr.decision_function(X[te])
        else:
            oof[te] = lr.decision_function(X[te])
    return roc_auc_score(yc, oof)

results = []
for c in range(234):
    if not (5 <= n_pos[c] < len(y) - 5):
        continue
    yc = y[:, c]
    native = roc_auc_score(yc, probs[:, c])

    # (1) full 234 features
    rec_full = cv_logreg(logits_all, yc, C=0.5)
    # (2) single feature
    rec_single = cv_logreg(logits_all[:, [c]], yc, C=0.5)
    # (3) permutation control: shuffle y, fit full features
    yshuf = rng.permutation(yc)
    rec_perm = cv_logreg(logits_all, yshuf, C=0.5)

    results.append({
        "class_id": classes[c],
        "taxon":    groups[c],
        "n_pos":    int(n_pos[c]),
        "native":   native,
        "rec_full": rec_full,
        "rec_single": rec_single,
        "rec_perm": rec_perm,
    })

df = pd.DataFrame(results)
print("=" * 80)
print("Per-taxon means")
print("=" * 80)
agg = df.groupby("taxon").agg(
    n_classes=("class_id", "count"),
    native=("native", "mean"),
    rec_full=("rec_full", "mean"),
    rec_single=("rec_single", "mean"),
    rec_perm=("rec_perm", "mean"),
).round(4)
print(agg.to_string())

print("\nDiagnostic guide:")
print("  rec_perm should be ~0.50 (no signal in shuffled labels).")
print("  If rec_perm > 0.70, the procedure is overfitting and rec_full numbers are inflated.")
print("  rec_single tests whether per-class threshold/scale alone recovers the gap.")
print("  rec_full > rec_single by a wide margin means OTHER logit channels carry the signal.")

print("\nNon-Aves detail (sorted by full-feature recovery):")
print("-" * 80)
nonaves = df[df["taxon"] != "Aves"].sort_values("rec_full", ascending=False)
print(nonaves[["taxon","class_id","n_pos","native","rec_single","rec_full","rec_perm"]].to_string(index=False))

out = ROOT / "data/probe_taxon_signal_v2_results.csv"
df.to_csv(out, index=False)
print(f"\nWrote {out}")
