"""Probe stage A: do v56's 234-dim logits carry within-val signal for non-Aves classes?

For each non-Aves class with >=5 positives in the OOF substrate, fit a 5-fold CV
LogReg using all 234 raw logits as features and the binary class label as target.
Compute mean OOF AUC. Compare against v56's native per-class AUC (which uses only
the diagonal logit, i.e. logit[c] for class c).

If recovered_AUC >> native_AUC, the encoder/logits have signal but the head
miscalibrates it — fix is head re-training, NOT pretrain corpus expansion.
If recovered_AUC ~= native_AUC, the signal isn't in the logit space, which
strongly suggests the encoder lacks features — pretrain corpus is the lever.
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.special import logit  # safe inverse-sigmoid

ROOT = Path("/home/swatson/work/kaggle/BirdCLEF/four_track")
TAX = pd.read_csv("/home/swatson/work/kaggle/BirdCLEF/data/raw/taxonomy.csv")
d = np.load(ROOT / "data/v56_soundscape_oof.npz", allow_pickle=True)
probs = d["probs_mean"].astype(np.float64)   # (1478, 234)
y     = d["y_true"].astype(np.int64)         # (1478, 234)

# Convert sigmoid probs -> raw logits for LR features (linear in logit space)
EPS = 1e-6
probs_clipped = np.clip(probs, EPS, 1 - EPS)
logits = logit(probs_clipped)                # (1478, 234)

# Class id ordering: train_a1 sorts primary_label as strings.
classes = sorted(TAX["primary_label"].astype(str).tolist())
TAX["primary_label"] = TAX["primary_label"].astype(str)
class_to_taxon = dict(zip(TAX["primary_label"], TAX["class_name"]))
groups = np.array([class_to_taxon[c] for c in classes])

n_pos = y.sum(axis=0)
print(f"OOF: N={len(y)}, classes={len(classes)}")
print(f"Native overall macro AUC (computable classes): "
      f"{np.nanmean([roc_auc_score(y[:,c], probs[:,c]) for c in range(234) if 0 < n_pos[c] < len(y)]):.4f}\n")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for c in range(234):
    if not (5 <= n_pos[c] < len(y) - 5):
        continue
    yc = y[:, c]
    # native AUC = AUC on diagonal probability
    native = roc_auc_score(yc, probs[:, c])
    # CV-recovered AUC = LR(234 features) -> binary target, OOF predictions
    oof_pred = np.zeros(len(yc))
    for tr, te in skf.split(logits, yc):
        lr = LogisticRegression(
            C=0.5, class_weight="balanced",
            solver="liblinear", max_iter=2000,
        )
        lr.fit(logits[tr], yc[tr])
        oof_pred[te] = lr.decision_function(logits[te])
    recovered = roc_auc_score(yc, oof_pred)
    results.append({
        "class_id": classes[c],
        "taxon": groups[c],
        "n_pos": int(n_pos[c]),
        "native_auc": native,
        "recovered_auc": recovered,
        "delta": recovered - native,
    })

df = pd.DataFrame(results)
print("=" * 78)
print("Per-taxon summary (mean over classes with >=5 positives)")
print("=" * 78)
agg = df.groupby("taxon").agg(
    n_classes=("class_id", "count"),
    native_mean=("native_auc", "mean"),
    recovered_mean=("recovered_auc", "mean"),
    delta_mean=("delta", "mean"),
).round(4)
print(agg.to_string())

print("\nPer-class detail (non-Aves only):")
print("-" * 78)
nonaves = df[df["taxon"] != "Aves"].sort_values("delta", ascending=False)
print(nonaves[["taxon", "class_id", "n_pos", "native_auc", "recovered_auc", "delta"]].to_string(index=False))

print("\n--- INTERPRETATION ---")
ins = df[df["taxon"] == "Insecta"]
if len(ins):
    nat = ins["native_auc"].mean()
    rec = ins["recovered_auc"].mean()
    print(f"Insecta: native {nat:.3f} -> recovered {rec:.3f} (delta {rec-nat:+.3f})")
    if rec - nat >= 0.10:
        print("  -> Logit space carries Insecta signal that v56's diagonal output misses.")
        print("     Head/loss/calibration fix is plausible. iNatSounds download might be unnecessary.")
    elif rec - nat >= 0.03:
        print("  -> Some recoverable signal; mixed verdict. Stage-B (true encoder probe) advisable.")
    else:
        print("  -> Logit space does NOT carry recoverable Insecta signal.")
        print("     Encoder genuinely lacks features. iNatSounds download is the right lever.")

# Save audit
out = ROOT / "data/probe_taxon_signal_results.csv"
df.to_csv(out, index=False)
print(f"\nWrote {out}")
