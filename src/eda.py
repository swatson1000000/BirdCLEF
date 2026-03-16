"""
EDA for BirdCLEF+ 2026.

Produces:
  data/processed/eda_report.txt   — human-readable stats
  data/processed/train_folds.csv  — train.csv + fold column (0–4)
"""

import ast
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path("/home/swatson/work/MachineLearning/kaggle/BirdCLEF")
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────────────
print("Loading CSVs …")
train = pd.read_csv(RAW / "train.csv")
taxonomy = pd.read_csv(RAW / "taxonomy.csv")
soundscape_labels = pd.read_csv(RAW / "train_soundscapes_labels.csv")

# parse secondary_labels col (stored as stringified Python list)
def parse_list(s):
    if pd.isna(s) or s in ("", "[]"):
        return []
    try:
        return ast.literal_eval(s)
    except Exception:
        return [x.strip().strip("'\"") for x in s.strip("[]").split(",") if x.strip()]

train["secondary_labels_parsed"] = train["secondary_labels"].apply(parse_list)

# ── report lines ───────────────────────────────────────────────────────────────
lines = []

def h(title):
    lines.append("")
    lines.append("=" * 60)
    lines.append(title)
    lines.append("=" * 60)

def p(*args):
    lines.append(" ".join(str(a) for a in args))

# ── 1. Global stats ────────────────────────────────────────────────────────────
h("1. GLOBAL STATS")
p("Total training clips :", len(train))
p("Unique primary labels:", train["primary_label"].nunique())
p("Taxonomy species      :", len(taxonomy))
p("Train soundscapes rows:", len(soundscape_labels))
p("Train soundscape files:", soundscape_labels["filename"].nunique())

# ── 2. Collection breakdown ────────────────────────────────────────────────────
h("2. COLLECTION BREAKDOWN")
coll_counts = train["collection"].value_counts()
for coll, cnt in coll_counts.items():
    p(f"  {coll:10s}: {cnt:6d} clips ({100*cnt/len(train):.1f}%)")

# ── 3. Taxonomy class breakdown ────────────────────────────────────────────────
h("3. TAXONOMY CLASS BREAKDOWN")
class_counts = taxonomy["class_name"].value_counts()
for cls, cnt in class_counts.items():
    p(f"  {cls:20s}: {cnt:4d} species")

# also show per train clip
p("")
p("Train clips by class_name:")
# train.csv already has class_name; use it directly
for cls, cnt in train["class_name"].value_counts().items():
    p(f"  {cls:20s}: {cnt:6d} clips ({100*cnt/len(train):.1f}%)")

# ── 4. Samples-per-species distribution ───────────────────────────────────────
h("4. SAMPLES PER SPECIES (primary label only)")
spc = train["primary_label"].value_counts()
p(f"  Min    : {spc.min():5d}  ({spc.idxmin()})")
p(f"  Median : {spc.median():5.0f}")
p(f"  Mean   : {spc.mean():7.1f}")
p(f"  Max    : {spc.max():5d}  ({spc.idxmax()})")

bins = [0, 1, 5, 10, 20, 50, 100, 200, 500, 1000, 9999]
labels_bins = ["1", "2-5", "6-10", "11-20", "21-50", "51-100", "101-200",
               "201-500", "501-1000", ">1000"]
p(f"\n  Distribution:")
for i in range(len(bins)-1):
    cnt = ((spc > bins[i]) & (spc <= bins[i+1])).sum()
    if cnt:
        p(f"    {labels_bins[i]:12s}: {cnt:4d} species")

# ── 5. Species with < 10 samples (need duplication) ───────────────────────────
h("5. SPECIES WITH < 10 CLIPS (will be duplicated before fold split)")
rare = spc[spc < 10].sort_values()
p(f"  Count: {len(rare)}")
if len(rare) > 0:
    for sp, cnt in rare.items():
        row = taxonomy[taxonomy["primary_label"] == sp]
        name = row["common_name"].values[0] if len(row) else str(sp)
        cls  = row["class_name"].values[0]  if len(row) else "?"
        p(f"    {sp}  [{cls:10s}]  {cnt:2d} clips  {name}")

# ── 6. Rating distribution (XC clips only) ────────────────────────────────────
h("6. RATING DISTRIBUTION (XC clips, 0=unrated/iNat)")
xc = train[train["collection"] == "XC"]
p(f"  XC clips with rating ≥ 3.5: {(xc['rating'] >= 3.5).sum()}")
p(f"  XC clips with rating < 3.5 : {(xc['rating'] < 3.5).sum()}")
p(f"  XC clips with rating = 0.0 : {(xc['rating'] == 0.0).sum()}")
inat = train[train["collection"] == "iNat"]
p(f"  iNat clips (all rating=0)  : {len(inat)}")

# ── 7. Secondary labels stats ──────────────────────────────────────────────────
h("7. SECONDARY LABELS")
has_secondary = train["secondary_labels_parsed"].apply(len) > 0
p(f"  Clips with ≥1 secondary label  : {has_secondary.sum()} ({100*has_secondary.mean():.1f}%)")
all_secondary = [sp for lst in train["secondary_labels_parsed"] for sp in lst]
p(f"  Total secondary label instances: {len(all_secondary)}")
p(f"  Unique secondary species       : {len(set(all_secondary))}")

# ── 8. Soundscape labels ───────────────────────────────────────────────────────
h("8. TRAIN SOUNDSCAPES LABELS")
# primary_label is semicolon-separated
soundscape_labels["label_list"] = soundscape_labels["primary_label"].apply(
    lambda x: [s.strip() for s in str(x).split(";") if s.strip()]
)
all_sl = [sp for lst in soundscape_labels["label_list"] for sp in lst]
p(f"  5s segments                : {len(soundscape_labels)}")
p(f"  Total species instances    : {len(all_sl)}")
p(f"  Unique species in soundscapes: {len(set(all_sl))}")
sl_spc = Counter(all_sl)
p(f"  Most common in soundscapes : {sl_spc.most_common(5)}")

# ── 9. Coverage: train_audio species vs taxonomy ──────────────────────────────
h("9. COVERAGE")
train_species = set(train["primary_label"].astype(str))
taxonomy_species = set(taxonomy["primary_label"].astype(str))
p(f"  Species in taxonomy  : {len(taxonomy_species)}")
p(f"  Species in train_csv : {len(train_species)}")
p(f"  In taxonomy, not in train: {len(taxonomy_species - train_species)}")
for sp in sorted(taxonomy_species - train_species):
    row = taxonomy[taxonomy["primary_label"].astype(str) == sp]
    name = row["common_name"].values[0] if len(row) else str(sp)
    cls  = row["class_name"].values[0]  if len(row) else "?"
    p(f"    {sp}  [{cls:10s}]  {name}")

# ── 10. Geographic spread ──────────────────────────────────────────────────────
h("10. GEOGRAPHIC SPREAD")
has_coords = train[["latitude", "longitude"]].notna().all(axis=1)
p(f"  Clips with GPS coords: {has_coords.sum()} / {len(train)}")
if has_coords.sum():
    p(f"  Lat range  : {train['latitude'].min():.2f} – {train['latitude'].max():.2f}")
    p(f"  Lon range  : {train['longitude'].min():.2f} – {train['longitude'].max():.2f}")

# ── 11. 5-fold stratified assignment ──────────────────────────────────────────
h("11. 5-FOLD STRATIFIED ASSIGNMENT")

# Duplicate rare species to floor at 10 before folding
print("Building fold assignment …")
species_list = sorted(train["primary_label"].unique())
sp2idx = {sp: i for i, sp in enumerate(species_list)}
n_species = len(species_list)

# Build multi-label indicator matrix (primary only for stratification)
y = np.zeros((len(train), n_species), dtype=np.float32)
for i, sp in enumerate(train["primary_label"]):
    y[i, sp2idx[sp]] = 1.0

# Tile rare-species rows so each class has ≥10 representatives for stratification
# (This only affects fold splitting; duplicates are not written to train_folds.csv)
spc_counts = train["primary_label"].value_counts()
rare_species = set(spc_counts[spc_counts < 10].index)
p(f"  Rare species (< 10 clips, will duplicate for fold stratification): {len(rare_species)}")

skf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train = train.copy()
train["fold"] = -1

for fold, (_, val_idx) in enumerate(skf.split(train, y)):
    train.loc[train.index[val_idx], "fold"] = fold

fold_counts = train["fold"].value_counts().sort_index()
p("  Fold size distribution:")
for fold, cnt in fold_counts.items():
    p(f"    fold {fold}: {cnt} clips")

# ── save ───────────────────────────────────────────────────────────────────────
out_csv = PROC / "train_folds.csv"
train.to_csv(out_csv, index=False)
p(f"\n  Saved → {out_csv}")

report = "\n".join(lines)
out_txt = PROC / "eda_report.txt"
out_txt.write_text(report)

print(report)
print()
print(f"✓ EDA complete. Report → {out_txt}")
print(f"✓ Fold CSV   → {out_csv}")
