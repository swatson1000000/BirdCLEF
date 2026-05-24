"""A2-2: Build pseudo-label manifest CSV from the A1 5-fold soundscape probs.

Reads data/processed/a1_pseudo_soundscape.npz (from a2_pseudo_label_a1.py)
and writes a CSV with the schema train_a1.py expects, in the BC2026_SS_PSEUDO
collection format.

Output schema (matches train_folds.csv columns + pseudo extras):
  filename                 BC2026_Train_NNNN_*.ogg
  primary_label            top-1 class above threshold (single species code)
  secondary_labels         "[]"  (handled by pseudo_positive_labels instead)
  pseudo_positive_labels   semicolon-joined species codes above threshold
  pseudo_window_start      0,5,10,...,55  (5s slot start)
  collection               "BC2026_SS_PSEUDO"
  fold                     -1 (training-only; never validated)
  rating                   0.0
  type                     "[]"

Filtering:
  --threshold 0.5     keep classes with prob ≥ 0.5
  --max-classes 5     limit per-window positives to top-K above threshold
  --min-classes 1     require ≥ N classes above threshold (else drop window)

Defaults are chosen to be conservative — high-confidence multi-label rows.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARENT = ROOT.parent

sys.path.insert(0, str(PARENT / 'src'))
from config import get_species_list  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probs', default=str(ROOT / 'data' / 'processed' / 'a1_pseudo_soundscape.npz'))
    ap.add_argument('--threshold', type=float, default=0.5,
                    help='Probability threshold for class to be considered positive')
    ap.add_argument('--max-classes', type=int, default=5,
                    help='Cap per-window positive count (top-K by prob)')
    ap.add_argument('--min-classes', type=int, default=1,
                    help='Drop windows with fewer than N positives')
    ap.add_argument('--out', default=str(ROOT / 'data' / 'processed' / 'a2_pseudo_manifest.csv'))
    args = ap.parse_args()

    species = get_species_list()
    n_cls = len(species)
    print(f'Species list: {n_cls} classes', flush=True)

    d = np.load(args.probs, allow_pickle=True)
    probs = d['probs']                      # (n_windows, 234)
    filenames = d['filenames']              # (n_windows,)
    start_sec = d['start_sec']              # (n_windows,)
    print(f'Loaded probs: {probs.shape}', flush=True)
    assert probs.shape[1] == n_cls, f'class count mismatch: {probs.shape[1]} vs {n_cls}'

    rows = []
    n_dropped = 0
    for i in range(probs.shape[0]):
        p = probs[i]
        # Find classes above threshold
        above = np.where(p >= args.threshold)[0]
        if len(above) < args.min_classes:
            n_dropped += 1
            continue
        # Cap to top-K by prob
        if len(above) > args.max_classes:
            above = above[np.argsort(-p[above])[:args.max_classes]]
        # Sort by descending prob so primary_label = top-1
        above = above[np.argsort(-p[above])]
        primary = species[above[0]]
        positives = ';'.join(species[c] for c in above)
        rows.append({
            'filename': str(filenames[i]),
            'primary_label': primary,
            'secondary_labels': '[]',
            'pseudo_positive_labels': positives,
            'pseudo_window_start': int(start_sec[i]),
            'collection': 'BC2026_SS_PSEUDO',
            'fold': -1,
            'rating': 0.0,
            'type': '[]',
        })

    print(f'Built {len(rows)} pseudo rows  ({n_dropped} dropped: < min_classes above threshold)',
          flush=True)
    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'Saved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)', flush=True)

    # Diagnostics
    n_pos_per_window = df['pseudo_positive_labels'].str.count(';') + 1
    print(f'\nPositives per window:  mean={n_pos_per_window.mean():.2f}  '
          f'p50={n_pos_per_window.median():.0f}  '
          f'p95={n_pos_per_window.quantile(0.95):.0f}  '
          f'max={n_pos_per_window.max()}')
    top10 = df['primary_label'].value_counts().head(10)
    print(f'\nTop-10 primary classes:')
    print(top10.to_string())


if __name__ == '__main__':
    main()
