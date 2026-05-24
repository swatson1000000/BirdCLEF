"""Inspect notebook + OOF artifacts for Probe B."""
import json
import sys
import numpy as np

print('=== d2_beta_oofs.npz ===', flush=True)
d = np.load('/home/swatson/work/kaggle/BirdCLEF/four_track/data/d2_beta_oofs.npz', allow_pickle=True)
for k in d.files:
    arr = d[k]
    print(f'  {k}: shape={arr.shape}, dtype={arr.dtype}')

print('\n=== v4_5fold_soundscape_oof.npz ===', flush=True)
d = np.load('/home/swatson/work/kaggle/BirdCLEF/four_track/data/v4_5fold_soundscape_oof.npz', allow_pickle=True)
for k in d.files:
    arr = d[k]
    print(f'  {k}: shape={arr.shape}, dtype={arr.dtype}')
    if k in ('filenames',):
        print('    sample:', arr[:3])
    if k in ('start_sec',):
        print('    sample:', arr[:5])

print('\n=== Notebook cell list ===', flush=True)
nb_path = '/home/swatson/work/kaggle/BirdCLEF/jupyter/protossm-postproc/birdclef2026-protossm-postproc.ipynb'
with open(nb_path) as f:
    nb = json.load(f)
print(f'Total cells: {len(nb["cells"])}')
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    first = src.split('\n', 1)[0][:140] if src else '(empty)'
    print(f'[{i:2d}] {c["cell_type"]:8s} | {first}')
