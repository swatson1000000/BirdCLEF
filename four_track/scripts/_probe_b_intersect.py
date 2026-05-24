"""Check 708 ↔ 1478 substrate intersection."""
import numpy as np, pandas as pd

m = pd.read_parquet('/home/swatson/work/kaggle/BirdCLEF/data/processed/perch_cache/full_perch_meta.parquet')
m['start_sec'] = m['row_id'].str.rsplit('_', n=1).str[-1].astype(int) - 5
m['cache_idx'] = np.arange(len(m))
print('d2_beta cache substrate:', len(m), 'files:', m['filename'].nunique())

v4 = np.load('/home/swatson/work/kaggle/BirdCLEF/four_track/data/v4_5fold_soundscape_oof.npz', allow_pickle=True)
v4_df = pd.DataFrame({'filename': v4['filenames'], 'start_sec': v4['start_sec'].astype(int)})
v4_df['v4_idx'] = np.arange(len(v4_df))
print('v4 1478 substrate:', len(v4_df), 'files:', v4_df['filename'].nunique())

# Join: each cache row may match multiple v4 rows (1478 has duplicates per labels CSV)
joined = m.merge(v4_df, on=['filename','start_sec'], how='inner')
print('inner-join total rows:', len(joined))
print('unique (file,start) pairs in v4 covered by cache:',
      v4_df.merge(m[['filename','start_sec']].drop_duplicates(), how='inner').shape[0])
mc = joined.groupby(['cache_idx']).size()
print(f'cache rows mapping to >=1 v4 row: {(mc>=1).sum()}/{len(m)}')
print(f'mean v4-rows per cache row: {mc.mean():.2f}')

# v4 rows with NO cache row (i.e. files in 1478 but not in 708)
v4_uncovered = v4_df[~v4_df['filename'].isin(m['filename'].unique())]
print(f'v4 rows from files not in 708 cache: {len(v4_uncovered)}')
print(f'v4 files not in 708: {sorted(v4_uncovered["filename"].unique())}')
