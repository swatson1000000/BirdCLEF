"""Extra: test A1-only and finer sweep around best."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

def macro_auc_skip_empty(y, p):
    aucs = []
    for c in range(y.shape[1]):
        yc = y[:, c]
        if yc.sum() == 0 or yc.sum() == len(yc):
            continue
        try:
            aucs.append(roc_auc_score(yc, p[:, c]))
        except Exception:
            pass
    return float(np.mean(aucs)), len(aucs)

def rank01_per_col(mat):
    n = mat.shape[0]
    order = np.argsort(mat, axis=0, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    rows = np.arange(n, dtype=np.float32)
    for c in range(mat.shape[1]):
        ranks[order[:, c], c] = rows
    if n > 1: ranks /= (n - 1)
    return ranks

m = pd.read_parquet('/home/swatson/work/kaggle/BirdCLEF/data/processed/perch_cache/full_perch_meta.parquet')
m['start_sec'] = m['row_id'].str.rsplit('_', n=1).str[-1].astype(int) - 5
m['cache_idx'] = np.arange(len(m))
v4 = np.load('/home/swatson/work/kaggle/BirdCLEF/four_track/data/v4_5fold_soundscape_oof.npz', allow_pickle=True)
v4_df = pd.DataFrame({'filename': v4['filenames'], 'start_sec': v4['start_sec'].astype(int)})
v4_df['v4_idx'] = np.arange(len(v4_df))
joined = m.merge(v4_df, on=['filename','start_sec'], how='inner')
cache_idx = joined['cache_idx'].to_numpy()
v4_idx = joined['v4_idx'].to_numpy()

d_beta = np.load('/home/swatson/work/kaggle/BirdCLEF/four_track/data/d2_beta_oofs.npz', allow_pickle=True)
v4_perfold = v4['probs_per_fold']
v4_y = v4['y_true'].astype(np.float32)
a1_ranks_full = np.stack([rank01_per_col(v4_perfold[i]) for i in range(5)], axis=0).mean(axis=0).astype(np.float32)
a1_r = a1_ranks_full[v4_idx]
y = v4_y[v4_idx]
b1_r = rank01_per_col(d_beta['b1_oof'].astype(np.float32))[cache_idx]
proto_r = rank01_per_col(d_beta['proto_oof'].astype(np.float32))[cache_idx]

# A1-only
auc_a1_only, _ = macro_auc_skip_empty(y, a1_r)
auc_proto_only, _ = macro_auc_skip_empty(y, proto_r)
auc_b1_only, _ = macro_auc_skip_empty(y, b1_r)
print(f'A1-only       : {auc_a1_only:.4f}')
print(f'Proto-only    : {auc_proto_only:.4f}')
print(f'B1-only       : {auc_b1_only:.4f}')

# Extended sweep including A1 > 0.5
print('\nExtended sweep A1 ∈ [0..1.0], B1 ∈ [0..0.3]:')
best = (None, -1)
for a1_w in np.round(np.arange(0.0, 1.01, 0.05),2):
    for b1_w in np.round(np.arange(0.0, 0.31, 0.05),2):
        proto_w = 1.0 - a1_w - b1_w
        if proto_w < -1e-9: continue
        fused = a1_w * a1_r + b1_w * b1_r + proto_w * proto_r
        auc, _ = macro_auc_skip_empty(y, fused)
        if auc > best[1]: best = ((float(a1_w), float(b1_w), float(proto_w)), auc)
print(f'BEST extended: A1={best[0][0]:.2f} B1={best[0][1]:.2f} Proto={best[0][2]:.2f} → {best[1]:.4f}')

# Compare:
# (1) production weights (rank-fused)
fused_prod = 0.20*a1_r + 0.10*b1_r + 0.70*proto_r
auc_prod, _ = macro_auc_skip_empty(y, fused_prod)
# (2) drop B1, keep A1=0.20 (ie A1=0.20 / Proto=0.80)
fused_no_b1 = 0.20*a1_r + 0.0*b1_r + 0.80*proto_r
auc_no_b1, _ = macro_auc_skip_empty(y, fused_no_b1)
# (3) production but with stronger A1
fused_a1_strong = 0.40*a1_r + 0.10*b1_r + 0.50*proto_r
auc_a1_strong, _ = macro_auc_skip_empty(y, fused_a1_strong)

print(f'\nProduction (A1=0.20 B1=0.10 Proto=0.70): {auc_prod:.4f}')
print(f'Drop B1   (A1=0.20 B1=0.00 Proto=0.80): {auc_no_b1:.4f}')
print(f'A1 stronger (A1=0.40 B1=0.10 Proto=0.50): {auc_a1_strong:.4f}')

# A1+Proto only (no B1) finer
print('\nFine sweep A1+Proto (B1=0):')
for a1_w in np.round(np.arange(0.0, 1.01, 0.025),3):
    proto_w = 1.0 - a1_w
    fused = a1_w*a1_r + proto_w*proto_r
    auc, _ = macro_auc_skip_empty(y, fused)
    print(f'  A1={a1_w:.3f} Proto={proto_w:.3f}  AUC={auc:.4f}')
