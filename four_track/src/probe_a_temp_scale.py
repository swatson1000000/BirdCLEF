"""Probe A — per-fold temperature scaling on A1 5-fold OOF (D1-b)."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

D = np.load('/home/swatson/work/kaggle/BirdCLEF/four_track/data/v4_5fold_soundscape_oof.npz', allow_pickle=True)
y = D['y_true']
ppf = D['probs_per_fold'].astype(np.float64)
filenames = D['filenames']
print(f'A1 5-fold OOF: {ppf.shape}, y {y.shape}', flush=True)

unique_files = np.unique(filenames)
N_CV = 5
rng = np.random.RandomState(42)
file_perm = rng.permutation(len(unique_files))
file_to_cvfold = {}
bin_size = len(unique_files) // N_CV
for f in range(N_CV):
    start = f * bin_size
    end = (f + 1) * bin_size if f < N_CV - 1 else len(unique_files)
    for fidx in file_perm[start:end]:
        file_to_cvfold[unique_files[fidx]] = f
cv_fold_ids = np.array([file_to_cvfold[fn] for fn in filenames], dtype=np.int64)


def macro_auc(y, p):
    n_pos = y.sum(axis=0).astype(int)
    active = np.where((n_pos > 0) & (n_pos < y.shape[0]))[0]
    aucs = [roc_auc_score(y[:, c], p[:, c]) for c in active]
    return float(np.mean(aucs)), len(active)


def sig_to_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def logit_to_sig(l):
    return 1.0 / (1.0 + np.exp(-l))


def fit_T(logits, targets, mask_active):
    L = logits[:, mask_active].ravel()
    Y = targets[:, mask_active].ravel()
    def nll(logT):
        T = np.exp(logT)
        sl = L / T
        return float(np.mean(np.maximum(sl, 0) - sl * Y + np.log1p(np.exp(-np.abs(sl)))))
    res = minimize_scalar(nll, bounds=(np.log(0.1), np.log(10.0)), method='bounded')
    return float(np.exp(res.x))


n_pos_all = y.sum(axis=0).astype(int)
active_mask = (n_pos_all > 0) & (n_pos_all < y.shape[0])
print(f'Active classes: {active_mask.sum()}/{y.shape[1]}', flush=True)

print('\n=== Baselines ===', flush=True)
auc_rank, _ = macro_auc(y, np.stack([rankdata(ppf[f], axis=0) for f in range(5)]).mean(axis=0))
auc_sig, _ = macro_auc(y, ppf.mean(axis=0))
print(f'  rank-mean (production):   {auc_rank:.4f}', flush=True)
print(f'  sigmoid-mean:             {auc_sig:.4f}  Δ vs prod: {auc_sig - auc_rank:+.4f}', flush=True)

print('\n=== CV-LOFO fit per-fold T ===', flush=True)
ppf_logits = sig_to_logit(ppf)
ppf_cal = np.zeros_like(ppf_logits)
T_log = np.zeros((N_CV, 5))
for k in range(N_CV):
    mask_tr = cv_fold_ids != k
    mask_va = cv_fold_ids == k
    for mf in range(5):
        T = fit_T(ppf_logits[mf, mask_tr], y[mask_tr], active_mask)
        T_log[k, mf] = T
        ppf_cal[mf, mask_va] = ppf_logits[mf, mask_va] / T

cal_sig = logit_to_sig(ppf_cal).mean(axis=0)
auc_temp_sig, _ = macro_auc(y, cal_sig)
print(f'  T per (cv_fold, model_fold):', flush=True)
for k in range(N_CV):
    print(f'    CV{k}: T = {T_log[k]}', flush=True)
print(f'  temp-scaled sigmoid-mean: {auc_temp_sig:.4f}  '
      f'Δ vs prod: {auc_temp_sig - auc_rank:+.4f}', flush=True)

# Sanity: temp-scaled rank-mean should equal rank-mean (rank-invariant)
ranks_cal = np.stack([rankdata(logit_to_sig(ppf_cal[f]), axis=0) for f in range(5)]).mean(axis=0)
auc_temp_rank, _ = macro_auc(y, ranks_cal)
print(f'  temp-scaled rank-mean:    {auc_temp_rank:.4f}  (rank-invariant sanity check)', flush=True)

print('\n=== SUMMARY (gate threshold: +0.005 vs prod 0.7672) ===', flush=True)
print(f'  production rank-mean:       {auc_rank:.4f}  baseline', flush=True)
print(f'  sigmoid-mean (no T):        {auc_sig:.4f}  Δ={auc_sig-auc_rank:+.4f}  '
      f'{"★ GATE-PASS" if (auc_sig-auc_rank)>=0.005 else "noise"}', flush=True)
print(f'  temp-scaled sigmoid-mean:   {auc_temp_sig:.4f}  Δ={auc_temp_sig-auc_rank:+.4f}  '
      f'{"★ GATE-PASS" if (auc_temp_sig-auc_rank)>=0.005 else "noise"}', flush=True)

# Refit T on ALL OOF data (final calibrators for inference)
print('\n=== Final T per fold (fit on all OOF) ===', flush=True)
T_final = np.zeros(5)
for mf in range(5):
    T_final[mf] = fit_T(ppf_logits[mf], y, active_mask)
print(f'  T_final = {T_final}', flush=True)

np.savez_compressed(
    '/home/swatson/work/kaggle/BirdCLEF/four_track/data/probe_a_temp_scale.npz',
    T_final=T_final, T_cv=T_log,
    auc_prod_rank=auc_rank, auc_sig=auc_sig, auc_temp_sig=auc_temp_sig,
)
print('saved → data/probe_a_temp_scale.npz', flush=True)
