#!/usr/bin/env python3
"""
Extended OOF grid search for Perch v2 probe hyperparameters.

Generates Perch cache (embeddings + logits) on labeled soundscapes if not
already cached, then runs a staged grid search over:
  1. Dual-probe blend weight (LogReg vs LightGBM)
  2. Smoothing alphas (texture + event)
  3. Prior lambdas (event + texture)
  4. Per-class isotonic calibration
  5. Ridge as third probe
  6. Probe blending alpha (base vs probe prediction)

Usage:
    python -u src/perch_oof_grid.py
"""

import gc
import json
import re
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────
BASE = Path("/home/swatson/work/MachineLearning/kaggle/BirdCLEF/data/raw")
MODEL_DIR = Path("/home/swatson/work/MachineLearning/kaggle/BirdCLEF/perch_v2/models/perch_v2")
CACHE_DIR = Path("/home/swatson/work/MachineLearning/kaggle/BirdCLEF/data/processed/perch_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SR = 32_000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12

# ── Load taxonomy ─────────────────────────────────────────────────────────
taxonomy = pd.read_csv(BASE / "taxonomy.csv")
sample_sub = pd.read_csv(BASE / "sample_submission.csv")
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
N_CLASSES = len(PRIMARY_LABELS)
label_to_idx = {l: i for i, l in enumerate(PRIMARY_LABELS)}

TEXTURE_TAXA = {"Amphibia", "Insecta"}

# ── Load Perch v2 via TFLite (CPU compatible) ────────────────────────────
print("Loading Perch v2 TFLite model...")
_tflite_path = str(MODEL_DIR / "model.tflite")
_interpreter = tf.lite.Interpreter(model_path=_tflite_path)
_interpreter.allocate_tensors()
_inp_detail = _interpreter.get_input_details()[0]
_out_details = {o["name"]: o for o in _interpreter.get_output_details()}
# Output names: StatefulPartitionedCall:0 = embedding (1536),
#               StatefulPartitionedCall:1 = logits (14795)
_emb_idx = _out_details["StatefulPartitionedCall:0"]["index"]
_logit_idx = _out_details["StatefulPartitionedCall:1"]["index"]

def perch_infer_batch(audio_batch):
    """Run TFLite inference on a batch of 5s audio windows. Returns logits, embeddings."""
    n = len(audio_batch)
    logits = np.zeros((n, 14795), dtype=np.float32)
    embs = np.zeros((n, 1536), dtype=np.float32)
    for i in range(n):
        _interpreter.set_tensor(_inp_detail["index"], audio_batch[i:i+1])
        _interpreter.invoke()
        embs[i] = _interpreter.get_tensor(_emb_idx)[0]
        logits[i] = _interpreter.get_tensor(_logit_idx)[0]
    return logits, embs

print("Perch TFLite loaded.")

# Get Perch label list (headerless CSV with first row 'inat2024_fsd50k')
bc_labels = (
    pd.read_csv(MODEL_DIR / "assets" / "labels.csv")
    .reset_index()
    .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
)
NO_LABEL_INDEX = len(bc_labels)

# Map competition labels to Perch indices via scientific_name merge
taxonomy_ = taxonomy.copy()
taxonomy_["scientific_name"] = taxonomy_["scientific_name"].astype(str)
mapping = taxonomy_.merge(
    bc_labels[["scientific_name", "bc_index"]],
    on="scientific_name", how="left",
)
mapping["bc_index"] = mapping["bc_index"].fillna(NO_LABEL_INDEX).astype(int)

label_to_bc = mapping.set_index("primary_label")["bc_index"]
BC_INDICES = np.array([int(label_to_bc.loc[c]) for c in PRIMARY_LABELS], dtype=np.int32)

MAPPED_MASK = BC_INDICES != NO_LABEL_INDEX
MAPPED_POS = np.where(MAPPED_MASK)[0].astype(np.int32)
UNMAPPED_POS = np.where(~MAPPED_MASK)[0].astype(np.int32)
MAPPED_BC = BC_INDICES[MAPPED_MASK].astype(np.int32)

CLASS_NAME_MAP = taxonomy_.set_index("primary_label")["class_name"].to_dict()

print(f"Mapped species: {MAPPED_MASK.sum()} / {N_CLASSES}")

# Proxy mapping for unmapped amphibians
proxy_map = {}
unmapped_df = mapping[mapping["bc_index"] == NO_LABEL_INDEX].copy()
unmapped_non_sonotype = unmapped_df[
    ~unmapped_df["primary_label"].astype(str).str.contains("son", na=False)
].copy()
for _, row in unmapped_non_sonotype.iterrows():
    genus = str(row["scientific_name"]).split()[0]
    hits = bc_labels[
        bc_labels["scientific_name"].str.match(rf"^{re.escape(genus)}\s", na=False)
    ]
    if len(hits) > 0:
        proxy_map[str(row["primary_label"])] = hits["bc_index"].astype(int).tolist()

SELECTED_PROXY_TARGETS = sorted(
    [t for t in proxy_map if CLASS_NAME_MAP.get(t) == "Amphibia"]
)
selected_proxy_pos = np.array(
    [label_to_idx[c] for c in SELECTED_PROXY_TARGETS], dtype=np.int32
)
selected_proxy_pos_to_bc = {
    label_to_idx[t]: np.array(proxy_map[t], dtype=np.int32)
    for t in SELECTED_PROXY_TARGETS
}

# ── Load soundscape labels (matching notebook logic exactly) ──────────────
FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")

def parse_labels(x):
    if pd.isna(x):
        return []
    return [t.strip() for t in str(x).split(";") if t.strip()]

def union_labels(series):
    return sorted(set(lbl for x in series for lbl in parse_labels(x)))

def parse_soundscape_filename(name):
    m = FNAME_RE.match(name)
    if not m:
        return {"site": None, "hour_utc": -1}
    _, site, _, hms = m.groups()
    return {"site": site, "hour_utc": int(hms[:2])}

soundscape_raw = pd.read_csv(BASE / "train_soundscapes_labels.csv")
soundscape_lbls = soundscape_raw.drop_duplicates().reset_index(drop=True)

sc_clean = (
    soundscape_lbls
    .groupby(["filename", "start", "end"])["primary_label"]
    .apply(union_labels)
    .reset_index(name="label_list")
)
sc_clean["end_sec"] = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
sc_clean["row_id"] = (
    sc_clean["filename"].str.replace(".ogg", "", regex=False)
    + "_" + sc_clean["end_sec"].astype(str)
)
meta_cols = sc_clean["filename"].apply(parse_soundscape_filename).apply(pd.Series)
sc_clean = pd.concat([sc_clean, meta_cols], axis=1)

# Identify fully-labeled files (all 12 windows annotated)
wpf = sc_clean.groupby("filename").size()
full_files = sorted(wpf[wpf == N_WINDOWS].index.tolist())
sc_clean["file_fully_labeled"] = sc_clean["filename"].isin(full_files)

# Multi-hot label matrix
Y_SC = np.zeros((len(sc_clean), N_CLASSES), dtype=np.uint8)
for i, labels in enumerate(sc_clean["label_list"]):
    for lbl in labels:
        if lbl in label_to_idx:
            Y_SC[i, label_to_idx[lbl]] = 1

# Ordered truth for fully-labeled files
full_truth = (
    sc_clean[sc_clean["file_fully_labeled"]]
    .sort_values(["filename", "end_sec"])
    .reset_index(drop=False)
)

# Active classes
ACTIVE_CLASSES = [PRIMARY_LABELS[i] for i in np.where(Y_SC.sum(axis=0) > 0)[0]]
idx_active_texture = np.array(
    [label_to_idx[c] for c in ACTIVE_CLASSES if CLASS_NAME_MAP.get(c) in TEXTURE_TAXA],
    dtype=np.int32,
)
idx_active_event = np.array(
    [label_to_idx[c] for c in ACTIVE_CLASSES if CLASS_NAME_MAP.get(c) not in TEXTURE_TAXA],
    dtype=np.int32,
)
idx_mapped_active_texture = idx_active_texture[MAPPED_MASK[idx_active_texture]]
idx_mapped_active_event = idx_active_event[MAPPED_MASK[idx_active_event]]
idx_unmapped_active_texture = idx_active_texture[~MAPPED_MASK[idx_active_texture]]
idx_unmapped_active_event = idx_active_event[~MAPPED_MASK[idx_active_event]]
idx_unmapped_inactive = np.array(
    [i for i in UNMAPPED_POS if PRIMARY_LABELS[i] not in ACTIVE_CLASSES], dtype=np.int32
)
idx_selected_proxy_active_texture = np.intersect1d(selected_proxy_pos, idx_active_texture)
idx_selected_prioronly_active_texture = np.setdiff1d(
    idx_unmapped_active_texture, selected_proxy_pos
)
idx_selected_prioronly_active_event = np.setdiff1d(
    idx_unmapped_active_event, selected_proxy_pos
)

print(f"Active texture: {len(idx_active_texture)}, Active event: {len(idx_active_event)}")
print(f"Total labeled windows: {len(sc_clean)}")

# ── Perch inference or cache load ─────────────────────────────────────────
cache_meta = CACHE_DIR / "full_perch_meta.parquet"
cache_arrays = CACHE_DIR / "full_perch_arrays.npz"

if cache_meta.exists() and cache_arrays.exists():
    print(f"Loading Perch cache from {CACHE_DIR}")
    meta_full = pd.read_parquet(cache_meta)
    arr = np.load(cache_arrays)
    scores_full_raw = arr["scores_full_raw"].astype(np.float32)
    emb_full = arr["emb_full"].astype(np.float32)
else:
    print("Generating Perch cache on labeled soundscapes...")
    n_files = len(full_files)
    n_rows = n_files * N_WINDOWS

    row_ids = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites = np.empty(n_rows, dtype=object)
    hours = np.empty(n_rows, dtype=np.int16)
    scores = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
    embeddings = np.zeros((n_rows, 1536), dtype=np.float32)

    write_row = 0
    BATCH_FILES = 8

    for start in tqdm(range(0, n_files, BATCH_FILES), desc="Perch inference"):
        batch = full_files[start : start + BATCH_FILES]
        bn = len(batch)
        x = np.empty((bn * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
        bstart = write_row

        for bi, fn in enumerate(batch):
            path = BASE / "train_soundscapes" / fn
            y, sr = sf.read(path, dtype="float32", always_2d=False)
            if y.ndim == 2:
                y = y.mean(axis=1)
            if len(y) < FILE_SAMPLES:
                y = np.pad(y, (0, FILE_SAMPLES - len(y)))
            y = y[:FILE_SAMPLES]
            x[bi * N_WINDOWS : (bi + 1) * N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)

            meta = parse_soundscape_filename(fn)
            site, hour = meta["site"], meta["hour_utc"]
            row_ids[write_row : write_row + N_WINDOWS] = [
                f"{fn.replace('.ogg', '')}_{t}" for t in range(5, 65, 5)
            ]
            filenames[write_row : write_row + N_WINDOWS] = fn
            sites[write_row : write_row + N_WINDOWS] = site
            hours[write_row : write_row + N_WINDOWS] = hour
            write_row += N_WINDOWS

        logits, emb = perch_infer_batch(x[: bn * N_WINDOWS])

        n_actual = write_row - bstart
        scores[bstart:write_row, MAPPED_POS] = logits[:n_actual, MAPPED_BC]
        embeddings[bstart:write_row] = emb[:n_actual]

        # Proxy scores for unmapped amphibians
        for comp_idx, bc_indices in selected_proxy_pos_to_bc.items():
            scores[bstart:write_row, comp_idx] = logits[:n_actual, bc_indices].max(axis=1)

    scores = scores[:write_row]
    embeddings = embeddings[:write_row]

    meta_full = pd.DataFrame({
        "row_id": row_ids[:write_row],
        "filename": filenames[:write_row],
        "site": sites[:write_row],
        "hour_utc": hours[:write_row],
    })

    scores_full_raw = scores
    emb_full = embeddings

    meta_full.to_parquet(cache_meta, index=False)
    np.savez_compressed(cache_arrays, scores_full_raw=scores_full_raw, emb_full=emb_full)
    print(f"Cache saved to {CACHE_DIR}")

# Align Y_FULL to cache row order
full_truth_aligned = (
    full_truth[["row_id", "index"]]
    .set_index("row_id")
    .loc[meta_full["row_id"]]
    .reset_index(drop=False)
)
Y_FULL = Y_SC[full_truth_aligned["index"].to_numpy()]

print(f"scores_full_raw: {scores_full_raw.shape}")
print(f"emb_full:        {emb_full.shape}")
print(f"Y_FULL:          {Y_FULL.shape}")

# Free model memory
del _interpreter
gc.collect()


# ── Prior tables ──────────────────────────────────────────────────────────
def fit_prior_tables(prior_df, Y_prior):
    prior_df = prior_df.reset_index(drop=True)
    global_p = Y_prior.mean(axis=0).astype(np.float32)

    site_keys = sorted(prior_df["site"].dropna().astype(str).unique())
    hour_keys = sorted(prior_df["hour_utc"].dropna().astype(int).unique())

    site_to_i, site_n, site_p = {}, [], []
    for s in site_keys:
        mask = prior_df["site"].astype(str).values == s
        site_to_i[s] = len(site_n)
        site_n.append(mask.sum())
        site_p.append(Y_prior[mask].mean(axis=0))
    site_n = np.array(site_n, dtype=np.float32)
    site_p = np.stack(site_p).astype(np.float32) if site_p else np.zeros((0, Y_prior.shape[1]), np.float32)

    hour_to_i, hour_n, hour_p = {}, [], []
    for h in hour_keys:
        mask = prior_df["hour_utc"].astype(int).values == h
        hour_to_i[h] = len(hour_n)
        hour_n.append(mask.sum())
        hour_p.append(Y_prior[mask].mean(axis=0))
    hour_n = np.array(hour_n, dtype=np.float32)
    hour_p = np.stack(hour_p).astype(np.float32) if hour_p else np.zeros((0, Y_prior.shape[1]), np.float32)

    sh_to_i, sh_n_list, sh_p_list = {}, [], []
    for (s, h), idx in prior_df.groupby(["site", "hour_utc"]).groups.items():
        sh_to_i[(str(s), int(h))] = len(sh_n_list)
        idx = np.array(list(idx))
        sh_n_list.append(len(idx))
        sh_p_list.append(Y_prior[idx].mean(axis=0))
    sh_n = np.array(sh_n_list, dtype=np.float32)
    sh_p = np.stack(sh_p_list).astype(np.float32) if sh_p_list else np.zeros((0, Y_prior.shape[1]), np.float32)

    return dict(
        global_p=global_p,
        site_to_i=site_to_i, site_n=site_n, site_p=site_p,
        hour_to_i=hour_to_i, hour_n=hour_n, hour_p=hour_p,
        sh_to_i=sh_to_i, sh_n=sh_n, sh_p=sh_p,
    )


def prior_logits(sites, hours, tables, eps=1e-4):
    n = len(sites)
    p = np.repeat(tables["global_p"][None, :], n, axis=0).astype(np.float32, copy=True)

    si = np.fromiter((tables["site_to_i"].get(str(s), -1) for s in sites), np.int32, n)
    hi = np.fromiter(
        (tables["hour_to_i"].get(int(h), -1) if int(h) >= 0 else -1 for h in hours),
        np.int32, n,
    )
    shi = np.fromiter(
        (
            tables["sh_to_i"].get((str(s), int(h)), -1) if int(h) >= 0 else -1
            for s, h in zip(sites, hours)
        ),
        np.int32, n,
    )

    valid = hi >= 0
    if valid.any():
        nh = tables["hour_n"][hi[valid]][:, None]
        p[valid] = nh / (nh + 8.0) * tables["hour_p"][hi[valid]] + (1.0 - nh / (nh + 8.0)) * p[valid]

    valid = si >= 0
    if valid.any():
        ns = tables["site_n"][si[valid]][:, None]
        p[valid] = ns / (ns + 8.0) * tables["site_p"][si[valid]] + (1.0 - ns / (ns + 8.0)) * p[valid]

    valid = shi >= 0
    if valid.any():
        nsh = tables["sh_n"][shi[valid]][:, None]
        p[valid] = nsh / (nsh + 4.0) * tables["sh_p"][shi[valid]] + (1.0 - nsh / (nsh + 4.0)) * p[valid]

    np.clip(p, eps, 1.0 - eps, out=p)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


def smooth_cols(scores, cols, alpha=0.35):
    if alpha <= 0 or len(cols) == 0:
        return scores.copy()
    s = scores.copy()
    view = s.reshape(-1, N_WINDOWS, s.shape[1])
    x = view[:, :, cols]
    prev = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
    nxt = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
    view[:, :, cols] = (1.0 - alpha) * x + 0.5 * alpha * (prev + nxt)
    return s


def smooth_events(scores, cols, alpha=0.15):
    if alpha <= 0 or len(cols) == 0:
        return scores.copy()
    s = scores.copy()
    view = s.reshape(-1, N_WINDOWS, s.shape[1])
    x = view[:, :, cols]
    prev = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
    nxt = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
    local_max = np.maximum(x, np.maximum(prev, nxt))
    view[:, :, cols] = (1.0 - alpha) * x + alpha * local_max
    return s


# ── Feature building ─────────────────────────────────────────────────────
def seq_features_1d(v):
    x = v.reshape(-1, N_WINDOWS)
    prev = np.concatenate([x[:, :1], x[:, :-1]], axis=1).reshape(-1)
    nxt = np.concatenate([x[:, 1:], x[:, -1:]], axis=1).reshape(-1)
    file_mean = np.repeat(x.mean(1), N_WINDOWS)
    file_max = np.repeat(x.max(1), N_WINDOWS)
    file_std = np.repeat(x.std(1), N_WINDOWS)
    padded = np.pad(x, ((0, 0), (1, 1)), mode="edge")
    local3 = np.array([padded[:, i : i + 3].mean(1) for i in range(N_WINDOWS)]).T.reshape(-1)
    delta = (x - np.concatenate([x[:, :1], x[:, :-1]], axis=1)).reshape(-1)
    return prev, nxt, file_mean, file_max, file_std, local3, delta


def build_class_features(Z, raw_col, prior_col, base_col):
    p, n, m, mx, std, l3, delta = seq_features_1d(base_col)
    _, _, _, raw_mx, _, _, _ = seq_features_1d(raw_col)
    return np.concatenate(
        [
            Z,
            raw_col[:, None], prior_col[:, None], base_col[:, None],
            p[:, None], n[:, None], m[:, None], mx[:, None],
            std[:, None], l3[:, None], delta[:, None], raw_mx[:, None],
        ],
        axis=1,
    ).astype(np.float32)


def macro_auc(y_true, y_score):
    keep = y_true.sum(axis=0) > 0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")


# ── OOF pipeline ─────────────────────────────────────────────────────────
def oof_pipeline(
    pca_dim=32, probe_alpha=0.40, lr_C=0.25, min_pos=8,
    lgbm_n_est=200, lgbm_depth=4, lgbm_lr=0.05,
    dual_weight_lr=0.5,
    use_ridge=False, ridge_alpha=1.0, ridge_weight=0.0,
    smooth_texture_alpha=0.35, smooth_event_alpha=0.15,
    lambda_event=0.4, lambda_texture=1.0, lambda_proxy_texture=0.8,
    use_isotonic=False,
):
    _scaler = StandardScaler()
    _pca = PCA(n_components=min(pca_dim, emb_full.shape[0] - 1, emb_full.shape[1]))
    _Z = _pca.fit_transform(_scaler.fit_transform(emb_full)).astype(np.float32)
    _gkf = GroupKFold(n_splits=5)
    _groups = meta_full["site"].to_numpy()

    _oof_base = np.zeros_like(scores_full_raw, dtype=np.float32)
    _oof_prior = np.zeros_like(scores_full_raw, dtype=np.float32)
    fold_indices = list(_gkf.split(scores_full_raw, groups=_groups))

    for _, va_idx in fold_indices:
        va_idx = np.sort(va_idx)
        val_sites = set(meta_full.iloc[va_idx]["site"].tolist())
        prior_m = ~sc_clean["site"].isin(val_sites).values
        tables = fit_prior_tables(
            sc_clean.loc[prior_m].reset_index(drop=True), Y_SC[prior_m]
        )
        _scores = scores_full_raw[va_idx].copy()
        _prior = prior_logits(
            meta_full.iloc[va_idx]["site"].to_numpy(),
            meta_full.iloc[va_idx]["hour_utc"].to_numpy(),
            tables,
        )
        if len(idx_mapped_active_event):
            _scores[:, idx_mapped_active_event] += lambda_event * _prior[:, idx_mapped_active_event]
        if len(idx_mapped_active_texture):
            _scores[:, idx_mapped_active_texture] += lambda_texture * _prior[:, idx_mapped_active_texture]
        if len(idx_selected_proxy_active_texture):
            _scores[:, idx_selected_proxy_active_texture] += (
                lambda_proxy_texture * _prior[:, idx_selected_proxy_active_texture]
            )
        if len(idx_selected_prioronly_active_event):
            _scores[:, idx_selected_prioronly_active_event] = (
                lambda_event * _prior[:, idx_selected_prioronly_active_event]
            )
        if len(idx_selected_prioronly_active_texture):
            _scores[:, idx_selected_prioronly_active_texture] = (
                lambda_texture * _prior[:, idx_selected_prioronly_active_texture]
            )
        if len(idx_unmapped_inactive):
            _scores[:, idx_unmapped_inactive] = -8.0
        _scores = smooth_cols(_scores, idx_active_texture, alpha=smooth_texture_alpha)
        _scores = smooth_events(_scores, idx_active_event, alpha=smooth_event_alpha)
        _oof_base[va_idx] = _scores.astype(np.float32)
        _oof_prior[va_idx] = _prior.astype(np.float32)

    # Train probes per fold
    _oof_lr = _oof_base.copy()
    _oof_lgbm = _oof_base.copy()
    _oof_ridge = _oof_base.copy() if use_ridge else None

    for _, va_idx in fold_indices:
        tr_idx = np.setdiff1d(np.arange(len(scores_full_raw)), va_idx)
        pos_cnt = Y_FULL[tr_idx].sum(axis=0)
        for ci in np.where(pos_cnt >= min_pos)[0]:
            y_tr = Y_FULL[tr_idx, ci]
            if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
                continue
            X_tr = build_class_features(
                _Z[tr_idx], scores_full_raw[tr_idx, ci],
                _oof_prior[tr_idx, ci], _oof_base[tr_idx, ci],
            )
            X_va = build_class_features(
                _Z[va_idx], scores_full_raw[va_idx, ci],
                _oof_prior[va_idx, ci], _oof_base[va_idx, ci],
            )
            # LogReg
            clf_lr = LogisticRegression(
                C=lr_C, max_iter=400, solver="liblinear", class_weight="balanced"
            )
            clf_lr.fit(X_tr, y_tr)
            pred_lr = clf_lr.decision_function(X_va).astype(np.float32)
            _oof_lr[va_idx, ci] = (
                (1.0 - probe_alpha) * _oof_base[va_idx, ci] + probe_alpha * pred_lr
            )
            # LightGBM
            n_pos_tr = int(y_tr.sum())
            n_neg_tr = len(y_tr) - n_pos_tr
            scale_tr = n_neg_tr / max(n_pos_tr, 1)
            clf_lgbm = lgb.LGBMClassifier(
                n_estimators=lgbm_n_est, max_depth=lgbm_depth,
                learning_rate=lgbm_lr, min_child_samples=5,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_tr,
                random_state=42, verbose=-1, n_jobs=1,
            )
            clf_lgbm.fit(X_tr, y_tr)
            pred_lgbm = clf_lgbm.predict_proba(X_va)[:, 1].astype(np.float32)
            pred_lgbm = np.log(pred_lgbm / (1.0 - pred_lgbm + 1e-7))
            _oof_lgbm[va_idx, ci] = (
                (1.0 - probe_alpha) * _oof_base[va_idx, ci] + probe_alpha * pred_lgbm
            )
            # Ridge
            if use_ridge:
                clf_ridge = RidgeClassifier(alpha=ridge_alpha, class_weight="balanced")
                clf_ridge.fit(X_tr, y_tr)
                pred_ridge = clf_ridge.decision_function(X_va).astype(np.float32)
                _oof_ridge[va_idx, ci] = (
                    (1.0 - probe_alpha) * _oof_base[va_idx, ci] + probe_alpha * pred_ridge
                )

    # Rank-average with tunable weights
    n = _oof_lr.shape[0]
    _oof_final = np.empty_like(_oof_lr)
    for c in range(_oof_lr.shape[1]):
        r_lr = rankdata(_oof_lr[:, c]) / n
        r_lgbm = rankdata(_oof_lgbm[:, c]) / n
        if use_ridge and ridge_weight > 0:
            r_ridge = rankdata(_oof_ridge[:, c]) / n
            w_lr = dual_weight_lr * (1.0 - ridge_weight)
            w_lgbm = (1.0 - dual_weight_lr) * (1.0 - ridge_weight)
            _oof_final[:, c] = w_lr * r_lr + w_lgbm * r_lgbm + ridge_weight * r_ridge
        else:
            _oof_final[:, c] = dual_weight_lr * r_lr + (1.0 - dual_weight_lr) * r_lgbm

    # Per-class isotonic calibration
    if use_isotonic:
        _oof_cal = _oof_final.copy()
        for _, va_idx in fold_indices:
            tr_idx = np.setdiff1d(np.arange(n), va_idx)
            for ci in range(_oof_final.shape[1]):
                if Y_FULL[tr_idx, ci].sum() < 3:
                    continue
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(_oof_final[tr_idx, ci], Y_FULL[tr_idx, ci])
                _oof_cal[va_idx, ci] = iso.predict(_oof_final[va_idx, ci])
        _oof_final = _oof_cal

    return macro_auc(Y_FULL, _oof_final)


# ── Run staged grid search ───────────────────────────────────────────────
t0 = time.time()

# Stage 1: Blend weight
print("\n" + "=" * 70)
print("Stage 1: Dual-probe blend weight (LR vs LGBM)")
print("=" * 70)
blend_results = []
for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
    auc = oof_pipeline(dual_weight_lr=w)
    blend_results.append({"lr_weight": w, "auc": auc})
    print(f"  LR weight={w:.1f}  LGBM weight={1-w:.1f}  AUC={auc:.6f}")
best_w = max(blend_results, key=lambda x: x["auc"])["lr_weight"]
print(f"  >>> Best blend weight: LR={best_w:.1f}")

# Stage 2: Smoothing alphas
print("\n" + "=" * 70)
print("Stage 2: Smoothing alpha sweep")
print("=" * 70)
smooth_results = []
for tex_a in [0.20, 0.30, 0.35, 0.40, 0.50]:
    for ev_a in [0.05, 0.10, 0.15, 0.20, 0.25]:
        auc = oof_pipeline(
            dual_weight_lr=best_w,
            smooth_texture_alpha=tex_a, smooth_event_alpha=ev_a,
        )
        smooth_results.append({"tex_alpha": tex_a, "ev_alpha": ev_a, "auc": auc})
        print(f"  texture={tex_a:.2f} event={ev_a:.2f}  AUC={auc:.6f}")
best_smooth = max(smooth_results, key=lambda x: x["auc"])
best_tex_a = best_smooth["tex_alpha"]
best_ev_a = best_smooth["ev_alpha"]
print(f"  >>> Best smoothing: texture={best_tex_a:.2f} event={best_ev_a:.2f}")

# Stage 3: Prior lambdas
print("\n" + "=" * 70)
print("Stage 3: Prior lambda sweep")
print("=" * 70)
lambda_results = []
for le in [0.2, 0.3, 0.4, 0.5, 0.6]:
    for lt in [0.6, 0.8, 1.0, 1.2, 1.4]:
        auc = oof_pipeline(
            dual_weight_lr=best_w,
            smooth_texture_alpha=best_tex_a, smooth_event_alpha=best_ev_a,
            lambda_event=le, lambda_texture=lt,
        )
        lambda_results.append({"lambda_ev": le, "lambda_tex": lt, "auc": auc})
        print(f"  lambda_event={le:.1f} lambda_texture={lt:.1f}  AUC={auc:.6f}")
best_lambda = max(lambda_results, key=lambda x: x["auc"])
best_le = best_lambda["lambda_ev"]
best_lt = best_lambda["lambda_tex"]
print(f"  >>> Best lambdas: event={best_le:.1f} texture={best_lt:.1f}")

# Stage 4: Isotonic calibration
print("\n" + "=" * 70)
print("Stage 4: Per-class isotonic calibration")
print("=" * 70)
auc_no_iso = oof_pipeline(
    dual_weight_lr=best_w,
    smooth_texture_alpha=best_tex_a, smooth_event_alpha=best_ev_a,
    lambda_event=best_le, lambda_texture=best_lt,
    use_isotonic=False,
)
auc_iso = oof_pipeline(
    dual_weight_lr=best_w,
    smooth_texture_alpha=best_tex_a, smooth_event_alpha=best_ev_a,
    lambda_event=best_le, lambda_texture=best_lt,
    use_isotonic=True,
)
print(f"  Without isotonic: {auc_no_iso:.6f}")
print(f"  With isotonic:    {auc_iso:.6f}")
use_iso = auc_iso > auc_no_iso
print(f"  >>> Use isotonic: {use_iso}")

# Stage 5: Ridge as third probe
print("\n" + "=" * 70)
print("Stage 5: Ridge as third probe")
print("=" * 70)
ridge_results = []
for rw in [0.0, 0.10, 0.20, 0.30]:
    for ra in [0.1, 1.0, 10.0]:
        if rw == 0.0 and ra != 0.1:
            continue
        auc = oof_pipeline(
            dual_weight_lr=best_w,
            smooth_texture_alpha=best_tex_a, smooth_event_alpha=best_ev_a,
            lambda_event=best_le, lambda_texture=best_lt,
            use_isotonic=use_iso,
            use_ridge=(rw > 0), ridge_weight=rw, ridge_alpha=ra,
        )
        ridge_results.append({"ridge_w": rw, "ridge_alpha": ra, "auc": auc})
        print(f"  ridge_weight={rw:.2f} ridge_alpha={ra:.1f}  AUC={auc:.6f}")
best_ridge = max(ridge_results, key=lambda x: x["auc"])
print(f"  >>> Best Ridge: weight={best_ridge['ridge_w']:.2f} alpha={best_ridge['ridge_alpha']:.1f}")

# Stage 6: Probe alpha
print("\n" + "=" * 70)
print("Stage 6: Probe blending alpha (base vs probe prediction)")
print("=" * 70)
_use_ridge = best_ridge["ridge_w"] > 0
alpha_results = []
for pa in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    auc = oof_pipeline(
        dual_weight_lr=best_w, probe_alpha=pa,
        smooth_texture_alpha=best_tex_a, smooth_event_alpha=best_ev_a,
        lambda_event=best_le, lambda_texture=best_lt,
        use_isotonic=use_iso,
        use_ridge=_use_ridge,
        ridge_weight=best_ridge["ridge_w"],
        ridge_alpha=best_ridge["ridge_alpha"],
    )
    alpha_results.append({"probe_alpha": pa, "auc": auc})
    print(f"  probe_alpha={pa:.2f}  AUC={auc:.6f}")
best_pa = max(alpha_results, key=lambda x: x["auc"])["probe_alpha"]
print(f"  >>> Best probe_alpha: {best_pa:.2f}")

# Final summary
final_auc = oof_pipeline(
    dual_weight_lr=best_w, probe_alpha=best_pa,
    smooth_texture_alpha=best_tex_a, smooth_event_alpha=best_ev_a,
    lambda_event=best_le, lambda_texture=best_lt,
    use_isotonic=use_iso,
    use_ridge=_use_ridge,
    ridge_weight=best_ridge["ridge_w"],
    ridge_alpha=best_ridge["ridge_alpha"],
)

elapsed = time.time() - t0
mins, secs = divmod(elapsed, 60)

print("\n" + "=" * 70)
print("FINAL BEST CONFIG")
print("=" * 70)
print(f"  dual_weight_lr     = {best_w}")
print(f"  probe_alpha        = {best_pa}")
print(f"  smooth_texture     = {best_tex_a}")
print(f"  smooth_event       = {best_ev_a}")
print(f"  lambda_event       = {best_le}")
print(f"  lambda_texture     = {best_lt}")
print(f"  use_isotonic       = {use_iso}")
print(f"  use_ridge          = {_use_ridge}")
if _use_ridge:
    print(f"  ridge_weight       = {best_ridge['ridge_w']}")
    print(f"  ridge_alpha        = {best_ridge['ridge_alpha']}")
print(f"  OOF Macro AUC      = {final_auc:.6f}")
print(f"  Total time         = {int(mins)}m {int(secs)}s")
print("=" * 70)

# Compare to current baseline
baseline_auc = oof_pipeline()
print(f"\nBaseline (current Settings) OOF AUC: {baseline_auc:.6f}")
print(f"Best config OOF AUC:                 {final_auc:.6f}")
print(f"Delta:                               {final_auc - baseline_auc:+.6f}")
