"""
Train ProtoSSM v4 locally on Kaggle-extracted Perch features.

Uses the jaejohn/perch-meta dataset (extracted on Kaggle) to avoid the
local-vs-Kaggle embedding mismatch. Trains with extended budget (200+
epochs, multi-seed, hyperparameter sweeps) that the 90-min Kaggle CPU
cap doesn't allow.

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle
    cd /home/swatson/work/kaggle/BirdCLEF/four_track
    python -u src/train_protossm_local.py [--epochs 200] [--seeds 3] [--d-model 320]
"""

import argparse
import gc
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent          # four_track/
BIRDCLEF = PROJECT.parent                                  # BirdCLEF/

PERCH_CACHE   = PROJECT / "data" / "kaggle_perch_cache"
RAW_DATA      = BIRDCLEF / "data" / "raw"
CKPT_DIR      = PROJECT / "models" / "protossm_pretrained_v2"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_WINDOWS = 12

# ── Model definitions (exact copy from notebook Cell 22) ──────────────

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, d_conv,
                                padding=d_conv - 1, groups=d_model)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B_size, T, D = x.shape
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)
        dt = F.softplus(self.dt_proj(x_conv))
        A = -torch.exp(self.A_log)
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)
        h = torch.zeros(B_size, D, self.d_state, device=x.device)
        ys = []
        for t in range(T):
            dt_t = dt[:, t, :]
            dA = torch.exp(A[None, :, :] * dt_t[:, :, None])
            dB = dt_t[:, :, None] * B[:, t, None, :]
            h = h * dA + x[:, t, :, None] * dB
            y_t = (h * C[:, t, None, :]).sum(-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        return y + x * self.D[None, None, :]


class TemporalCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + attn_out
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return x


class ProtoSSMv2(nn.Module):
    def __init__(self, d_input=1536, d_model=192, d_state=16,
                 n_ssm_layers=2, n_classes=234, n_windows=12,
                 dropout=0.2, n_sites=20, meta_dim=16,
                 use_cross_attn=True, cross_attn_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.n_windows = n_windows
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)
        self.ssm_fwd = nn.ModuleList()
        self.ssm_bwd = nn.ModuleList()
        self.ssm_merge = nn.ModuleList()
        self.ssm_norm = nn.ModuleList()
        for _ in range(n_ssm_layers):
            self.ssm_fwd.append(SelectiveSSM(d_model, d_state))
            self.ssm_bwd.append(SelectiveSSM(d_model, d_state))
            self.ssm_merge.append(nn.Linear(2 * d_model, d_model))
            self.ssm_norm.append(nn.LayerNorm(d_model))
        self.ssm_drop = nn.Dropout(dropout)
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = TemporalCrossAttention(d_model, n_heads=cross_attn_heads, dropout=dropout)
        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp = nn.Parameter(torch.tensor(5.0))
        self.class_bias = nn.Parameter(torch.zeros(n_classes))
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))
        self.n_families = 0
        self.family_head = None

    def init_prototypes_from_data(self, embeddings, labels):
        with torch.no_grad():
            h = self.input_proj(embeddings)
            for c in range(self.n_classes):
                mask = labels[:, c] > 0.5
                if mask.sum() > 0:
                    self.prototypes.data[c] = F.normalize(h[mask].mean(0), dim=0)

    def init_family_head(self, n_families, class_to_family):
        self.n_families = n_families
        self.family_head = nn.Linear(self.d_model, n_families)
        self.register_buffer('class_to_family', torch.tensor(class_to_family, dtype=torch.long))

    def forward(self, emb, perch_logits=None, site_ids=None, hours=None):
        B, T, _ = emb.shape
        h = self.input_proj(emb)
        h = h + self.pos_enc[:, :T, :]
        if site_ids is not None and hours is not None:
            s_emb = self.site_emb(site_ids)
            h_emb = self.hour_emb(hours)
            meta = self.meta_proj(torch.cat([s_emb, h_emb], dim=-1))
            h = h + meta[:, None, :]
        for fwd, bwd, merge, norm in zip(
            self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm
        ):
            residual = h
            h_f = fwd(h)
            h_b = bwd(h.flip(1)).flip(1)
            h = merge(torch.cat([h_f, h_b], dim=-1))
            h = self.ssm_drop(h)
            h = norm(h + residual)
        if self.use_cross_attn:
            h = self.cross_attn(h)
        h_norm = F.normalize(h, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        temp = F.softplus(self.proto_temp)
        sim = torch.matmul(h_norm, p_norm.T) * temp + self.class_bias[None, None, :]
        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
            species_logits = alpha * sim + (1 - alpha) * perch_logits
        else:
            species_logits = sim
        family_logits = None
        if self.family_head is not None:
            h_pool = h.mean(dim=1)
            family_logits = self.family_head(h_pool)
        return species_logits, family_logits, h

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Loss functions ────────────────────────────────────────────────────

def focal_bce_with_logits(logits, targets, gamma=2.0, pos_weight=None, reduction="mean"):
    if pos_weight is not None:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction="none")
    else:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = targets * p + (1 - targets) * (1 - p)
    focal_weight = (1 - pt) ** gamma
    loss = focal_weight * bce
    if reduction == "mean":
        return loss.mean()
    return loss


def macro_auc_skip_empty(y_true, y_score):
    keep = y_true.sum(axis=0) > 0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")


# ── TTA ───────────────────────────────────────────────────────────────

def temporal_shift_tta(emb_files, logits_files, model, site_ids, hours,
                       shifts=(0, 1, -1), device=DEVICE):
    all_preds = []
    model.eval()
    for shift in shifts:
        e = np.roll(emb_files, shift, axis=1) if shift != 0 else emb_files
        l = np.roll(logits_files, shift, axis=1) if shift != 0 else logits_files
        with torch.no_grad():
            out, _, _ = model(
                torch.tensor(e, dtype=torch.float32).to(device),
                torch.tensor(l, dtype=torch.float32).to(device),
                site_ids=torch.tensor(site_ids, dtype=torch.long).to(device),
                hours=torch.tensor(hours, dtype=torch.long).to(device),
            )
            pred = out.cpu().numpy()
        if shift != 0:
            pred = np.roll(pred, -shift, axis=1)
        all_preds.append(pred)
    return np.mean(all_preds, axis=0)


# ── Data helpers ──────────────────────────────────────────────────────

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")


def reshape_to_files(flat_array, meta_df, n_windows=N_WINDOWS):
    filenames = meta_df["filename"].to_numpy()
    unique_files = list(dict.fromkeys(filenames))
    n_files = len(unique_files)
    assert len(flat_array) == n_files * n_windows
    new_shape = (n_files, n_windows) + flat_array.shape[1:]
    return flat_array.reshape(new_shape), unique_files


def build_taxonomy_groups(taxonomy_df, primary_labels):
    for col in ["family", "order", "class_name"]:
        if col in taxonomy_df.columns:
            group_map = taxonomy_df.set_index("primary_label")[col].to_dict()
            break
    else:
        group_map = {label: "Unknown" for label in primary_labels}
    groups = sorted(set(group_map.values()))
    grp_to_idx = {g: i for i, g in enumerate(groups)}
    class_to_group = [grp_to_idx.get(group_map.get(label, "Unknown"), 0)
                      for label in primary_labels]
    return len(groups), class_to_group, grp_to_idx


def build_site_mapping(meta_df):
    sites = meta_df["site"].unique().tolist()
    site_to_idx = {s: i + 1 for i, s in enumerate(sites)}
    return site_to_idx, len(sites) + 1


def get_file_metadata(meta_df, file_list, site_to_idx, n_sites_max):
    file_to_row = {}
    filenames = meta_df["filename"].to_numpy()
    sites = meta_df["site"].to_numpy()
    hours = meta_df["hour_utc"].to_numpy()
    for i, f in enumerate(filenames):
        if f not in file_to_row:
            file_to_row[f] = i
    site_ids = np.zeros(len(file_list), dtype=np.int64)
    hour_ids = np.zeros(len(file_list), dtype=np.int64)
    for fi, fname in enumerate(file_list):
        row = file_to_row.get(fname)
        if row is not None:
            site_ids[fi] = min(site_to_idx.get(sites[row], 0), n_sites_max - 1)
            hour_ids[fi] = int(hours[row]) % 24
    return site_ids, hour_ids


def mixup_files(emb, logits, labels, site_ids, hours, families, alpha=0.3):
    n = len(emb)
    if alpha <= 0 or n < 2:
        return emb, logits, labels, site_ids, hours, families
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)
    perm = np.random.permutation(n)
    emb_mix = lam * emb + (1 - lam) * emb[perm]
    logits_mix = lam * logits + (1 - lam) * logits[perm]
    labels_mix = lam * labels + (1 - lam) * labels[perm]
    families_mix = lam * families + (1 - lam) * families[perm] if families is not None else None
    return emb_mix, logits_mix, labels_mix, site_ids, hours, families_mix


# ── Training loop ─────────────────────────────────────────────────────

def train_proto_ssm_single(model, emb_train, logits_train, labels_train,
                           site_ids_train, hours_train,
                           emb_val=None, logits_val=None, labels_val=None,
                           site_ids_val=None, hours_val=None,
                           file_families_train=None, file_families_val=None,
                           cfg=None, verbose=True, device=DEVICE):
    label_smoothing = cfg.get("label_smoothing", 0.0)
    mixup_alpha = cfg.get("mixup_alpha", 0.0)
    focal_gamma = cfg.get("focal_gamma", 0.0)
    swa_start_frac = cfg.get("swa_start_frac", 1.0)
    n_epochs = cfg["n_epochs"]
    swa_start_epoch = int(n_epochs * swa_start_frac)

    labels_np = labels_train.copy()
    if label_smoothing > 0:
        labels_np = labels_np * (1.0 - label_smoothing) + label_smoothing / 2.0

    has_val = emb_val is not None
    if has_val:
        emb_v = torch.tensor(emb_val, dtype=torch.float32).to(device)
        logits_v = torch.tensor(logits_val, dtype=torch.float32).to(device)
        labels_v = torch.tensor(labels_val, dtype=torch.float32).to(device)
        site_v = torch.tensor(site_ids_val, dtype=torch.long).to(device)
        hour_v = torch.tensor(hours_val, dtype=torch.long).to(device)

    labels_tr_t = torch.tensor(labels_np, dtype=torch.float32).to(device)
    pos_counts = labels_tr_t.sum(dim=(0, 1))
    total = labels_tr_t.shape[0] * labels_tr_t.shape[1]
    pos_weight = ((total - pos_counts) / (pos_counts + 1)).clamp(max=cfg["pos_weight_cap"])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["lr"],
        epochs=n_epochs, steps_per_epoch=1,
        pct_start=0.1, anneal_strategy='cos')

    best_val_loss = float('inf')
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    swa_state = None
    swa_count = 0

    for epoch in range(n_epochs):
        if mixup_alpha > 0 and epoch > 5:
            emb_mix, logits_mix, labels_mix, _, _, fam_mix = mixup_files(
                emb_train, logits_train, labels_np,
                site_ids_train, hours_train, file_families_train,
                alpha=mixup_alpha)
        else:
            emb_mix, logits_mix, labels_mix = emb_train, logits_train, labels_np
            fam_mix = file_families_train

        emb_tr = torch.tensor(emb_mix, dtype=torch.float32).to(device)
        logits_tr = torch.tensor(logits_mix, dtype=torch.float32).to(device)
        labels_tr = torch.tensor(labels_mix, dtype=torch.float32).to(device)
        site_tr = torch.tensor(site_ids_train, dtype=torch.long).to(device)
        hour_tr = torch.tensor(hours_train, dtype=torch.long).to(device)
        fam_tr = torch.tensor(fam_mix, dtype=torch.float32).to(device) if fam_mix is not None else None

        model.train()
        species_out, family_out, _ = model(emb_tr, logits_tr, site_ids=site_tr, hours=hour_tr)

        if focal_gamma > 0:
            loss_main = focal_bce_with_logits(
                species_out, labels_tr, gamma=focal_gamma,
                pos_weight=pos_weight[None, None, :])
        else:
            loss_main = F.binary_cross_entropy_with_logits(
                species_out, labels_tr, pos_weight=pos_weight[None, None, :])

        loss_distill = F.mse_loss(species_out, logits_tr)
        loss = loss_main + cfg["distill_weight"] * loss_distill

        if family_out is not None and fam_tr is not None:
            loss += 0.1 * F.binary_cross_entropy_with_logits(family_out, fam_tr)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch >= swa_start_epoch:
            if swa_state is None:
                swa_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                swa_count = 1
            else:
                for k in swa_state:
                    swa_state[k] += model.state_dict()[k].cpu()
                swa_count += 1

        model.eval()
        with torch.no_grad():
            if has_val:
                val_out, _, _ = model(emb_v, logits_v, site_ids=site_v, hours=hour_v)
                val_loss = F.binary_cross_entropy_with_logits(
                    val_out, labels_v, pos_weight=pos_weight[None, None, :])
                val_pred = val_out.reshape(-1, val_out.shape[-1]).cpu().numpy()
                val_true = labels_v.reshape(-1, labels_v.shape[-1]).cpu().numpy()
                try:
                    val_auc = macro_auc_skip_empty(val_true, val_pred)
                except Exception:
                    val_auc = 0.0
            else:
                val_loss = loss
                val_auc = 0.0

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss.item())
        history["val_auc"].append(val_auc)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if verbose and (epoch + 1) % 10 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            best_marker = " ★ BEST" if wait == 0 else ""
            swa_info = f" swa={swa_count}" if swa_count > 0 else ""
            elapsed = time.time() - _epoch_t0 if '_epoch_t0' in dir() else 0
            print(f"  Epoch {epoch+1:3d}/{n_epochs}: train={loss.item():.4f} "
                  f"val={val_loss.item():.4f} auc={val_auc:.4f} "
                  f"lr={lr_now:.6f} wait={wait}{swa_info}{best_marker}")

        if wait >= cfg["patience"]:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.4f})")
            break

    if swa_state is not None and swa_count >= 3:
        if verbose:
            print(f"  Applying SWA (averaged {swa_count} checkpoints)")
        avg_state = {k: v / swa_count for k, v in swa_state.items()}
        model.load_state_dict(avg_state)
    elif best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ── Data loading ──────────────────────────────────────────────────────

def load_data():
    print("Loading Kaggle-extracted Perch features...")
    arr = np.load(PERCH_CACHE / "full_perch_arrays.npz")
    emb_full = arr["emb_full"].astype(np.float32)
    scores_full_raw = arr["scores_full_raw"].astype(np.float32)
    meta_full = pd.read_parquet(PERCH_CACHE / "full_perch_meta.parquet")
    print(f"  emb_full: {emb_full.shape}, scores: {scores_full_raw.shape}")

    taxonomy = pd.read_csv(RAW_DATA / "taxonomy.csv")
    sample_sub = pd.read_csv(RAW_DATA / "sample_submission.csv")
    soundscape_labels = pd.read_csv(RAW_DATA / "train_soundscapes_labels.csv")
    taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
    soundscape_labels["primary_label"] = soundscape_labels["primary_label"].astype(str)

    PRIMARY_LABELS = sample_sub.columns[1:].tolist()
    N_CLASSES = len(PRIMARY_LABELS)
    print(f"  {N_CLASSES} classes, {len(meta_full)} windows")

    def parse_soundscape_labels(x):
        if pd.isna(x):
            return []
        return [t.strip() for t in str(x).split(";") if t.strip()]

    sc_clean = (
        soundscape_labels
        .groupby(["filename", "start", "end"])["primary_label"]
        .apply(lambda s: sorted(set(lbl for x in s for lbl in parse_soundscape_labels(x))))
        .reset_index(name="label_list")
    )
    sc_clean["start_sec"] = pd.to_timedelta(sc_clean["start"]).dt.total_seconds().astype(int)
    sc_clean["end_sec"] = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
    sc_clean["row_id"] = (sc_clean["filename"].str.replace(".ogg", "", regex=False)
                          + "_" + sc_clean["end_sec"].astype(str))

    meta_parsed = sc_clean["filename"].apply(lambda name: {
        "site": (m := FNAME_RE.match(name)) and m.group(2),
        "hour_utc": int(m.group(4)[:2]) if m else -1,
    }).apply(pd.Series)
    sc_clean = pd.concat([sc_clean, meta_parsed], axis=1)

    windows_per_file = sc_clean.groupby("filename").size()
    full_files = sorted(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())
    full_truth = (sc_clean[sc_clean["filename"].isin(full_files)]
                  .sort_values(["filename", "end_sec"])
                  .reset_index(drop=False))

    label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}
    Y_SC = np.zeros((len(sc_clean), N_CLASSES), dtype=np.uint8)
    for i, labels in enumerate(sc_clean["label_list"]):
        idxs = [label_to_idx[lbl] for lbl in labels if lbl in label_to_idx]
        if idxs:
            Y_SC[i, idxs] = 1

    Y_FULL = Y_SC[full_truth["index"].to_numpy()]
    full_truth_aligned = full_truth.set_index("row_id").loc[meta_full["row_id"]].reset_index()
    Y_FULL = Y_SC[full_truth_aligned["index"].to_numpy()]

    emb_files, file_list = reshape_to_files(emb_full, meta_full)
    logits_files, _ = reshape_to_files(scores_full_raw, meta_full)
    labels_files, _ = reshape_to_files(Y_FULL, meta_full)

    print(f"  Reshaped: emb={emb_files.shape}, labels={labels_files.shape}")
    print(f"  Files: {len(file_list)}, active classes: {int((Y_FULL.sum(axis=0) > 0).sum())}")

    n_families, class_to_family, _ = build_taxonomy_groups(taxonomy, PRIMARY_LABELS)
    site_to_idx, n_sites_mapped = build_site_mapping(meta_full)
    site_ids_all, hours_all = get_file_metadata(meta_full, file_list, site_to_idx, 20)

    file_families = np.zeros((len(file_list), n_families), dtype=np.float32)
    for fi in range(len(file_list)):
        active_classes = np.where(labels_files[fi].sum(axis=0) > 0)[0]
        for ci in active_classes:
            file_families[fi, class_to_family[ci]] = 1.0

    file_groups = np.array([f.split("_")[3] if len(f.split("_")) > 3 else f
                            for f in file_list])

    return dict(
        emb_files=emb_files, logits_files=logits_files,
        labels_files=labels_files, site_ids_all=site_ids_all,
        hours_all=hours_all, file_families=file_families,
        file_groups=file_groups, n_families=n_families,
        class_to_family=class_to_family, N_CLASSES=N_CLASSES,
        PRIMARY_LABELS=PRIMARY_LABELS,
    )


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=320)
    parser.add_argument("--d-state", type=int, default=32)
    parser.add_argument("--n-ssm-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--oof-splits", type=int, default=5)
    parser.add_argument("--tta-shifts", type=int, nargs="+", default=[0, 1, -1])
    parser.add_argument("--no-oof", action="store_true")
    parser.add_argument(
        "--cv-mode",
        choices=("site", "file"),
        default="site",
        help="OOF split mode: 'site' = GroupKFold by recording site (default, "
             "OOD-style); 'file' = random file-level KFold (within-pool, "
             "measures teacher-strength rather than cross-site generalization).",
    )
    parser.add_argument(
        "--save-oof-path",
        type=str,
        default=None,
        help="If set, save per-seed OOF logits + meta to this NPZ path. "
             "Keys: oof_per_seed (S,F,T,C), oof_mean (F,T,C), file_list, "
             "labels (F,T,C), per_seed_auc (S,), site_ids (F,), hour_ids (F,)",
    )
    args = parser.parse_args()

    data = load_data()
    N_CLASSES = data["N_CLASSES"]

    ssm_cfg = {
        "d_model": args.d_model,
        "d_state": args.d_state,
        "n_ssm_layers": args.n_ssm_layers,
        "dropout": 0.12,
        "n_sites": 20,
        "meta_dim": 24,
        "use_cross_attn": True,
        "cross_attn_heads": 8,
    }
    train_cfg = {
        "n_epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": 1e-3,
        "patience": args.patience,
        "pos_weight_cap": 25.0,
        "distill_weight": 0.15,
        "proto_margin": 0.15,
        "label_smoothing": 0.03,
        "mixup_alpha": 0.4,
        "focal_gamma": 2.5,
        "swa_start_frac": 0.65,
        "swa_lr": 4e-4,
    }

    print(f"\nConfig: d_model={args.d_model}, n_ssm_layers={args.n_ssm_layers}, "
          f"epochs={args.epochs}, seeds={args.seeds}, device={DEVICE}")
    print(f"OOF splits: {args.oof_splits}, TTA shifts: {args.tta_shifts}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ── OOF evaluation ────────────────────────────────────────────────
    if not args.no_oof:
        print("\n" + "=" * 60)
        print("OOF Cross-Validation")
        print("=" * 60)

        n_splits = args.oof_splits
        n_files = len(data["emb_files"])
        if args.cv_mode == "site":
            n_unique_groups = len(set(data["file_groups"]))
            if n_unique_groups < n_splits:
                print(f"  WARNING: {n_unique_groups} groups < {n_splits} splits, reducing")
                n_splits = n_unique_groups
            gkf = GroupKFold(n_splits=n_splits)
            print(f"  CV mode: site-grouped GroupKFold (n_splits={n_splits})")
        else:
            # File-level shuffle KFold — measures within-pool teacher strength
            # rather than cross-site generalization.
            if n_files < n_splits:
                print(f"  WARNING: {n_files} files < {n_splits} splits, reducing")
                n_splits = n_files
            gkf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            print(f"  CV mode: file-level shuffle KFold (n_splits={n_splits})")
        dummy_y = np.zeros(n_files)

        all_seed_aucs = []
        all_oof_preds = []  # accumulate per-seed OOF logits
        for seed in range(args.seeds):
            print(f"\n--- Seed {seed} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)

            n_files = len(data["emb_files"])
            oof_preds = np.zeros((n_files, N_WINDOWS, N_CLASSES), dtype=np.float32)

            if args.cv_mode == "site":
                splits = gkf.split(dummy_y, dummy_y, data["file_groups"])
            else:
                splits = gkf.split(dummy_y)
            for fold_i, (train_idx, val_idx) in enumerate(splits):
                print(f"\n  Fold {fold_i+1}/{n_splits} "
                      f"(train={len(train_idx)}, val={len(val_idx)})")
                t0 = time.time()

                fold_model = ProtoSSMv2(
                    d_input=data["emb_files"].shape[2],
                    d_model=ssm_cfg["d_model"],
                    d_state=ssm_cfg["d_state"],
                    n_ssm_layers=ssm_cfg["n_ssm_layers"],
                    n_classes=N_CLASSES, n_windows=N_WINDOWS,
                    dropout=ssm_cfg["dropout"],
                    n_sites=ssm_cfg["n_sites"],
                    meta_dim=ssm_cfg["meta_dim"],
                    use_cross_attn=ssm_cfg["use_cross_attn"],
                    cross_attn_heads=ssm_cfg["cross_attn_heads"],
                ).to(DEVICE)

                emb_flat = data["emb_files"][train_idx].reshape(-1, data["emb_files"].shape[2])
                lab_flat = data["labels_files"][train_idx].reshape(-1, N_CLASSES)
                fold_model.init_prototypes_from_data(
                    torch.tensor(emb_flat, dtype=torch.float32).to(DEVICE),
                    torch.tensor(lab_flat, dtype=torch.float32).to(DEVICE))
                fold_model.init_family_head(data["n_families"], data["class_to_family"])
                fold_model.to(DEVICE)

                fold_model, _ = train_proto_ssm_single(
                    fold_model,
                    data["emb_files"][train_idx],
                    data["logits_files"][train_idx],
                    data["labels_files"][train_idx].astype(np.float32),
                    site_ids_train=data["site_ids_all"][train_idx],
                    hours_train=data["hours_all"][train_idx],
                    emb_val=data["emb_files"][val_idx],
                    logits_val=data["logits_files"][val_idx],
                    labels_val=data["labels_files"][val_idx].astype(np.float32),
                    site_ids_val=data["site_ids_all"][val_idx],
                    hours_val=data["hours_all"][val_idx],
                    file_families_train=data["file_families"][train_idx],
                    file_families_val=data["file_families"][val_idx],
                    cfg=train_cfg, verbose=True, device=DEVICE)

                if len(args.tta_shifts) > 1:
                    oof_preds[val_idx] = temporal_shift_tta(
                        data["emb_files"][val_idx],
                        data["logits_files"][val_idx],
                        fold_model,
                        data["site_ids_all"][val_idx],
                        data["hours_all"][val_idx],
                        shifts=args.tta_shifts, device=DEVICE)
                else:
                    fold_model.eval()
                    with torch.no_grad():
                        out, _, _ = fold_model(
                            torch.tensor(data["emb_files"][val_idx], dtype=torch.float32).to(DEVICE),
                            torch.tensor(data["logits_files"][val_idx], dtype=torch.float32).to(DEVICE),
                            site_ids=torch.tensor(data["site_ids_all"][val_idx], dtype=torch.long).to(DEVICE),
                            hours=torch.tensor(data["hours_all"][val_idx], dtype=torch.long).to(DEVICE))
                        oof_preds[val_idx] = out.cpu().numpy()

                elapsed = time.time() - t0
                mins, secs = divmod(elapsed, 60)
                print(f"  Fold {fold_i+1} done: {int(mins)}m{int(secs)}s")
                del fold_model; gc.collect()

            oof_flat = oof_preds.reshape(-1, N_CLASSES)
            y_flat = data["labels_files"].reshape(-1, N_CLASSES).astype(np.float32)
            seed_auc = macro_auc_skip_empty(y_flat, oof_flat)
            all_seed_aucs.append(seed_auc)
            all_oof_preds.append(oof_preds.copy())
            print(f"\n  Seed {seed} OOF macro AUC: {seed_auc:.4f}")

        print(f"\n{'='*60}")
        print(f"OOF summary: mean={np.mean(all_seed_aucs):.4f} "
              f"std={np.std(all_seed_aucs):.4f} "
              f"seeds={all_seed_aucs}")
        print(f"{'='*60}")

        if args.save_oof_path is not None:
            oof_per_seed = np.stack(all_oof_preds, axis=0)  # (S, F, T, C)
            oof_mean = oof_per_seed.mean(axis=0)            # (F, T, C) — mean of logits
            out_path = Path(args.save_oof_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Get the file_list as it was during cross-val (same order as
            # data["emb_files"] / labels_files)
            arr = np.load(PERCH_CACHE / "full_perch_arrays.npz")
            meta_full = pd.read_parquet(PERCH_CACHE / "full_perch_meta.parquet")
            _, file_list = reshape_to_files(arr["emb_full"].astype(np.float32), meta_full)
            np.savez_compressed(
                out_path,
                oof_per_seed=oof_per_seed.astype(np.float32),
                oof_mean=oof_mean.astype(np.float32),
                file_list=np.array(file_list),
                labels=data["labels_files"].astype(np.float32),
                per_seed_auc=np.array(all_seed_aucs, dtype=np.float32),
                site_ids=data["site_ids_all"].astype(np.int32),
                hour_ids=data["hours_all"].astype(np.int32),
            )
            print(f"[save-oof] wrote {out_path}  oof_per_seed={oof_per_seed.shape}")

    # ── Train final model (multi-seed average) ────────────────────────
    print(f"\n{'='*60}")
    print(f"Training {args.seeds} final model(s) on all data")
    print(f"{'='*60}")

    final_states = []
    for seed in range(args.seeds):
        print(f"\n--- Final model seed {seed} ---")
        torch.manual_seed(seed + 1000)
        np.random.seed(seed + 1000)

        model = ProtoSSMv2(
            d_input=data["emb_files"].shape[2],
            d_model=ssm_cfg["d_model"],
            d_state=ssm_cfg["d_state"],
            n_ssm_layers=ssm_cfg["n_ssm_layers"],
            n_classes=N_CLASSES, n_windows=N_WINDOWS,
            dropout=ssm_cfg["dropout"],
            n_sites=ssm_cfg["n_sites"],
            meta_dim=ssm_cfg["meta_dim"],
            use_cross_attn=ssm_cfg["use_cross_attn"],
            cross_attn_heads=ssm_cfg["cross_attn_heads"],
        ).to(DEVICE)

        emb_flat = data["emb_files"].reshape(-1, data["emb_files"].shape[2])
        lab_flat = data["labels_files"].reshape(-1, N_CLASSES)
        model.init_prototypes_from_data(
            torch.tensor(emb_flat, dtype=torch.float32).to(DEVICE),
            torch.tensor(lab_flat, dtype=torch.float32).to(DEVICE))
        model.init_family_head(data["n_families"], data["class_to_family"])
        model.to(DEVICE)

        print(f"  Parameters: {model.count_parameters():,}")

        t0 = time.time()
        model, hist = train_proto_ssm_single(
            model,
            data["emb_files"], data["logits_files"],
            data["labels_files"].astype(np.float32),
            site_ids_train=data["site_ids_all"],
            hours_train=data["hours_all"],
            file_families_train=data["file_families"],
            cfg=train_cfg, verbose=True, device=DEVICE)
        elapsed = time.time() - t0
        mins, secs = divmod(elapsed, 60)
        print(f"  Training time: {int(mins)}m{int(secs)}s")

        state = {k: v.cpu() for k, v in model.state_dict().items()}
        final_states.append(state)

        ckpt_path = CKPT_DIR / f"protossm_seed{seed}.pt"
        torch.save(state, ckpt_path)
        print(f"  Saved: {ckpt_path}")

    # Save averaged checkpoint
    if len(final_states) > 1:
        avg_state = {}
        for k in final_states[0]:
            avg_state[k] = sum(s[k] for s in final_states) / len(final_states)
        avg_path = CKPT_DIR / "protossm_pretrained.pt"
        torch.save(avg_state, avg_path)
        print(f"\nSaved multi-seed averaged checkpoint: {avg_path}")
    else:
        avg_path = CKPT_DIR / "protossm_pretrained.pt"
        torch.save(final_states[0], avg_path)
        print(f"\nSaved checkpoint: {avg_path}")

    # Save config for reproducibility
    import json
    config_path = CKPT_DIR / "config.json"
    config_path.write_text(json.dumps({
        "ssm_cfg": ssm_cfg,
        "train_cfg": train_cfg,
        "seeds": args.seeds,
        "tta_shifts": args.tta_shifts,
        "feature_source": "jaejohn/perch-meta (Kaggle-extracted 2026-03-13)",
    }, indent=2))
    print(f"Saved config: {config_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
