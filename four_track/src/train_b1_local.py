"""
Train B1 PerceiverIO Perch consumer locally with extended budget.

Mirrors `src/train_protossm_local.py` to produce a pre-trained B1
checkpoint that cell 31b can guard-load in submit mode instead of
retraining from scratch. See `new_plan.md` §12 for rationale and gates.

Architecture + training config are locked to `CFG["b1_perceiver"]` and
`CFG["b1_perceiver_train"]` defaults from `src/b1_perceiver.py` so the
saved state dict loads cleanly into cell 31b's `b1_model`:

    d_latent=256, n_latents=16, n_cross_layers=2, n_self_layers=4,
    n_heads=8, meta_dim=16, n_sites=20, dropout=0.3

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle
    cd /home/swatson/work/kaggle/BirdCLEF/four_track
    nohup python -u src/train_b1_local.py \
        --epochs 200 --patience 40 --seeds 3 \
        > log/train_b1_local_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""

import argparse
import gc
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent          # four_track/
BIRDCLEF = PROJECT.parent                                  # BirdCLEF/

PERCH_CACHE = PROJECT / "data" / "kaggle_perch_cache"
RAW_DATA    = BIRDCLEF / "data" / "raw"
CKPT_DIR    = PROJECT / "models" / "b1_pretrained"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_WINDOWS = 12


# ── PerceiverIO building blocks (verbatim copy from b1_perceiver.py) ──

class _PreNormCrossBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn    = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv):
        q_n  = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        attn_out, _ = self.attn(q_n, kv_n, kv_n)
        q = q + attn_out
        q = q + self.ffn(self.norm_ff(q))
        return q


class _PreNormSelfBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.norm_a = nn.LayerNorm(d_model)
        self.attn   = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_f = nn.LayerNorm(d_model)
        self.ffn    = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_n = self.norm_a(x)
        attn_out, _ = self.attn(x_n, x_n, x_n)
        x = x + attn_out
        x = x + self.ffn(self.norm_f(x))
        return x


class PerceiverIOHead(nn.Module):
    def __init__(self, d_input=1536, d_logits=234, d_latent=256, n_latents=16,
                 n_cross_layers=2, n_self_layers=4, n_heads=8,
                 n_classes=234, n_windows=12,
                 n_sites=20, meta_dim=16, dropout=0.3):
        super().__init__()
        self.n_classes = n_classes
        self.n_windows = n_windows
        self.d_latent  = d_latent

        self.emb_proj    = nn.Linear(d_input,  d_latent)
        self.logits_proj = nn.Linear(d_logits, d_latent)
        self.site_emb    = nn.Embedding(n_sites, meta_dim)
        self.hour_emb    = nn.Embedding(24,      meta_dim)
        self.meta_proj   = nn.Linear(2 * meta_dim, d_latent)

        self.window_pos = nn.Parameter(torch.randn(1, n_windows, d_latent) * 0.02)
        self.input_norm = nn.LayerNorm(d_latent)
        self.input_drop = nn.Dropout(dropout)

        self.latents = nn.Parameter(torch.randn(n_latents, d_latent) * 0.02)

        self.cross_blocks = nn.ModuleList([
            _PreNormCrossBlock(d_latent, n_heads, dropout)
            for _ in range(n_cross_layers)
        ])
        self.self_blocks = nn.ModuleList([
            _PreNormSelfBlock(d_latent, n_heads, dropout)
            for _ in range(n_self_layers)
        ])

        self.query_pos     = nn.Parameter(torch.randn(1, n_windows, d_latent) * 0.02)
        self.decoder_cross = _PreNormCrossBlock(d_latent, n_heads, dropout)
        self.decoder_norm  = nn.LayerNorm(d_latent)

        self.classifier = nn.Linear(d_latent, n_classes)

    def forward(self, emb, perch_logits, site_ids=None, hours=None):
        B, T, _ = emb.shape
        e = self.emb_proj(emb)
        l = self.logits_proj(perch_logits)
        x = e + l
        if site_ids is not None and hours is not None:
            s = self.site_emb(site_ids)
            h = self.hour_emb(hours)
            m = self.meta_proj(torch.cat([s, h], dim=-1))
            x = x + m[:, None, :]
        x = x + self.window_pos[:, :T, :]
        x = self.input_drop(self.input_norm(x))
        latents = self.latents[None, :, :].expand(B, -1, -1).contiguous()
        for blk in self.cross_blocks:
            latents = blk(latents, x)
        for blk in self.self_blocks:
            latents = blk(latents)
        queries = self.query_pos[:, :T, :].expand(B, -1, -1).contiguous()
        out = self.decoder_cross(queries, latents)
        out = self.decoder_norm(out)
        return self.classifier(out)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Loss + metrics ────────────────────────────────────────────────────

def focal_bce_with_logits(logits, targets, gamma=2.0, pos_weight=None):
    if pos_weight is not None:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction="none")
    else:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = targets * p + (1 - targets) * (1 - p)
    focal_weight = (1 - pt) ** gamma
    return (focal_weight * bce).mean()


def macro_auc_skip_empty(y_true, y_score):
    keep = y_true.sum(axis=0) > 0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")


def mixup_files(emb, logits, labels, alpha=0.3):
    n = len(emb)
    if alpha <= 0 or n < 2:
        return emb, logits, labels
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)
    perm = np.random.permutation(n)
    return (
        lam * emb    + (1 - lam) * emb[perm],
        lam * logits + (1 - lam) * logits[perm],
        lam * labels + (1 - lam) * labels[perm],
    )


# ── Training loop ─────────────────────────────────────────────────────

def train_b1_single(model, emb_train, logits_train, labels_train,
                    site_ids_train, hours_train,
                    emb_val=None, logits_val=None, labels_val=None,
                    site_ids_val=None, hours_val=None,
                    cfg=None, verbose=True, device=DEVICE):
    label_smoothing = cfg.get("label_smoothing", 0.0)
    mixup_alpha     = cfg.get("mixup_alpha", 0.0)
    focal_gamma     = cfg.get("focal_gamma", 0.0)
    swa_start_frac  = cfg.get("swa_start_frac", 1.0)
    n_epochs        = cfg["n_epochs"]
    swa_start_epoch = int(n_epochs * swa_start_frac)

    labels_np = labels_train.copy()
    if label_smoothing > 0:
        labels_np = labels_np * (1.0 - label_smoothing) + label_smoothing / 2.0

    has_val = emb_val is not None
    if has_val:
        emb_v    = torch.tensor(emb_val,    dtype=torch.float32).to(device)
        logits_v = torch.tensor(logits_val, dtype=torch.float32).to(device)
        labels_v = torch.tensor(labels_val, dtype=torch.float32).to(device)
        site_v   = torch.tensor(site_ids_val, dtype=torch.long).to(device)
        hour_v   = torch.tensor(hours_val,    dtype=torch.long).to(device)

    labels_tr_t = torch.tensor(labels_np, dtype=torch.float32).to(device)
    pos_counts  = labels_tr_t.sum(dim=(0, 1))
    total       = labels_tr_t.shape[0] * labels_tr_t.shape[1]
    pos_weight  = ((total - pos_counts) / (pos_counts + 1)).clamp(max=cfg["pos_weight_cap"])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["lr"],
        epochs=n_epochs, steps_per_epoch=1,
        pct_start=0.1, anneal_strategy="cos")

    best_val_loss = float("inf")
    best_state    = None
    wait          = 0
    swa_state     = None
    swa_count     = 0
    history       = {"train_loss": [], "val_loss": [], "val_auc": []}

    epoch_t0 = time.time()

    for epoch in range(n_epochs):
        if mixup_alpha > 0 and epoch > 5:
            emb_mix, logits_mix, labels_mix = mixup_files(
                emb_train, logits_train, labels_np, alpha=mixup_alpha)
        else:
            emb_mix, logits_mix, labels_mix = emb_train, logits_train, labels_np

        emb_tr    = torch.tensor(emb_mix,    dtype=torch.float32).to(device)
        logits_tr = torch.tensor(logits_mix, dtype=torch.float32).to(device)
        labels_tr = torch.tensor(labels_mix, dtype=torch.float32).to(device)
        site_tr   = torch.tensor(site_ids_train, dtype=torch.long).to(device)
        hour_tr   = torch.tensor(hours_train,    dtype=torch.long).to(device)

        model.train()
        species_out = model(emb_tr, logits_tr, site_ids=site_tr, hours=hour_tr)

        if focal_gamma > 0:
            loss_main = focal_bce_with_logits(
                species_out, labels_tr, gamma=focal_gamma,
                pos_weight=pos_weight[None, None, :])
        else:
            loss_main = F.binary_cross_entropy_with_logits(
                species_out, labels_tr, pos_weight=pos_weight[None, None, :])

        loss_distill = F.mse_loss(species_out, logits_tr)
        loss = loss_main + cfg["distill_weight"] * loss_distill

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch >= swa_start_epoch:
            if swa_state is None:
                swa_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                swa_count = 1
            else:
                for k in swa_state:
                    swa_state[k] += model.state_dict()[k].detach().cpu()
                swa_count += 1

        model.eval()
        with torch.no_grad():
            if has_val:
                val_out  = model(emb_v, logits_v, site_ids=site_v, hours=hour_v)
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
                val_auc  = 0.0

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss.item())
        history["val_auc"].append(val_auc)

        is_best = val_loss.item() < best_val_loss
        if is_best:
            best_val_loss = val_loss.item()
            best_state    = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if verbose and ((epoch + 1) % 10 == 0 or is_best and epoch < 20):
            elapsed = time.time() - epoch_t0
            mins, secs = divmod(elapsed, 60)
            lr_now = optimizer.param_groups[0]["lr"]
            swa_info = f" swa={swa_count}" if swa_count > 0 else ""
            marker = " ★ BEST" if is_best else ""
            stamp  = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"  B1 Epoch {epoch+1:3d}/{n_epochs}: train={loss.item():.4f} "
                  f"val={val_loss.item():.4f} auc={val_auc:.4f} "
                  f"lr={lr_now:.6f} wait={wait}{swa_info} "
                  f"time={int(mins)}m{int(secs):02d}s  {stamp}{marker}", flush=True)

        if wait >= cfg["patience"]:
            if verbose:
                print(f"  B1 early stop at epoch {epoch+1} (best val_loss={best_val_loss:.4f})", flush=True)
            break

    if swa_state is not None and swa_count >= 3:
        if verbose:
            print(f"  B1 applying SWA (averaged {swa_count} checkpoints)", flush=True)
        avg_state = {k: v / swa_count for k, v in swa_state.items()}
        model.load_state_dict(avg_state)
    elif best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ── Data loading (mirror of train_protossm_local.py) ──────────────────

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")


def reshape_to_files(flat_array, meta_df, n_windows=N_WINDOWS):
    filenames = meta_df["filename"].to_numpy()
    unique_files = list(dict.fromkeys(filenames))
    n_files = len(unique_files)
    assert len(flat_array) == n_files * n_windows
    new_shape = (n_files, n_windows) + flat_array.shape[1:]
    return flat_array.reshape(new_shape), unique_files


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


def load_data():
    print("Loading Kaggle-extracted Perch features...", flush=True)
    arr = np.load(PERCH_CACHE / "full_perch_arrays.npz")
    emb_full        = arr["emb_full"].astype(np.float32)
    scores_full_raw = arr["scores_full_raw"].astype(np.float32)
    meta_full = pd.read_parquet(PERCH_CACHE / "full_perch_meta.parquet")
    print(f"  emb_full: {emb_full.shape}, scores: {scores_full_raw.shape}", flush=True)

    sample_sub        = pd.read_csv(RAW_DATA / "sample_submission.csv")
    soundscape_labels = pd.read_csv(RAW_DATA / "train_soundscapes_labels.csv")
    soundscape_labels["primary_label"] = soundscape_labels["primary_label"].astype(str)

    PRIMARY_LABELS = sample_sub.columns[1:].tolist()
    N_CLASSES = len(PRIMARY_LABELS)
    print(f"  {N_CLASSES} classes, {len(meta_full)} windows", flush=True)

    def parse_sc_labels(x):
        if pd.isna(x):
            return []
        return [t.strip() for t in str(x).split(";") if t.strip()]

    sc_clean = (
        soundscape_labels
        .groupby(["filename", "start", "end"])["primary_label"]
        .apply(lambda s: sorted(set(lbl for x in s for lbl in parse_sc_labels(x))))
        .reset_index(name="label_list")
    )
    sc_clean["start_sec"] = pd.to_timedelta(sc_clean["start"]).dt.total_seconds().astype(int)
    sc_clean["end_sec"]   = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
    sc_clean["row_id"] = (
        sc_clean["filename"].str.replace(".ogg", "", regex=False)
        + "_" + sc_clean["end_sec"].astype(str)
    )

    label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}
    Y_SC = np.zeros((len(sc_clean), N_CLASSES), dtype=np.uint8)
    for i, labels in enumerate(sc_clean["label_list"]):
        idxs = [label_to_idx[lbl] for lbl in labels if lbl in label_to_idx]
        if idxs:
            Y_SC[i, idxs] = 1

    sc_indexed = sc_clean.reset_index()  # "index" column holds original row
    full_truth_aligned = (
        sc_indexed.set_index("row_id").loc[meta_full["row_id"]].reset_index()
    )
    Y_FULL = Y_SC[full_truth_aligned["index"].to_numpy()]

    emb_files,    file_list = reshape_to_files(emb_full,        meta_full)
    logits_files, _         = reshape_to_files(scores_full_raw, meta_full)
    labels_files, _         = reshape_to_files(Y_FULL,          meta_full)
    print(f"  Reshaped: emb={emb_files.shape}, labels={labels_files.shape}", flush=True)
    print(f"  Files: {len(file_list)}, active classes: {int((Y_FULL.sum(axis=0) > 0).sum())}", flush=True)

    site_to_idx, _ = build_site_mapping(meta_full)
    site_ids_all, hours_all = get_file_metadata(meta_full, file_list, site_to_idx, 20)

    # file_groups: group by date-stamp (4th underscore field in BC2026 filename)
    # — same grouping the ProtoSSM local trainer uses.
    file_groups = np.array([
        f.split("_")[3] if len(f.split("_")) > 3 else f for f in file_list
    ])

    return dict(
        emb_files=emb_files,
        logits_files=logits_files,
        labels_files=labels_files,
        site_ids_all=site_ids_all,
        hours_all=hours_all,
        file_groups=file_groups,
        N_CLASSES=N_CLASSES,
        PRIMARY_LABELS=PRIMARY_LABELS,
    )


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",    type=int, default=200)
    parser.add_argument("--patience",  type=int, default=40)
    parser.add_argument("--seeds",     type=int, default=3)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--oof-splits", type=int, default=5)
    parser.add_argument("--no-oof",    action="store_true")
    args = parser.parse_args()

    data = load_data()
    N_CLASSES = data["N_CLASSES"]

    # Architecture locked to notebook CFG["b1_perceiver"] defaults.
    b1_arch = {
        "d_input":        data["emb_files"].shape[2],
        "d_logits":       N_CLASSES,
        "d_latent":       256,
        "n_latents":      16,
        "n_cross_layers": 2,
        "n_self_layers":  4,
        "n_heads":        8,
        "n_classes":      N_CLASSES,
        "n_windows":      N_WINDOWS,
        "n_sites":        20,
        "meta_dim":       16,
        "dropout":        0.3,
    }
    train_cfg = {
        "n_epochs":        args.epochs,
        "lr":              args.lr,
        "weight_decay":    1e-2,
        "patience":        args.patience,
        "pos_weight_cap":  25.0,
        "focal_gamma":     2.0,
        "label_smoothing": 0.0,
        "mixup_alpha":     0.3,
        "swa_start_frac":  0.65,
        "distill_weight":  0.05,
    }

    print(f"\nConfig: arch={b1_arch}", flush=True)
    print(f"Train : {train_cfg}", flush=True)
    print(f"Seeds : {args.seeds}, OOF splits: {args.oof_splits}, device: {DEVICE}", flush=True)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ── OOF evaluation ────────────────────────────────────────────────
    oof_seed_aucs = []
    if not args.no_oof:
        print("\n" + "=" * 60, flush=True)
        print("OOF Cross-Validation", flush=True)
        print("=" * 60, flush=True)

        n_splits = args.oof_splits
        n_unique = len(set(data["file_groups"]))
        if n_unique < n_splits:
            print(f"  WARNING: {n_unique} groups < {n_splits} splits, reducing", flush=True)
            n_splits = n_unique

        gkf = GroupKFold(n_splits=n_splits)
        dummy_y = np.zeros(len(data["emb_files"]))

        for seed in range(args.seeds):
            print(f"\n--- Seed {seed} ---", flush=True)
            torch.manual_seed(seed)
            np.random.seed(seed)

            n_files = len(data["emb_files"])
            oof_preds = np.zeros((n_files, N_WINDOWS, N_CLASSES), dtype=np.float32)

            for fold_i, (train_idx, val_idx) in enumerate(
                gkf.split(dummy_y, dummy_y, data["file_groups"])
            ):
                t0 = time.time()
                print(f"\n  Fold {fold_i+1}/{n_splits} "
                      f"(train={len(train_idx)}, val={len(val_idx)})", flush=True)

                fold_model = PerceiverIOHead(**b1_arch).to(DEVICE)

                fold_model, _ = train_b1_single(
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
                    cfg=train_cfg, verbose=True, device=DEVICE,
                )

                fold_model.eval()
                with torch.no_grad():
                    val_emb    = torch.tensor(data["emb_files"][val_idx],    dtype=torch.float32).to(DEVICE)
                    val_logits = torch.tensor(data["logits_files"][val_idx], dtype=torch.float32).to(DEVICE)
                    val_sites  = torch.tensor(data["site_ids_all"][val_idx], dtype=torch.long).to(DEVICE)
                    val_hours  = torch.tensor(data["hours_all"][val_idx],    dtype=torch.long).to(DEVICE)
                    out = fold_model(val_emb, val_logits, site_ids=val_sites, hours=val_hours)
                    oof_preds[val_idx] = out.cpu().numpy()

                elapsed = time.time() - t0
                mins, secs = divmod(elapsed, 60)
                print(f"  Fold {fold_i+1} done: {int(mins)}m{int(secs):02d}s", flush=True)
                del fold_model; gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            oof_flat = oof_preds.reshape(-1, N_CLASSES)
            y_flat   = data["labels_files"].reshape(-1, N_CLASSES).astype(np.float32)
            seed_auc = macro_auc_skip_empty(y_flat, oof_flat)
            oof_seed_aucs.append(seed_auc)
            print(f"\n  Seed {seed} OOF macro AUC: {seed_auc:.4f}", flush=True)

        print(f"\n{'='*60}", flush=True)
        print(f"OOF summary: mean={np.mean(oof_seed_aucs):.4f} "
              f"std={np.std(oof_seed_aucs):.4f} "
              f"seeds={oof_seed_aucs}", flush=True)
        print(f"{'='*60}", flush=True)

    # ── Train final model on all data (multi-seed) ────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"Training {args.seeds} final B1 model(s) on all soundscapes", flush=True)
    print(f"{'='*60}", flush=True)

    final_states = []
    for seed in range(args.seeds):
        print(f"\n--- Final B1 seed {seed} ---", flush=True)
        torch.manual_seed(seed + 1000)
        np.random.seed(seed + 1000)

        model = PerceiverIOHead(**b1_arch).to(DEVICE)
        print(f"  Parameters: {model.count_parameters():,}", flush=True)

        t0 = time.time()
        model, _ = train_b1_single(
            model,
            data["emb_files"],
            data["logits_files"],
            data["labels_files"].astype(np.float32),
            site_ids_train=data["site_ids_all"],
            hours_train=data["hours_all"],
            cfg=train_cfg, verbose=True, device=DEVICE,
        )
        elapsed = time.time() - t0
        mins, secs = divmod(elapsed, 60)
        print(f"  Training time: {int(mins)}m{int(secs):02d}s", flush=True)

        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        final_states.append(state)

        ckpt_path = CKPT_DIR / f"b1_seed{seed}.pt"
        torch.save(state, ckpt_path)
        print(f"  Saved: {ckpt_path}", flush=True)

        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Multi-seed averaged checkpoint (the one we upload) ────────────
    if len(final_states) > 1:
        avg_state = {k: sum(s[k] for s in final_states) / len(final_states)
                     for k in final_states[0]}
    else:
        avg_state = final_states[0]
    avg_path = CKPT_DIR / "b1_pretrained.pt"
    torch.save(avg_state, avg_path)
    print(f"\nSaved averaged checkpoint: {avg_path}", flush=True)

    # ── Config snapshot for reproducibility ───────────────────────────
    config_path = CKPT_DIR / "config.json"
    config_path.write_text(json.dumps({
        "arch":              b1_arch,
        "train_cfg":         train_cfg,
        "seeds":             args.seeds,
        "oof_splits":        args.oof_splits,
        "oof_seed_aucs":     oof_seed_aucs,
        "oof_mean":          float(np.mean(oof_seed_aucs)) if oof_seed_aucs else None,
        "oof_std":           float(np.std(oof_seed_aucs)) if oof_seed_aucs else None,
        "feature_source":    "jaejohn/perch-meta (Kaggle-extracted)",
    }, indent=2))
    print(f"Saved config: {config_path}", flush=True)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
