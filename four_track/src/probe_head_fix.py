"""Lever A head-fix probe — §14.22.7.1.

Freezes v56 fold-0 encoder, re-inits cls_conv + att_conv from random,
retrains head with class-balanced sampling on focal data, evaluates per-taxon
AUC on train_soundscapes val per epoch.

Target: Insecta OOF AUC >= 0.65 (probe v3 GroupKFold ceiling 0.73).
Negative control: Aves head-fix should NOT lift Aves above its native ~0.89
(probe v3 confirmed head fix slightly hurts Aves under proper grouping).
"""
from __future__ import annotations
import sys, os, time, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score

# Disable JIT fuser per feedback_gb10_nvrtc_jit
torch._C._jit_set_nvfuser_enabled(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

HERE = Path(__file__).resolve().parent
FT_ROOT = HERE.parent
ROOT = FT_ROOT.parent
PARENT_SRC = ROOT / "src"
for p in (str(PARENT_SRC), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config  # noqa: E402
from config import RAW, SAMPLE_RATE, CHUNK_SAMPLES, N_MELS, N_CLASSES, get_species_index  # noqa: E402
from utils import load_audio, pad_or_crop, waveform_to_mel  # noqa: E402
from model_a1 import BirdSEDModelA1  # noqa: E402
from dataset_a1 import BirdTrainDatasetA1  # noqa: E402

EAGER_CKPT  = FT_ROOT / "models/a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_hybrid.pt"
TRAIN_FOLDS = ROOT / "data/processed/train_folds.csv"
TAXONOMY    = RAW / "taxonomy.csv"
OUT_DIR     = FT_ROOT / "data"
SEED        = 42
BATCH       = 32
PER_CLASS_CAP = 50
EPOCHS      = 12
LR          = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)


def build_val_features(model, sp2idx):
    """Run frozen backbone+gem on val mels once; return cached features + labels."""
    df = pd.read_csv(RAW / "train_soundscapes_labels.csv")
    soundsc_dir = RAW / "train_soundscapes"
    mels = []
    val_labels = np.zeros((len(df), N_CLASSES), dtype=np.float32)
    for i, row in df.iterrows():
        h, m, s_ = str(row["start"]).split(":"); t_start = int(h)*3600+int(m)*60+int(s_)
        h, m, s_ = str(row["end"]).split(":");   t_end   = int(h)*3600+int(m)*60+int(s_)
        path = soundsc_dir / str(row["filename"])
        try:
            wav = load_audio(path)
            s, e = int(t_start * SAMPLE_RATE), int(t_end * SAMPLE_RATE)
            seg = wav[s:e] if e <= len(wav) else wav[s:]
            seg = pad_or_crop(seg, CHUNK_SAMPLES, random_crop=False)
            mel = waveform_to_mel(seg)
        except Exception as ex:
            print(f"[warn] {row['filename']}@{t_start}: {ex}", flush=True)
            mel = torch.zeros(3, N_MELS, 512)
        mels.append(mel)
        for sp in str(row["primary_label"]).split(";"):
            sp = sp.strip()
            if sp in sp2idx:
                val_labels[i, sp2idx[sp]] = 1.0

    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(mels), BATCH):
            batch = torch.stack(mels[i:i+BATCH]).to(DEVICE)
            f_list = model.backbone(batch)
            feat = model.gem_pool(f_list[-1])  # (B, c_out, T')
            feats.append(feat.cpu())
    return torch.cat(feats, 0), val_labels


def per_taxon_auc(probs, labels, taxa):
    n_pos = labels.sum(axis=0)
    out = {}
    for tax in ("Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"):
        mask = (taxa == tax) & (n_pos > 0) & (n_pos < len(labels))
        aucs = []
        for c in np.where(mask)[0]:
            try:
                aucs.append(roc_auc_score(labels[:, c], probs[:, c]))
            except ValueError:
                pass
        out[tax] = float(np.mean(aucs)) if aucs else float("nan")
    overall_aucs = [roc_auc_score(labels[:, c], probs[:, c])
                    for c in range(labels.shape[1]) if 0 < n_pos[c] < len(labels)]
    out["_overall"] = float(np.mean(overall_aucs)) if overall_aucs else float("nan")
    return out


def head_forward(model, feat):
    """Forward through cls_conv+att_conv → clip_logits."""
    frame = model.cls_conv(feat).permute(0, 2, 1)         # (B, T', K)
    att   = model.att_conv(feat).permute(0, 2, 1)
    wt    = torch.softmax(att, dim=1)
    return (frame * wt).sum(dim=1)                         # (B, K)


def main():
    print("=== Lever A head-fix probe ===", flush=True)
    print(f"device={DEVICE}  EPOCHS={EPOCHS}  PER_CLASS_CAP={PER_CLASS_CAP}", flush=True)

    sp2idx = get_species_index()
    classes_sorted = sorted(sp2idx.keys())
    tax_df = pd.read_csv(TAXONOMY)
    tax_df["primary_label"] = tax_df["primary_label"].astype(str)
    cls2tax = dict(zip(tax_df["primary_label"], tax_df["class_name"]))
    taxa = np.array([cls2tax[c] for c in classes_sorted])

    print(f"loading eager v56 fold-0 ckpt: {EAGER_CKPT.name}", flush=True)
    model = BirdSEDModelA1(mixstyle_p=0.0)
    sd = torch.load(EAGER_CKPT, map_location="cpu", weights_only=False)
    miss, unex = model.load_state_dict(sd, strict=False)
    print(f"  missing={len(miss)}  unexpected={len(unex)}", flush=True)
    model = model.to(DEVICE)

    print("caching val features (frozen encoder forward) ...", flush=True)
    t0 = time.time()
    val_feat, val_labels = build_val_features(model, sp2idx)
    print(f"  val_feat={tuple(val_feat.shape)}  in {time.time()-t0:.1f}s", flush=True)

    # native baseline
    model.eval()
    with torch.no_grad():
        probs_native = []
        for i in range(0, len(val_feat), BATCH):
            f = val_feat[i:i+BATCH].to(DEVICE)
            probs_native.append(torch.sigmoid(head_forward(model, f)).cpu().numpy())
    probs_native = np.concatenate(probs_native).astype(np.float32)
    base = per_taxon_auc(probs_native, val_labels, taxa)
    print(f"native v56: overall={base['_overall']:.4f}  " +
          " ".join(f"{k}={v:.4f}" for k, v in base.items() if not k.startswith("_")),
          flush=True)

    # re-init head
    print("re-initializing cls_conv + att_conv from scratch ...", flush=True)
    c_out = model.cls_conv.in_channels
    model.cls_conv = nn.Conv1d(c_out, N_CLASSES, kernel_size=1).to(DEVICE)
    model.att_conv = nn.Sequential(
        nn.Conv1d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm1d(c_out),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Conv1d(c_out, N_CLASSES, 1),
    ).to(DEVICE)
    for n, p in model.named_parameters():
        p.requires_grad = ("cls_conv" in n) or ("att_conv" in n)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"  trainable params: {n_trainable:,} / {n_total:,}", flush=True)

    # focal training data
    print(f"loading focal train (fold!=0, cap={PER_CLASS_CAP}/class) ...", flush=True)
    folds = pd.read_csv(TRAIN_FOLDS)
    folds["primary_label"] = folds["primary_label"].astype(str)
    train_df = folds[folds["fold"] != 0].copy()
    train_df = train_df[train_df["primary_label"].isin(sp2idx)]
    train_df = (train_df.groupby("primary_label", group_keys=False)
                       .apply(lambda g: g.sample(min(len(g), PER_CLASS_CAP),
                                                  random_state=SEED))
                       .reset_index(drop=True))
    print(f"  train_df: {len(train_df)} rows over "
          f"{train_df['primary_label'].nunique()} classes", flush=True)

    # class-balanced sampler
    cls_counts = train_df["primary_label"].value_counts().to_dict()
    weights = train_df["primary_label"].map(lambda s: 1.0 / cls_counts[s]).values
    sampler = WeightedRandomSampler(weights, num_samples=len(train_df), replacement=True)

    train_ds = BirdTrainDatasetA1(
        train_df, augment=True, bg_noise_dir=None, min_samples_per_class=0,
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        multiprocessing_context="spawn",
    )
    print(f"  loader: BS={BATCH}  steps/epoch~{len(train_df)//BATCH}", flush=True)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    bce = nn.BCEWithLogitsLoss()

    metrics_log = []
    best_insecta = -1.0

    print("=" * 40)
    for epoch in range(1, EPOCHS + 1):
        t_ep = time.time()
        # backbone always eval (frozen)
        model.backbone.eval()
        if hasattr(model, "gem_pool"):
            model.gem_pool.eval()
        # head in train mode
        model.cls_conv.train()
        model.att_conv.train()

        running, n_steps = 0.0, 0
        for batch in train_loader:
            mel, labels, _mask = batch
            mel = mel.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            with torch.no_grad():
                feat = model.gem_pool(model.backbone(mel)[-1])
            clip = head_forward(model, feat)
            loss = bce(clip, labels.float())
            optim.zero_grad(); loss.backward(); optim.step()
            running += loss.item(); n_steps += 1

        # eval on cached val
        model.cls_conv.eval(); model.att_conv.eval()
        with torch.no_grad():
            probs = []
            for i in range(0, len(val_feat), BATCH):
                f = val_feat[i:i+BATCH].to(DEVICE)
                probs.append(torch.sigmoid(head_forward(model, f)).cpu().numpy())
        probs = np.concatenate(probs).astype(np.float32)
        m = per_taxon_auc(probs, val_labels, taxa)
        m["_train_loss"] = running / max(n_steps, 1)
        m["_epoch"] = epoch
        m["_time"]  = time.time() - t_ep
        metrics_log.append(m)

        ins = m.get("Insecta", float("nan"))
        is_best = (not np.isnan(ins)) and ins > best_insecta
        if is_best:
            best_insecta = ins
            torch.save({
                "cls_conv": model.cls_conv.state_dict(),
                "att_conv": model.att_conv.state_dict(),
                "epoch": epoch, "metrics": m,
            }, OUT_DIR / "probe_head_fix_best.pt")

        mins, secs = divmod(int(m["_time"]), 60)
        marker = " ★ BEST" if is_best else ""
        print("=" * 40)
        print(f"Epoch {epoch:>2}/{EPOCHS}: train_loss={m['_train_loss']:.4f}  "
              f"overall={m['_overall']:.4f}  Aves={m.get('Aves',np.nan):.4f}  "
              f"Insecta={m.get('Insecta',np.nan):.4f}  "
              f"Amph={m.get('Amphibia',np.nan):.4f}  "
              f"Mam={m.get('Mammalia',np.nan):.4f}  "
              f"time={mins}m{secs:02d}s  {time.strftime('%Y-%m-%d %H:%M:%S')}{marker}",
              flush=True)
        print("=" * 40, flush=True)

        gc.collect(); torch.cuda.empty_cache()

    pd.DataFrame(metrics_log).to_csv(OUT_DIR / "probe_head_fix_epoch_metrics.csv",
                                     index=False)
    print(f"\nbest Insecta OOF AUC: {best_insecta:.4f}  "
          f"(target ≥ 0.65; probe v3 GroupKFold ceiling 0.73; native {base.get('Insecta', np.nan):.4f})",
          flush=True)
    print(f"[done] saved {OUT_DIR/'probe_head_fix_best.pt'} + epoch_metrics.csv",
          flush=True)


if __name__ == "__main__":
    main()
