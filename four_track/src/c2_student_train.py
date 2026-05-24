"""
§13 Phase 3 (draft) — Train a ProtoSSM student on soundscape + Phase-2
pseudo-labeled focal clips.

This is the single-seed draft. It omits SWA, multi-seed averaging, mixup,
and temporal-shift TTA so we can iterate fast on the two knobs that most
affect the Phase 3 gate: label format and focal:soundscape mix ratio. Once
the gate passes, scale up using the patterns from src/train_protossm_local.py.

Label format (hard-primary + soft-secondary distillation)
---------------------------------------------------------
  target[primary_label] = 1.0                     (ground truth from data dir)
  target[c]             = teacher_prob[c], c ≠ primary_label  (Phase-2 soft)
Rationale: teacher's mean max_conf is 0.991 → near-hard anyway; this keeps
ground truth exact while using teacher only for multi-label co-occurrence.

Focal:soundscape mix (50/50 batch-level balanced)
-------------------------------------------------
Each batch = B/2 soundscape windows + B/2 focal-clip chunks. "Epoch" =
one pass over the 33,516 retained focal clips (~2,000 batches at B=32).
Soundscape sampled with replacement to match focal batch size.
Rationale: 325× more focal windows than soundscape windows would drown
the only LB-relevant substrate (soundscape) without rebalancing.

Gate (§13 Phase 3): student val ROC-AUC on a held-out soundscape split
≥ teacher's val ROC-AUC on the same split.

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle
    cd /home/swatson/work/kaggle/BirdCLEF/four_track
    python -u src/c2_student_train.py [--epochs 30] [--batch-size 32]
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from train_protossm_local import (
    DEVICE, N_WINDOWS, ProtoSSMv2,
    load_data as load_soundscape_data,
)

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT  = Path(__file__).resolve().parent.parent
BIRDCLEF = PROJECT.parent

FOCAL_FEAT_DIR    = BIRDCLEF / "kaggle_datasets" / "train-audio-perch"
PSEUDO_LABEL_DIR  = PROJECT  / "data" / "processed" / "c2_pseudo_labels_kagglefeat"
STUDENT_CKPT_DIR  = PROJECT  / "models" / "c2_student"
TEACHER_CKPT_DIR  = PROJECT  / "models" / "protossm_pretrained_v2"


def _build_protossm_from_teacher_config(n_classes):
    cfg = json.loads((TEACHER_CKPT_DIR / "config.json").read_text())["ssm_cfg"]
    return ProtoSSMv2(
        d_input=1536, d_model=cfg["d_model"], d_state=cfg["d_state"],
        n_ssm_layers=cfg["n_ssm_layers"], n_classes=n_classes,
        n_windows=N_WINDOWS, dropout=cfg["dropout"],
        n_sites=cfg["n_sites"], meta_dim=cfg["meta_dim"],
        use_cross_attn=cfg["use_cross_attn"],
        cross_attn_heads=cfg["cross_attn_heads"],
    )


def _load_teacher_state_into(model):
    state = torch.load(TEACHER_CKPT_DIR / "protossm_pretrained.pt",
                       map_location=DEVICE)
    missing, unexpected = model.load_state_dict(state, strict=False)
    extra = [k for k in unexpected
             if not k.startswith(("family_head", "class_to_family"))]
    if missing or extra:
        raise RuntimeError(
            f"teacher ckpt load mismatch: missing={missing}, extra={extra}")

# ── Focal data loading ─────────────────────────────────────────────────

def load_focal_retained(n_classes: int):
    """Load Phase-1 focal features, filter to Phase-2 retained clips, and
    return per-clip window-ranges + hybrid labels.

    Returns:
        emb        (N_total_windows, 1536)  float32  — raw per-window Perch emb
        scores     (N_total_windows, 234)   float32  — raw per-window Perch logits
        stem_starts (n_clips + 1,)          int64    — window-range [s:e] per clip
        targets    (n_clips, 234)           float32  — hybrid labels
    """
    print(f"[{time.strftime('%H:%M:%S')}] loading focal features from {FOCAL_FEAT_DIR}")
    meta = pd.read_parquet(FOCAL_FEAT_DIR / "full_train_audio_meta.parquet")
    npz  = np.load(FOCAL_FEAT_DIR / "full_train_audio_perch.npz")
    emb, scores = npz["emb"], npz["scores"]
    meta["orig_idx"] = np.arange(len(meta))
    meta = meta.sort_values(["stem", "window_idx"], kind="stable").reset_index(drop=True)
    reorder = meta["orig_idx"].to_numpy()
    emb, scores = emb[reorder], scores[reorder]

    print(f"[{time.strftime('%H:%M:%S')}] loading Phase-2 pseudo-labels")
    pseudo = np.load(PSEUDO_LABEL_DIR / "pseudo_soft_labels.npz")
    pseudo_stems      = pseudo["stems"]
    pseudo_soft       = pseudo["soft_labels"]
    pseudo_species_id = pseudo["species_id"]
    pseudo_df = pd.read_parquet(PSEUDO_LABEL_DIR / "pseudo_labels.parquet")

    retained_mask = pseudo_df["retained"].to_numpy()
    primary_idx   = pseudo_df["primary_label_idx"].to_numpy()
    n_retained = int(retained_mask.sum())
    print(f"  retained: {n_retained:,} / {len(pseudo_df):,} ({n_retained/len(pseudo_df):.1%})")

    retained_stems = set(pseudo_stems[retained_mask].tolist())
    keep_rows = meta["stem"].isin(retained_stems).to_numpy()
    meta   = meta.loc[keep_rows].reset_index(drop=True)
    emb    = emb[keep_rows]
    scores = scores[keep_rows]

    stems = meta["stem"].to_numpy()
    stem_starts = np.concatenate(
        [[0], np.where(stems[1:] != stems[:-1])[0] + 1, [len(stems)]])
    unique_stems = stems[stem_starts[:-1]]
    n_clips = len(unique_stems)
    assert n_clips == n_retained, (n_clips, n_retained)

    stem_to_pseudo_row = {s: i for i, s in enumerate(pseudo_stems)}
    pseudo_rows = np.array([stem_to_pseudo_row[s] for s in unique_stems])
    soft  = pseudo_soft[pseudo_rows]             # (n_clips, 234) float32
    prim  = primary_idx[pseudo_rows]             # (n_clips,) int64

    targets = soft.copy()
    assert (prim >= 0).all(), "unmapped species among retained clips"
    targets[np.arange(n_clips), prim] = np.maximum(targets[np.arange(n_clips), prim], 1.0)
    print(f"  focal clips ready: {n_clips:,} clips, "
          f"{len(emb):,} total windows (mean {len(emb)/n_clips:.1f}/clip)")
    return emb, scores, stem_starts, targets


def sample_focal_chunks(emb_all, scores_all, stem_starts, clip_idxs, rng):
    """For each clip in clip_idxs, sample a random 12-window contiguous chunk
    (zero-padded if T<12).

    Returns chunk_emb (B, 12, 1536), chunk_scores (B, 12, 234).
    """
    B = len(clip_idxs)
    chunk_emb    = np.zeros((B, N_WINDOWS, emb_all.shape[1]), dtype=np.float32)
    chunk_scores = np.zeros((B, N_WINDOWS, scores_all.shape[1]), dtype=np.float32)
    for i, ci in enumerate(clip_idxs):
        s, e = stem_starts[ci], stem_starts[ci + 1]
        T = e - s
        if T <= N_WINDOWS:
            chunk_emb[i, :T]    = emb_all[s:e]
            chunk_scores[i, :T] = scores_all[s:e]
        else:
            offset = rng.integers(0, T - N_WINDOWS + 1)
            chunk_emb[i]    = emb_all[s + offset : s + offset + N_WINDOWS]
            chunk_scores[i] = scores_all[s + offset : s + offset + N_WINDOWS]
    return chunk_emb, chunk_scores


# ── Training ───────────────────────────────────────────────────────────

def train_student(sc, focal, args):
    """Fine-tune a student initialized from the §10 teacher.

    Training matches src/train_protossm_local.py: full-batch soundscape GD,
    one optimizer step per epoch, OneCycleLR with steps_per_epoch=1.
    Focal pseudo-labels (if provided) are added as a *separate* forward pass
    whose loss is accumulated before each optimizer.step(). This keeps the
    soundscape signal clean (with site/hour metadata) while still pulling
    extra signal from focal clips.
    """
    no_focal = focal is None
    if not no_focal:
        emb_fo, scores_fo, stem_starts, targets_fo = focal
        n_focal = len(targets_fo)
    emb_sc     = sc["emb_files"].astype(np.float32)        # (n_files, 12, 1536)
    logits_sc  = sc["logits_files"].astype(np.float32)
    labels_sc  = sc["labels_files"].astype(np.float32)
    sites_all  = sc["site_ids_all"].astype(np.int64)
    hours_all  = sc["hours_all"].astype(np.int64)
    n_classes  = sc["N_CLASSES"]

    rng = np.random.default_rng(args.seed)
    n_sc = len(emb_sc)
    perm = rng.permutation(n_sc)
    n_val = max(6, int(round(0.2 * n_sc)))
    val_idx = np.sort(perm[:n_val])
    tr_idx  = np.sort(perm[n_val:])
    print(f"[{time.strftime('%H:%M:%S')}] soundscape split: "
          f"train={len(tr_idx)}, val={len(val_idx)}")

    # Static on-device tensors (full-batch, one step per epoch)
    def _to(x, dtype):
        return torch.tensor(x, dtype=dtype).to(DEVICE)

    emb_tr, scores_tr, labels_tr = _to(emb_sc[tr_idx], torch.float32), \
        _to(logits_sc[tr_idx], torch.float32), _to(labels_sc[tr_idx], torch.float32)
    site_tr = _to(sites_all[tr_idx], torch.long)
    hour_tr = _to(hours_all[tr_idx], torch.long)
    emb_v, scores_v, labels_v = _to(emb_sc[val_idx], torch.float32), \
        _to(logits_sc[val_idx], torch.float32), _to(labels_sc[val_idx], torch.float32)
    site_v = _to(sites_all[val_idx], torch.long)
    hour_v = _to(hours_all[val_idx], torch.long)

    # Build + initialize from teacher
    model = _build_protossm_from_teacher_config(n_classes).to(DEVICE)
    if args.init_from_teacher:
        _load_teacher_state_into(model)
        print(f"  student initialized from teacher ckpt")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  student params: {n_params:,}")

    pos_counts = labels_tr.sum(dim=(0, 1))
    total = len(tr_idx) * N_WINDOWS
    pos_weight = ((total - pos_counts) / (pos_counts + 1)).clamp(max=25.0)

    print(f"[{time.strftime('%H:%M:%S')}] training: {args.epochs} epochs × 1 step/epoch"
          f" (full-batch soundscape{'' if no_focal else f' + {args.focal_batch_size} focal chunks'})"
          f", lr={args.lr}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        epochs=args.epochs, steps_per_epoch=1,
        pct_start=0.1, anneal_strategy="cos")

    best_val_auc = -1.0
    best_state   = None
    history      = []

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        optimizer.zero_grad()

        # Soundscape full-batch forward
        species_out, _, _ = model(emb_tr, perch_logits=scores_tr,
                                  site_ids=site_tr, hours=hour_tr)
        loss_sc = F.binary_cross_entropy_with_logits(
            species_out, labels_tr, pos_weight=pos_weight[None, None, :])
        loss_distill = F.mse_loss(species_out, scores_tr)
        loss = loss_sc + args.distill_weight * loss_distill
        parts = {"sc": loss_sc.item(), "distill": loss_distill.item()}

        # Focal forward (if enabled) — separate pass, no site/hour metadata
        if not no_focal and args.focal_batch_size > 0:
            fb = rng.integers(0, n_focal, size=args.focal_batch_size)
            emb_fb, scores_fb = sample_focal_chunks(
                emb_fo, scores_fo, stem_starts, fb, rng)
            labels_fb = np.broadcast_to(
                targets_fo[fb, None, :],
                (len(fb), N_WINDOWS, n_classes)).copy()
            emb_ft    = torch.tensor(emb_fb).to(DEVICE)
            scores_ft = torch.tensor(scores_fb).to(DEVICE)
            labels_ft = torch.tensor(labels_fb).to(DEVICE)
            species_out_f, _, _ = model(emb_ft, perch_logits=scores_ft,
                                        site_ids=None, hours=None)
            loss_focal = F.binary_cross_entropy_with_logits(
                species_out_f, labels_ft, pos_weight=pos_weight[None, None, :])
            loss = loss + args.focal_weight * loss_focal
            parts["focal"] = loss_focal.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Val
        model.eval()
        with torch.no_grad():
            out_v, _, _ = model(emb_v, perch_logits=scores_v,
                                site_ids=site_v, hours=hour_v)
            val_probs = torch.sigmoid(out_v).reshape(-1, n_classes).cpu().numpy()
            val_true  = labels_v.reshape(-1, n_classes).cpu().numpy()
            keep = val_true.sum(axis=0) > 0
            try:
                val_auc = roc_auc_score(val_true[:, keep], val_probs[:, keep],
                                        average="macro")
            except Exception:
                val_auc = 0.0
            val_loss = F.binary_cross_entropy_with_logits(
                out_v, labels_v, pos_weight=pos_weight[None, None, :]).item()

        elapsed = time.time() - t0
        mins, secs = divmod(int(elapsed), 60)
        is_best = val_auc > best_val_auc
        if is_best:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        marker = " ★ BEST" if is_best else ""

        parts_str = "  ".join(f"{k}={v:.4f}" for k, v in parts.items())
        line = (f"Epoch {epoch+1:3d}/{args.epochs}: "
                f"{parts_str}  val_loss={val_loss:.4f}  "
                f"val_roc_auc={val_auc:.4f}  time={mins}m{secs:02d}s  "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}{marker}")
        print("=" * 40)
        print(line)
        print("=" * 40)
        history.append({
            "epoch": epoch + 1, **{f"train_{k}": v for k, v in parts.items()},
            "val_loss": val_loss, "val_roc_auc": val_auc,
            "is_best": bool(is_best),
        })

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val_auc, val_idx


# ── Teacher baseline on the same val split ─────────────────────────────

def teacher_val_auc(sc, val_idx, args):
    """Score the §10 teacher on the student's val split (Phase 3 gate reference)."""
    model = _build_protossm_from_teacher_config(sc["N_CLASSES"]).to(DEVICE)
    _load_teacher_state_into(model)
    model.eval()
    emb_v    = torch.tensor(sc["emb_files"][val_idx].astype(np.float32)).to(DEVICE)
    scores_v = torch.tensor(sc["logits_files"][val_idx].astype(np.float32)).to(DEVICE)
    labels_v = torch.tensor(sc["labels_files"][val_idx].astype(np.float32)).to(DEVICE)
    site_v   = torch.tensor(sc["site_ids_all"][val_idx].astype(np.int64)).to(DEVICE)
    hour_v   = torch.tensor(sc["hours_all"][val_idx].astype(np.int64)).to(DEVICE)
    with torch.no_grad():
        out, _, _ = model(emb_v, perch_logits=scores_v,
                          site_ids=site_v, hours=hour_v)
        probs = torch.sigmoid(out).reshape(-1, sc["N_CLASSES"]).cpu().numpy()
        true  = labels_v.reshape(-1, sc["N_CLASSES"]).cpu().numpy()
        keep  = true.sum(axis=0) > 0
        return roc_auc_score(true[:, keep], probs[:, keep], average="macro")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",            type=int,   default=100)
    ap.add_argument("--lr",                type=float, default=2e-4)
    ap.add_argument("--distill-weight",    type=float, default=0.15)
    ap.add_argument("--focal-batch-size",  type=int,   default=64,
                    help="Focal chunks per step (0 = no focal).")
    ap.add_argument("--focal-weight",      type=float, default=0.25,
                    help="Scale for focal loss when added to soundscape loss.")
    ap.add_argument("--init-from-teacher", action="store_true", default=True)
    ap.add_argument("--no-init-from-teacher", dest="init_from_teacher",
                    action="store_false")
    ap.add_argument("--seed",              type=int,   default=0)
    ap.add_argument("--output-dir",        type=Path,  default=STUDENT_CKPT_DIR)
    ap.add_argument("--no-focal",          action="store_true",
                    help="Diagnostic: skip focal; train on soundscape only.")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    sc    = load_soundscape_data()
    focal = None if args.no_focal else load_focal_retained(sc["N_CLASSES"])

    model, history, best_val_auc, val_idx = train_student(sc, focal, args)
    teacher_auc = teacher_val_auc(sc, val_idx, args)
    gate_pass = best_val_auc >= teacher_auc

    ckpt_path = args.output_dir / "c2_student.pt"
    torch.save(model.state_dict(), ckpt_path)
    summary = {
        "best_val_roc_auc":        float(best_val_auc),
        "teacher_val_roc_auc":     float(teacher_auc),
        "gate_pass":               bool(gate_pass),
        "n_epochs":                args.epochs,
        "lr":                      args.lr,
        "distill_weight":          args.distill_weight,
        "focal_batch_size":        args.focal_batch_size,
        "focal_weight":            args.focal_weight,
        "init_from_teacher":       bool(args.init_from_teacher),
        "no_focal":                bool(args.no_focal),
        "seed":                    args.seed,
        "val_idx":                 [int(i) for i in val_idx],
        "elapsed_sec":             float(time.time() - t0),
        "ckpt":                    str(ckpt_path),
        "timestamp":               time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (args.output_dir / "c2_student_summary.json").write_text(
        json.dumps(summary, indent=2))
    (args.output_dir / "c2_student_history.json").write_text(
        json.dumps(history, indent=2))

    print(f"\n[{time.strftime('%H:%M:%S')}] student best val ROC-AUC: {best_val_auc:.4f}")
    print(f"[{time.strftime('%H:%M:%S')}] teacher val ROC-AUC (same split): {teacher_auc:.4f}")
    print(f"[{time.strftime('%H:%M:%S')}] Phase-3 gate (student ≥ teacher): "
          f"{'PASS' if gate_pass else 'FAIL'}")
    print(f"[{time.strftime('%H:%M:%S')}] ckpt: {ckpt_path}")


if __name__ == "__main__":
    main()
