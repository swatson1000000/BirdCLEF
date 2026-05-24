"""
§13 Phase 2 — Teacher scoring + pseudo-labeling on Kaggle-consistent Perch features.

Runs the §10 pre-trained ProtoSSM teacher over all 35,549 focal clips' Perch
embeddings (extracted in Phase 1 on a Kaggle kernel for call-path consistency)
and produces per-clip pseudo soft-labels.

Inputs
------
  kaggle_datasets/train-audio-perch/full_train_audio_perch.npz   (emb, scores)
  kaggle_datasets/train-audio-perch/full_train_audio_meta.parquet (stem, species, window_idx, ...)
  four_track/models/protossm_pretrained_v2/protossm_pretrained.pt (teacher)
  four_track/models/protossm_pretrained_v2/config.json           (teacher hyperparams)
  BirdCLEF/data/raw/sample_submission.csv                         (234-class order)

Outputs (four_track/data/processed/c2_pseudo_labels_kagglefeat/)
----------------------------------------------------------------
  pseudo_soft_labels.npz  {stems: (N,) <U32, species_id: (N,) int64,
                           soft_labels: (N, 234) float32}
  pseudo_labels.parquet   {stem, species_id, primary_label_idx, max_conf,
                           top1, top2, top3, retained}
  summary.json            {n_stems, n_retained, retention_rate, ...}

Filter gate (§13 Phase 2): retained = (max_conf > 0.6) ∧ (primary_label ∈ top3).
Phase-2 pass criterion: retention rate ≥ 80%.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_protossm_local import ProtoSSMv2, N_WINDOWS

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT  = Path(__file__).resolve().parent.parent
BIRDCLEF = PROJECT.parent

DEFAULT_INPUT_DIR  = BIRDCLEF / "kaggle_datasets" / "train-audio-perch"
DEFAULT_CKPT_DIR   = PROJECT  / "models" / "protossm_pretrained_v2"
DEFAULT_OUTPUT_DIR = PROJECT  / "data" / "processed" / "c2_pseudo_labels_kagglefeat"
SAMPLE_SUB_CSV     = BIRDCLEF / "data" / "raw" / "sample_submission.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Helpers ────────────────────────────────────────────────────────────

def load_primary_labels() -> list[str]:
    cols = pd.read_csv(SAMPLE_SUB_CSV, nrows=0).columns.tolist()
    labels = [str(c) for c in cols[1:]]
    assert len(labels) == 234, f"expected 234 primary labels, got {len(labels)}"
    return labels


def build_teacher(ckpt_dir: Path) -> ProtoSSMv2:
    cfg = json.loads((ckpt_dir / "config.json").read_text())["ssm_cfg"]
    model = ProtoSSMv2(
        d_input=1536,
        d_model=cfg["d_model"],
        d_state=cfg["d_state"],
        n_ssm_layers=cfg["n_ssm_layers"],
        n_classes=234,
        n_windows=N_WINDOWS,
        dropout=cfg["dropout"],
        n_sites=cfg["n_sites"],
        meta_dim=cfg["meta_dim"],
        use_cross_attn=cfg["use_cross_attn"],
        cross_attn_heads=cfg["cross_attn_heads"],
    ).to(DEVICE)
    state = torch.load(ckpt_dir / "protossm_pretrained.pt", map_location=DEVICE)
    missing, unexpected = model.load_state_dict(state, strict=False)
    extra = [k for k in unexpected if not k.startswith(("family_head", "class_to_family"))]
    if missing or extra:
        raise RuntimeError(
            f"state_dict mismatch: missing={missing}, unexpected={extra}")
    model.eval()
    return model


def chunk_clip(emb_clip: np.ndarray, scores_clip: np.ndarray):
    """Slice a clip's windows into 12-window chunks with zero-padding.

    Returns (emb_chunks, scores_chunks, valid_mask) with shapes
    (n_chunks, 12, 1536), (n_chunks, 12, 234), (n_chunks, 12) uint8.
    """
    T = emb_clip.shape[0]
    n_chunks = max(1, math.ceil(T / N_WINDOWS))
    pad_len = n_chunks * N_WINDOWS - T
    if pad_len > 0:
        emb_clip = np.concatenate(
            [emb_clip, np.zeros((pad_len, emb_clip.shape[1]), dtype=emb_clip.dtype)], axis=0)
        scores_clip = np.concatenate(
            [scores_clip, np.zeros((pad_len, scores_clip.shape[1]), dtype=scores_clip.dtype)], axis=0)
    emb_chunks    = emb_clip.reshape(n_chunks, N_WINDOWS, -1)
    scores_chunks = scores_clip.reshape(n_chunks, N_WINDOWS, -1)
    valid = np.ones((n_chunks, N_WINDOWS), dtype=np.uint8)
    if pad_len > 0:
        valid[-1, N_WINDOWS - pad_len:] = 0
    return emb_chunks, scores_chunks, valid


# ── Main ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir",  type=Path, default=DEFAULT_INPUT_DIR)
    ap.add_argument("--ckpt-dir",   type=Path, default=DEFAULT_CKPT_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--batch-size", type=int,  default=64,
                    help="chunks per forward pass")
    ap.add_argument("--conf-thresh", type=float, default=0.6,
                    help="max_conf threshold for retained filter")
    ap.add_argument("--smoke", type=int, default=0,
                    help="limit to first N stems (0 = all)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading features from {args.input_dir}")
    meta = pd.read_parquet(args.input_dir / "full_train_audio_meta.parquet")
    npz  = np.load(args.input_dir / "full_train_audio_perch.npz")
    emb_all    = npz["emb"]     # (N, 1536) float32
    scores_all = npz["scores"]  # (N, 234)  float32
    assert len(meta) == len(emb_all) == len(scores_all)

    primary_labels = load_primary_labels()
    species_to_idx = {sp: i for i, sp in enumerate(primary_labels)}

    meta["orig_idx"] = np.arange(len(meta))
    meta = meta.sort_values(["stem", "window_idx"], kind="stable").reset_index(drop=True)
    reorder = meta["orig_idx"].to_numpy()
    emb_all    = emb_all[reorder]
    scores_all = scores_all[reorder]

    stems = meta["stem"].to_numpy()
    species_ids = meta["species"].to_numpy()
    stem_starts = np.concatenate([[0], np.where(stems[1:] != stems[:-1])[0] + 1, [len(stems)]])
    unique_stems = stems[stem_starts[:-1]]
    unique_species = species_ids[stem_starts[:-1]]
    n_stems = len(unique_stems)
    print(f"  {n_stems} unique stems, {len(meta)} total windows")

    if args.smoke > 0:
        n_stems = min(args.smoke, n_stems)
        unique_stems = unique_stems[:n_stems]
        unique_species = unique_species[:n_stems]
        stem_starts = stem_starts[:n_stems + 1]
        print(f"  SMOKE MODE: limiting to first {n_stems} stems")

    print(f"[{time.strftime('%H:%M:%S')}] loading teacher from {args.ckpt_dir}")
    model = build_teacher(args.ckpt_dir)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  teacher params: {n_params:,}")

    # ── Inference ──────────────────────────────────────────────────────
    print(f"[{time.strftime('%H:%M:%S')}] running teacher on {n_stems} stems (batch={args.batch_size})")
    clip_probs = np.zeros((n_stems, 234), dtype=np.float32)

    # Accumulate chunks across stems until batch_size is reached; then forward.
    buf_emb:    list[np.ndarray] = []
    buf_scores: list[np.ndarray] = []
    buf_valid:  list[np.ndarray] = []
    buf_owner:  list[int]        = []  # stem index per chunk

    def flush():
        if not buf_emb:
            return
        emb_b    = torch.from_numpy(np.stack(buf_emb)).to(DEVICE)
        scores_b = torch.from_numpy(np.stack(buf_scores)).to(DEVICE)
        valid_b  = np.stack(buf_valid).astype(bool)
        with torch.no_grad():
            logits, _, _ = model(emb_b, perch_logits=scores_b, site_ids=None, hours=None)
            probs = torch.sigmoid(logits).cpu().numpy()  # (B, 12, 234)
        for i, owner in enumerate(buf_owner):
            v = valid_b[i]           # (12,) bool
            if v.any():
                chunk_max = probs[i, v].max(axis=0)  # (234,)
                np.maximum(clip_probs[owner], chunk_max, out=clip_probs[owner])
        buf_emb.clear(); buf_scores.clear(); buf_valid.clear(); buf_owner.clear()

    t_inf = time.time()
    last_log = t_inf
    for si in range(n_stems):
        s, e = stem_starts[si], stem_starts[si + 1]
        emb_clip    = emb_all[s:e]
        scores_clip = scores_all[s:e]
        e_chunks, s_chunks, v_chunks = chunk_clip(emb_clip, scores_clip)
        for c in range(e_chunks.shape[0]):
            buf_emb.append(e_chunks[c])
            buf_scores.append(s_chunks[c])
            buf_valid.append(v_chunks[c])
            buf_owner.append(si)
            if len(buf_emb) >= args.batch_size:
                flush()
        now = time.time()
        if now - last_log > 30 or si == n_stems - 1:
            rate = (si + 1) / (now - t_inf + 1e-9)
            eta  = (n_stems - si - 1) / max(rate, 1e-9)
            print(f"  [{time.strftime('%H:%M:%S')}] stem {si + 1}/{n_stems} "
                  f"({rate:.1f}/s, eta {eta:.0f}s)", flush=True)
            last_log = now
    flush()
    print(f"[{time.strftime('%H:%M:%S')}] inference done in {time.time() - t_inf:.1f}s")

    # ── Reduce + filter ────────────────────────────────────────────────
    max_conf = clip_probs.max(axis=1)
    top3 = np.argsort(clip_probs, axis=1)[:, -3:][:, ::-1]  # (n_stems, 3)

    primary_idx = np.array(
        [species_to_idx.get(str(sp), -1) for sp in unique_species], dtype=np.int64)
    n_unmapped = int((primary_idx < 0).sum())
    if n_unmapped:
        print(f"  WARNING: {n_unmapped} stems have species_id not in the 234-class space")

    in_top3  = np.array(
        [pi in t3 for pi, t3 in zip(primary_idx, top3)], dtype=bool)
    retained = (max_conf > args.conf_thresh) & in_top3

    retention = retained.mean()
    print(f"[{time.strftime('%H:%M:%S')}] retained {retained.sum():,} / {n_stems:,} "
          f"({retention:.1%})")
    gate_pass = retention >= 0.80
    print(f"  Phase-2 gate (≥80%): {'PASS' if gate_pass else 'FAIL'}")

    # ── Write outputs ──────────────────────────────────────────────────
    np.savez_compressed(
        args.output_dir / "pseudo_soft_labels.npz",
        stems=np.array(unique_stems, dtype="<U32"),
        species_id=np.array(unique_species, dtype="<U16"),
        soft_labels=clip_probs,
    )

    df = pd.DataFrame({
        "stem":              unique_stems,
        "species_id":        unique_species.astype(str),
        "primary_label_idx": primary_idx,
        "max_conf":          max_conf,
        "top1":              top3[:, 0],
        "top2":              top3[:, 1],
        "top3":              top3[:, 2],
        "retained":          retained,
    })
    df.to_parquet(args.output_dir / "pseudo_labels.parquet", index=False)

    summary = {
        "n_stems":               int(n_stems),
        "n_retained":            int(retained.sum()),
        "retention_rate":        float(retention),
        "gate_pass":             bool(gate_pass),
        "conf_thresh":           args.conf_thresh,
        "n_species_unmapped":    n_unmapped,
        "teacher_ckpt":          str(args.ckpt_dir / "protossm_pretrained.pt"),
        "input_dir":             str(args.input_dir),
        "elapsed_sec":           float(time.time() - t0),
        "timestamp":             time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{time.strftime('%H:%M:%S')}] outputs written to {args.output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
