"""Run trained ProtoSSM (3-seed average) on the full Kaggle-extracted Perch
features (10658 train_soundscapes files, 12 windows each).

Loads the merged train-soundscapes-perch dataset (~127k windows), reshapes to
per-file batches of 12 windows, parses site/hour from filenames, runs the
3-seed averaged ProtoSSM forward, and saves predictions.

Important caveat: ProtoSSM was trained on 59 labeled files covering 8 sites
and 71 of 234 classes. The 10658-file pool has 23 sites; unseen-site inputs
get site_id=0 (the model's "default" slot). The 163 untrained classes get
random-init prototypes → near-uniform predictions (no signal). Pseudo
generation downstream filters by threshold to manage noise.

Output: data/processed/protossm_pseudo_soundscape.npz with keys
    probs   (n_files * 12, 234)   — sigmoid(logits), float32
    filenames (n_files * 12,)     — file per window
    start_sec (n_files * 12,)     — int32, start in seconds
"""
from __future__ import annotations
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parent.parent
BIRDCLEF = PROJECT.parent

MERGED = PROJECT / "kaggle_datasets" / "train-soundscapes-perch"
CACHE = PROJECT / "data" / "kaggle_perch_cache"
CKPT_DIR = PROJECT / "models" / "protossm_pretrained_v2"
OUT_PATH = PROJECT / "data" / "processed" / "protossm_pseudo_soundscape.npz"

N_WINDOWS = 12
BATCH_SIZE = 64

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")


def _parse_site_hour(fname: str) -> tuple[str, int]:
    m = FNAME_RE.match(fname)
    if m is None:
        return "", -1
    return m.group(2), int(m.group(4)[:2]) % 24


def main() -> int:
    sys.path.insert(0, str(PROJECT / "src"))
    # Import after sys.path mutated so we get train_protossm_local's classes
    from train_protossm_local import ProtoSSMv2, reshape_to_files  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    print("[load] merged Perch features…", flush=True)
    t0 = time.time()
    arr = np.load(MERGED / "full_train_soundscapes_perch.npz")
    emb = arr["emb"].astype(np.float32)        # (N_windows, 1536)
    scores = arr["scores"].astype(np.float32)  # (N_windows, 234)
    meta = pd.read_parquet(MERGED / "full_train_soundscapes_meta.parquet")
    assert len(meta) == emb.shape[0] == scores.shape[0]
    print(f"  emb={emb.shape}  scores={scores.shape}  rows={len(meta)}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    emb_files, file_list = reshape_to_files(emb, meta, n_windows=N_WINDOWS)
    scores_files, _ = reshape_to_files(scores, meta, n_windows=N_WINDOWS)
    n_files = len(file_list)
    print(f"  reshape: emb_files={emb_files.shape}  scores_files={scores_files.shape}  "
          f"files={n_files}", flush=True)

    # Build site_to_idx and n_sites_max identical to training-time
    print("[meta] loading site_to_idx from training cache…", flush=True)
    cache_meta = pd.read_parquet(CACHE / "full_perch_meta.parquet")
    train_sites = list(dict.fromkeys(cache_meta["site"].tolist()))
    site_to_idx = {s: i + 1 for i, s in enumerate(train_sites)}
    n_sites = 20  # value used at train-time
    print(f"  train sites: {sorted(set(train_sites))}", flush=True)

    site_ids = np.zeros(n_files, dtype=np.int64)
    hour_ids = np.zeros(n_files, dtype=np.int64)
    unseen_sites = 0
    for fi, fname in enumerate(file_list):
        site, hour = _parse_site_hour(fname)
        if site not in site_to_idx:
            unseen_sites += 1
            site_ids[fi] = 0
        else:
            site_ids[fi] = min(site_to_idx[site], n_sites - 1)
        hour_ids[fi] = hour % 24 if hour >= 0 else 0
    print(f"  unseen-site files (mapped to site_id=0): {unseen_sites} of {n_files} "
          f"({100*unseen_sites/n_files:.1f}%)", flush=True)

    # Load 3-seed average ckpt
    print(f"[ckpt] {CKPT_DIR / 'protossm_pretrained.pt'}", flush=True)
    state = torch.load(CKPT_DIR / "protossm_pretrained.pt", map_location=device)

    # Build the model with default hyperparams matching train recipe
    model = ProtoSSMv2(
        d_input=1536, d_model=320, d_state=32,
        n_ssm_layers=4, n_classes=234, n_windows=N_WINDOWS,
        dropout=0.12, n_sites=n_sites, meta_dim=24,
        use_cross_attn=True, cross_attn_heads=8,
    ).to(device)
    # Family-head was added during training; recreate to match state dict keys.
    # Number of families = number of unique values in class_to_family;
    # we don't have it here but can infer from state dict shape.
    if "family_head.weight" in state:
        n_families = state["family_head.weight"].shape[0]
        # class_to_family is stored as a buffer; we won't predict family logits
        # for pseudo generation, so passing zeros is fine.
        model.init_family_head(n_families, [0] * 234)
        model.to(device)
        print(f"  family head present: n_families={n_families}", flush=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)} sample={missing[:3]}", flush=True)
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)} sample={unexpected[:3]}", flush=True)
    model.eval()

    print(f"[infer] {n_files} files, batch={BATCH_SIZE}", flush=True)
    t0 = time.time()
    all_logits = np.zeros((n_files, N_WINDOWS, 234), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n_files, BATCH_SIZE):
            j = min(i + BATCH_SIZE, n_files)
            emb_b = torch.from_numpy(emb_files[i:j]).to(device)
            sc_b = torch.from_numpy(scores_files[i:j]).to(device)
            si_b = torch.from_numpy(site_ids[i:j]).to(device)
            hr_b = torch.from_numpy(hour_ids[i:j]).to(device)
            logits, _, _ = model(emb_b, sc_b, site_ids=si_b, hours=hr_b)
            all_logits[i:j] = logits.cpu().numpy()
            if i % (BATCH_SIZE * 16) == 0 and i > 0:
                print(f"  [{i}/{n_files}] {time.time()-t0:.0f}s", flush=True)
    print(f"[infer] done {time.time()-t0:.1f}s", flush=True)

    probs = 1.0 / (1.0 + np.exp(-all_logits))  # sigmoid
    # Reshape to flat (n_files * 12, 234)
    probs_flat = probs.reshape(-1, 234).astype(np.float32)
    filenames_flat = np.array(
        [f for f in file_list for _ in range(N_WINDOWS)],
        dtype="<U64",
    )
    start_sec_flat = np.tile(
        np.arange(0, N_WINDOWS * 5, 5, dtype=np.int32), n_files,
    )
    assert probs_flat.shape[0] == filenames_flat.shape[0] == start_sec_flat.shape[0]
    print(f"[flat] probs={probs_flat.shape}", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        probs=probs_flat,
        filenames=filenames_flat,
        start_sec=start_sec_flat,
    )
    print(f"[save] {OUT_PATH}  ({OUT_PATH.stat().st_size/1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
