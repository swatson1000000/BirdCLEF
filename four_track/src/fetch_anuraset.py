"""L5b-Amphibia: AnuraSet fetch + preprocess.

Pulls the 10 Pantanal-overlap species from AnuraSet (Cañas et al. 2023,
CC-BY-1.0, doi:10.5281/zenodo.8342596) and writes them as additional focal
clips matching the `train_audio` layout, plus an extension CSV that mirrors
`train.csv` schema for later concatenation into `train_folds.csv`.

Pipeline (gated by CLI phase flags; default = preview only):

    --phase meta      Download species.csv, weak_labels.csv, strong_labels.zip
    --phase plan      Build the 10-species filter, list which raw recordings
                      will be touched, dry-run the per-species clip yield
    --phase audio     Download raw_data.zip (7.21 GB), selectively unpack only
                      the recordings touched by the plan
    --phase cut       For each strong-label segment of a target species, cut a
                      5 s window centered on the segment midpoint, resample to
                      32 kHz, write OGG into data/external/anuraset_focal/
    --phase csv       Emit data/processed/anuraset_supplement.csv (train.csv
                      schema, ready for concat into train_folds.csv)
    --phase all       Run every phase end-to-end

Idempotent: each phase skips work whose outputs already exist.

Usage:
    # Preview only (no downloads)
    python -u src/fetch_anuraset.py --phase meta
    python -u src/fetch_anuraset.py --phase plan

    # Full pipeline once preview looks right
    python -u src/fetch_anuraset.py --phase all
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T

# ── Path wiring ───────────────────────────────────────────────────────────────
HERE    = Path(__file__).resolve().parent
FT_ROOT = HERE.parent
ROOT    = FT_ROOT.parent

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from config import RAW, PROC, SAMPLE_RATE, CHUNK_SAMPLES  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
ZENODO_BASE = "https://zenodo.org/records/8342596/files"
ANURA_SR    = 22050  # AnuraSet native rate

EXT_ROOT    = FT_ROOT / "data" / "external" / "anuraset"
META_DIR    = EXT_ROOT / "_meta"
RAW_AUDIO   = EXT_ROOT / "raw_audio"          # extracted .wav files (60 s)
FOCAL_OUT   = FT_ROOT / "data" / "external" / "anuraset_focal"
SUPPL_CSV   = FT_ROOT / "data" / "processed" / "anuraset_supplement.csv"

# 10-species AnuraSet ↔ Pantanal overlap (verified 2026-04-18 via species.csv
# join against probe_class_name_auc.csv). Maps the 6-letter AnuraSet CODE
# (used in strong-label .txt files) → (binomial, Pantanal primary_label).
OVERLAP_CODES = {
    "BOARAN": ("Boana raniceps",            "555146"),
    "DENMIN": ("Dendropsophus minutus",     "65377"),
    "DENNAN": ("Dendropsophus nanus",       "65380"),
    "ELABIC": ("Elachistocleis bicolor",    "25092"),
    "LEPELE": ("Leptodactylus elenae",      "22967"),
    "LEPFUS": ("Leptodactylus fuscus",      "22973"),
    "LEPPOD": ("Leptodactylus podicipinus", "22961"),
    "PHYALB": ("Physalaemus albonotatus",   "23158"),
    "PITAZU": ("Pithecopus azureus",        "517063"),
    "SCINAS": ("Scinax nasicus",            "24279"),
}
OVERLAP_SPECIES = {sp: pl for (sp, pl) in OVERLAP_CODES.values()}


def _download(filename: str, dest_dir: Path) -> Path:
    """Stream a Zenodo file to dest_dir; skip if already present + sized."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / filename
    if out.exists() and out.stat().st_size > 0:
        print(f"  [skip] {out.name} already present ({out.stat().st_size/1e6:.1f} MB)", flush=True)
        return out
    url = f"{ZENODO_BASE}/{filename}"
    print(f"  [get ] {url}", flush=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(out, "wb") as f:
            done = 0
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total > 0 and done % (50 << 20) < (1 << 20):
                    print(f"    {done/1e6:.0f}/{total/1e6:.0f} MB", flush=True)
    print(f"  [ok  ] {out.name} ({out.stat().st_size/1e6:.1f} MB)", flush=True)
    return out


# ── Phase: meta ───────────────────────────────────────────────────────────────

def phase_meta() -> None:
    print("[phase=meta] Downloading AnuraSet metadata", flush=True)
    _download("species.csv",       META_DIR)
    _download("weak_labels.csv",   META_DIR)
    z = _download("strong_labels.zip", META_DIR)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(META_DIR)
    txts = list((META_DIR / "strong_labels").rglob("*.txt"))
    assert txts, "strong_labels/*.txt missing after unzip"
    print(f"  strong_labels: {len(txts)} per-recording .txt files (Audacity TSV)", flush=True)


# ── Phase: plan ───────────────────────────────────────────────────────────────

def _load_strong_labels() -> pd.DataFrame:
    """Walk strong_labels/<site>/<stem>.txt files and concat into a single DF.

    Each .txt is Audacity-format TSV: t_start \\t t_end \\t code_quality
    where code_quality looks like 'BOARAN_M' (CODE + '_' + Q ∈ {L,M,H}).
    Returns a frame with columns: filename, t_start, t_end, code, quality.
    """
    root = META_DIR / "strong_labels"
    assert root.exists(), "Run --phase meta first"
    rows = []
    for txt in root.rglob("*.txt"):
        stem = txt.stem  # e.g. INCT20955_20191014_020000
        wav_name = f"{stem}.wav"
        try:
            df = pd.read_csv(txt, sep="\t", header=None,
                             names=["t_start", "t_end", "code_q"],
                             dtype={"t_start": float, "t_end": float, "code_q": str})
        except Exception as ex:
            print(f"  [warn] parse fail {txt.name}: {ex}", flush=True)
            continue
        if df.empty:
            continue
        cq = df["code_q"].str.rsplit("_", n=1, expand=True)
        df["code"]    = cq[0]
        df["quality"] = cq[1]
        df["filename"] = wav_name
        df["site"]     = txt.parent.name
        rows.append(df[["filename", "site", "t_start", "t_end", "code", "quality"]])
    out = pd.concat(rows, ignore_index=True)
    return out


def phase_plan() -> dict:
    print("[phase=plan] Building per-species clip plan", flush=True)
    sl = _load_strong_labels()
    print(f"  strong_labels parsed: {len(sl)} segments across {sl['filename'].nunique()} recordings", flush=True)
    print(f"  quality breakdown: {sl['quality'].value_counts().to_dict()}", flush=True)

    keep = sl[sl["code"].isin(OVERLAP_CODES)].copy()
    print(f"  10-species filter: {len(keep)} / {len(sl)} segments", flush=True)
    keep_q = keep[keep["quality"].isin(["H", "M"])]
    print(f"  quality H+M filter: {len(keep_q)} / {len(keep)} segments", flush=True)

    plan = {}
    for code, sub in keep_q.groupby("code"):
        sp, primary = OVERLAP_CODES[code]
        plan[sp] = {
            "code":          code,
            "primary_label": primary,
            "n_segments":    int(len(sub)),
            "n_recordings":  int(sub["filename"].nunique()),
            "sites":         sorted(sub["site"].unique().tolist()),
        }
    plan_path = META_DIR / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    print("\n  per-species plan (H+M only):", flush=True)
    for sp, info in sorted(plan.items(), key=lambda x: -x[1]["n_segments"]):
        print(f"    {sp:30s} {info['code']}  segs={info['n_segments']:5d}  "
              f"recordings={info['n_recordings']:4d}  sites={','.join(info['sites'])}  → {info['primary_label']}", flush=True)
    n_unique_files = keep_q["filename"].nunique()
    print(f"\n  total unique recordings to extract from raw_data.zip: {n_unique_files}", flush=True)
    print(f"  [ok] plan.json → {plan_path}", flush=True)
    return plan


# ── Phase: audio (selective extract) ──────────────────────────────────────────

def phase_audio() -> None:
    print("[phase=audio] Downloading raw_data.zip + selective extract", flush=True)
    z = _download("raw_data.zip", EXT_ROOT)
    sl = _load_strong_labels()
    keep = sl[sl["code"].isin(OVERLAP_CODES) & sl["quality"].isin(["H", "M"])]
    needed = set(keep["filename"].unique())
    print(f"  recordings to extract: {len(needed)}", flush=True)
    RAW_AUDIO.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        # Match by basename — raw_data.zip nests under site dirs
        wanted = [n for n in names if Path(n).name in needed]
        print(f"  matched in zip: {len(wanted)}/{len(needed)}", flush=True)
        extracted = 0
        for n in wanted:
            out = RAW_AUDIO / Path(n).name
            if out.exists():
                continue
            with zf.open(n) as src, open(out, "wb") as dst:
                dst.write(src.read())
            extracted += 1
        print(f"  extracted {extracted} new files; total in raw_audio/: {len(list(RAW_AUDIO.glob('*.wav')))}", flush=True)


# ── Phase: cut focal clips ────────────────────────────────────────────────────

def _resample_cache():
    return T.Resample(orig_freq=ANURA_SR, new_freq=SAMPLE_RATE)


def phase_cut() -> None:
    print("[phase=cut] Cutting 5 s focal clips around strong-label segments", flush=True)
    sl   = _load_strong_labels()
    keep = sl[sl["code"].isin(OVERLAP_CODES) & sl["quality"].isin(["H", "M"])].copy()
    print(f"  segments to cut: {len(keep)} from {keep['filename'].nunique()} recordings", flush=True)

    resampler = _resample_cache()
    half_chunk_sec = (CHUNK_SAMPLES / SAMPLE_RATE) / 2.0   # 2.5 s
    written = {code: 0 for code in OVERLAP_CODES}

    for src_name, segs in keep.groupby("filename"):
        wav_path = RAW_AUDIO / src_name
        if not wav_path.exists():
            print(f"  [warn] missing {src_name} (skipping {len(segs)} segs)", flush=True)
            continue
        wav, sr = sf.read(str(wav_path), dtype="float32")
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if sr != ANURA_SR:
            wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, ANURA_SR).numpy()
        wav32 = resampler(torch.from_numpy(wav).unsqueeze(0)).squeeze(0).numpy()
        n_total = len(wav32)
        for _, row in segs.iterrows():
            code   = row["code"]
            t_mid  = (float(row["t_start"]) + float(row["t_end"])) / 2.0
            s = int((t_mid - half_chunk_sec) * SAMPLE_RATE)
            e = s + CHUNK_SAMPLES
            if s < 0:
                s, e = 0, CHUNK_SAMPLES
            if e > n_total:
                e = n_total
                s = max(0, e - CHUNK_SAMPLES)
            seg = wav32[s:e]
            if len(seg) < CHUNK_SAMPLES:
                pad = np.zeros(CHUNK_SAMPLES - len(seg), dtype=np.float32)
                seg = np.concatenate([seg, pad])
            _, primary = OVERLAP_CODES[code]
            out_dir = FOCAL_OUT / primary
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(src_name).stem
            clip_id = f"AS_{stem}_{int(row['t_start']*1000):07d}.ogg"
            out = out_dir / clip_id
            if out.exists():
                continue
            sf.write(str(out), seg, SAMPLE_RATE, format="OGG", subtype="VORBIS")
            written[code] += 1
    print("\n  per-species clips written this run:", flush=True)
    for code, n in sorted(written.items(), key=lambda x: -x[1]):
        sp, primary = OVERLAP_CODES[code]
        total = len(list((FOCAL_OUT / primary).glob("*.ogg"))) if (FOCAL_OUT / primary).exists() else 0
        print(f"    {sp:30s} +{n:4d}  (total on disk: {total})", flush=True)


# ── Phase: csv (train.csv schema supplement) ──────────────────────────────────

def phase_csv() -> None:
    print(f"[phase=csv] Emitting {SUPPL_CSV.name}", flush=True)
    tax = pd.read_csv(RAW / "taxonomy.csv")
    tax["primary_label"] = tax["primary_label"].astype(str)
    tax_lookup = tax.set_index("primary_label").to_dict(orient="index")

    rows = []
    for sp, primary in OVERLAP_SPECIES.items():
        sp_dir = FOCAL_OUT / primary
        if not sp_dir.exists():
            continue
        meta = tax_lookup.get(primary, {})
        for clip in sorted(sp_dir.glob("*.ogg")):
            rows.append({
                "primary_label":   primary,
                "secondary_labels": "[]",
                "type":             "[]",
                "latitude":         "",
                "longitude":        "",
                "scientific_name":  sp,
                "common_name":      meta.get("common_name", sp),
                "class_name":       "Amphibia",
                "inat_taxon_id":    meta.get("inat_taxon_id", ""),
                "author":           "AnuraSet (Canas et al. 2023)",
                "license":          "cc-by",
                "rating":           0.0,
                "url":              "https://doi.org/10.5281/zenodo.8342596",
                "filename":         f"{primary}/{clip.name}",
                "collection":       "AnuraSet",
            })
    if not rows:
        print("  [warn] no clips on disk; run --phase cut first", flush=True)
        return
    df = pd.DataFrame(rows)
    SUPPL_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SUPPL_CSV, index=False)
    print(f"  wrote {len(df)} rows → {SUPPL_CSV}", flush=True)
    print("  per-species counts:", flush=True)
    for sp, n in df["scientific_name"].value_counts().items():
        print(f"    {sp:30s} {n:4d}", flush=True)


# ── Driver ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["meta", "plan", "audio", "cut", "csv", "all"])
    args = ap.parse_args()

    if args.phase in ("meta", "all"):
        phase_meta()
    if args.phase in ("plan", "all"):
        phase_plan()
    if args.phase in ("audio", "all"):
        phase_audio()
    if args.phase in ("cut", "all"):
        phase_cut()
    if args.phase in ("csv", "all"):
        phase_csv()


if __name__ == "__main__":
    main()
