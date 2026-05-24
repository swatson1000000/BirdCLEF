"""Phase A of §33 — build Cerrado-geo-filtered XC v3 pretrain manifest.

Filters the xeno-canto v3 bulk corpus at
`data/external/xenocanto_bulk/{species_code}/_meta.json` by:
  - Cerrado biome bbox: lat ∈ [-24, -2], lon ∈ [-60, -41]
  - length 5..120 seconds (drop stubs and outliers)
  - audio file actually present on disk at `{species_code}/XC{id}.ogg`
  - dedupe by XC id (safety; shouldn't occur within the bulk download)

Emits `data/processed/xc_cerrado_pretrain_manifest.csv` with columns
  species_code, xc_id, audio_path, lat, lon, length_sec, quality, country, gen, sp

Per-species count histogram printed at the end. Gate per §33 PICK UP HERE
action sequence:
  PASS:  >= 1000 species with >= 5 recordings each   (proceed to Phase B)
  FAIL:  fewer — corpus too long-tailed; manifest written for inspection
         but the next step is to abort / pivot, not pretrain.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

XC_ROOT = Path(
    "/home/swatson/work/kaggle/BirdCLEF/data/external/xenocanto_bulk"
)
FT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = FT_ROOT / "data" / "processed" / "xc_cerrado_pretrain_manifest.csv"

# Cerrado biome bbox (chosen 2026-05-17 per new_plan.md §33.2; broader than
# strict Pantanal-only which had only 1634 recs, narrower than all-Brazil
# which drifts toward the killed L2-redux mechanism).
LAT_LO, LAT_HI = -24.0, -2.0
LON_LO, LON_HI = -60.0, -41.0

LEN_MIN_SEC = 5.0
LEN_MAX_SEC = 120.0

# Phase B gate (per new_plan.md §33 action 1)
MIN_SPECIES_AT_MIN_COUNT = 1000
MIN_RECORDS_PER_SPECIES = 5


def parse_length_seconds(length: str | None) -> float | None:
    """XC length field is 'mm:ss' or 'h:mm:ss' or sometimes weird."""
    if not length or not isinstance(length, str):
        return None
    parts = length.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except (ValueError, TypeError):
        return None
    return None


def main() -> int:
    if not XC_ROOT.exists():
        sys.exit(f"missing XC root: {XC_ROOT}")

    t0 = time.time()
    species_dirs = sorted([p for p in XC_ROOT.iterdir() if p.is_dir()])
    print(f"[scan] {len(species_dirs)} species dirs at {XC_ROOT}", flush=True)

    rows: list[dict] = []
    seen_ids: set[str] = set()

    # Counters for end-of-run audit
    n_total = 0
    n_geo_pass = 0
    n_len_pass = 0
    n_file_present = 0
    n_dedup_drop = 0

    for i, sp_dir in enumerate(species_dirs):
        if i % 500 == 0 and i > 0:
            print(
                f"  [progress] {i}/{len(species_dirs)} species  "
                f"rows so far={len(rows):,}  ({time.time() - t0:.1f}s)",
                flush=True,
            )
        meta_path = sp_dir / "_meta.json"
        if not meta_path.exists():
            continue
        try:
            with meta_path.open() as f:
                records = json.load(f)
        except Exception as ex:
            print(f"  [warn] {meta_path.name}: {ex}", flush=True)
            continue
        if not isinstance(records, list):
            records = [records]

        for r in records:
            n_total += 1
            lat = r.get("lat")
            lon = r.get("lon")
            try:
                lat_f = float(lat)
                lon_f = float(lon)
            except (TypeError, ValueError):
                continue
            if not (LAT_LO <= lat_f <= LAT_HI and LON_LO <= lon_f <= LON_HI):
                continue
            n_geo_pass += 1

            length_sec = parse_length_seconds(r.get("length"))
            if length_sec is None or not (
                LEN_MIN_SEC <= length_sec <= LEN_MAX_SEC
            ):
                continue
            n_len_pass += 1

            xc_id = str(r.get("id") or "").strip()
            if not xc_id:
                continue
            audio_path = sp_dir / f"XC{xc_id}.ogg"
            if not audio_path.exists():
                continue
            n_file_present += 1

            if xc_id in seen_ids:
                n_dedup_drop += 1
                continue
            seen_ids.add(xc_id)

            rows.append(
                {
                    "species_code": sp_dir.name,
                    "xc_id": xc_id,
                    "audio_path": str(audio_path),
                    "lat": lat_f,
                    "lon": lon_f,
                    "length_sec": length_sec,
                    "quality": (r.get("q") or "").strip(),
                    "country": (r.get("cnt") or "").strip(),
                    "gen": (r.get("gen") or "").strip(),
                    "sp": (r.get("sp") or "").strip(),
                    # author (rec field) for GroupShuffleSplit anti-leakage
                    # in §33 Phase B pretrain — must not be empty for the
                    # split to work, fall back to xc_id (unique) if missing.
                    "author": (r.get("rec") or f"__xc{xc_id}").strip(),
                }
            )

    elapsed = time.time() - t0
    print(f"[scan complete] {elapsed:.1f}s", flush=True)
    print("", flush=True)
    print("=" * 60, flush=True)
    print("Filter pass-through audit", flush=True)
    print("=" * 60, flush=True)
    print(f"  total records seen :   {n_total:>10,}", flush=True)
    print(f"  geo bbox pass      :   {n_geo_pass:>10,}", flush=True)
    print(f"  length 5..120s pass:   {n_len_pass:>10,}", flush=True)
    print(f"  file present on disk:  {n_file_present:>10,}", flush=True)
    print(f"  dedup drops        :   {n_dedup_drop:>10,}", flush=True)
    print(f"  final manifest rows:   {len(rows):>10,}", flush=True)

    # Per-species count histogram + gate check
    counts = Counter(r["species_code"] for r in rows)
    n_species = len(counts)
    n_species_ge5 = sum(1 for c in counts.values() if c >= MIN_RECORDS_PER_SPECIES)

    print("", flush=True)
    print("=" * 60, flush=True)
    print(f"Per-species coverage ({n_species:,} species in manifest)", flush=True)
    print("=" * 60, flush=True)

    # Bucket distribution
    buckets = [
        (1, 1, "1 rec"),
        (2, 4, "2-4 recs"),
        (5, 9, "5-9 recs"),
        (10, 49, "10-49 recs"),
        (50, 199, "50-199 recs"),
        (200, 10_000, "200+ recs"),
    ]
    for lo, hi, label in buckets:
        n = sum(1 for c in counts.values() if lo <= c <= hi)
        print(f"  {label:>12} : {n:>5} species", flush=True)

    print("", flush=True)
    print("  Top 20 species by record count:", flush=True)
    for sp, c in counts.most_common(20):
        print(f"    {sp:>10} : {c:>5}", flush=True)

    # Phase B gate check
    print("", flush=True)
    print("=" * 60, flush=True)
    print("§33 Phase A → B GATE", flush=True)
    print("=" * 60, flush=True)
    print(
        f"  species with >= {MIN_RECORDS_PER_SPECIES} recordings: "
        f"{n_species_ge5:>5} / required {MIN_SPECIES_AT_MIN_COUNT}",
        flush=True,
    )
    if n_species_ge5 >= MIN_SPECIES_AT_MIN_COUNT:
        verdict = (
            f"PASS  ({n_species_ge5} >= {MIN_SPECIES_AT_MIN_COUNT})  -> "
            f"proceed to §33 Phase B (pretrain B0 on Cerrado corpus)"
        )
    else:
        verdict = (
            f"FAIL  ({n_species_ge5} < {MIN_SPECIES_AT_MIN_COUNT})  -> "
            f"corpus too long-tailed; pivot to (ii) accept LB 0.933"
        )
    print(f"  verdict: {verdict}", flush=True)
    print("=" * 60, flush=True)

    # Write manifest
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "species_code", "xc_id", "audio_path",
        "lat", "lon", "length_sec",
        "quality", "country", "gen", "sp", "author",
    ]
    with OUT_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[save] {OUT_PATH}  ({len(rows):,} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
