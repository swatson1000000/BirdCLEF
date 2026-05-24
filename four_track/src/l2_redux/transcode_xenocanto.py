"""L2-redux Phase 2c.2b — filter + transcode rohanrao bulk to ogg vorbis.

Per new_plan.md §14.17.16.7. Walks our 6,971 target eBird codes, finds
matching directories in data/external/xenocanto_raw/{A-M,N-Z}/<code>/
(direct code match, or sciname-fallback for revision-drifted codes),
then ffmpeg-transcodes each .mp3 to 32 kHz mono ogg vorbis q4 with
duration cap (30s non-rare / 60s rare iff < 20 recs in source dir),
per-species cap 500 recordings.

Output layout:
    data/external/xenocanto_bulk/
        <target_code>/
            XC<id>.ogg     (transcoded)
            ...
        _failures.json     (failed-transcode log)

Resumable: skip target ogg if it already exists. Parallelizable via
thread pool (default 4 workers — ffmpeg is CPU-bound but releases the
GIL, so threading helps).

Usage:
    python -u src/l2_redux/transcode_xenocanto.py
    python -u src/l2_redux/transcode_xenocanto.py --max-species 5  # smoke
    python -u src/l2_redux/transcode_xenocanto.py --workers 8       # parallel
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).resolve().parent
FT_ROOT   = HERE.parent.parent
ROOT      = FT_ROOT.parent
PROC      = FT_ROOT / "data" / "processed"

RAW_ROOT  = ROOT / "data" / "external" / "xenocanto_raw"   # rohanrao dump
OUT_ROOT  = ROOT / "data" / "external" / "xenocanto_bulk"

TARGETS_JSON   = PROC / "l2_redux_xc_targets.json"
EBIRD_CSV      = PROC / "ebird_taxonomy_v2021.csv"
UNMATCHED_JSON = PROC / "l2_redux_unmatched_codes.json"
FAILURES_JSON  = OUT_ROOT / "_failures.json"

# Spec defaults from §14.17.16
DEFAULT_PER_SPECIES_CAP = 500
DEFAULT_RARE_THRESHOLD  = 20
DEFAULT_DUR_CAP_NORMAL  = 30
DEFAULT_DUR_CAP_RARE    = 60
DEFAULT_WORKERS         = 4

# Abort gates
DISK_CAP_PCT       = 92
WALL_CLOCK_CAP_SEC = 7 * 24 * 3600   # 7 days


# ── Disk + time gates ─────────────────────────────────────────────────────────

def disk_pct() -> int:
    out = subprocess.run(["df", "-P", str(Path.home())],
                         capture_output=True, text=True, check=True)
    line = out.stdout.strip().splitlines()[-1]
    return int(line.split()[4].rstrip("%"))


def check_gates(start_time: float) -> None:
    if (time.time() - start_time) > WALL_CLOCK_CAP_SEC:
        sys.exit(f"  [gate] wall-clock > 7d cap; aborting.")
    used = disk_pct()
    if used > DISK_CAP_PCT:
        sys.exit(f"  [gate] disk {used}% > {DISK_CAP_PCT}% cap; aborting.")


# ── eBird code ↔ scientific name ──────────────────────────────────────────────

def load_ebird_table() -> tuple:
    """Return (code2sci, sci2code) dicts from eBird v2021 taxonomy."""
    if not EBIRD_CSV.exists():
        sys.exit(f"  [setup] missing {EBIRD_CSV}")
    df = pd.read_csv(EBIRD_CSV, encoding="utf-8-sig")
    df["SPECIES_CODE"] = df["SPECIES_CODE"].astype(str)
    df["SCI_NAME"]     = df["SCI_NAME"].astype(str)
    code2sci = dict(zip(df["SPECIES_CODE"], df["SCI_NAME"]))
    sci2code = dict(zip(df["SCI_NAME"], df["SPECIES_CODE"]))
    return code2sci, sci2code


def list_rohanrao_codes() -> dict:
    """Walk data/external/xenocanto_raw/{A-M,N-Z}/<code>/ → {code: dir}."""
    out = {}
    for sub in ("A-M", "N-Z"):
        sub_dir = RAW_ROOT / sub
        if not sub_dir.exists():
            continue
        for d in sub_dir.iterdir():
            if d.is_dir():
                out[d.name] = d
    return out


def build_target_to_rohanrao_map(targets: list, code2sci: dict,
                                  rohanrao_codes: dict) -> tuple:
    """Per target_code, find a rohanrao directory.

    Strategy:
        (a) Direct: target_code matches a rohanrao directory name.
        (b) Fallback: look up target_code's sciname (eBird v2021),
            search rohanrao codes for one with the same sciname.

    Returns ({target_code: src_dir}, [unmatched_codes]).
    """
    # Build rohanrao_code → sciname using eBird v2021 (assumes 2020 codes
    # are still present in v2021; usually true).
    rohanrao_sci2code = {}
    for r_code in rohanrao_codes:
        sci = code2sci.get(r_code)
        if sci:
            rohanrao_sci2code[sci] = r_code

    mapping = {}
    unmatched = []
    n_direct = n_fallback = 0
    for tc in targets:
        if tc in rohanrao_codes:
            mapping[tc] = rohanrao_codes[tc]
            n_direct += 1
            continue
        sci = code2sci.get(tc)
        if sci and sci in rohanrao_sci2code:
            r_code = rohanrao_sci2code[sci]
            mapping[tc] = rohanrao_codes[r_code]
            n_fallback += 1
            continue
        unmatched.append(tc)

    return mapping, unmatched, n_direct, n_fallback


# ── ffmpeg transcode ──────────────────────────────────────────────────────────

def transcode(src_mp3: Path, dst_ogg: Path, duration_cap_sec: int) -> bool:
    """ffmpeg one-shot: 32 kHz mono ogg vorbis q4 + duration cap."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src_mp3),
        "-ac", "1",
        "-ar", "32000",
        "-t", str(duration_cap_sec),
        "-c:a", "libvorbis", "-q:a", "4",
        str(dst_ogg),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except subprocess.CalledProcessError as e:
        return False
    except subprocess.TimeoutExpired:
        return False
    return dst_ogg.exists() and dst_ogg.stat().st_size > 0


# ── Per-species pipeline ──────────────────────────────────────────────────────

def process_species(target_code: str, src_dir: Path, args) -> dict:
    """Transcode (capped) recordings from src_dir → out_dir per spec."""
    out_dir = OUT_ROOT / target_code
    out_dir.mkdir(parents=True, exist_ok=True)

    # List source mp3s, sorted for determinism (stable cap selection).
    src_mp3s = sorted(src_dir.glob("*.mp3"))
    n_total  = len(src_mp3s)
    if n_total == 0:
        return dict(target_code=target_code, n_total=0, n_kept=0,
                    n_done=0, n_skipped=0, n_failed=0,
                    is_rare=False, dur_cap=0)

    is_rare = n_total < args.rare_threshold
    dur_cap = args.dur_cap_rare if is_rare else args.dur_cap_normal

    kept = src_mp3s[:args.per_species_cap]
    n_done = n_skip = n_fail = 0

    for src in kept:
        # Output filename: same XC<id>.ogg pattern.
        rid = src.stem  # 'XC133197'
        dst = out_dir / f"{rid}.ogg"
        if dst.exists() and dst.stat().st_size > 0:
            n_skip += 1
            continue
        if transcode(src, dst, dur_cap):
            n_done += 1
        else:
            n_fail += 1
            dst.unlink(missing_ok=True)

    return dict(
        target_code=target_code, n_total=n_total, n_kept=len(kept),
        n_done=n_done, n_skipped=n_skip, n_failed=n_fail,
        is_rare=is_rare, dur_cap=dur_cap,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="L2-redux Phase 2c.2b — transcode rohanrao bulk")
    p.add_argument("--per-species-cap", type=int, default=DEFAULT_PER_SPECIES_CAP)
    p.add_argument("--rare-threshold",  type=int, default=DEFAULT_RARE_THRESHOLD)
    p.add_argument("--dur-cap-normal",  type=int, default=DEFAULT_DUR_CAP_NORMAL)
    p.add_argument("--dur-cap-rare",    type=int, default=DEFAULT_DUR_CAP_RARE)
    p.add_argument("--workers",         type=int, default=DEFAULT_WORKERS,
                   help="Thread pool workers for parallel ffmpeg")
    p.add_argument("--max-species",     type=int, default=None,
                   help="Smoke-test limit on number of species processed")
    p.add_argument("--start-idx",       type=int, default=0)
    args = p.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("  [setup] ffmpeg not on PATH (conda install -c conda-forge ffmpeg)")

    if not RAW_ROOT.exists() or not any(RAW_ROOT.iterdir()):
        sys.exit(f"  [setup] missing rohanrao bulk at {RAW_ROOT} — run Phase 2c.2a first")

    if not TARGETS_JSON.exists():
        sys.exit(f"  [setup] missing {TARGETS_JSON} — run Phase 2c.1 first")

    targets = json.loads(TARGETS_JSON.read_text())
    code2sci, _ = load_ebird_table()
    rohanrao_codes = list_rohanrao_codes()
    print(f"  [setup] rohanrao species dirs found: {len(rohanrao_codes)}", flush=True)

    mapping, unmatched, n_direct, n_fallback = build_target_to_rohanrao_map(
        targets, code2sci, rohanrao_codes,
    )
    print(f"  [match] direct code:  {n_direct}", flush=True)
    print(f"  [match] sciname fbk:  {n_fallback}", flush=True)
    print(f"  [match] unmatched:    {len(unmatched)} → {UNMATCHED_JSON}", flush=True)
    UNMATCHED_JSON.parent.mkdir(parents=True, exist_ok=True)
    UNMATCHED_JSON.write_text(json.dumps(unmatched, indent=2))

    if len(unmatched) > 500:
        print(f"  [warn] unmatched count ({len(unmatched)}) > 500 — ~{100*len(unmatched)/len(targets):.1f}% "
              f"of targets uncovered. Proceeding with partial coverage; review log.",
              flush=True)

    runnable = list(mapping.items())
    if args.start_idx:
        runnable = runnable[args.start_idx:]
    if args.max_species is not None:
        runnable = runnable[:args.max_species]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print(f"L2-redux Phase 2c.2b — transcode rohanrao → ogg vorbis", flush=True)
    print(f"  targets total       : {len(targets)}", flush=True)
    print(f"  matched (runnable)  : {len(mapping)}", flush=True)
    print(f"  to process now      : {len(runnable)}", flush=True)
    print(f"  workers             : {args.workers}", flush=True)
    print(f"  per-species cap     : {args.per_species_cap}", flush=True)
    print(f"  dur cap normal/rare : {args.dur_cap_normal}s / {args.dur_cap_rare}s "
          f"(rare iff < {args.rare_threshold} recs)", flush=True)
    print(f"  output              : {OUT_ROOT}", flush=True)
    print("=" * 60, flush=True)

    failures = []
    if FAILURES_JSON.exists():
        try:
            failures = json.loads(FAILURES_JSON.read_text())
        except json.JSONDecodeError:
            failures = []

    start_time = time.time()
    cum_done = cum_skip = cum_fail = 0

    # Parallelize across species — each species's mp3s run sequentially within one
    # worker, but multiple species can be in-flight at once. ffmpeg subprocess
    # releases GIL so threads scale.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_species, tc, src_dir, args): tc
            for tc, src_dir in runnable
        }
        n_total_species = len(futures)
        for i, fut in enumerate(as_completed(futures), 1):
            tc = futures[fut]
            try:
                summary = fut.result()
            except Exception as e:
                print(f"  [{i}/{n_total_species}] {tc:>14}: EXCEPTION "
                      f"{type(e).__name__}: {e}", flush=True)
                failures.append({"target_code": tc, "error": str(e)})
                FAILURES_JSON.write_text(json.dumps(failures, indent=2))
                continue

            cum_done += summary["n_done"]
            cum_skip += summary["n_skipped"]
            cum_fail += summary["n_failed"]

            # Periodic gates (every 25 species).
            if i % 25 == 0:
                check_gates(start_time)

            elapsed_h = (time.time() - start_time) / 3600.0
            print(
                f"  [{i:>4}/{n_total_species}] {summary['target_code']:>14} "
                f"src={summary['n_total']:4} kept={summary['n_kept']:4} "
                f"done={summary['n_done']:4} skip={summary['n_skipped']:4} "
                f"fail={summary['n_failed']:3} rare={'Y' if summary['is_rare'] else 'N'} "
                f"cum_done={cum_done} t={elapsed_h:.2f}h",
                flush=True,
            )

    elapsed = time.time() - start_time
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    used = disk_pct()
    print("=" * 60, flush=True)
    print(f"Phase 2c.2b done at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"  wall-clock         : {h}h {m}m {s}s", flush=True)
    print(f"  cum_done (transcoded): {cum_done}", flush=True)
    print(f"  cum_skip (existing) : {cum_skip}", flush=True)
    print(f"  cum_fail            : {cum_fail}", flush=True)
    print(f"  failures recorded   : {len(failures)} → {FAILURES_JSON}", flush=True)
    print(f"  disk usage          : {used}%", flush=True)


if __name__ == "__main__":
    main()
