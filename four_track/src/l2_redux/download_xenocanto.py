"""L2-redux Phase 2c.2 — Xeno-Canto v3 bulk download (unmatched-codes pass).

Per new_plan.md §14.17.16 + §14.17.16.7. Phase 2c.2 was first attempted via
the rohanrao Kaggle bulk dump, but that snapshot only covered 252 of our
6,971 target species (BirdCLEF-2020 NA-only). This downloader covers the
remaining 6,719 codes via the Xeno-Canto v3 API.

API:
    Endpoint:     https://xeno-canto.org/api/3/recordings
    Auth:         ?key=<API_KEY> on every metadata call AND on every file
                  download (mandatory since 2025-10-10).
    Schema:       same field names as v2 (id, gen, sp, ssp, en, file,
                  length, q, cnt, loc, lat, lng, type), with pagination
                  via numRecordings + numPages.

API key is read from the XC_API_KEY environment variable. It is never
printed, never written to log, never persisted to disk.

Output layout:
    data/external/xenocanto_bulk/
        <ebird_code>/
            _meta.json      (cached XC API response — KEY STRIPPED)
            XC<id>.ogg      (resampled clip)
            ...
        _failures.json      (per-species failure log)

This output dir already contains 252 species (~18,668 ogg files) populated
from the rohanrao 2020 snapshot. Those species do NOT appear in the input
list (`l2_redux_unmatched_codes.json`) so we won't re-fetch them.

Resumable: per-species `_meta.json` cached; per-recording output existence
is the skip predicate. A re-run picks up where the last one stopped.

Throttling: default 1 req/sec (conservative; XC v3 doesn't publish a hard
rate limit but client convention is 1s sleep between paged calls).

Usage:
    export XC_API_KEY="..."          # required, never logged
    python -u src/l2_redux/download_xenocanto.py
    python -u src/l2_redux/download_xenocanto.py --max-species 5  # smoke
    python -u src/l2_redux/download_xenocanto.py --rate 0.5       # politer
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).resolve().parent
FT_ROOT   = HERE.parent.parent
ROOT      = FT_ROOT.parent
PROC      = FT_ROOT / "data" / "processed"
OUT_ROOT  = ROOT / "data" / "external" / "xenocanto_bulk"
LOG_FAILURES = OUT_ROOT / "_failures.json"

UNMATCHED_JSON = PROC / "l2_redux_unmatched_codes.json"
EBIRD_CSV      = PROC / "ebird_taxonomy_v2021.csv"

XC_API   = "https://xeno-canto.org/api/3/recordings"
XC_FETCH_TIMEOUT = 30
XC_DOWNLOAD_TIMEOUT = 60
USER_AGENT = "BirdCLEF-2026-research-bot/0.2 (academic; contact via github)"

# Spec defaults from §14.17.16
DEFAULT_RATE_PER_SEC    = 1.0
DEFAULT_PER_SPECIES_CAP = 500
DEFAULT_RARE_THRESHOLD  = 20    # < 20 recs ⇒ "rare" → use 60 s clip cap
DEFAULT_DUR_CAP_NORMAL  = 30
DEFAULT_DUR_CAP_RARE    = 60

# Abort gates
DISK_CAP_PCT = 90
WALL_CLOCK_WARN_SEC = 7 * 24 * 3600   # warn-only after 7 days (per user 2026-04-30)


# ── Throttler ─────────────────────────────────────────────────────────────────

class Throttler:
    """Min-interval throttler: blocks until at least 1/rate seconds since last call."""

    def __init__(self, rate_per_sec: float):
        self.min_interval = 1.0 / max(rate_per_sec, 0.01)
        self._next_ok_at = 0.0

    def wait(self) -> None:
        now = time.time()
        if now < self._next_ok_at:
            time.sleep(self._next_ok_at - now)
        self._next_ok_at = time.time() + self.min_interval


# ── Disk + time gates ─────────────────────────────────────────────────────────

def disk_pct() -> int:
    """Return percent used of /home/swatson partition."""
    out = subprocess.run(
        ["df", "-P", str(Path.home())],
        capture_output=True, text=True, check=True,
    )
    line = out.stdout.strip().splitlines()[-1]
    pct_str = line.split()[4].rstrip("%")
    return int(pct_str)


_warned_wall_clock = False

def check_gates(start_time: float) -> None:
    global _warned_wall_clock
    elapsed = time.time() - start_time
    if elapsed > WALL_CLOCK_WARN_SEC and not _warned_wall_clock:
        days = elapsed / 86400
        print(f"  [gate] wall-clock {days:.1f}d > 7d warn threshold; continuing per user override (2026-04-30).", flush=True)
        _warned_wall_clock = True
    used = disk_pct()
    if used > DISK_CAP_PCT:
        sys.exit(f"  [gate] disk {used}% > {DISK_CAP_PCT}% cap; aborting.")


# ── eBird code → scientific name ──────────────────────────────────────────────

def load_code2sci() -> dict:
    """Build a SPECIES_CODE → SCI_NAME mapping from the eBird taxonomy CSV."""
    if not EBIRD_CSV.exists():
        sys.exit(f"  [setup] missing {EBIRD_CSV} — pull from BC2024 train_audio.zip")
    df = pd.read_csv(EBIRD_CSV, encoding="utf-8-sig")
    return dict(zip(df["SPECIES_CODE"].astype(str), df["SCI_NAME"].astype(str)))


# ── XC API ────────────────────────────────────────────────────────────────────

def parse_sciname(sci: str) -> tuple:
    """'Turdus tephronotus' → ('Turdus', 'tephronotus'). Handles multi-word
    epithet (some species, e.g. 'Vermivora chrysoptera bachmanii') by joining
    everything after the genus into the species term."""
    parts = sci.split()
    if len(parts) < 2:
        return (parts[0] if parts else "", "")
    return (parts[0], " ".join(parts[1:]))


def _redact_url(url: str) -> str:
    """Strip ?key=... from a URL for safe logging."""
    if "key=" not in url:
        return url
    head, _, tail = url.partition("?")
    parts = [p for p in tail.split("&") if not p.startswith("key=")]
    parts.append("key=<redacted>")
    return f"{head}?{'&'.join(parts)}"


def xc_fetch_species(sci: str, api_key: str, throttler: Throttler,
                     session: requests.Session) -> list:
    """Page through XC API for all recordings of a single species. Returns list
    of recording dicts (the raw XC entries, with no key fields). Raises
    `SystemExit` on auth failure (401/403) since that's terminal."""
    gen, sp = parse_sciname(sci)
    if not gen or not sp:
        return []
    params_base = {"query": f"gen:{gen} sp:{sp}", "key": api_key}

    all_recs = []
    page = 1
    while True:
        throttler.wait()
        params = dict(params_base, page=page)
        try:
            resp = session.get(XC_API, params=params, timeout=XC_FETCH_TIMEOUT)
        except requests.RequestException as e:
            print(f"    [xc-meta] {sci} page {page} network error: {type(e).__name__}",
                  flush=True)
            break

        if resp.status_code in (401, 403):
            sys.exit(f"  [auth] XC v3 returned {resp.status_code} on metadata fetch — "
                     f"check XC_API_KEY env var")
        if resp.status_code == 429:
            print(f"    [xc-meta] {sci} page {page}: 429 rate-limited; sleeping 30s",
                  flush=True)
            time.sleep(30)
            continue
        if not resp.ok:
            print(f"    [xc-meta] {sci} page {page}: HTTP {resp.status_code}",
                  flush=True)
            break

        try:
            data = resp.json()
        except ValueError:
            print(f"    [xc-meta] {sci} page {page}: non-JSON response", flush=True)
            break

        recs = data.get("recordings", [])
        all_recs.extend(recs)

        n_pages = int(data.get("numPages", 1) or 1)
        if page >= n_pages:
            break
        page += 1
    return all_recs


def parse_length_seconds(length_str: str) -> int:
    """XC's 'length' field is 'M:SS' or 'MM:SS' or 'H:MM:SS'. Returns total seconds."""
    if not length_str:
        return 0
    parts = length_str.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return 0
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    if len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    return 0


def xc_download_url(rec: dict, api_key: str) -> str:
    """Resolve a recording's download URL from the XC API entry, with key
    appended (XC v3 download endpoint requires key since 2025-10-10)."""
    f = rec.get("file") or ""
    if not f:
        return ""
    if f.startswith("//"):
        url = "https:" + f
    elif f.startswith("http"):
        url = f
    else:
        url = "https://xeno-canto.org" + f

    sep = "&" if "?" in url else "?"
    return f"{url}{sep}key={api_key}"


# ── ffmpeg transcode ──────────────────────────────────────────────────────────

def transcode(src_mp3: Path, dst_ogg: Path, duration_cap_sec: int) -> bool:
    """Use ffmpeg to read mp3, resample to 32 kHz mono, cap duration, write
    ogg vorbis q4. Returns True on success."""
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
        print(f"    [transcode] ffmpeg failed: {e.stderr.decode(errors='ignore')[:200]}",
              flush=True)
        return False
    except subprocess.TimeoutExpired:
        print(f"    [transcode] ffmpeg timeout on {src_mp3}", flush=True)
        return False
    return dst_ogg.exists() and dst_ogg.stat().st_size > 0


# ── Per-species pipeline ──────────────────────────────────────────────────────

def process_species(
    code: str,
    sci: str,
    out_dir: Path,
    api_key: str,
    args,
    throttler: Throttler,
    session: requests.Session,
    start_time: float,
) -> dict:
    """Fetch metadata (cached), then download + transcode up to `cap` recordings.

    Returns a per-species summary dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "_meta.json"

    # Phase 1: metadata (resume if cached).
    if meta_path.exists():
        with meta_path.open() as f:
            recs = json.load(f)
    else:
        recs = xc_fetch_species(sci, api_key, throttler, session)
        with meta_path.open("w") as f:
            json.dump(recs, f)

    n_total = len(recs)
    if n_total == 0:
        return {"code": code, "sci": sci, "n_total": 0, "n_kept": 0,
                "n_downloaded": 0, "n_skipped_existing": 0, "n_failed": 0,
                "is_rare": True, "dur_cap": args.dur_cap_rare}

    # Sort by quality (q='A' best, then 'B', etc.) so caps keep the best clips.
    recs.sort(key=lambda r: (str(r.get("q") or "Z"),
                              -parse_length_seconds(r.get("length") or "")))

    # Apply rare/non-rare clip cap.
    is_rare = n_total < args.rare_threshold
    dur_cap = args.dur_cap_rare if is_rare else args.dur_cap_normal

    kept = recs[:args.per_species_cap]
    n_dl = n_skip = n_fail = 0

    for rec in kept:
        check_gates(start_time)

        rid = str(rec.get("id") or "").strip()
        if not rid:
            n_fail += 1
            continue
        dst = out_dir / f"XC{rid}.ogg"
        if dst.exists() and dst.stat().st_size > 0:
            n_skip += 1
            continue

        url = xc_download_url(rec, api_key)
        if not url:
            n_fail += 1
            continue

        # Download mp3 to a temp path next to dst.
        tmp_mp3 = out_dir / f".tmp_XC{rid}.mp3"
        throttler.wait()
        try:
            with session.get(url, timeout=XC_DOWNLOAD_TIMEOUT, stream=True) as resp:
                if resp.status_code in (401, 403):
                    sys.exit(f"  [auth] XC v3 returned {resp.status_code} on file "
                             f"download — check XC_API_KEY env var")
                if resp.status_code == 429:
                    print(f"    [dl] XC{rid} ({code}): 429 rate-limited; sleeping 30s",
                          flush=True)
                    time.sleep(30)
                    n_fail += 1   # treat this attempt as failed; will retry on next run
                    tmp_mp3.unlink(missing_ok=True)
                    continue
                resp.raise_for_status()
                with tmp_mp3.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=64 * 1024):
                        f.write(chunk)
        except requests.RequestException as e:
            print(f"    [dl] XC{rid} ({code}) failed: {type(e).__name__}", flush=True)
            n_fail += 1
            tmp_mp3.unlink(missing_ok=True)
            continue

        # Transcode + cap.
        if transcode(tmp_mp3, dst, dur_cap):
            n_dl += 1
        else:
            n_fail += 1
            dst.unlink(missing_ok=True)
        tmp_mp3.unlink(missing_ok=True)

    return {
        "code":               code,
        "sci":                sci,
        "n_total":            n_total,
        "n_kept":             len(kept),
        "n_downloaded":       n_dl,
        "n_skipped_existing": n_skip,
        "n_failed":           n_fail,
        "is_rare":            is_rare,
        "dur_cap":            dur_cap,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="L2-redux Phase 2c.2 — XC v3 bulk download")
    p.add_argument("--rate", type=float, default=DEFAULT_RATE_PER_SEC,
                   help="HTTP request rate per second (default 1.0)")
    p.add_argument("--per-species-cap", type=int, default=DEFAULT_PER_SPECIES_CAP)
    p.add_argument("--rare-threshold",  type=int, default=DEFAULT_RARE_THRESHOLD)
    p.add_argument("--dur-cap-normal",  type=int, default=DEFAULT_DUR_CAP_NORMAL)
    p.add_argument("--dur-cap-rare",    type=int, default=DEFAULT_DUR_CAP_RARE)
    p.add_argument("--max-species",     type=int, default=None,
                   help="Smoke-test limit on number of species processed")
    p.add_argument("--start-idx",       type=int, default=0,
                   help="Resume from this index in the targets list")
    p.add_argument("--targets-json",    type=str,
                   default=str(UNMATCHED_JSON),
                   help="Path to JSON list of eBird codes to download "
                        "(default: l2_redux_unmatched_codes.json — the 6,719 "
                        "codes the rohanrao snapshot didn't cover)")
    args = p.parse_args()

    api_key = os.environ.get("XC_API_KEY", "").strip()
    if not api_key:
        sys.exit("  [setup] XC_API_KEY env var not set; export it before launch")

    if shutil.which("ffmpeg") is None:
        sys.exit("  [setup] ffmpeg not on PATH (conda install -c conda-forge ffmpeg)")

    targets_path = Path(args.targets_json)
    if not targets_path.exists():
        sys.exit(f"  [setup] missing {targets_path} — run Phase 2c.1 first")

    targets = json.load(targets_path.open())
    code2sci = load_code2sci()

    # Filter to codes we can map to scientific names.
    runnable = [(c, code2sci.get(c)) for c in targets]
    runnable = [(c, s) for c, s in runnable if s]
    n_unmapped = len(targets) - len(runnable)
    if n_unmapped:
        print(f"  [setup] {n_unmapped} codes had no eBird sciname mapping; skipping")

    if args.start_idx:
        runnable = runnable[args.start_idx:]
    if args.max_species is not None:
        runnable = runnable[:args.max_species]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_FAILURES.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print(f"L2-redux Phase 2c.2 — XC v3 bulk download", flush=True)
    print(f"  endpoint           : {XC_API}", flush=True)
    print(f"  api key            : {'set (' + str(len(api_key)) + ' chars)' if api_key else 'MISSING'}",
          flush=True)
    print(f"  targets file       : {targets_path}", flush=True)
    print(f"  targets total      : {len(targets)}", flush=True)
    print(f"  runnable (mapped)  : {len(runnable)}", flush=True)
    print(f"  rate (req/sec)     : {args.rate}", flush=True)
    print(f"  per-species cap    : {args.per_species_cap}", flush=True)
    print(f"  dur cap normal/rare: {args.dur_cap_normal}s / {args.dur_cap_rare}s "
          f"(rare iff < {args.rare_threshold} recs)", flush=True)
    print(f"  output             : {OUT_ROOT}", flush=True)
    print(f"  start_idx          : {args.start_idx}", flush=True)
    print("=" * 60, flush=True)

    throttler = Throttler(args.rate)
    session   = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    failures = []
    if LOG_FAILURES.exists():
        try:
            failures = json.loads(LOG_FAILURES.read_text())
        except json.JSONDecodeError:
            failures = []

    start_time = time.time()
    cum_dl = cum_skip = cum_fail = 0

    for i, (code, sci) in enumerate(runnable):
        check_gates(start_time)

        out_dir = OUT_ROOT / code
        try:
            summary = process_species(
                code, sci, out_dir, api_key, args, throttler, session, start_time,
            )
        except SystemExit:
            raise
        except Exception as e:
            print(f"  [{i}] {code:>14} ({sci}): EXCEPTION {type(e).__name__}",
                  flush=True)
            failures.append({"code": code, "sci": sci, "error": type(e).__name__})
            LOG_FAILURES.write_text(json.dumps(failures, indent=2))
            continue

        cum_dl   += summary["n_downloaded"]
        cum_skip += summary["n_skipped_existing"]
        cum_fail += summary["n_failed"]

        elapsed_h = (time.time() - start_time) / 3600.0
        used_pct  = disk_pct()
        print(
            f"  [{i+1}/{len(runnable)}] {code:>14} ({sci[:32]:32}) "
            f"total={summary['n_total']:4} "
            f"kept={summary['n_kept']:4} "
            f"dl={summary['n_downloaded']:4} "
            f"skip={summary['n_skipped_existing']:4} "
            f"fail={summary['n_failed']:3} "
            f"rare={'Y' if summary['is_rare'] else 'N'} "
            f"cum_dl={cum_dl} disk={used_pct}% t={elapsed_h:.1f}h",
            flush=True,
        )

    elapsed = time.time() - start_time
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print("=" * 60, flush=True)
    print(f"Phase 2c.2 done at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"  wall-clock: {h}h {m}m {s}s", flush=True)
    print(f"  cum_dl   : {cum_dl}", flush=True)
    print(f"  cum_skip : {cum_skip}", flush=True)
    print(f"  cum_fail : {cum_fail}", flush=True)
    print(f"  failures recorded: {len(failures)} → {LOG_FAILURES}", flush=True)


if __name__ == "__main__":
    main()
