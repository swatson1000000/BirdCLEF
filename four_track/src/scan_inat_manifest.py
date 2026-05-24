"""Scan iNat manifest for unreadable / zero-length .wav files.

Reads `inat_root/inat_manifest.csv`, runs `sf.info` on each row's audio
file, and drops rows where the file fails to open or has 0 frames. The
zero-frame case is the one that crashed the e14 dataloader worker
(`torchaudio.functional.resample` cannot reshape `[1, 0]`).

Backs up the original manifest to `inat_manifest_orig.csv` (only on the
first run; subsequent runs preserve the backup) and overwrites
`inat_manifest.csv` with the cleaned rows. The pretrain script picks up
the cleaned version automatically — no flag needed.
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import soundfile as sf


def _check(abs_path: str) -> str | None:
    try:
        info = sf.info(abs_path)
        if info.frames == 0:
            return "zero-frame"
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def main() -> None:
    p = argparse.ArgumentParser(description="Scan iNat manifest for bad audio files")
    p.add_argument("--inat-root", type=Path, required=True,
                   help="Directory containing inat_manifest.csv")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--chunksize", type=int, default=128)
    args = p.parse_args()

    manifest_path = args.inat_root / "inat_manifest.csv"
    backup_path = args.inat_root / "inat_manifest_orig.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest missing: {manifest_path}")

    df = pd.read_csv(manifest_path, dtype=str)
    abs_paths = df["file_path"].apply(lambda p: str(args.inat_root / p)).tolist()
    print(f"  [manifest] {len(df)} rows from {manifest_path}", flush=True)

    bad: list[tuple[int, str, str]] = []
    t0 = time.time()
    last_print = t0
    n = len(abs_paths)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, err in enumerate(
            ex.map(_check, abs_paths, chunksize=args.chunksize)
        ):
            if err is not None:
                bad.append((i, df.iloc[i]["file_path"], err))
            now = time.time()
            if now - last_print > 30:
                rate = (i + 1) / (now - t0)
                eta = (n - i - 1) / max(rate, 1.0)
                print(
                    f"  [scan] {i + 1}/{n} ({rate:.0f}/s)  "
                    f"bad so far: {len(bad)}  eta: {eta:.0f}s",
                    flush=True,
                )
                last_print = now

    elapsed = time.time() - t0
    pct = 100.0 * len(bad) / max(n, 1)
    print(
        f"  [scan] done in {elapsed:.0f}s  ({n / max(elapsed, 1):.0f}/s)",
        flush=True,
    )
    print(f"  [scan] bad: {len(bad)} / {n} ({pct:.4f}%)", flush=True)

    if bad:
        print("  [scan] first 30 bad rows:", flush=True)
        for _, fp, err in bad[:30]:
            print(f"    {err}  {fp}", flush=True)
        if len(bad) > 30:
            print(f"    ... and {len(bad) - 30} more", flush=True)

    if not backup_path.exists():
        df.to_csv(backup_path, index=False)
        print(f"  [backup] original → {backup_path}", flush=True)
    else:
        print(f"  [backup] preserved existing {backup_path}", flush=True)

    if bad:
        bad_idx = {idx for idx, _, _ in bad}
        clean = df.drop(index=bad_idx).reset_index(drop=True)
        clean.to_csv(manifest_path, index=False)
        print(
            f"  [clean] wrote {len(clean)} rows → {manifest_path} "
            f"(removed {len(bad)})",
            flush=True,
        )
    else:
        print(f"  [clean] no bad rows; manifest unchanged", flush=True)


if __name__ == "__main__":
    main()
