"""Idempotently inject (or replace) the D2-β S1 stacker cell into the LB notebook.

Reads the cell source from `d2_s1_kernel_cell.py:CELL_SOURCE` and inserts
it immediately AFTER the A1 fusion cell (Cell 17 marker — historical
"cell 37"), so S1 sees the rank arrays produced by the A1 cell at
`_a1_ranks`, the B1 ranks from cell 36b at `_b1_ranks`, and the raw
ProtoSSM scores from cell 36b at `_proto_scores_before`.

Idempotent: replaces in place if the S1 marker is already present;
otherwise inserts after the A1 anchor.

Also adds `stevewatson999/birdclef-2026-d2b-s1-coefs` to
`kernel-metadata.json:dataset_sources` so the kernel mounts the
coefficients file at `/kaggle/input/birdclef-2026-d2b-s1-coefs/`.

Usage:
    python four_track/src/inject_d2_s1_cell.py
"""

import json
import sys
from pathlib import Path

HERE    = Path(__file__).resolve().parent
FT_ROOT = HERE.parent
ROOT    = FT_ROOT.parent

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from d2_s1_kernel_cell import CELL_SOURCE  # noqa: E402

NB_PATH   = ROOT / "jupyter" / "protossm-postproc" / "birdclef2026-protossm-postproc.ipynb"
META_PATH = NB_PATH.parent / "kernel-metadata.json"

MARKER          = "# Cell 37b — D2-β S1 stacker"
A1_CELL_MARKER  = "# Cell 17 — Track A1 SED fusion"
S1_DATASET_ID   = "stevewatson999/birdclef-2026-d2b-s1-coefs"


def _split_source(text: str) -> list:
    lines = text.splitlines(keepends=True)
    return lines if lines else [text]


def inject_cell() -> None:
    nb = json.loads(NB_PATH.read_text())

    new_cell = {
        "cell_type": "code",
        "metadata": {},
        "source": _split_source(CELL_SOURCE.lstrip("\n")),
        "outputs": [],
        "execution_count": None,
    }

    # 1. Replace-in-place if the S1 marker is already present.
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if MARKER in src:
            print(f"  replacing existing S1 cell at index {i}", flush=True)
            nb["cells"][i] = new_cell
            NB_PATH.write_text(json.dumps(nb, indent=1) + "\n")
            return

    # 2. Otherwise insert immediately after the A1 cell.
    insert_at = None
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if A1_CELL_MARKER in src:
            insert_at = i + 1
            break
    if insert_at is None:
        sys.exit(
            "Couldn't find the A1 fusion cell to insert after. "
            "Run `python four_track/src/inject_a1_cell.py` first."
        )

    print(
        f"  inserting new S1 cell at index {insert_at} "
        f"(after A1 cell at {insert_at - 1})",
        flush=True,
    )
    nb["cells"].insert(insert_at, new_cell)
    NB_PATH.write_text(json.dumps(nb, indent=1) + "\n")


def update_dataset_sources() -> None:
    meta = json.loads(META_PATH.read_text())
    sources = meta.get("dataset_sources", [])
    if S1_DATASET_ID in sources:
        print(f"  dataset_sources already contains {S1_DATASET_ID}", flush=True)
        return
    sources.append(S1_DATASET_ID)
    meta["dataset_sources"] = sources
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  appended {S1_DATASET_ID} to dataset_sources", flush=True)


def main() -> None:
    print(f"Notebook: {NB_PATH}", flush=True)
    if not NB_PATH.exists():
        sys.exit(f"Notebook not found: {NB_PATH}")
    inject_cell()
    print(f"Metadata: {META_PATH}", flush=True)
    update_dataset_sources()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
