"""Idempotently insert (or replace) the A1 fusion cell into the LB notebook.

Reads the cell source from `a1_notebook_cell.py:CELL_SOURCE`, opens the
notebook JSON, and:
  - If a cell containing the marker string already exists, replaces it
    in place (idempotent — safe to run after every edit).
  - Otherwise, inserts it immediately after cell 36 (Score Fusion).
  - Also adds the `stevewatson999/birdclef-2026-a1-effb0-ckpts` dataset
    to `kernel-metadata.json:dataset_sources` if not already present.
"""

import json
import sys
from pathlib import Path

HERE    = Path(__file__).resolve().parent
FT_ROOT = HERE.parent
ROOT    = FT_ROOT.parent

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from a1_notebook_cell import CELL_SOURCE

NB_PATH   = ROOT / "jupyter" / "protossm-postproc" / "birdclef2026-protossm-postproc.ipynb"
META_PATH = NB_PATH.parent / "kernel-metadata.json"

MARKER = "# Cell 17 — Track A1 SED fusion"
A1_DATASET_ID = "stevewatson999/birdclef-2026-a1-effb0-ckpts"


def _split_source(text: str) -> list:
    """Jupyter stores cell source as a list of lines with trailing newlines."""
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

    # 1. Try replace-in-place if the marker already exists
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if MARKER in src:
            print(f"  replacing existing A1 cell at index {i}", flush=True)
            nb["cells"][i] = new_cell
            NB_PATH.write_text(json.dumps(nb, indent=1) + "\n")
            return

    # 2. Otherwise, insert after the "Score Fusion" cell (cell 36 by position,
    #    but search for its content so the index is robust to prior edits)
    insert_at = None
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if "Score Fusion: ProtoSSM" in src and "final_test_scores" in src:
            insert_at = i + 1
            break
    if insert_at is None:
        sys.exit("Couldn't find the Score Fusion cell to insert after.")

    print(f"  inserting new A1 cell at index {insert_at} "
          f"(after 'Score Fusion' cell at {insert_at - 1})", flush=True)
    nb["cells"].insert(insert_at, new_cell)
    NB_PATH.write_text(json.dumps(nb, indent=1) + "\n")


def update_dataset_sources() -> None:
    meta = json.loads(META_PATH.read_text())
    sources = meta.get("dataset_sources", [])
    if A1_DATASET_ID in sources:
        print(f"  dataset_sources already contains {A1_DATASET_ID}", flush=True)
        return
    sources.append(A1_DATASET_ID)
    meta["dataset_sources"] = sources
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  appended {A1_DATASET_ID} to dataset_sources", flush=True)


def main() -> None:
    print(f"Notebook: {NB_PATH}", flush=True)
    inject_cell()
    print(f"Metadata: {META_PATH}", flush=True)
    update_dataset_sources()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
