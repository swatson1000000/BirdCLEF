"""Idempotently insert (or replace) the three Track B1 cells in the LB notebook.

Reads cell sources from `b1_perceiver.py:{CELL_24B, CELL_31B, CELL_36B}` and,
for each one:
  - Replaces in place if a cell containing the marker string already exists
    (idempotent — safe to run after every edit).
  - Otherwise, inserts the new cell immediately after a content-matched
    anchor cell.

No `kernel-metadata.json` change is needed — B1 trains in-notebook and
mounts no new dataset. Order of operations matters: 24b is inserted first
(after cell 24), then 31b (after the cell that calls `run_proto_ssm_oof`),
then 36b (after the Score Fusion cell, *before* the existing A1 cell 37).
"""

import json
import sys
from pathlib import Path

HERE    = Path(__file__).resolve().parent
FT_ROOT = HERE.parent
ROOT    = FT_ROOT.parent

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from b1_perceiver import CELL_24B, CELL_31B, CELL_36B

NB_PATH = ROOT / "jupyter" / "protossm-postproc" / "birdclef2026-protossm-postproc.ipynb"

# (marker_in_new_cell, anchor_substring_in_existing_cell, human_label)
# anchor_substring is searched in cell *source*; the new cell is inserted
# at index `anchor + 1`. The marker is also what we search for to do
# replace-in-place on subsequent runs.
B1_CELLS = [
    (
        "# Cell 24b — Track B1 PerceiverIO training (def + OOF)",
        "# ProtoSSM v4 Training Loop",                       # cell 24
        "B1 train defs (cell 24b)",
        CELL_24B,
    ),
    (
        "# Cell 31b — Track B1 PerceiverIO instantiate + OOF + retrain on full",
        "# Instantiate and train ProtoSSM v4",               # cell 31 — unique header
        "B1 OOF + retrain (cell 31b)",
        CELL_31B,
    ),
    (
        "# Cell 36b — Track B1 inference + rank fusion",
        "Score Fusion: ProtoSSM",                            # cell 36 — must land BEFORE cell 37 (A1)
        "B1 test fusion (cell 36b)",
        CELL_36B,
    ),
]


def _split_source(text: str) -> list:
    """Jupyter stores cell source as a list of lines with trailing newlines."""
    lines = text.splitlines(keepends=True)
    return lines if lines else [text]


def _build_cell(source_text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": _split_source(source_text.lstrip("\n")),
        "outputs": [],
        "execution_count": None,
    }


def _find_marker(nb_cells, marker: str):
    for i, c in enumerate(nb_cells):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if marker in src:
            return i
    return None


def _find_anchor(nb_cells, anchor_substring: str, skip_marker: str):
    """Find the index of an existing cell containing `anchor_substring`,
    skipping any cell that already contains `skip_marker` (so we don't
    accidentally use the previously-injected B1 cell as its own anchor)."""
    for i, c in enumerate(nb_cells):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if skip_marker in src:
            continue
        if anchor_substring in src:
            return i
    return None


def inject_all() -> None:
    nb = json.loads(NB_PATH.read_text())
    cells = nb["cells"]

    for marker, anchor_sub, label, source_text in B1_CELLS:
        new_cell = _build_cell(source_text)

        existing_idx = _find_marker(cells, marker)
        if existing_idx is not None:
            print(f"  [{label}] replacing existing cell at index {existing_idx}", flush=True)
            cells[existing_idx] = new_cell
            continue

        anchor_idx = _find_anchor(cells, anchor_sub, marker)
        if anchor_idx is None:
            sys.exit(
                f"[{label}] couldn't find anchor cell matching {anchor_sub!r}. "
                "The notebook layout may have changed."
            )
        insert_at = anchor_idx + 1
        print(f"  [{label}] inserting at index {insert_at} "
              f"(after anchor at index {anchor_idx})", flush=True)
        cells.insert(insert_at, new_cell)

    NB_PATH.write_text(json.dumps(nb, indent=1) + "\n")


def main() -> None:
    print(f"Notebook: {NB_PATH}", flush=True)
    inject_all()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
