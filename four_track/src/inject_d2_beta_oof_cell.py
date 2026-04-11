"""Idempotently insert (or replace) the D2-β OOF-dump cell into the LB notebook.

Reads the cell source from `d2_beta_oof_cell.py:CELL_SOURCE` and injects
it immediately after the B1 OOF cell (cell 31b) — i.e. at the point
where `oof_proto_flat`, `oof_b1_flat`, `y_flat`, and `file_groups` are
all first in scope on the shared 59-file OOF substrate.

Idempotent: if a cell containing the D2-β OOF marker already exists,
replace it in place. Otherwise, find cell 31b by its marker and insert
immediately after. Mirrors the pattern in `inject_a1_cell.py` and
`inject_b1_cell.py`.

This injector does NOT add any new dataset to `kernel-metadata.json` —
the only external dataset the D2-β cell needs is the A1 checkpoint pack
at `stevewatson999/birdclef-2026-a1-effb0-ckpts`, which is already
listed there by `inject_a1_cell.py`. If this ever changes, add the
corresponding `update_dataset_sources` block here.

Usage:
    python four_track/src/inject_d2_beta_oof_cell.py

After injection, the run-to-dump-OOFs recipe is:
    1. Set cell 1 of the notebook to `MODE = "train"`.
    2. Push the notebook to Kaggle and run it (one non-submitting
       kernel run → no slot consumed).
    3. Download `/kaggle/working/d2_beta_oofs.npz` from the kernel
       output and save as `four_track/data/d2_beta_oofs.npz`.
    4. Phase 2: run `four_track/src/d2_lgbm_stack.py` locally.
"""

import json
import sys
from pathlib import Path

HERE    = Path(__file__).resolve().parent
FT_ROOT = HERE.parent
ROOT    = FT_ROOT.parent

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from d2_beta_oof_cell import CELL_SOURCE  # noqa: E402

NB_PATH = ROOT / "jupyter" / "protossm-postproc" / "birdclef2026-protossm-postproc.ipynb"

# Marker for the D2-β OOF dump cell (in CELL_SOURCE). Must be stable /
# unique across all cells in the notebook so replace-in-place is
# unambiguous.
MARKER = "# Cell 31c — D2-β OOF dump"

# Anchor marker identifying cell 31b (B1 OOF). This must match the text
# that `inject_b1_cell.py` writes into the notebook via
# `b1_perceiver.py:CELL_31B` — see b1_perceiver.py:478.
B1_CELL_MARKER = "# Cell 31b — Track B1 PerceiverIO instantiate"


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

    # 1. Replace-in-place if the D2-β cell is already present.
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if MARKER in src:
            print(f"  replacing existing D2-β OOF cell at index {i}", flush=True)
            nb["cells"][i] = new_cell
            NB_PATH.write_text(json.dumps(nb, indent=1) + "\n")
            return

    # 2. Otherwise insert immediately after cell 31b (B1 OOF). Search by
    #    marker so the index is robust to other injections that may have
    #    shifted cell positions.
    insert_at = None
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if B1_CELL_MARKER in src:
            insert_at = i + 1
            break
    if insert_at is None:
        sys.exit(
            "Couldn't find the B1 OOF cell (cell 31b) to insert after. "
            "Run `python four_track/src/inject_b1_cell.py` first."
        )

    print(
        f"  inserting new D2-β OOF cell at index {insert_at} "
        f"(after 'Cell 31b — B1 OOF' at {insert_at - 1})",
        flush=True,
    )
    nb["cells"].insert(insert_at, new_cell)
    NB_PATH.write_text(json.dumps(nb, indent=1) + "\n")


def main() -> None:
    print(f"Notebook: {NB_PATH}", flush=True)
    if not NB_PATH.exists():
        sys.exit(f"Notebook not found: {NB_PATH}")
    inject_cell()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
