"""§33 Phase B wrapper — Cerrado-geo XC pretrain on B0.

Thin wrapper around pretrain_l2_redux that:
  1. Loads the pre-built Cerrado manifest from
     data/processed/xc_cerrado_pretrain_manifest.csv
  2. Renames columns to the L2ReduxDataset schema
     (audio_path → abs_path, species_code → primary_label)
  3. Overrides pretrain_l2_redux.build_manifest / build_species_list /
     PRETRAIN_DIR so ckpts land at models/xc_cerrado_pretrain/ instead of
     models/l2_redux/, and the species cache JSON is namespaced separately.
  4. Calls pretrain_l2_redux.main() so the dataset, model, training loop,
     focal-BCE loss, scheduler, and per-epoch summary stay shared.

This mirrors the wrapper pattern used by a5_train_xarch.py for a2_train.
The L2-redux pretrain pipeline is the canonical multi-taxon focal-BCE
recipe (per new_plan_history.md §14.17.15) and §33.3 explicitly calls
for forking it.

Usage:
    # Smoke (1 ep, 2 train batches — wiring check)
    python -u src/pretrain_xc_cerrado.py --epochs 1 --smoke-test

    # Phase B full (epoch count TBD — start at 20 per §21 time-box;
    # pretrain_l2_redux's CosineAnnealingLR collapses to single-cycle)
    python -u src/pretrain_xc_cerrado.py --epochs 20
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pretrain_l2_redux as p  # noqa: E402

FT_ROOT = HERE.parent
CERRADO_MANIFEST_CSV = (
    FT_ROOT / "data" / "processed" / "xc_cerrado_pretrain_manifest.csv"
)
XC_CERRADO_DIR = FT_ROOT / "models" / "xc_cerrado_pretrain"


def _load_cerrado_manifest(include_xc_bulk: bool = False) -> pd.DataFrame:
    """Replacement for pretrain_l2_redux.build_manifest.

    Loads the §33 Phase A manifest, renames to the L2ReduxDataset column
    schema, adds the source_comp tag the parent script's logging expects.
    Ignores include_xc_bulk (not applicable — corpus IS XC).
    """
    if not CERRADO_MANIFEST_CSV.exists():
        sys.exit(
            f"missing manifest: {CERRADO_MANIFEST_CSV}\n"
            "Run src/build_xc_cerrado_manifest.py first (§33 Phase A)."
        )
    df = pd.read_csv(CERRADO_MANIFEST_CSV)
    df = df.rename(
        columns={"audio_path": "abs_path", "species_code": "primary_label"}
    )
    df["source_comp"] = "xc_v3_cerrado"
    # author column already populated by build_xc_cerrado_manifest.py
    df["author"] = df["author"].fillna("__unknown__").astype(str)
    print(
        f"  [manifest] xc_v3_cerrado: {len(df):,} clips, "
        f"{df['primary_label'].nunique():,} species, "
        f"{df['author'].nunique():,} unique recordists",
        flush=True,
    )
    return df


def _load_cerrado_species(
    manifest: pd.DataFrame, include_xc_bulk: bool = False
) -> list[str]:
    species = sorted(manifest["primary_label"].astype(str).unique())
    print(
        f"  [species] {len(species)} unique XC species codes (no JSON cache)",
        flush=True,
    )
    return species


if __name__ == "__main__":
    # Monkey-patch the parent's path constants + factory functions BEFORE
    # main() runs. main() reads PRETRAIN_DIR at the module level for the
    # save path and calls build_manifest / build_species_list at startup.
    p.PRETRAIN_DIR = XC_CERRADO_DIR
    p.build_manifest = _load_cerrado_manifest
    p.build_species_list = _load_cerrado_species
    p.main()
