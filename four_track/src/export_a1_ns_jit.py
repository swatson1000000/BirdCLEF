"""TorchScript-export the A1-NS (§14.9 L1 noisy-student) fold checkpoint(s).

Analogue of `export_a1_jit.py` but sources from `four_track/models/a1_ns/`
and writes to `four_track/kaggle_datasets/a1-effb0-ns-ckpts/`. The NS student
shares architecture with A1 baseline (`BirdSEDModelA1` with default backbone,
no stochastic depth — see project_l1_recipe_2026_04_17 note 7), so the same
trace + wrapper pipeline applies.

Usage
-----
    python -u src/export_a1_ns_jit.py --folds 0

Only folds actually present on disk are exported.
"""

import argparse
import sys
from pathlib import Path

import torch

HERE    = Path(__file__).resolve().parent       # four_track/src/
FT_ROOT = HERE.parent                           # four_track/
ROOT    = FT_ROOT.parent                        # BirdCLEF/

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config  # noqa: E402
from model_a1 import BirdSEDModelA1  # noqa: E402

CKPT_DIR = FT_ROOT / "models" / "a1_ns"
OUT_DIR  = FT_ROOT / "kaggle_datasets" / "a1-effb0-ns-ckpts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_TEMPLATE = "a1_ns_{backbone}_fold{fold}_seed42_bce.pt"


class A1Wrapper(torch.nn.Module):
    def __init__(self, inner: BirdSEDModelA1):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)["clip_logits"]


def export_one(fold: int) -> Path:
    src = CKPT_DIR / CKPT_TEMPLATE.format(backbone=config.BACKBONE, fold=fold)
    if not src.exists():
        sys.exit(f"missing NS checkpoint: {src}")

    model = BirdSEDModelA1(
        backbone_name=config.BACKBONE,
        mixstyle_p=0.0,
    )
    state = torch.load(src, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    model.mixstyle.active = False

    wrapper = A1Wrapper(model).eval()

    example = torch.zeros(1, 3, config.N_MELS, 512)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, strict=False)

    out = traced(example)
    assert out.shape == (1, config.N_CLASSES), f"unexpected output shape {out.shape}"

    out_path = OUT_DIR / f"a1_fold{fold}.pt"
    traced.save(str(out_path))
    print(f"  fold {fold}: traced → {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)",
          flush=True)
    return out_path


def write_dataset_metadata() -> None:
    meta = OUT_DIR / "dataset-metadata.json"
    meta.write_text(
        '{\n'
        '  "title": "BirdCLEF 2026 — A1 EffNet-B0 NS (noisy student, §14.9 L1)",\n'
        '  "id": "stevewatson999/birdclef-2026-a1-effb0-ns-ckpts",\n'
        '  "licenses": [{"name": "CC0-1.0"}]\n'
        '}\n'
    )
    print(f"  metadata → {meta}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, nargs="+", default=[0])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"A1-NS TorchScript export — folds {args.folds}", flush=True)
    print(f"  src dir : {CKPT_DIR}", flush=True)
    print(f"  out dir : {OUT_DIR}", flush=True)

    for f in args.folds:
        export_one(f)

    write_dataset_metadata()
    print("\nDone. Next step:", flush=True)
    print(f"  kaggle datasets create -p {OUT_DIR}", flush=True)
    print("  (or `kaggle datasets version -p {dir} -m 'msg'` if it already exists)",
          flush=True)


if __name__ == "__main__":
    main()
