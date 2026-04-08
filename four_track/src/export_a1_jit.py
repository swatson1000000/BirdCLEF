"""TorchScript-export the four A1 fold checkpoints used in the LB ensemble.

Drops fold 3 (calibration outlier — see new_plan.md "A1 training results
2026-04-07"). Each surviving fold is loaded into a `BirdSEDModelA1` instance,
set to `.eval()`, traced with a representative `(1, 3, N_MELS, T)` input, and
saved as a self-contained TorchScript `.pt` file. The traced files have no
timm or four_track package dependency and can be loaded inside the Kaggle
inference notebook with `torch.jit.load(...)`.

Outputs land in `four_track/kaggle_datasets/a1-effb0-ckpts/` ready for
`kaggle datasets create`.
"""

import shutil
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

# We keep folds {0, 1, 2, 4}; fold 3 dropped per new_plan.md analysis.
KEEP_FOLDS = [0, 1, 2, 4]

CKPT_DIR = FT_ROOT / "models" / "a1"
OUT_DIR  = FT_ROOT / "kaggle_datasets" / "a1-effb0-ckpts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class A1Wrapper(torch.nn.Module):
    """Strip the dict return of BirdSEDModelA1 down to a single tensor.

    TorchScript tracing prefers tensor I/O over Python dicts; the consumer
    only needs `clip_logits` for inference.
    """

    def __init__(self, inner: BirdSEDModelA1):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)["clip_logits"]


def export_one(fold: int) -> Path:
    src = CKPT_DIR / f"a1_{config.BACKBONE}_fold{fold}_seed42_hybrid.pt"
    if not src.exists():
        sys.exit(f"missing checkpoint: {src}")

    model = BirdSEDModelA1(
        backbone_name=config.BACKBONE,
        mixstyle_p=0.0,           # disable so trace produces a no-op hook path
    )
    state = torch.load(src, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    # Belt-and-braces: explicitly switch off the mixstyle module too.
    model.mixstyle.active = False

    wrapper = A1Wrapper(model).eval()

    # Tracing input: (1, 3, N_MELS, T) — T is whatever a 20s chunk produces
    # at HOP_LENGTH=1252 on SR=32000 → 512 frames. Match training shape.
    example = torch.zeros(1, 3, config.N_MELS, 512)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, strict=False)

    # Sanity check: tensor in → tensor out, expected shape
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
        '  "title": "BirdCLEF 2026 — A1 EffNet-B0 SED 4-fold ensemble",\n'
        '  "id": "stevewatson999/birdclef-2026-a1-effb0-ckpts",\n'
        '  "licenses": [{"name": "CC0-1.0"}]\n'
        '}\n'
    )
    print(f"  metadata → {meta}", flush=True)


def main() -> None:
    print(f"A1 TorchScript export — keep folds {KEEP_FOLDS} (drop fold 3)", flush=True)
    print(f"  src dir : {CKPT_DIR}", flush=True)
    print(f"  out dir : {OUT_DIR}", flush=True)

    for f in KEEP_FOLDS:
        export_one(f)

    write_dataset_metadata()
    print("\nDone. Next step:", flush=True)
    print(f"  kaggle datasets create -p {OUT_DIR}", flush=True)
    print("  (or `kaggle datasets version -p {dir} -m 'msg'` if it already exists)",
          flush=True)


if __name__ == "__main__":
    main()
