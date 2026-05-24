"""TorchScript-export the five B2 fold checkpoints used in the LB ensemble.

Mirrors `export_a1_jit.py`. B2 is the ConvNeXt-tiny SED branch built for
§14.16 (architectural diversity from A1's EffNet-B0). All 5 folds passed
the 0.7414 hybrid gate (mean val_roc_auc 0.7904 across folds 0–4), so
none are dropped here — see `new_plan.md` §14.16 and the
`b2_5fold_status_*.txt` report.

Each fold is loaded into a `BirdSEDModelB2` instance, set to `.eval()`,
traced with a representative `(1, 3, N_MELS, T)` input, and saved as a
self-contained TorchScript `.pt` file. The traced files have no timm or
four_track package dependency and can be loaded inside the Kaggle
inference notebook with `torch.jit.load(...)`.

Outputs land in `four_track/kaggle_datasets/b2-convnext-ckpts/` ready for
`kaggle datasets create`.
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
from model_b2 import BirdSEDModelB2  # noqa: E402

KEEP_FOLDS = [0, 1, 2, 3, 4]

CKPT_DIR = FT_ROOT / "models" / "b2"
OUT_DIR  = FT_ROOT / "kaggle_datasets" / "b2-convnext-ckpts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class B2Wrapper(torch.nn.Module):
    """Strip the dict return of BirdSEDModelB2 down to a single tensor.

    TorchScript tracing prefers tensor I/O over Python dicts; the consumer
    only needs `clip_logits` for inference.
    """

    def __init__(self, inner: BirdSEDModelB2):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)["clip_logits"]


def _short_backbone_tag(backbone_name: str) -> str:
    """Match `train_b2.py`'s checkpoint filename convention.

    `train_b2.py` saves with a short tag (e.g. "convnext_tiny") rather than
    the full timm slug. Trim trailing pretrain qualifiers if the user passes
    a full timm name.
    """
    return backbone_name.split(".")[0]


def export_one(fold: int, backbone_name: str, loss_suffix: str) -> Path:
    tag = _short_backbone_tag(backbone_name)
    src = CKPT_DIR / f"b2_{tag}_fold{fold}_seed42_{loss_suffix}.pt"
    if not src.exists():
        sys.exit(f"missing checkpoint: {src}")

    model = BirdSEDModelB2(
        backbone_name=backbone_name,
        mixstyle_p=0.0,           # disable so trace produces a no-op hook path
    )
    state = torch.load(src, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    model.mixstyle.active = False

    wrapper = B2Wrapper(model).eval()

    example = torch.zeros(1, 3, config.N_MELS, 512)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, strict=False)

    out = traced(example)
    assert out.shape == (1, config.N_CLASSES), f"unexpected output shape {out.shape}"

    out_path = OUT_DIR / f"b2_fold{fold}.pt"
    traced.save(str(out_path))
    print(f"  fold {fold}: traced → {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)",
          flush=True)
    return out_path


def write_dataset_metadata() -> None:
    meta = OUT_DIR / "dataset-metadata.json"
    meta.write_text(
        '{\n'
        '  "title": "BirdCLEF 2026 — B2 ConvNeXt-Tiny 5-fold ensemble",\n'
        '  "id": "stevewatson999/birdclef-2026-b2-convnext-ckpts",\n'
        '  "licenses": [{"name": "CC0-1.0"}]\n'
        '}\n'
    )
    print(f"  metadata → {meta}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=KEEP_FOLDS,
                        help=f"folds to export (default: {KEEP_FOLDS})")
    parser.add_argument("--backbone", type=str,
                        default=BirdSEDModelB2.DEFAULT_BACKBONE,
                        help="timm backbone slug (must match training)")
    parser.add_argument("--loss-suffix", type=str, default="hybrid",
                        help="ckpt filename loss suffix (hybrid|asl|bce)")
    parser.add_argument("--skip-metadata", action="store_true",
                        help="don't overwrite dataset-metadata.json")
    args = parser.parse_args()

    print(f"B2 TorchScript export — folds {args.folds} loss={args.loss_suffix} "
          f"backbone={args.backbone}", flush=True)
    print(f"  src dir : {CKPT_DIR}", flush=True)
    print(f"  out dir : {OUT_DIR}", flush=True)

    for f in args.folds:
        export_one(f, backbone_name=args.backbone, loss_suffix=args.loss_suffix)

    if not args.skip_metadata:
        write_dataset_metadata()
    print("\nDone. Next step:", flush=True)
    print(f"  kaggle datasets create -p {OUT_DIR}", flush=True)
    print("  (or `kaggle datasets version -p {dir} -m 'msg'` if it already exists)",
          flush=True)


if __name__ == "__main__":
    main()
