"""Export Stage 1 BirdSEDModel checkpoints to ONNX for Kaggle CPU inference.

Each fold checkpoint is exported to:
  models/sed_<backbone>_fold<N>_seed<S>.onnx

The ONNX model accepts a (B, 3, N_MELS, T) float32 mel-spectrogram tensor
and returns a single (B, N_CLASSES) float32 sigmoid probabilities tensor.

Notes:
  - `torch.compile` is cleared before export (compile + ONNX are incompatible).
  - Dynamic batch axis only; H and W are fixed to N_MELS=224, T=512.
  - Uses opset 17.

Usage:
  python src/export_onnx.py [--backbone NAME] [--seed N] [--folds 0,1,2,3,4]
"""

import argparse
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

import config
from model import BirdSEDModel


class BirdSEDModelONNX(nn.Module):
    """Thin wrapper that returns only sigmoid clip probs — no dict output."""

    def __init__(self, model: BirdSEDModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return torch.sigmoid(out["clip_logits"])  # (B, N_CLASSES)


def load_model(ckpt_path: Path, device: torch.device) -> BirdSEDModel:
    model = BirdSEDModel(
        backbone_name=config.BACKBONE,
        n_classes=config.N_CLASSES,
    )
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def export_fold(
    fold: int,
    seed: int,
    backbone: str,
    device: torch.device,
) -> Path:
    ckpt_path = config.MODELS / f"sed_{backbone}_fold{fold}_seed{seed}.pt"
    onnx_path = config.MODELS / f"sed_{backbone}_fold{fold}_seed{seed}.onnx"

    if not ckpt_path.exists():
        print(f"  [SKIP] checkpoint not found: {ckpt_path}")
        return None

    print(f"  fold {fold} — loading {ckpt_path.name} …", flush=True)
    model = load_model(ckpt_path, device).to(device)
    wrapper = BirdSEDModelONNX(model)
    wrapper.eval()

    # Fixed H=N_MELS, T=512, dynamic batch
    dummy = torch.randn(1, 3, config.N_MELS, 512, device=device)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        torch.onnx.export(
            wrapper,
            dummy,
            str(onnx_path),
            opset_version=17,
            input_names=["mel"],
            output_names=["probs"],
            dynamic_axes={"mel": {0: "batch_size"}, "probs": {0: "batch_size"}},
            dynamo=False,   # use TorchScript exporter — embeds weights correctly
        )

    # Quick validation
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dummy_np = dummy.cpu().numpy()
    out = sess.run(["probs"], {"mel": dummy_np})[0]
    assert out.shape == (1, config.N_CLASSES), f"Unexpected output shape: {out.shape}"
    assert 0.0 <= out.min() and out.max() <= 1.0, "Output outside [0,1]"

    size_mb = onnx_path.stat().st_size / 1e6
    print(f"  fold {fold} — exported → {onnx_path.name}  ({size_mb:.1f} MB)  ✓", flush=True)
    return onnx_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default=config.BACKBONE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--folds", default=None,
                        help="Comma-separated fold indices, e.g. '0,1,2,3,4'")
    args = parser.parse_args()

    # Always export from CPU — avoids CUDA graph issues and matches Kaggle env
    device = torch.device("cpu")

    fold_ids = list(range(config.N_FOLDS))
    if args.folds is not None:
        fold_ids = [int(f) for f in args.folds.split(",")]

    print(f"Exporting {len(fold_ids)} fold(s) to ONNX (opset 17) …")
    exported = []
    for fold in fold_ids:
        path = export_fold(fold, args.seed, args.backbone, device)
        if path:
            exported.append(path)

    print(f"\nDone — {len(exported)}/{len(fold_ids)} models exported.")
    for p in exported:
        print(f"  {p}")


if __name__ == "__main__":
    main()
