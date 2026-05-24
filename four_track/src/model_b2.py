"""Track B2 SED model — ConvNeXt-tiny backbone + A1-style SED head.

Second SED branch for §14.16. The goal is **architectural diversity**
from A1 (EffNet-B0) while reusing A1's data pipeline (PCEN mel, 3-channel
tile, 20s chunks) and SED head (GeM freq pool → cls_conv + att_conv
attention-weighted sum).

ConvNeXt vs EffNet — where the diversity comes from:
- Stem: ConvNeXt uses a 4×4 patch-style conv stride 4; EffNet stem is
  3×3 stride 2 + inverted residual.
- Main block: ConvNeXt uses 7×7 depthwise + LayerNorm + GELU in an
  inverted-bottleneck MLP; EffNet uses 3×3 depthwise + SE + swish.
- Normalization: ConvNeXt is LayerNorm throughout; EffNet is BatchNorm.
- Downsampling: ConvNeXt uses separate 2×2 stride-2 conv between
  stages; EffNet downsamples inside the first block of each stage.

Forward interface is identical to BirdSEDModelA1 so downstream code
(losses, val, JIT export) can switch models by class reference only.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import timm

PARENT_SRC = Path(__file__).resolve().parents[2] / "src"
if str(PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(PARENT_SRC))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import config  # noqa: E402
from mixstyle import FrequencyMixStyle  # noqa: E402
from model_a1 import GEMFrequencyPool  # noqa: E402  (shared head component)


class BirdSEDModelB2(nn.Module):
    """B2 SED model — ConvNeXt-tiny + GeM freq pool + attention head.

    Forward returns the same dict shape as BirdSEDModelA1:
        clip_logits  : (B, N_CLASSES)
        frame_logits : (B, T', N_CLASSES)
        att_weights  : (B, T', N_CLASSES)
    """

    DEFAULT_BACKBONE = "convnext_tiny.fb_in22k_ft_in1k"

    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        n_classes: int = config.N_CLASSES,
        in_channels: int = 3,
        gem_p: float = 3.0,
        att_dropout: float = 0.3,
        mixstyle_p: float = 0.5,
        mixstyle_alpha: float = 0.3,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            out_indices=(3,),      # deepest stage — 768 channels for tiny
            in_chans=in_channels,
        )

        self.mixstyle = FrequencyMixStyle(p=mixstyle_p, alpha=mixstyle_alpha)
        self._register_mixstyle_hook()

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, config.N_MELS, 512)
            c_out = self.backbone(dummy)[-1].shape[1]

        self.gem_pool = GEMFrequencyPool(p=gem_p)
        self.cls_conv = nn.Conv1d(c_out, n_classes, kernel_size=1)
        self.att_conv = nn.Sequential(
            nn.Conv1d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.Dropout(att_dropout),
            nn.Conv1d(c_out, n_classes, kernel_size=1),
        )

    def _register_mixstyle_hook(self) -> None:
        """Hook MixStyle on ConvNeXt `stages[0]` output — analogous to
        EffNet's `blocks[1]` (early enough that channel statistics are
        already semantic, but before the heavy spectral processing).
        """
        target = None
        if hasattr(self.backbone, "stages") and len(self.backbone.stages) > 0:
            target = self.backbone.stages[0]
        else:
            for _name, module in self.backbone.named_modules():
                if isinstance(module, nn.Conv2d):
                    target = module
                    break
        if target is None:
            raise RuntimeError("Could not locate a layer to attach MixStyle hook")

        def _hook(_module, _inp, output):
            return self.mixstyle(output)

        target.register_forward_hook(_hook)

    def forward(self, x: torch.Tensor) -> dict:
        feats = self.backbone(x)                      # [(B, C, F', T')]
        feat  = self.gem_pool(feats[-1])              # (B, C, T')
        frame_logits = self.cls_conv(feat).permute(0, 2, 1)
        att_logits   = self.att_conv(feat).permute(0, 2, 1)
        att_weights  = torch.softmax(att_logits, dim=1)
        clip_logits  = (frame_logits * att_weights).sum(dim=1)
        return {
            "clip_logits":  clip_logits,
            "frame_logits": frame_logits,
            "att_weights":  att_weights,
        }
