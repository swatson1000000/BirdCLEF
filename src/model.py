"""BirdSEDModel — Sound Event Detection model for BirdCLEF+ 2026.

Architecture:
  timm CNN backbone (features_only, out_indices=(4,))
  → GEM pooling over frequency dimension
  → Conv1d attention head over time frames
  → clip-level logits via attention-weighted sum
"""

import warnings

import torch
import torch.nn as nn
import timm

import config


class GEMFrequencyPool(nn.Module):
    """Generalised Mean (GEM) pooling over the frequency (height) dimension.

    Input:  (B, C, F, T)
    Output: (B, C, T)

    The pooling exponent p is a learnable parameter initialised to 3.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=self.eps).pow(self.p).mean(dim=2).pow(1.0 / self.p)


class BirdSEDModel(nn.Module):
    """Sound Event Detection model.

    Forward returns a dict:
      'clip_logits'  : (B, N_CLASSES)     — used for training loss
      'frame_logits' : (B, T', N_CLASSES) — per-frame predictions
      'att_weights'  : (B, T', N_CLASSES) — attention weights (sum to 1 over T')
    """

    def __init__(
        self,
        backbone_name: str = config.BACKBONE,
        n_classes: int = config.N_CLASSES,
        in_channels: int = 3,
        gem_p: float = 3.0,
        att_dropout: float = 0.3,
    ):
        super().__init__()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unexpected keys.*found while loading pretrained weights")
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=True,
                features_only=True,
                out_indices=(4,),
                in_chans=in_channels,
            )

        # Infer backbone output channel count via one dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, config.N_MELS, 512)
            feat  = self.backbone(dummy)[-1]          # (1, C, F', T')
            c_out = feat.shape[1]

        self.gem_pool = GEMFrequencyPool(p=gem_p)

        # Classifier head: maps backbone features → per-frame logits
        self.cls_conv = nn.Conv1d(c_out, n_classes, kernel_size=1)

        # Attention head: produces per-frame, per-class attention weights
        self.att_conv = nn.Sequential(
            nn.Conv1d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.Dropout(att_dropout),
            nn.Conv1d(c_out, n_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        x: (B, 3, N_MELS, T)
        """
        feat = self.backbone(x)[-1]           # (B, C, F', T')
        feat = self.gem_pool(feat)            # (B, C, T')

        # Per-frame predictions
        frame_logits = self.cls_conv(feat).permute(0, 2, 1)   # (B, T', n_classes)

        # Attention weights (softmax over time dimension)
        att_logits  = self.att_conv(feat).permute(0, 2, 1)    # (B, T', n_classes)
        att_weights = torch.softmax(att_logits, dim=1)         # normalised over T'

        # Clip-level logits = attention-weighted sum of frame logits
        clip_logits = (frame_logits * att_weights).sum(dim=1)  # (B, n_classes)

        return {
            "clip_logits"  : clip_logits,
            "frame_logits" : frame_logits,
            "att_weights"  : att_weights,
        }
