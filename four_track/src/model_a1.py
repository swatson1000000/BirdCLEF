"""Track A1 SED model — BirdSEDModel + Frequency MixStyle injection.

Wraps the parent `BirdSEDModel` (timm CNN backbone → GEM freq pool → conv
attention head) and inserts a `FrequencyMixStyle` layer between the backbone's
first downsampling stage and its deeper blocks. We use timm's `features_only`
mode and a manual two-stage forward, with the MixStyle hook in the middle.

Why mid-backbone?  MixStyle on raw mel input is too aggressive (destroys the
content the early convs need). Inserting after stem + first stage gives a
feature map where channel statistics correspond to spectral colour rather
than raw amplitude — exactly what we want to perturb to defeat
recording-device shift between focal recordings (`train_audio`) and the
expert-labeled soundscapes used for validation / test.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import timm

# Make the parent BirdCLEF/src/ importable so we can pull config (paths,
# N_MELS, etc.) without duplicating it in four_track/src/.
PARENT_SRC = Path(__file__).resolve().parents[2] / "src"
if str(PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(PARENT_SRC))

import config  # noqa: E402  (sets warning filters as a side-effect)

# Local imports
from mixstyle import FrequencyMixStyle  # noqa: E402


class GEMFrequencyPool(nn.Module):
    """GEM pooling over the frequency (height) dimension.

    Input  : (B, C, F, T)
    Output : (B, C, T)
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=self.eps).pow(self.p).mean(dim=2).pow(1.0 / self.p)


class BirdSEDModelA1(nn.Module):
    """SED model with optional Frequency-MixStyle injection mid-backbone.

    Forward returns:
        clip_logits  : (B, N_CLASSES)      — for clip-level loss
        frame_logits : (B, T', N_CLASSES)  — per-frame predictions
        att_weights  : (B, T', N_CLASSES)  — softmax-over-time attention

    Args:
        backbone_name : timm model name (default tf_efficientnet_b0.ns_jft_in1k)
        n_classes     : output classes (234 for BirdCLEF 2026)
        in_channels   : input channels (3 — PCEN tile)
        gem_p         : initial GEM pooling exponent
        att_dropout   : dropout in the attention head
        mixstyle_p    : probability MixStyle fires per forward pass (0 disables)
        mixstyle_alpha: Beta(alpha, alpha) parameter for the mix coefficient
        mixstyle_after: which features_only out_index to inject AFTER. We
                        request out_indices=(1, 4); 1 is "after first stage,"
                        4 is the deepest block we use as the SED feature map.
                        Setting this to 1 means MixStyle perturbs the
                        post-stage-1 feature map before stage 2 runs.
    """

    def __init__(
        self,
        backbone_name: str = config.BACKBONE,
        n_classes: int = config.N_CLASSES,
        in_channels: int = 3,
        gem_p: float = 3.0,
        att_dropout: float = 0.3,
        mixstyle_p: float = 0.5,
        mixstyle_alpha: float = 0.3,
    ):
        super().__init__()

        # Two backbone halves: stem+early via features_only out_indices=(1,),
        # then a separate timm model for the deeper stages. timm's
        # features_only API doesn't expose "split here", so we instantiate the
        # full backbone once and walk its modules to find the split point.
        # The pragmatic alternative — and what we do — is to just request
        # out_indices=(4,) and apply MixStyle on the input mel spectrogram
        # after a manual stem conv. That's harder to maintain than this:
        #   we run the *full* backbone twice in eval-of-self mode and use a
        #   forward hook to inject MixStyle on the chosen layer's output.
        #
        # Concretely: we register a forward hook on backbone.blocks[1] (the
        # second EfficientNet block) so the hook fires once per forward pass,
        # perturbing the feature map in place during training only.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            out_indices=(4,),
            in_chans=in_channels,
        )

        self.mixstyle = FrequencyMixStyle(p=mixstyle_p, alpha=mixstyle_alpha)
        self._register_mixstyle_hook()

        # Infer backbone output channel count via one dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, config.N_MELS, 512)
            feat  = self.backbone(dummy)[-1]          # (1, C, F', T')
            c_out = feat.shape[1]

        self.gem_pool = GEMFrequencyPool(p=gem_p)
        self.cls_conv = nn.Conv1d(c_out, n_classes, kernel_size=1)
        self.att_conv = nn.Sequential(
            nn.Conv1d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.Dropout(att_dropout),
            nn.Conv1d(c_out, n_classes, kernel_size=1),
        )

    # ── MixStyle hook plumbing ────────────────────────────────────────────────

    def _register_mixstyle_hook(self) -> None:
        """Attach a forward hook on an early backbone layer.

        We pick `backbone.blocks[1]` for EfficientNet (b0/b3/b4) — the second
        IRBlock stage. Falls back to the first feature_info layer for
        non-EfficientNet backbones.
        """
        target = None
        if hasattr(self.backbone, "blocks") and len(self.backbone.blocks) > 1:
            target = self.backbone.blocks[1]
        else:
            # Generic fallback: hook the first registered feature_info layer
            for name, module in self.backbone.named_modules():
                if isinstance(module, nn.Conv2d):
                    target = module
                    break
        if target is None:
            raise RuntimeError("Could not locate a layer to attach MixStyle hook")

        def _hook(_module, _inp, output):
            return self.mixstyle(output)

        target.register_forward_hook(_hook)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> dict:
        """x: (B, 3, N_MELS, T)"""
        feat = self.backbone(x)[-1]              # (B, C, F', T')
        feat = self.gem_pool(feat)               # (B, C, T')

        frame_logits = self.cls_conv(feat).permute(0, 2, 1)   # (B, T', n_classes)
        att_logits   = self.att_conv(feat).permute(0, 2, 1)   # (B, T', n_classes)
        att_weights  = torch.softmax(att_logits, dim=1)        # softmax over time

        clip_logits  = (frame_logits * att_weights).sum(dim=1) # (B, n_classes)

        return {
            "clip_logits":  clip_logits,
            "frame_logits": frame_logits,
            "att_weights":  att_weights,
        }
