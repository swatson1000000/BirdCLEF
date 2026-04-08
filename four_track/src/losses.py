"""Asymmetric Loss for multi-label SED — Track A1.

Reference: Ben-Baruch et al. 2020 — "Asymmetric Loss For Multi-Label
Classification" (https://arxiv.org/abs/2009.14119).

ASL is a focal-style loss with separate gamma exponents for positive and
negative samples and a learnable probability shift on the negative side.
On long-tailed multi-label problems (BirdCLEF: 234 classes, most rare) ASL
typically beats vanilla BCE by suppressing the gradient from easy negatives
and from likely false negatives produced by missing secondary labels.

Notes for BirdCLEF use:
  - We mask out secondary-label positions before computing the loss in the
    training loop (consistent with parent `BirdSEDModel` BCE pipeline).
  - `clip=0.05` is the original paper's recommended probability shift on the
    negative branch — drops easy negatives whose predicted prob is already
    below 0.05.
"""

import torch
import torch.nn as nn


class AsymmetricLossOptimized(nn.Module):
    """Asymmetric Loss (optimized, in-place version) for multi-label tasks.

    Args:
        gamma_neg : focusing exponent on negative samples (default 4)
        gamma_pos : focusing exponent on positive samples (default 1)
        clip      : probability margin shift on negatives (default 0.05)
        eps       : numerical stability (default 1e-8)
        reduction : 'none' (default — for use with secondary mask),
                    'mean', or 'sum'
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "none",
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip      = clip
        self.eps       = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits, targets: (B, N_CLASSES) — float32 / float."""
        # Sigmoid in fp32 for numerical stability under bf16/fp16 autocast.
        x_sig = torch.sigmoid(logits.float())
        xs_pos = x_sig
        xs_neg = 1.0 - x_sig

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Cross-entropy components
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1.0 - targets)
            pt  = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            loss = loss * one_sided_w

        loss = -loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss  # 'none' — caller masks/aggregates


class HybridBceAsl(nn.Module):
    """Convex combination of BCE and ASL.

    Useful for stable warm-up: ASL alone can underflow gradients on rare
    classes early in training. Schedule by epoch in the trainer if desired.
    """

    def __init__(self, bce_weight: float = 0.5, **asl_kwargs):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.asl = AsymmetricLossOptimized(reduction="none", **asl_kwargs)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets.float())
        asl_loss = self.asl(logits, targets.float())
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * asl_loss
