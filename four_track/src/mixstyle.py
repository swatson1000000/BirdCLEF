"""Frequency-axis MixStyle for SED backbones — Track A1.

MixStyle (Zhou et al. 2021, https://arxiv.org/abs/2104.02008) perturbs the
per-channel feature statistics of an instance using statistics drawn from
another randomly-paired instance in the batch. It improves cross-domain
generalisation by simulating recording-device / habitat shifts at the
feature-statistics level.

For audio SED on mel spectrograms the most useful axis to "mix" is
**frequency** — the per-frequency-band mean/std encodes the spectral colour
of the recording (microphone response, ambient noise, distance to source).
Time-axis statistics are mostly bird-call content and should not be mixed.

Insertion point: after the first one or two backbone stages, before the
deeper convolutional blocks. This is consistent with the original paper.

Toggle `mixstyle.training_mode = False` (or `model.eval()`) for inference;
the layer becomes a no-op.
"""

import random
import torch
import torch.nn as nn


class FrequencyMixStyle(nn.Module):
    """MixStyle applied along the frequency (height) dimension of a 4-D feature
    map produced by a 2-D conv backbone consuming mel spectrograms.

    Input  : (B, C, F, T)   — frequency-major feature map
    Output : (B, C, F, T)   — same shape, statistics-perturbed when active

    Args:
        p     : probability the layer fires per forward pass (paper default 0.5)
        alpha : Beta(alpha, alpha) parameter for the mix coefficient
        eps   : numerical stability for std
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.3, eps: float = 1e-6):
        super().__init__()
        self.p     = p
        self.alpha = alpha
        self.eps   = eps
        self._beta = torch.distributions.Beta(alpha, alpha)
        # External toggle: disable from outside without flipping module.training,
        # useful when you want eval-mode for BatchNorm but still want MixStyle
        # active (we don't, but kept for symmetry with the paper code).
        self.active = True

    def extra_repr(self) -> str:
        return f"p={self.p}, alpha={self.alpha}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (self.training and self.active):
            return x
        if random.random() > self.p:
            return x

        B = x.size(0)
        # Per-instance, per-channel stats over the FREQUENCY dimension only
        # (not time): preserves temporal structure of bird calls while mixing
        # the spectral colour of the recording.
        mu  = x.mean(dim=2, keepdim=True)                       # (B, C, 1, T)
        var = x.var(dim=2, keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        x_normed = (x - mu) / sig                               # (B, C, F, T)

        # Random pair index within the batch
        perm = torch.randperm(B, device=x.device)
        mu2  = mu[perm]
        sig2 = sig[perm]

        # Mix coefficient ~ Beta(alpha, alpha)
        lam = self._beta.sample((B, 1, 1, 1)).to(x.device)
        mu_mix  = lam * mu  + (1.0 - lam) * mu2
        sig_mix = lam * sig + (1.0 - lam) * sig2

        return x_normed * sig_mix + mu_mix
