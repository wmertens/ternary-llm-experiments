"""Bonsai-style ternary linear layer with a mutable curriculum level.

Forward: y = (x @ q(W).T) * s + bias
where q(W) = round(clamp(W, -1, 1) * half) / half, half = (levels-1)//2
and the gradient passes through q via the straight-through estimator.

`levels` is mutated by the training loop to step through the odd-level
curriculum (e.g. 257 -> 129 -> ... -> 3).  At levels=3 this matches
Bonsai's `model/qlinear.py` (clamp(-1,1).round()) with a per-row scale.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def quantize_levels(w: torch.Tensor, levels: int) -> torch.Tensor:
    """Quantize w to `levels` evenly spaced values in [-1, 1] including 0.

    levels must be odd so 0 is in the codebook.
    """
    if levels < 3 or levels % 2 == 0:
        raise ValueError(f"levels must be an odd integer >= 3, got {levels}")
    half = (levels - 1) // 2
    return torch.round(w.clamp(-1.0, 1.0) * half) / half


class QLinear(nn.Linear):
    """Linear with per-output-row fp32 scale and STE-quantized weight."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 levels: int = 3, **kwargs) -> None:
        super().__init__(in_features, out_features, bias=bias, **kwargs)
        self.scales = nn.Parameter(torch.ones(out_features))
        self.levels = int(levels)

    def quantized_weight(self) -> torch.Tensor:
        q = quantize_levels(self.weight, self.levels)
        # STE: forward uses q, backward routes gradient to self.weight as identity.
        return self.weight + (q - self.weight).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.quantized_weight()
        y = F.linear(x, w)
        y = y * self.scales.to(y.dtype)
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, bias={self.bias is not None}, levels={self.levels}"


def set_levels(model: nn.Module, levels: int) -> int:
    """Set `levels` on every QLinear in `model`. Returns count updated."""
    n = 0
    for m in model.modules():
        if isinstance(m, QLinear):
            m.levels = int(levels)
            n += 1
    return n


@torch.no_grad()
def clamp_qlinear_weights(model: nn.Module, lo: float = -1.0, hi: float = 1.0) -> int:
    """Project every QLinear latent weight back into [lo, hi].

    The quantizer clamps to [-1,1] anyway, so anything outside that range is
    dead capacity (the rounded value can't change until the latent re-enters
    the box). Lion's sign-based update has no implicit norm control, so this
    projection is needed every step.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, QLinear):
            m.weight.data.clamp_(lo, hi)
            n += 1
    return n
