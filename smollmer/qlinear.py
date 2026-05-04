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


def quantize_sherry(w: torch.Tensor, levels: int, block: int = 4) -> torch.Tensor:
    """`quantize_levels` plus the Sherry constraint: in each contiguous block
    of `block` weights along the last dim, exactly one slot is 0 and the rest
    are nonzero. Two rules per block:

      1. The smallest-|latent| slot is forced to 0 ("designated zero").
      2. Any *other* slot that would have quantized to 0 is bumped to
         sign(w) * (1/half), the smallest nonzero quantization step.

    Both rules use the latent w, not the post-quant value (low-L quantization
    has many ties at 0, so the latent breaks them deterministically).

    Applied throughout the curriculum, not just at L=3: at high L rule 1
    zeroes an already-tiny weight (cheap) and rule 2 almost never fires
    (few latents round to 0 with a fine codebook); as L drops, the gradient
    keeps pushing the non-designated weights away from 0 so by L=3 the
    constraint is satisfied by structure rather than by the bump itself.

    Pack format: see `pack.pack_sherry` (5 bits per block = 1.25 bpw, vs
    1.6 bpw for plain `pack_ternary_158`).
    """
    if w.shape[-1] % block != 0:
        raise ValueError(f"last dim {w.shape[-1]} not divisible by block={block}")
    if levels < 3 or levels % 2 == 0:
        raise ValueError(f"levels must be an odd integer >= 3, got {levels}")
    half = (levels - 1) // 2
    min_step = 1.0 / half
    q = quantize_levels(w, levels)
    blocks = q.reshape(*q.shape[:-1], -1, block)
    w_blocks = w.reshape(*w.shape[:-1], -1, block)
    min_idx = w_blocks.abs().argmin(dim=-1, keepdim=True)
    is_min = torch.zeros_like(blocks, dtype=torch.bool).scatter_(-1, min_idx, True)
    # sign(0) = 0 in torch; treat the zero-latent edge case as +1 so the
    # bumped value is well-defined.
    sign = torch.where(w_blocks >= 0,
                       torch.ones_like(w_blocks),
                       -torch.ones_like(w_blocks))
    bumped = torch.where(~is_min & (blocks == 0),
                         sign * min_step, blocks)
    bumped = torch.where(is_min, torch.zeros_like(bumped), bumped)
    return bumped.reshape(w.shape)


class QLinear(nn.Linear):
    """Linear with per-output-row fp32 scale and STE-quantized weight."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 levels: int = 3, sherry: bool = False, **kwargs) -> None:
        super().__init__(in_features, out_features, bias=bias, **kwargs)
        self.scales = nn.Parameter(torch.ones(out_features))
        self.levels = int(levels)
        self.sherry = bool(sherry)

    def quantized_weight(self) -> torch.Tensor:
        q = (quantize_sherry(self.weight, self.levels) if self.sherry
             else quantize_levels(self.weight, self.levels))
        # STE: forward uses q, backward routes gradient to self.weight as identity.
        return self.weight + (q - self.weight).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Latents may be stored in fp16 while activations flow as fp32 / bf16
        # (autocast may or may not be active). Cast so F.linear gets matching
        # dtypes; values are bounded [-1,1] so the down/up cast is safe.
        w = self.quantized_weight()
        y = F.linear(x, w.to(x.dtype))
        y = y * self.scales.to(y.dtype)
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bias={self.bias is not None}, levels={self.levels}, "
                f"sherry={self.sherry}")


def set_levels(model: nn.Module, levels: int) -> int:
    """Set `levels` on every QLinear in `model`. Returns count updated."""
    n = 0
    for m in model.modules():
        if isinstance(m, QLinear):
            m.levels = int(levels)
            n += 1
    return n


def set_sherry(model: nn.Module, on: bool) -> int:
    """Toggle the Sherry constraint on every QLinear. Returns count updated."""
    n = 0
    for m in model.modules():
        if isinstance(m, QLinear):
            m.sherry = bool(on)
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
