"""Bonsai-style ternary linear layer with per-(row, column-group) scales.

Forward: y = x @ (q(W) ⊙ S)^T + bias
  q(W) = round(clamp(W, -1, 1) * half) / half     # half = (levels - 1) // 2
  S    = scales broadcast from [out, n_groups] to [out, in]
         (each group of `group_size` consecutive input columns shares one scale)

`levels` is mutated by the training loop to step through the odd-level
curriculum (e.g. 257 -> 129 -> ... -> 3). At levels=3, q(W) ∈ {-1, 0, +1}
per element; combined with the per-(row, group) scale we recover Bonsai's
deployment layout where each (row, group) has values in {-s, 0, +s}.

Per-(row, group) scaling — instead of plain per-row — is the key fidelity
lever: a finer-grained scale gives each group of 128 input columns its own
dynamic range, so mid-magnitude weights round to ±1 instead of 0. Bonsai's
trained Ternary-1.7B has ~62% nonzero density vs ~15% nonzero with per-row
scales (verified empirically on prism-ml/Ternary-Bonsai-1.7B-unpacked).

Gradient flow:
- self.weight (latent in [-1, 1] per (row, group)): receives gradient via
  the STE identity through quantize_levels.
- self.scales: receives gradient from the post-quant multiply directly.
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
    """Linear with per-(row × column-group) scales and STE-quantized weight.

    Caches the quantized output `q` (unscaled, in [-1, 1] per element)
    across the grad_accum window — latents don't change between opt.step
    calls, so q is identical for all grad_accum × 2 (fwd+bwd) calls within
    an opt.step. Scales are NOT cached because they're trainable too;
    they're applied each forward (cheap multiply on a [out × in] tensor
    relative to the matmul cost).

    Cache invalidated by clamp_qlinear_weights, set_levels — anything that
    mutates the latent or codebook.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 levels: int = 3, group_size: int = 128, **kwargs) -> None:
        super().__init__(in_features, out_features, bias=bias, **kwargs)
        if in_features % group_size != 0:
            raise ValueError(
                f"in_features={in_features} not divisible by group_size={group_size}; "
                "pick a group_size that divides every projection's in_features "
                "(SmolLM2-135M: try 32 or 64; Qwen3-1.7B: 128)")
        self.group_size = int(group_size)
        self.n_groups = in_features // self.group_size
        # Replaces nn.Linear's per-row scale (which we never used as a scale
        # anyway since scales were a separate Parameter). The scale tensor
        # is [out_features, n_groups]; broadcast to [out_features, in_features]
        # at forward time by repeating each group's scale across its columns.
        self.scales = nn.Parameter(torch.ones(out_features, self.n_groups))
        self.levels = int(levels)
        self._q_cache: torch.Tensor | None = None

    def invalidate_q_cache(self) -> None:
        self._q_cache = None

    def quantized_weight(self) -> torch.Tensor:
        """STE-quantized latent weight, in [-1, 1] per element. The
        per-(row, group) scale is NOT applied here — that happens in
        forward() so the matmul sees the actual scaled weight."""
        if self._q_cache is None:
            with torch.no_grad():
                q = quantize_levels(self.weight, self.levels)
            self._q_cache = q
        return self.weight + (self._q_cache - self.weight).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_ste = self.quantized_weight()  # [out, in], in [-1, 1]
        # Apply per-(row, group) scales: expand scales [out, n_groups] to
        # [out, in] by broadcasting along the group's columns.
        q_blocks = q_ste.view(self.out_features, self.n_groups, self.group_size)
        scales_b = self.scales.unsqueeze(-1).to(q_blocks.dtype)
        w_scaled = (q_blocks * scales_b).view(self.out_features, self.in_features)
        y = F.linear(x, w_scaled)
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bias={self.bias is not None}, levels={self.levels}, "
                f"group_size={self.group_size}, n_groups={self.n_groups}")


def set_levels(model: nn.Module, levels: int) -> int:
    """Set `levels` on every QLinear in `model`. Returns count updated."""
    n = 0
    for m in model.modules():
        if isinstance(m, QLinear):
            m.levels = int(levels)
            m.invalidate_q_cache()
            n += 1
    return n


@torch.no_grad()
def clamp_qlinear_weights(model: nn.Module, lo: float = -1.0, hi: float = 1.0) -> int:
    """Project every QLinear latent weight back into [lo, hi]. The quantizer
    clamps to [-1, 1] anyway, so anything outside that range is dead capacity
    (the rounded value can't change until the latent re-enters the box). Lion's
    sign-based update has no implicit norm control, so this projection is
    needed every step.

    Also invalidates the per-module quantized-output cache, since the latent
    just changed.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, QLinear):
            m.weight.data.clamp_(lo, hi)
            m.invalidate_q_cache()
            n += 1
    return n


def invalidate_q_cache(model: nn.Module) -> int:
    """Invalidate every QLinear's quantized-output cache. Use after any
    out-of-band weight mutation (load_state_dict, etc.) that bypassed
    clamp_qlinear_weights / set_levels."""
    n = 0
    for m in model.modules():
        if isinstance(m, QLinear):
            m.invalidate_q_cache()
            n += 1
    return n
