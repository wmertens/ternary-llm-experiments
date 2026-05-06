"""Bonsai-style ternary linear layer with per-(row, column-group) scales.

Forward: y = x @ (q(W) ⊙ S)^T + bias
  q(W) = round(clamp(W, -1, 1) * half) / half     # half = (levels - 1) // 2
  S    = scales broadcast from [out, n_groups] to [out, in]
         (each group of `group_size` consecutive input columns shares one scale)

Two quantization modes (per QLinear, controlled by `mode`):

* `mode="levels"` (default): `levels` is mutated by the training loop to
  step through the odd-level curriculum (e.g. 257 -> 129 -> ... -> 3). At
  levels=3, q(W) ∈ {-1, 0, +1} per element; combined with the per-(row,
  group) scale we recover Bonsai's deployment layout where each (row,
  group) has values in {-s, 0, +s}.

* `mode="soft"`: continuous attractor toward {-1, 0, +1} via residual
  contraction. forward weight = c(w) + (1-α)·(w - c(w)) where c(w) is the
  nearest of {-1, 0, +1} (boundaries at ±1/3). α=0 is identity (FP);
  α=1 is hard ternary; values near ±1/3 in output become unreachable as α
  grows ("forbidden bands"). Gradient through the blend is (1-α); the
  parallel L2 attractor penalty (`attractor_l2`) provides the basin force.

Per-(row, group) scaling — instead of plain per-row — is the key fidelity
lever: a finer-grained scale gives each group of 128 input columns its own
dynamic range, so mid-magnitude weights round to ±1 instead of 0. Bonsai's
trained Ternary-1.7B has ~62% nonzero density vs ~15% nonzero with per-row
scales (verified empirically on prism-ml/Ternary-Bonsai-1.7B-unpacked).

Gradient flow:
- self.weight (latent in [-1, 1] per (row, group)): receives gradient via
  the STE identity through quantize_levels (levels mode) or the smooth
  (1-α) slope of soft_ternary (soft mode).
- self.scales: receives gradient from the post-quant multiply directly.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


# Decision boundaries for nearest-ternary on values in [-1, 1] split into
# three equal bins. The +1 attractor's bin is [1/3, 1], etc.
TERNARY_BOUNDARY = 1.0 / 3.0


def quantize_levels(w: torch.Tensor, levels: int) -> torch.Tensor:
    """Quantize w to `levels` evenly spaced values in [-1, 1] including 0.

    levels must be odd so 0 is in the codebook.
    """
    if levels < 3 or levels % 2 == 0:
        raise ValueError(f"levels must be an odd integer >= 3, got {levels}")
    half = (levels - 1) // 2
    return torch.round(w.clamp(-1.0, 1.0) * half) / half


def nearest_ternary(w: torch.Tensor) -> torch.Tensor:
    """c(w): nearest of {-1, 0, +1} with bin boundaries at ±1/3. Result has
    no gradient flow back to w (the rounding is a step function)."""
    with torch.no_grad():
        return torch.where(
            w > TERNARY_BOUNDARY, torch.ones_like(w),
            torch.where(w < -TERNARY_BOUNDARY, -torch.ones_like(w),
                        torch.zeros_like(w)))


def soft_ternary(w: torch.Tensor, alpha: float,
                 c: torch.Tensor | None = None) -> torch.Tensor:
    """Residual contraction reparameterization toward {-1, 0, +1}.

    T_α(w) = c(w) + (1-α)·(w - c(w))

    α ∈ [0, 1]. α=0 → identity, α=1 → hard ternary. As α grows, output
    values near ±1/3 become unreachable (the "forbidden bands" forming
    just before each bin boundary). Gradient w.r.t. w is (1-α) per element,
    which vanishes at α=1 — the parallel L2 attractor penalty
    (`attractor_l2`) is what keeps the basin sharp through the late game.

    Pass a precomputed `c` to avoid recomputing nearest_ternary across the
    grad_accum window (latents are constant between opt.step calls).
    """
    if c is None:
        c = nearest_ternary(w)
    return c + (1.0 - alpha) * (w - c)


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
                 levels: int = 3, group_size: int = 128,
                 mode: str = "levels", alpha: float = 0.0, **kwargs) -> None:
        super().__init__(in_features, out_features, bias=bias, **kwargs)
        if in_features % group_size != 0:
            raise ValueError(
                f"in_features={in_features} not divisible by group_size={group_size}; "
                "pick a group_size that divides every projection's in_features "
                "(SmolLM2-135M: try 32 or 64; Qwen3-1.7B: 128)")
        if mode not in ("levels", "soft"):
            raise ValueError(f"mode must be 'levels' or 'soft', got {mode!r}")
        self.group_size = int(group_size)
        self.n_groups = in_features // self.group_size
        # Replaces nn.Linear's per-row scale (which we never used as a scale
        # anyway since scales were a separate Parameter). The scale tensor
        # is [out_features, n_groups]; broadcast to [out_features, in_features]
        # at forward time by repeating each group's scale across its columns.
        self.scales = nn.Parameter(torch.ones(out_features, self.n_groups))
        self.levels = int(levels)
        self.mode = mode
        self.alpha = float(alpha)
        # In levels mode caches the quantized output; in soft mode caches
        # the nearest-ternary c(w). Both depend only on the latent (and
        # the level/mode setting), not on α — so α can change without
        # invalidating the cache.
        self._q_cache: torch.Tensor | None = None

    def invalidate_q_cache(self) -> None:
        self._q_cache = None

    def quantized_weight(self) -> torch.Tensor:
        """Quantized latent weight, in [-1, 1] per element. Per-(row,
        group) scale is NOT applied here — that happens in forward()."""
        if self.mode == "soft":
            if self._q_cache is None:
                self._q_cache = nearest_ternary(self.weight)
            return soft_ternary(self.weight, self.alpha, self._q_cache)
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
                f"bias={self.bias is not None}, mode={self.mode}, "
                f"levels={self.levels}, alpha={self.alpha}, "
                f"group_size={self.group_size}, n_groups={self.n_groups}")


def set_levels(model: nn.Module, levels: int) -> int:
    """Set `levels` on every QLinear in `model` and switch to levels mode.
    Returns count updated."""
    n = 0
    for m in model.modules():
        if isinstance(m, QLinear):
            m.levels = int(levels)
            m.mode = "levels"
            m.invalidate_q_cache()
            n += 1
    return n


def set_soft_mode(model: nn.Module, alpha: float = 0.0) -> int:
    """Switch every QLinear to soft-ternary mode and seed alpha. Cache
    invalidated so the next forward recomputes c(w) for the new mode.
    Returns count updated."""
    n = 0
    for m in model.modules():
        if isinstance(m, QLinear):
            m.mode = "soft"
            m.alpha = float(alpha)
            m.invalidate_q_cache()
            n += 1
    return n


def set_soft_alpha(model: nn.Module, alpha: float) -> int:
    """Update α on every QLinear (assumed already in soft mode). Does NOT
    invalidate the c(w) cache — c depends on the latent, not α. Returns
    count updated."""
    n = 0
    for m in model.modules():
        if isinstance(m, QLinear):
            m.alpha = float(alpha)
            n += 1
    return n


def attractor_l2(model: nn.Module) -> torch.Tensor:
    """Mean ‖w - c(w)‖² across all QLinear latents (per-element mean, so
    the scale is independent of model size). Differentiable in w; gradient
    is 2/N · (w - c(w)), the basin force pulling each latent toward its
    nearest ternary attractor. Multiply by your λ(α) coefficient and add
    to the loss before backward.

    Reuses the per-module `_q_cache` (== c(w) in soft mode) when available
    so we don't recompute the nearest-ternary every call."""
    parts: list[torch.Tensor] = []
    n_total = 0
    device = None
    for m in model.modules():
        if isinstance(m, QLinear):
            w = m.weight
            device = w.device
            if m.mode == "soft" and m._q_cache is not None:
                c = m._q_cache
            else:
                c = nearest_ternary(w)
            parts.append(((w - c) ** 2).sum())
            n_total += w.numel()
    if not parts or n_total == 0:
        return torch.tensor(0.0, device=device or "cpu")
    return torch.stack(parts).sum() / n_total


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
