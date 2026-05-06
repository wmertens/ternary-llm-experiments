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


def quantize_top1(w: torch.Tensor, levels: int, block: int = 4,
                  alpha: float = 1.0) -> torch.Tensor:
    """Constrain each block of `block` consecutive weights to have AT MOST
    ONE weight at the codebook extreme (±1, i.e. q*half = ±half). All
    other slots are unaffected.

    Level-aware behavior across the curriculum:

      * **L=3** (half=1): every nonzero codebook value IS ±1, so the
        constraint becomes "at most 1 nonzero per block of 4" — the
        strict top1 form that pairs with `pack_top1` for 1.0 bpw
        packing (1 + 4×2 = 9 codes per block).
      * **L>3**: most weights are not at the codebook boundary
        (saturation is rare with fine codebooks), so the constraint
        rarely fires and training proceeds essentially unconstrained.
        It tightens smoothly as L drops through the curriculum.

    Demotion: extreme slots beyond the first per block are pushed to
    sign·(half-1)/half — the next-lower codebook value. At L=3 this is
    0 (full top1). At L=257 it's ±127/128 ≈ ±0.992 (a tiny precision
    nudge, not a destructive zero-out).

    `alpha` blends between off (alpha=0, plain `quantize_levels`) and
    full constraint (alpha=1). The argmax-among-extremes is computed
    on the LATENT magnitude `|w|`, not the post-quant value, so the
    "kept" extreme is the slot the gradient is pushing hardest.
    """
    if w.shape[-1] % block != 0:
        raise ValueError(f"last dim {w.shape[-1]} not divisible by block={block}")
    if levels < 3 or levels % 2 == 0:
        raise ValueError(f"levels must be an odd integer >= 3, got {levels}")
    q = quantize_levels(w, levels)
    if alpha <= 0.0:
        return q
    half = (levels - 1) // 2
    next_lower = (half - 1) / half  # 0 at L=3, 0.5 at L=5, ..., 127/128 at L=257
    blocks = q.reshape(*q.shape[:-1], -1, block)
    w_blocks = w.reshape(*w.shape[:-1], -1, block)
    # Slots at the codebook extreme.
    is_extreme = blocks.abs() >= 1.0
    # Among extremes, pick the one with the largest latent |w|. Mask out
    # non-extremes with -1 so argmax ignores them; if a block has zero
    # extremes argmax returns 0 by default, but the demote-mask below
    # then evaluates to all-False so the block is unchanged.
    masked_abs = torch.where(is_extreme, w_blocks.abs(),
                              torch.full_like(w_blocks, -1.0))
    keep_idx = masked_abs.argmax(dim=-1, keepdim=True)
    is_keep = torch.zeros_like(blocks, dtype=torch.bool).scatter_(-1, keep_idx, True)
    # Demote the OTHER extreme slots toward ±next_lower; alpha-blend.
    demoted = torch.sign(blocks) * next_lower
    new_blocks = torch.where(
        is_extreme & ~is_keep,
        alpha * demoted + (1.0 - alpha) * blocks,
        blocks,
    )
    return new_blocks.reshape(w.shape)


def quantize_sherry(w: torch.Tensor, levels: int, block: int = 4,
                    alpha: float = 1.0) -> torch.Tensor:
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

    `alpha` linearly interpolates between plain `quantize_levels` (alpha=0)
    and full Sherry (alpha=1). Use to ramp the constraint in over the first
    N steps of a stage, so the network has a chance to find the right
    block-wise sign pattern before being pinned to a Sherry-valid
    configuration. (At L=3, abruptly snapping the constraint can lock the
    model into a locally-suboptimal configuration that the LR-decayed
    optimizer can't escape — see the flip_rate cliff.)

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
    if alpha <= 0.0:
        return q
    blocks = q.reshape(*q.shape[:-1], -1, block)
    w_blocks = w.reshape(*w.shape[:-1], -1, block)
    min_idx = w_blocks.abs().argmin(dim=-1, keepdim=True)
    is_min = torch.zeros_like(blocks, dtype=torch.bool).scatter_(-1, min_idx, True)
    # sign(0) = 0 in torch; treat the zero-latent edge case as +1 so the
    # bumped value is well-defined.
    sign = torch.where(w_blocks >= 0,
                       torch.ones_like(w_blocks),
                       -torch.ones_like(w_blocks))
    # Rule 1 (min slot → 0), blended: alpha=1 forces 0, alpha=0 leaves q.
    # Equivalent to (1-alpha)*q for the min slot.
    blocks = torch.where(is_min, (1.0 - alpha) * blocks, blocks)
    # Rule 2 (non-min zero slot → ±min_step), blended: alpha=1 fully bumps,
    # alpha=0 leaves at 0 (no bump). Note this uses the original `q`-derived
    # `blocks==0` mask, but blocks has been updated for the min slot already
    # — that's fine because is_min and (~is_min & (q == 0)) are disjoint by
    # construction (a slot is either the min or not).
    blocks = torch.where(~is_min & (q.reshape_as(blocks) == 0),
                         alpha * sign * min_step, blocks)
    return blocks.reshape(w.shape)


class QLinear(nn.Linear):
    """Linear with per-output-row fp32 scale and STE-quantized weight.

    Caches the quantized output `q` across the grad_accum window. Latents
    don't change between opt.step calls, so q is identical for all
    grad_accum × 2 (fwd+bwd) calls within an opt.step — no point recomputing
    quantize_sherry / quantize_levels each time. Cache is invalidated by
    clamp_qlinear_weights, set_levels, set_sherry — anything that mutates
    weight or codebook.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 levels: int = 3, sherry: bool = False, top1: bool = False,
                 **kwargs) -> None:
        super().__init__(in_features, out_features, bias=bias, **kwargs)
        self.scales = nn.Parameter(torch.ones(out_features))
        self.levels = int(levels)
        # `sherry` and `top1` are mutually exclusive block-structure
        # constraints. Sherry: 1 zero per block of 4 (75% nonzero). top1:
        # 1 nonzero per block of 4 (25% nonzero, matches the natural ~85%
        # sparsity of trained ternary far better). At most one should be
        # True at a time — set_sherry / set_top1 enforce this mutex.
        self.sherry = bool(sherry)
        self.top1 = bool(top1)
        if self.sherry and self.top1:
            raise ValueError("sherry and top1 are mutually exclusive")
        # alpha=1.0 is the active constraint at full strength; 0.0 disables
        # it (equivalent to plain quantize_levels). Use the corresponding
        # setter to ramp during a stage's warmup.
        self.sherry_alpha: float = 1.0
        self.top1_alpha: float = 1.0
        self._q_cache: torch.Tensor | None = None

    def invalidate_q_cache(self) -> None:
        self._q_cache = None

    def quantized_weight(self) -> torch.Tensor:
        if self._q_cache is None:
            with torch.no_grad():
                if self.top1:
                    q = quantize_top1(self.weight, self.levels,
                                      alpha=self.top1_alpha)
                elif self.sherry:
                    q = quantize_sherry(self.weight, self.levels,
                                        alpha=self.sherry_alpha)
                else:
                    q = quantize_levels(self.weight, self.levels)
            self._q_cache = q
        # STE: forward uses q, backward routes gradient to self.weight as identity.
        return self.weight + (self._q_cache - self.weight).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No explicit cast: autocast handles fp16-latent / fp32-activation
        # mismatch inside F.linear (the in-kernel cast in cuBLAS is free).
        # Without autocast, callers should set latent dtype to match
        # activation dtype (build_student / chat do this).
        w = self.quantized_weight()
        y = F.linear(x, w)
        y = y * self.scales.to(y.dtype)
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bias={self.bias is not None}, levels={self.levels}, "
                f"sherry={self.sherry}, top1={self.top1}")


def set_levels(model: nn.Module, levels: int) -> int:
    """Set `levels` on every QLinear in `model`. Returns count updated."""
    n = 0
    for m in model.modules():
        if isinstance(m, QLinear):
            m.levels = int(levels)
            m.invalidate_q_cache()
            n += 1
    return n


def set_sherry(model: nn.Module, on: bool) -> int:
    """Toggle the Sherry constraint on every QLinear. Mutually exclusive with
    top1; turning sherry on clears top1. Returns count updated."""
    n = 0
    for m in model.modules():
        if isinstance(m, QLinear):
            m.sherry = bool(on)
            if on:
                m.top1 = False
            m.invalidate_q_cache()
            n += 1
    return n


def set_top1(model: nn.Module, on: bool) -> int:
    """Toggle the top1 constraint on every QLinear. Mutually exclusive with
    sherry; turning top1 on clears sherry. Returns count updated."""
    n = 0
    for m in model.modules():
        if isinstance(m, QLinear):
            m.top1 = bool(on)
            if on:
                m.sherry = False
            m.invalidate_q_cache()
            n += 1
    return n


def set_sherry_alpha(model: nn.Module, alpha: float) -> int:
    """Set the Sherry blend factor on every QLinear. alpha=1 → full Sherry,
    alpha=0 → no Sherry (plain quantize_levels). Use to ramp the constraint
    in over the first N steps of a stage. Invalidates the q-cache so the
    new alpha takes effect on the next forward."""
    n = 0
    a = float(alpha)
    for m in model.modules():
        if isinstance(m, QLinear):
            m.sherry_alpha = a
            m.invalidate_q_cache()
            n += 1
    return n


def set_top1_alpha(model: nn.Module, alpha: float) -> int:
    """Set the top1 blend factor on every QLinear. alpha=1 → full top1
    (only argmax slot per block keeps a nonzero value), alpha=0 → no top1
    (plain quantize_levels). Use to ramp the constraint in. Invalidates
    the q-cache."""
    n = 0
    a = float(alpha)
    for m in model.modules():
        if isinstance(m, QLinear):
            m.top1_alpha = a
            m.invalidate_q_cache()
            n += 1
    return n


@torch.no_grad()
def clamp_qlinear_weights(model: nn.Module, lo: float = -1.0, hi: float = 1.0) -> int:
    """Project every QLinear latent weight back into [lo, hi].

    The quantizer clamps to [-1,1] anyway, so anything outside that range is
    dead capacity (the rounded value can't change until the latent re-enters
    the box). Lion's sign-based update has no implicit norm control, so this
    projection is needed every step.

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
    out-of-band weight mutation (load_state_dict, permute, etc.) that
    bypassed clamp_qlinear_weights / set_levels / set_sherry."""
    n = 0
    for m in model.modules():
        if isinstance(m, QLinear):
            m.invalidate_q_cache()
            n += 1
    return n
