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

import math

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


def nearest_ternary_quantile(w: torch.Tensor,
                             target_zero_frac: float,
                             group_size: int) -> torch.Tensor:
    """c(w) using a per-(row, group) |w|-quantile cutoff: weights below the
    cutoff map to 0, above to sign(w). By construction, target_zero_frac of
    weights per group end up at 0.

    Why this exists: the fixed ±1/3 boundary classifies a uniform [-1, 1]
    distribution as 33% zero, but real weight distributions concentrate
    near 0 (Gaussian-ish from teacher init), so |w|<1/3 is true for far
    more than 33% — typically 50-70%. The L2 attractor then pulls those
    extra weights all the way to 0, giving a too-sparse deployment.
    Bonsai's trained ternary model is ~62% non-zero (~38% zero).

    Quantile is computed in fp32 and is wrapped in `torch.no_grad`, so c(w)
    is a constant from autograd's perspective — same semantics as
    nearest_ternary."""
    with torch.no_grad():
        if target_zero_frac <= 0 or target_zero_frac >= 1:
            return nearest_ternary(w)
        out_f, in_f = w.shape
        n_groups = in_f // group_size
        wb = w.view(out_f, n_groups, group_size)
        abs_wb = wb.abs().float()
        # quantile() reduces along the in-group axis; keepdim for broadcast.
        cutoff = abs_wb.quantile(float(target_zero_frac),
                                 dim=-1, keepdim=True)
        is_zero = abs_wb <= cutoff
        c = torch.where(is_zero, torch.zeros_like(wb), torch.sign(wb))
        return c.view(out_f, in_f).to(w.dtype)


def module_ternary(m: "QLinear") -> torch.Tensor:
    """c(w) for QLinear m. Uses the per-(row, group) quantile cutoff if
    m.target_zero_frac is set in (0, 1); otherwise falls back to the fixed
    ±1/3 boundary in nearest_ternary. Centralized so every consumer (the
    soft-mode forward cache, attractor_l2, flip-rate metrics, soft-stage
    bin counts) sees the same c(w)."""
    frac = getattr(m, "target_zero_frac", None)
    if frac is not None and 0.0 < frac < 1.0:
        return nearest_ternary_quantile(m.weight, frac, m.group_size)
    return nearest_ternary(m.weight)


def module_ternary_fixed(m: "QLinear") -> torch.Tensor:
    """c(w) using the per-(row, group) well saddle as a FROZEN classifier:
    threshold = well_a/√3 per group. Unlike module_ternary's quantile
    cutoff which moves with the current |w| distribution, this threshold
    is fixed at init by init_well_a — so flips against it represent
    actual basin migrations rather than distribution-relative reshuffling.

    With well_a all 1.0 (default / no calibration), the threshold is
    1/√3 ≈ 0.577 — wider than nearest_ternary's ±1/3, so this matches
    the well's geometric basin rather than the L2 form's classifier.

    Used by the flip_rate_fixed metric to detect actual saddle crossings
    that the moving-quantile classifier misses (it rescales with the
    distribution and so shows ~0 flips even when weights are reorganizing
    around a stable proportion)."""
    with torch.no_grad():
        out_f = m.weight.shape[0]
        wb = m.weight.view(out_f, m.n_groups, m.group_size)
        threshold = (m.well_a.float() / math.sqrt(3.0)).unsqueeze(-1)
        is_zero = wb.abs().float() <= threshold
        c = torch.where(is_zero, torch.zeros_like(wb), torch.sign(wb))
        return c.view(out_f, -1).to(m.weight.dtype)


def gumbel_softmax_binary(logit: torch.Tensor, tau: float,
                          kappa: float) -> torch.Tensor:
    """Binary Gumbel-Softmax relaxation with a single logit.

    Returns soft p(positive outcome) in (0, 1):
        p = σ((2κ·logit + g₁ − g₂) / τ)    g₁, g₂ ~ Gumbel(0,1)

    As τ→0: converges to 1{logit ≥ 0} (hard assignment).
    Gradient flows through σ w.r.t. logit; Gumbel noise is treated
    as a constant (no gradient through the sampling).

    Lion optimizer handles the vanishing gradient magnitude at low τ
    (near-saturated σ) because it uses the sign of the gradient.
    """
    g1 = -torch.log(-torch.log(
        torch.rand_like(logit).clamp_(1e-10, 1.0 - 1e-10)))
    g2 = -torch.log(-torch.log(
        torch.rand_like(logit).clamp_(1e-10, 1.0 - 1e-10)))
    return torch.sigmoid((2.0 * kappa * logit + g1 - g2) / tau)


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


class _HardTernaryWithSmoothGrad(torch.autograd.Function):
    """Hard ternary forward, smooth gradient backward.

    Used when T is at the floor (post-anneal): the loss sees the actual
    deployed {-1,0,+1} weights, but the backward uses the smooth gradient
    at T_floor instead of the dead-zone STE.

    Forward:  sign(w_norm) if |w_norm| > 0.5 else 0  →  exactly {-1, 0, +1}
    Backward: same analytical gradient as _SmoothTernary at temperature T:
              d(w_q)/d(w_norm) = (2/T) * ((1 - p_zero) - w_q_soft²)
    """

    @staticmethod
    def forward(ctx, w_norm: torch.Tensor, temp: float) -> torch.Tensor:
        w_q = torch.where(w_norm.abs() > 0.5,
                          torch.sign(w_norm),
                          torch.zeros_like(w_norm))
        codebook = w_norm.new_tensor([-1.0, 0.0, 1.0])
        probs = torch.softmax(
            -(w_norm.unsqueeze(-1) - codebook).pow(2) / temp, dim=-1)
        w_q_soft = (probs * codebook).sum(-1)
        ctx.save_for_backward(w_q_soft, probs[..., 1].clone())
        ctx.temp = temp
        return w_q

    @staticmethod
    def backward(ctx, grad_w_q: torch.Tensor):
        w_q_soft, p_zero = ctx.saved_tensors
        var_p = (1.0 - p_zero) - w_q_soft.pow(2)
        return grad_w_q * (2.0 / ctx.temp) * var_p, None


class _SmoothTernary(torch.autograd.Function):
    """Smooth ternary expectation with memory-efficient backward.

    Saves w_q and p_zero (2× weight size) instead of the full [O,G,gs,3]
    probs tensor, using the analytical gradient:
        d(w_q)/d(w_norm) = (2/T) * ((1 - p_zero) - w_q²)
    where p_zero = P(k=0) and (1-p_zero) = E[k²].
    """

    @staticmethod
    def forward(ctx, w_norm: torch.Tensor, temp: float) -> torch.Tensor:
        codebook = w_norm.new_tensor([-1.0, 0.0, 1.0])
        dists_sq = (w_norm.unsqueeze(-1) - codebook).pow(2)
        probs = torch.softmax(-dists_sq / temp, dim=-1)
        w_q = (probs * codebook).sum(-1)
        ctx.save_for_backward(w_q, probs[..., 1].clone())
        ctx.temp = temp
        return w_q

    @staticmethod
    def backward(ctx, grad_w_q: torch.Tensor):
        w_q, p_zero = ctx.saved_tensors
        var_p = (1.0 - p_zero) - w_q.pow(2)
        return grad_w_q * (2.0 / ctx.temp) * var_p, None


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
                 mode: str = "levels", alpha: float = 0.0,
                 target_zero_frac: float | None = None,
                 **kwargs) -> None:
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
        # Per-(row, group) well minima location for the triple-well attractor.
        # Default 1.0 → canonical U(w)=w²(w²-1)² with minima at ±1. Set by
        # init_well_a() to √3·quantile(|w_init|, target_zero_frac) per group
        # so the saddle a/√3 lands at the target |w| quantile. Persistent
        # buffer so resume restores the calibration that was used.
        self.register_buffer(
            "well_a",
            torch.ones(out_features, self.n_groups, dtype=torch.float32),
        )
        self.levels = int(levels)
        self.mode = mode
        self.alpha = float(alpha)
        self.target_zero_frac = (float(target_zero_frac)
                                 if target_zero_frac is not None else None)
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
                self._q_cache = module_ternary(self)
            return soft_ternary(self.weight, self.alpha, self._q_cache)
        if self._q_cache is None:
            with torch.no_grad():
                q = quantize_levels(self.weight, self.levels)
            self._q_cache = q
        return self.weight + (self._q_cache - self.weight).detach()

    def _smooth_qat_effective_weight(self) -> torch.Tensor:
        """Smooth ternary QAT via temperature-scaled softmax over {-1, 0, +1}.

        Scale source — set by self.use_hestia_scale:
          * False (legacy):  γ = self.codepoint_c (learnable then frozen)
          * True (Hestia):   γ_inner = mean(|w_latent|) per (row, group),
                             recomputed each forward, then DETACHED from
                             autograd (Hestia §3.1: γ is treated as a
                             constant during backpropagation). This gives
                             each weight a clean independent gradient
                             instead of entangling per-group neighbors via
                             the shared scale. The OUTER self.scales (=
                             amax(|w_orig|), preserved from build_student)
                             restores the original-weight magnitude: full
                             forward uses w_q · γ_inner · amax = w_q ·
                             mean(|w_orig|), the BitNet/Hestia recipe.

        w_norm  = latent / γ_inner
        w_quant = E_{probs}[k]  where probs_k ∝ exp(-‖w_norm-k‖²/T)
        eff     = w_quant · γ_inner   (returned in latent magnitude; outer
                                       self.scales=amax restores w_orig
                                       magnitude in QLinear.forward.)

        When smooth_alpha > 0, blends with the FP32 latent to suppress the
        initial quantization shock:
            eff = α·w + (1-α)·(w_quant·γ)

        Backward via _SmoothTernary: saves w_q and p_zero (2× weight size)
        instead of the full [O,G,gs,3] probs tensor. Falls through to STE
        when smooth_temp ≤ 1e-4.
        """
        temp = float(getattr(self, "smooth_temp", 0.0))
        if temp <= 1e-4:
            return self._full_qat_effective_weight()

        w = self.weight
        out_f, in_f = w.shape
        wb = w.view(out_f, self.n_groups, self.group_size)

        if getattr(self, "use_hestia_scale", False):
            # Detach γ from autograd (Hestia §3.1: "treat γ as a constant
            # during backpropagation"). Without this, the gradient flows
            # back through |w| and entangles every weight in a (row, group)
            # with its neighbors via the shared scale — each weight's
            # update direction then depends on its neighbors' positions.
            # Detaching gives each weight a clean independent gradient.
            s_eff = wb.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8).detach()
        else:
            s_eff = self.codepoint_c.unsqueeze(-1).to(wb.dtype).clamp_min(1e-8)

        if getattr(self, "smooth_at_floor", False):
            w_q = _HardTernaryWithSmoothGrad.apply(wb / s_eff, temp)
        else:
            w_q = _SmoothTernary.apply(wb / s_eff, temp)
        eff = (w_q * s_eff).view(out_f, in_f)

        smooth_alpha = float(getattr(self, "smooth_alpha", 0.0))
        if smooth_alpha > 0.0:
            eff = smooth_alpha * w + (1.0 - smooth_alpha) * eff
        return eff

    def _full_qat_effective_weight(self) -> torch.Tensor:
        """Single-stage QAT (qat_distill mode): every slot is ternarized.
        codepoint_c is a learnable nn.Parameter (one per row, per group).

        Forward at slot (r, c):
            sign(w)·c_{r,g}   if |w| > c_{r,g}/2
            0                 otherwise

        Backward (sign-cancellation-free routing):
          d/dw: 1 at nonzero-target slots, 0 at zero-target slots.
          d/dc: 1 at every nonzero-target slot (regardless of sign).
                The natural derivative is sign(w), which cancels across a
                (row, group) when positive and negative band weights are
                balanced — leaving c with ~0 gradient and stuck. We
                redirect via .detach() trickery so each non-zero band
                slot contributes +1 to c's gradient, summing to the
                count-of-non-zero-slots × upstream gradient. c can
                actually learn now.

        c-noise (set via c_noise_sigma attribute): during training, add
        independent per-group Gaussian noise to the threshold only (not
        the output magnitude). Weights at |w| ≈ c/2 stochastically flip
        between zero and nonzero across steps → gradient pushes them
        decisively away from the boundary. c's gradient remains clean
        (uses the unnoised c_b for output), so c still gets a clear
        magnitude signal.
        """
        w = self.weight
        out_f, in_f = w.shape
        wb = w.view(out_f, self.n_groups, self.group_size)
        c_b = self.codepoint_c.unsqueeze(-1).to(w.dtype)  # [out, ng, 1]
        with torch.no_grad():
            sign_wb = torch.sign(wb)
            c_noise_sigma = getattr(self, "c_noise_sigma", 0.0)
            if self.training and c_noise_sigma > 0.0:
                noise = torch.randn(out_f, self.n_groups, 1,
                                    dtype=c_b.dtype, device=c_b.device)
                c_b_thresh = (c_b + c_noise_sigma * noise).clamp_min(1e-8)
                is_nonzero = wb.abs() > c_b_thresh * 0.5
            else:
                is_nonzero = wb.abs() > c_b * 0.5
        # Forward = sign(w)·c (carried by the detached product).
        # (c_b - c_b.detach()) is 0 in forward, but back-prop sees identity
        # on c_b — and because c_b broadcasts across the gs dim, each
        # non-zero slot inside torch.where contributes +1 to c_b[r,g,0]'s
        # gradient. Same trick for w: (wb - wb.detach()) is the STE.
        sign_wb_c = (sign_wb * c_b).detach()
        nonzero = (sign_wb_c
                   + (c_b - c_b.detach())
                   + (wb - wb.detach()))
        eff = torch.where(is_nonzero, nonzero, torch.zeros_like(wb))
        return eff.view(out_f, in_f)

    def _qat_effective_weight(self) -> torch.Tensor:
        """Progressive QAT: per element, blend STE-ternary (at promoted
        slots) with the latent (elsewhere). The ternary target in latent
        space is sign(w)·c_{r,g} if |w| > c_{r,g}/2, else 0. STE makes
        the forward equal to that target while the backward sees identity,
        so the optimizer keeps moving the latent freely. As the latent
        crosses c_{r,g}/2 the ternary value flips — promotion is not a
        one-way trap.

        Lazily caches `_c_elem` (codepoint_c broadcast to weight shape,
        in weight dtype) on the module; invalidate by deleting attr.
        """
        w = self.weight
        c_elem = getattr(self, "_c_elem", None)
        if (c_elem is None
                or c_elem.shape != w.shape
                or c_elem.dtype != w.dtype
                or c_elem.device != w.device):
            out_f, in_f = w.shape
            self._c_elem = (self.codepoint_c.unsqueeze(-1)
                            .expand(out_f, self.n_groups, self.group_size)
                            .reshape(out_f, in_f)
                            .to(w.dtype)
                            .contiguous())
            c_elem = self._c_elem
        target = torch.where(w.abs() > c_elem * 0.5,
                             torch.sign(w) * c_elem,
                             torch.zeros_like(w))
        # STE: forward = target, backward = identity-on-w
        w_ste = w + (target - w).detach()
        return torch.where(self.qat_mask, w_ste, w)

    def _gsq_effective_weight(self) -> torch.Tensor:
        """GSQ ternary forward: c · GS_mask · GS_sign.

        mask_logits ≥ 0 → nonzero;  sign_logits ≥ 0 → +1, else −1.
        At tau≤1e-4 or eval: hard assignment (no Gumbel noise).
        Gradient flows via the sigmoid in gumbel_softmax_binary.
        """
        tau = float(getattr(self, "gsq_tau", 1.0))
        kappa = float(getattr(self, "gsq_kappa", 1.0))
        out_f, in_f = self.weight.shape
        mask_l = self.mask_logits   # [out, n_groups, group_size]
        sign_l = self.sign_logits
        c_b = self.codepoint_c.unsqueeze(-1).to(mask_l.dtype).clamp_min(1e-8)

        if tau <= 1e-4 or not self.training:
            mask = (mask_l >= 0).to(c_b.dtype)
            sign = torch.where(sign_l >= 0,
                               torch.ones_like(mask_l),
                               -torch.ones_like(mask_l))
        else:
            mask = gumbel_softmax_binary(mask_l, tau, kappa)
            sign = 2.0 * gumbel_softmax_binary(sign_l, tau, kappa) - 1.0

        return (c_b * mask * sign).view(out_f, in_f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Full-precision control path: use the raw weight directly — no
        # quantize, no STE, no per-group scales. CMuon trains this weight as
        # an ordinary FP matrix. Used to isolate whether the HRM recurrence /
        # fixpoint behaviour is a ternary artefact (--fp-weights in the
        # trainer sets m.fp_weights = True). Does not combine with int8 act.
        if getattr(self, "fp_weights", False):
            y = F.linear(x, self.weight.to(dtype=x.dtype))
            if self.bias is not None:
                y = y + self.bias
            return y
        # Optional BitNet-style per-token absmax int8 activation quantization.
        # Enabled when `self.int8_activations = True` is set by the trainer.
        # STE: forward sees quantized, backward sees identity.
        if getattr(self, "int8_activations", False):
            Qp = 127.0
            with torch.no_grad():
                # Per-token absmax. amax over the last dim only (the input
                # feature axis); broadcasts across the rest.
                absmax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
                s = Qp / absmax
                xq = (x * s).round().clamp_(-Qp, Qp) / s
            x = x + (xq - x).detach()
        # Five paths, picked by what's attached:
        #   mask_logits Parameter present               → GSQ PTQ
        #   codepoint_c is a Parameter + smooth_temp > 1e-4 → smooth QAT
        #   codepoint_c is a Parameter, no qat_mask → full-QAT / STE
        #   codepoint_c + qat_mask buffer (any True) → progressive QAT
        #   otherwise → standard soft/levels (original Bonsai flow)
        if getattr(self, "mask_logits", None) is not None:
            q_ste = self._gsq_effective_weight()
        else:
            c = getattr(self, "codepoint_c", None)
            if (c is not None and isinstance(c, nn.Parameter)
                    and getattr(self, "smooth_temp", 0.0) > 1e-4):
                q_ste = self._smooth_qat_effective_weight()
            elif c is not None and isinstance(c, nn.Parameter):
                q_ste = self._full_qat_effective_weight()
            elif (c is not None and hasattr(self, "qat_mask")
                    and self.qat_mask.any()):
                q_ste = self._qat_effective_weight()
            else:
                q_ste = self.quantized_weight()  # [out, in], in [-1, 1]
        # Apply per-(row, group) scales: expand scales [out, n_groups] to
        # [out, in] by broadcasting along the group's columns.
        q_blocks = q_ste.view(self.out_features, self.n_groups, self.group_size)
        scales_b = self.scales.unsqueeze(-1).to(q_blocks.dtype)
        w_scaled = (q_blocks * scales_b).view(self.out_features, self.in_features)
        # F.linear requires matching dtypes; outside autocast (or with
        # latent-dtype != activation dtype) we have to align. .to() is a
        # no-op when dtypes already match.
        y = F.linear(x, w_scaled.to(dtype=x.dtype))
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


def set_soft_mode(model: nn.Module, alpha: float = 0.0,
                  target_zero_frac: float | None = None) -> int:
    """Switch every QLinear to soft-ternary mode, seed alpha, and configure
    the c(w) classifier:
      * target_zero_frac=None → fixed ±1/3 boundary (nearest_ternary).
      * target_zero_frac in (0, 1) → per-(row, group) |w|-quantile cutoff
        so exactly that fraction of weights per group classify as zero.
    Cache invalidated so the next forward recomputes c(w) for the new
    settings. Returns count updated."""
    n = 0
    for m in model.modules():
        if isinstance(m, QLinear):
            m.mode = "soft"
            m.alpha = float(alpha)
            m.target_zero_frac = (float(target_zero_frac)
                                  if target_zero_frac is not None else None)
            m.invalidate_q_cache()
            n += 1
    return n


def set_c_noise(model: nn.Module, sigma: float) -> int:
    """Set c_noise_sigma on every QLinear (training-only threshold noise).
    sigma=0 disables it. Returns count updated."""
    n = 0
    for m in model.modules():
        if isinstance(m, QLinear):
            m.c_noise_sigma = float(sigma)
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


def triple_well_potential(w: torch.Tensor,
                          a: torch.Tensor | float = 1.0) -> torch.Tensor:
    """U_a(w) = U(w/a) where U(u) = u²·(u²-1)². Triple-well with minima of
    value 0 at {-a, 0, +a} and saddle points of height 4/27 ≈ 0.148 at
    ±a/√3. a=1 (default) recovers the original ±1 minima. `a` may be a
    scalar or a tensor broadcastable to `w` (e.g. a per-(row, group)
    tensor reshaped to [out, n_groups, 1] against w shaped [out, n_groups,
    group_size]).

    Saddle height is preserved (still 4/27 in U units) regardless of a;
    only the basin width scales with a. Smaller a → narrower 0-basin,
    fewer weights captured to 0.

    Use as a regularizer (`triple_well_loss`) when --soft-attractor=well:
    minimizing L_kl + α·U_a(W) over latents folds the reparam and the
    penalty into one C^∞ function. With per-group a calibrated to a
    target zero-frac at init, the 0-basin lines up with the natural |w|
    quantile of each group — same intent as the L2 form's quantile cutoff,
    just frozen at init instead of recomputed every step.

    Latents converge near {-a, 0, +a}; deploy-time `rescale_well_for_deploy`
    absorbs a into the per-group scales (math-preserving) so the rest of
    the pipeline sees the standard {-1, 0, +1} codebook.
    """
    if isinstance(a, float) and a == 1.0:
        u = w
    else:
        u = w / a
    u2 = u * u
    return u2 * (u2 - 1.0) * (u2 - 1.0)


def triple_well_loss(model: nn.Module) -> torch.Tensor:
    """Mean U_a(w) across all QLinear latents using each module's per-(row,
    group) `well_a` buffer (default 1.0 = canonical ±1 wells; populated by
    init_well_a for per-group calibration). Per-element mean, so the
    coefficient stays scale-invariant across model sizes.

    Sum is computed in fp32 even when latents are fp16: with millions of
    weights per layer and per-element U values up to ~4/27 inside basins
    (much larger if latents drift outside [-a, a]), the per-layer sum
    overflows fp16's 65504 ceiling and goes to inf. The .float() cast
    propagates a fp32 gradient back to the fp16 latent at backward time.
    """
    parts: list[torch.Tensor] = []
    n_total = 0
    device = None
    for m in model.modules():
        if isinstance(m, QLinear):
            w = m.weight
            device = w.device
            wb = w.float().view(w.shape[0], m.n_groups, m.group_size)
            ab = m.well_a.float().view(w.shape[0], m.n_groups, 1)
            parts.append(triple_well_potential(wb, ab).sum())
            n_total += w.numel()
    if not parts or n_total == 0:
        return torch.tensor(0.0, device=device or "cpu")
    return torch.stack(parts).sum() / n_total


@torch.no_grad()
def init_well_a(model: nn.Module, target_zero_frac: float) -> int:
    """For each QLinear, fill `well_a` per (row, group) with
    √3·quantile(|w|, target_zero_frac) so the saddle a/√3 lands at the
    target |w| quantile. Result: ~target_zero_frac of init weights per
    group sit inside the 0-basin naturally — the well analogue of
    nearest_ternary_quantile's per-group cutoff.

    Returns the count of QLinear modules updated. No-op (returns 0) when
    target_zero_frac is outside (0, 1)."""
    if not (0.0 < target_zero_frac < 1.0):
        return 0
    sqrt3 = math.sqrt(3.0)
    n = 0
    for m in model.modules():
        if isinstance(m, QLinear):
            w = m.weight.detach().float()
            wb = w.view(w.shape[0], m.n_groups, m.group_size).abs()
            cutoff = wb.quantile(float(target_zero_frac), dim=-1)
            m.well_a.copy_((sqrt3 * cutoff).clamp_min(1e-6))
            n += 1
    return n


@torch.no_grad()
def rescale_well_for_deploy(model: nn.Module) -> int:
    """Math-preserving rescale: per (row, group), `latent /= a` and
    `scales *= a` using each module's `well_a`. Forward output
    `latent · scales` is unchanged. Latents that drifted past ±a (e.g.
    sat near the ±1 clamp) get clipped back to ±1 after the rescale,
    which is a small information loss only for weights the well never
    fully captured. After this, `well_a` is reset to 1.0 and
    set_soft_mode + module_ternary classify against the ±1/3 (or
    quantile) boundary as usual; finalize.py and chat.py work unchanged.

    Returns count of QLinear modules actually rescaled (skips ones whose
    well_a is already 1.0). Invalidates the per-module q cache."""
    n = 0
    for m in model.modules():
        if not isinstance(m, QLinear):
            continue
        a = m.well_a.float()  # [out, n_groups]
        if torch.equal(a, torch.ones_like(a)):
            continue
        out_f = m.weight.shape[0]
        wb = m.weight.data.float().view(out_f, m.n_groups, m.group_size)
        wb = (wb / a.unsqueeze(-1)).clamp_(-1.0, 1.0)
        m.weight.data.copy_(wb.view(out_f, -1).to(m.weight.dtype))
        m.scales.data.mul_(a.to(m.scales.dtype))
        m.well_a.fill_(1.0)
        m.invalidate_q_cache()
        n += 1
    return n


def attractor_l2(model: nn.Module) -> torch.Tensor:
    """Mean ‖w - c(w)‖² across all QLinear latents (per-element mean, so
    the scale is independent of model size). Differentiable in w; gradient
    is 2/N · (w - c(w)), the basin force pulling each latent toward its
    nearest ternary attractor. Multiply by your λ(α) coefficient and add
    to the loss before backward.

    Reuses the per-module `_q_cache` (== c(w) in soft mode) when available
    so we don't recompute the nearest-ternary every call.

    Sum is computed in fp32 even when latents are fp16: with millions of
    weights per layer and per-element residuals up to 1, the per-layer sum
    overflows fp16's 65504 ceiling and goes to inf."""
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
                c = module_ternary(m)
            diff = (w - c).float()
            parts.append((diff * diff).sum())
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


def set_smooth_temp(model: nn.Module, T_global: float,
                    temp_scales: dict[str, float] | None = None,
                    at_floor: bool = False) -> None:
    """Set smooth_temp on every QLinear to T_global × per-layer scale.

    temp_scales maps module names (as in model.named_modules()) to
    per-layer multipliers from Hessian calibration. Missing names get 1.0.
    Call once per optimizer step before the forward pass.

    at_floor=True switches to _HardTernaryWithSmoothGrad: forward is exactly
    {-1,0,+1} (the deployed model), backward is still the smooth gradient at
    T_global (avoids dead zones). Use once the cosine anneal has completed.
    """
    for name, m in model.named_modules():
        if isinstance(m, QLinear):
            scale = (temp_scales.get(name, 1.0)
                     if temp_scales is not None else 1.0)
            m.smooth_temp = T_global * scale
            m.smooth_at_floor = at_floor


def set_smooth_alpha(model: nn.Module, alpha: float) -> None:
    """Set smooth_alpha on every QLinear.

    alpha=1 → FP32-equivalent forward (zero quantization shock at init).
    alpha=0 → pure smooth ternary (normal training).
    Intermediate values blend: α·w + (1-α)·c·E_T[k].

    Anneal from 1→0 over --blend-steps optimizer steps at the start of
    training to avoid the ~500-step quantization shock.
    """
    for m in model.modules():
        if isinstance(m, QLinear):
            m.smooth_alpha = float(alpha)


def init_gsq_logits(model: nn.Module, alpha: float = 3.0,
                    std: float = 0.01) -> int:
    """Attach mask_logits and sign_logits Parameters to every QLinear.

    Warm-starts from the nearest-ternary of the current weights
    (following GSQ paper Eq. 3–4, α=3 for ternary):

        signal_mask = +1 if t≠0, −1 if t=0
        signal_sign = +1 if t>0, −1 if t<0, 0 if t=0
        logit = std × (ε + α × signal)

    codepoint_c must already be attached (call attach_learnable_c first).
    Returns the number of QLinear modules updated.
    """
    n = 0
    for m in model.modules():
        if not (isinstance(m, QLinear) and hasattr(m, "codepoint_c")):
            continue
        out_f, in_f = m.weight.shape
        w = m.weight.detach().float()
        c_b = (m.codepoint_c.detach().float()
               .unsqueeze(-1)
               .expand(out_f, m.n_groups, m.group_size)
               .reshape(out_f, in_f))
        is_nonzero = w.abs() > c_b * 0.5

        mask_sig = torch.where(is_nonzero,
                               torch.ones_like(w), -torch.ones_like(w))
        sign_sig = torch.where(is_nonzero & (w > 0),
                               torch.ones_like(w),
                               torch.where(is_nonzero & (w < 0),
                                           -torch.ones_like(w),
                                           torch.zeros_like(w)))

        mask_l = std * (torch.randn_like(w) + alpha * mask_sig)
        sign_l = std * (torch.randn_like(w) + alpha * sign_sig)

        m.mask_logits = nn.Parameter(
            mask_l.view(out_f, m.n_groups, m.group_size).to(m.weight.dtype))
        m.sign_logits = nn.Parameter(
            sign_l.view(out_f, m.n_groups, m.group_size).to(m.weight.dtype))
        n += 1
    return n


def set_gsq_temp(model: nn.Module, tau: float, kappa: float) -> None:
    """Set gsq_tau and gsq_kappa on every QLinear with GSQ logits."""
    for m in model.modules():
        if isinstance(m, QLinear) and getattr(m, "mask_logits", None) is not None:
            m.gsq_tau = float(tau)
            m.gsq_kappa = float(kappa)


def snap_gsq_to_weight(model: nn.Module) -> int:
    """Hard-snap GSQ logits to {-1, 0, +1} and store in self.weight.

    After this call, mask_logits and sign_logits are removed. The model
    is back to the standard QLinear representation with ternary weight
    ∈ {-1, 0, +1} and calibrated codepoint_c. Run _deploy_fold (from
    finalize_smooth.py) to absorb codepoint_c into scales.
    Returns count snapped.
    """
    n = 0
    for m in model.modules():
        if not (isinstance(m, QLinear)
                and getattr(m, "mask_logits", None) is not None):
            continue
        out_f, in_f = m.weight.shape
        with torch.no_grad():
            mask = (m.mask_logits.detach() >= 0).float()
            sign = torch.where(m.sign_logits.detach() >= 0,
                               torch.ones_like(mask),
                               -torch.ones_like(mask))
            t = (mask * sign).view(out_f, in_f)
            m.weight.data.copy_(t.to(m.weight.dtype))
        del m.mask_logits
        del m.sign_logits
        n += 1
    return n


@torch.no_grad()
def compute_T_init(model: nn.Module, target_odds: float = 9.0) -> float:
    """Temperature where the median weight has target_odds:1 probability
    on its nearest codebook value vs second-nearest in normalized space.

    T_init = median(d2² - d1²) / log(target_odds)

    where d1² and d2² are the squared distances from each latent (normalized
    by codepoint_c) to its nearest and second-nearest codebook values
    {-1, 0, +1}. At T=T_init, the typical weight is already mostly committed
    but gradients still flow smoothly.
    """
    gaps: list[torch.Tensor] = []
    for m in model.modules():
        if not (isinstance(m, QLinear) and hasattr(m, "codepoint_c")):
            continue
        w = m.weight.detach().float()
        c = m.codepoint_c.detach().float()
        out_f = w.shape[0]
        wb = w.view(out_f, m.n_groups, m.group_size)
        c_b = c.unsqueeze(-1).clamp_min(1e-8)
        w_norm = wb / c_b
        codebook = w_norm.new_tensor([-1.0, 0.0, 1.0])
        dists_sq = (w_norm.unsqueeze(-1) - codebook).pow(2)  # [..., 3]
        sorted_d, _ = dists_sq.sort(dim=-1)
        gaps.append((sorted_d[..., 1] - sorted_d[..., 0]).flatten().cpu())
    if not gaps:
        return 1.0
    median_gap = float(torch.cat(gaps).median())
    T_init = median_gap / math.log(max(target_odds, 1.001))
    return max(T_init, 1e-4)


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
