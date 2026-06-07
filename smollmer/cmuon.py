"""Cautious Muon (C-Muon) for STE-quantized ternary weights.

Muon (Keller Jordan, 2024) projects the SGD momentum direction onto the
nearest "orthogonal" matrix via 5 Newton–Schulz iterations of the matrix
sign function. The intuition: for a matrix-valued parameter the right
metric on updates is the operator norm, not the elementwise L2 — so
each step should move the parameter in a direction whose singular values
are bounded (≈ 1), not in whatever per-coord direction has the largest
gradient.

Cautious mask (Liang et al. 2024, arXiv:2411.16085): after computing the
update direction, zero out coordinates where the direction's sign
disagrees with the current gradient's sign. The rationale: a saturated
momentum that current gradient contradicts is likely about to oscillate;
better to not apply that coord this step. The mask is mean-normalized so
the effective per-step displacement budget is preserved.

This module pairs the two for use on STE-quantized ternary latents:
QLinear stores a continuous latent in [-1, 1]; forward quantizes via
`quantize_levels(w, 3)` → {-1, 0, +1} with identity STE; gradient flows
back to the latent. C-Muon updates the latent matrices each step;
training-side `clamp_qlinear_weights` keeps them in box.

Only 2D parameters get the Muon path. Pass 1D / non-2D params to a
separate optimizer (Lion, etc.) — this is the standard Muon convention,
since orthogonalization is undefined for non-matrices.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def newton_schulz5(G: torch.Tensor, steps: int = 5,
                   eps: float = 1e-7) -> torch.Tensor:
    """Newton–Schulz iteration approximating G·(G^T G)^{-1/2}.

    Returns a matrix U of the same shape as G whose left singular vectors
    match G and whose singular values are ≈ 1. Quintic coefficients
    (3.4445, -4.7750, 2.0315) are the standard Jordan / Bernstein–Newman
    tuning that converges in 5 steps with reasonable error.

    Operates in fp32 throughout for numerical stability. Always reads the
    input on its native device; result is on the same device, in fp32.

    Transposes the matrix to be "wide" (rows ≤ cols) before iterating
    so each `X @ X.T` is the cheaper square. Transposes back at the end.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + eps)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class CMuon(torch.optim.Optimizer):
    """Cautious Muon for 2D parameters.

    Per-step:
      m ← β·m + (1-β)·g
      U ← newton_schulz5(m, ns_steps)
      if cautious:
          mask ← 1 where sign(U) == sign(g), 0 otherwise
          mask ← mask / mean(mask).clamp_min(1e-3)   # preserve effective LR
          U ← U * mask
      w ← w - lr · U

    Cautious-mask convention follows CautiousAdamW: keep coords where the
    update direction (-U on subtract) agrees with the loss-reducing direction
    (-g). I.e. agreement iff sign(U) == sign(g) ↔ U·g > 0.

    Only 2D params are accepted; pass non-2D params to a separate optimizer
    (Lion for embeddings, etc.). 1D in same group raises in __init__.

    State per param:
      m   : fp32 momentum buffer, shape == p.shape
    """

    def __init__(self, params, lr: float = 1e-3, beta: float = 0.95,
                 ns_steps: int = 5, cautious: bool = True,
                 state_dtype: torch.dtype = torch.float32) -> None:
        if not 0.0 < lr:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be >= 1, got {ns_steps}")
        defaults = dict(lr=float(lr), beta=float(beta),
                        ns_steps=int(ns_steps), cautious=bool(cautious))
        super().__init__(params, defaults)
        self.state_dtype = state_dtype
        # Sanity-check shapes once at construction so the per-step loop is
        # branch-free.
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2 or min(p.shape) < 2:
                    raise ValueError(
                        "CMuon expects 2D params with both dims >= 2; got "
                        f"shape {tuple(p.shape)}. Send 1D/embedding/norm "
                        "params to a separate Lion/AdamW optimizer.")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        n_zeroed = 0
        n_total = 0
        for group in self.param_groups:
            lr = float(group["lr"])
            beta = float(group["beta"])
            ns_steps = int(group["ns_steps"])
            cautious = bool(group["cautious"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p, dtype=self.state_dtype)
                m = state["m"]
                # EMA in m's dtype. For fp16 state this accrues naive
                # rounding error per step but Muon's NS5 normalizes the
                # update direction, so per-coord precision drift on m
                # doesn't propagate to the same per-step magnitude error
                # that AdamW would suffer. If fp16 m underflows in late
                # training, we can revisit with stochastic rounding.
                m.mul_(beta).add_(g.to(m.dtype), alpha=1.0 - beta)
                # NS5 always in fp32 for orthogonalization precision.
                m_fp32 = m if m.dtype == torch.float32 else m.float()
                U = newton_schulz5(m_fp32, ns_steps)
                if cautious:
                    # sign(U) == sign(g)  iff  U * g > 0
                    agree = (U * g > 0).to(U.dtype)
                    mean_agree = agree.mean().clamp_min(1e-3)
                    mask = agree / mean_agree
                    n_zeroed += int((agree == 0).sum().item())
                    n_total += agree.numel()
                    U = U * mask
                p.add_(U.to(p.dtype), alpha=-lr)
        # Return a meta-metric for caller telemetry (fraction of coords the
        # cautious mask zeroed this step). 0 if cautious is off.
        return (loss if loss is not None else 0.0,
                n_zeroed / max(1, n_total))
