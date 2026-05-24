"""Distillation curriculum loop.

For each (levels, n_steps) stage in the curriculum:
  1. Set every QLinear.levels to `levels`.
  2. Train n_steps with KL-distillation against cached teacher top-K + rest_mass.
  3. Save a checkpoint with metadata.

Loss (per position):
  KL(p_teacher || p_student)
    = sum_i p_t[i] * log(p_t[i] / p_s[i])           over top-K i
    + p_rest_t * log(p_rest_t / p_rest_s)
where p_rest_s = 1 - sum(p_s[topk_idx]).  Drop teacher constants.
"""
from __future__ import annotations

import argparse
import math
import random
import signal
import sys
from datetime import datetime
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .build_student import load_student
from .qlinear import (QLinear, attractor_l2, clamp_qlinear_weights,
                      init_well_a, module_ternary, module_ternary_fixed,
                      nearest_ternary, quantize_levels,
                      rescale_well_for_deploy, set_levels, set_soft_alpha,
                      set_soft_mode, triple_well_loss)
from .teacher_floor import load_or_compute as load_teacher_floor


_INTERRUPT = {"flag": False}


def _install_sigint_handler() -> None:
    def handler(signum, frame):
        # Flag is set BEFORE any I/O so a broken-pipe print can't lose the
        # save signal — happens when stdout is piped to a tee that died of
        # the same SIGINT (same foreground process group). All prints in
        # this handler are wrapped to make signal delivery best-effort.
        if _INTERRUPT["flag"]:
            try:
                print("\n[!!] second SIGINT — hard exit, no save", flush=True)
            except Exception:
                pass
            sys.exit(130)
        _INTERRUPT["flag"] = True
        try:
            print("\n[!] SIGINT — finishing current step, then saving resume "
                  "state. Press Ctrl-C again to hard-exit.", flush=True)
        except Exception:
            pass
    signal.signal(signal.SIGINT, handler)


def save_resume(path: Path, model, opt, next_step: int,
                best_snapshot: dict[str, torch.Tensor] | None,
                ctrl_state: dict | None,
                run_name: str | None,
                samples_consumed: int = 0,
                soft_state: dict | None = None) -> None:
    """Atomic write of the resume snapshot. Used both by SIGINT and by the
    periodic auto-checkpoint — same file (`interrupted.pt`), latest wins.
    The atomic rename via `.tmp` -> rename means a crash mid-write leaves
    the previous good snapshot intact."""
    payload = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "opt": opt.state_dict(),
        "next_step": int(next_step),
        "samples_consumed": int(samples_consumed),
        "best_snapshot": best_snapshot,
        "ctrl_state": ctrl_state,
        "run_name": run_name,
        "soft_state": soft_state,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(tmp))
    tmp.replace(path)


@torch.no_grad()
def snapshot_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Clone all model params to CPU. Used to remember the best-so-far weights
    within a stage so we can roll back if Lion thrashes a near-optimal model."""
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


@torch.no_grad()
def restore_from_snapshot(model: torch.nn.Module, snap: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(snap, strict=False)


class BestEmaTracker:
    """Track loss EMA and remember the best-so-far step. Used purely for
    snapshotting the lowest-loss model state — there's no exit gate
    coupled to plateau detection (the soft-stage exit is α-saturation +
    tail; the safety cap is --soft-steps).

    `ema_warmup` skips best-EMA tracking for the first N steps so a low
    early-warmup loss (often U-shaped past the LR-warmup window) doesn't
    lock in as forever-best. EMA still updates during warmup; only the
    `best` latch is frozen until step >= warmup.
    """

    def __init__(self, ema_alpha: float = 0.05, rel_threshold: float = 1e-3,
                 ema_warmup: int = 0) -> None:
        self.ema_alpha = ema_alpha
        self.rel_threshold = rel_threshold
        self.ema_warmup = ema_warmup
        self.ema: float | None = None
        self.best_ema: float = float("inf")
        self.best_step: int = 0

    def update(self, step: int, loss: float) -> bool:
        """Update EMA. Returns True iff this step set a new best (i.e. the
        caller should snapshot the model)."""
        self.ema = loss if self.ema is None else \
            (1.0 - self.ema_alpha) * self.ema + self.ema_alpha * loss
        if step < self.ema_warmup:
            return False
        if self.ema < self.best_ema * (1.0 - self.rel_threshold):
            self.best_ema = self.ema
            self.best_step = step
            return True
        return False

    def state_dict(self) -> dict:
        return {
            "ema": self.ema,
            "best_ema": self.best_ema,
            "best_step": self.best_step,
        }

    def load_state_dict(self, state: dict) -> bool:
        """Restore EMA tracking. Returns True if a stale (within-warmup) best
        was discarded; the caller should also drop its best_snapshot in that
        case so a poisoned warmup-era model isn't restored at stage end."""
        self.ema = state.get("ema")
        bs = int(state.get("best_step", 0))
        if bs < self.ema_warmup:
            self.best_ema = float("inf")
            self.best_step = 0
            return True
        self.best_ema = state.get("best_ema", float("inf"))
        self.best_step = bs
        return False


# Legacy alias so callers in this file keep working through the rename.
PlateauController = BestEmaTracker


class AlphaSchedule:
    """Adaptive α schedule: bump α after `patience` consecutive stable steps.

    Stability check: a fast loss-EMA (window ~1/ema_alpha steps) hasn't
    outrun a slow loss-EMA (window slow_ratio× longer) by more than
    `tolerance`. This maps directly to "loss didn't go up for X steps":
    if loss is flat, fast ≈ slow, diff ≤ tolerance, steady_count grows.
    If loss is rising, fast leads slow, diff > tolerance, steady_count
    resets. If loss is falling, fast < slow, diff < 0, steady_count grows
    (we welcome falling loss).

    Why not "EMA vs latched-at-last-bump baseline": that latches forever
    if loss can never return below the baseline. The slow-EMA baseline
    drifts with the trend, so a model that converges to a *new* stable
    plateau (higher than where the last bump happened) still counts as
    stable — its bumps continue.

    Cumulative-max semantics: α only ever increases. Failure mode is a
    visible stall (`steady_count` wedged at 0, no new bumps in TB) rather
    than silent overshoot. The --soft-steps flag is a safety cap.

    `alpha_max < 1` keeps the (1-α) slope on KL gradient flowing to the
    latent — at α=1 the only force is the attractor penalty. Final hard
    rounding to {-1,0,+1} happens in finalize.py.
    """

    def __init__(self, alpha_max: float = 0.95,
                 patience: int = 200,
                 bump: float = 0.02,
                 tolerance: float = 0.02,
                 ema_alpha: float = 0.05,
                 slow_ratio: float = 10.0,
                 alpha_init: float = 0.0) -> None:
        self.alpha_max = float(alpha_max)
        self.patience = int(patience)
        self.bump = float(bump)
        self.tolerance = float(tolerance)
        self.ema_alpha = float(ema_alpha)
        self.slow_ratio = float(slow_ratio)
        self.slow_ema_alpha = self.ema_alpha / max(1.0, self.slow_ratio)
        self.alpha = float(alpha_init)
        self.ema: float | None = None       # fast loss-EMA
        self.slow_ema: float | None = None  # slow loss-EMA (drifting baseline)
        self.steady_count = 0
        self.bumps = 0

    def step(self, step_loss: float) -> tuple[float, int, bool]:
        """Update with the latest step loss. Returns (alpha, steady_count,
        bumped). `bumped` is True iff α was incremented this step.

        Non-finite losses are dropped (steady_count resets) so a single
        NaN can't poison the EMAs."""
        x = float(step_loss)
        if not (x == x and x != float("inf") and x != float("-inf")):
            self.steady_count = 0
            return self.alpha, self.steady_count, False
        if self.ema is None:
            self.ema = x
            self.slow_ema = x
        else:
            self.ema = (1.0 - self.ema_alpha) * self.ema + self.ema_alpha * x
            self.slow_ema = ((1.0 - self.slow_ema_alpha) * self.slow_ema
                             + self.slow_ema_alpha * x)
        if self.ema <= self.slow_ema + self.tolerance:
            self.steady_count += 1
        else:
            self.steady_count = 0
        bumped = False
        if (self.steady_count >= self.patience
                and self.alpha < self.alpha_max):
            self.alpha = min(self.alpha + self.bump, self.alpha_max)
            self.steady_count = 0
            self.bumps += 1
            bumped = True
        return self.alpha, self.steady_count, bumped

    def state_dict(self) -> dict:
        return {
            "alpha_max": self.alpha_max,
            "patience": self.patience,
            "bump": self.bump,
            "tolerance": self.tolerance,
            "ema_alpha": self.ema_alpha,
            "slow_ratio": self.slow_ratio,
            "slow_ema_alpha": self.slow_ema_alpha,
            "alpha": self.alpha,
            "ema": self.ema,
            "slow_ema": self.slow_ema,
            "steady_count": self.steady_count,
            "bumps": self.bumps,
        }

    def load_state_dict(self, state: dict) -> None:
        self.alpha_max = float(state.get("alpha_max", self.alpha_max))
        self.patience = int(state.get("patience", self.patience))
        self.bump = float(state.get("bump", self.bump))
        self.tolerance = float(state.get("tolerance", self.tolerance))
        self.ema_alpha = float(state.get("ema_alpha", self.ema_alpha))
        self.slow_ratio = float(state.get("slow_ratio", self.slow_ratio))
        self.slow_ema_alpha = float(state.get(
            "slow_ema_alpha", self.ema_alpha / max(1.0, self.slow_ratio)))
        self.alpha = float(state.get("alpha", self.alpha))
        self.ema = state.get("ema")
        # Migrate from legacy state where the baseline was latched at bump
        # time: seed slow_ema from "baseline" if "slow_ema" is absent.
        self.slow_ema = state.get("slow_ema", state.get("baseline"))
        self.steady_count = int(state.get("steady_count", 0))
        self.bumps = int(state.get("bumps", 0))


class Lion32(torch.optim.Optimizer):
    """Lion (Chen et al. 2023, arXiv:2302.06675) with fp32 momentum buffer
    regardless of parameter dtype.

    Why fp32 state on fp16 params: sign() is exact, but the EMA
    `beta * m + (1-beta) * g` accumulates rounding error in fp16 once
    grads drop below the buffer's ULP. fp32 m, fp16 p costs an extra
    cast per step but avoids silent stall on small late-stage grads.
    """

    def __init__(self, params, lr: float = 3e-4,
                 betas: tuple[float, float] = (0.9, 0.99),
                 weight_decay: float = 0.0) -> None:
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                m = state["exp_avg"]
                if wd != 0:
                    p.mul_(1 - lr * wd)
                # Update direction: sign(beta1*m + (1-beta1)*g), in fp32.
                upd = m.mul(beta1).add_(grad, alpha=1 - beta1).sign_()
                p.add_(upd.to(p.dtype), alpha=-lr)
                # EMA update: beta2*m + (1-beta2)*g.
                m.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


class AdamW32(torch.optim.Optimizer):
    """AdamW with fp32 state regardless of parameter dtype.

    Necessary on fp16 latents: `exp_avg_sq = (1-b2)*g^2` puts the squared
    grad through a near-1e-3 multiplier, so for grads < ~sqrt(fp16_min) ≈ 8e-3
    the buffer underflows to 0 and the divisor sqrt(v_hat) collapses to eps —
    silent stall once the model nears a minimum (where ternary L=3 spends
    most of its time). fp32 v fixes it.
    """

    def __init__(self, params, lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                state["step"] += 1
                step = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]
                if wd != 0:
                    p.mul_(1 - lr * wd)
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bc1 = 1 - beta1 ** step
                bc2 = 1 - beta2 ** step
                denom = (v / bc2).sqrt().add_(eps)
                upd = (m / bc1) / denom
                p.add_(upd.to(p.dtype), alpha=-lr)
        return loss


class CautiousAdamW(torch.optim.Optimizer):
    """AdamW with the one-line "cautious" modification from
    Liang et al. 2024 (arXiv:2411.16085). Each step, mask out coordinates
    where the momentum sign disagrees with the current gradient sign, then
    rescale the mask by its mean to preserve the effective LR.

    This is the optimizer used by Deepgrove's Bonsai (paper §2.1: lr=0.01,
    cosine + linear warmup) -- the recipe PrismML's Ternary Bonsai is built on.
    Sample-efficiency gain reported: ~1.47x over plain AdamW.

    State is allocated in fp32 regardless of param dtype (see AdamW32 docstring
    for the underflow argument). Usage: AdamW LRs are much higher than Lion's;
    Bonsai used 1e-2, sweep 1e-3..1e-2 for QAT distillation.
    """

    def __init__(self, params, lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                state["step"] += 1
                step = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]
                if wd != 0:
                    p.mul_(1 - lr * wd)
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bc1 = 1 - beta1 ** step
                bc2 = 1 - beta2 ** step
                denom = (v / bc2).sqrt().add_(eps)
                # Cautious mask: 1 where momentum and grad agree, 0 otherwise.
                # Mean-normalize so total update magnitude is preserved (per paper).
                mask = (m * grad > 0).to(m.dtype)
                mask.div_(mask.mean().clamp_min(1e-3))
                upd = (m * mask) / denom / bc1
                p.add_(upd.to(p.dtype), alpha=-lr)
        return loss


class PRM32(torch.optim.Optimizer):
    """Population Risk Minimization (Litman & Guo 2026,
    github.com/elonlit/PopRiskMinimization).

    AdamW + one extra fp32 state tensor s (Welford EMA of centered
    minibatch gradient variance) + one extra step: multiply the Adam
    direction by a per-parameter SNR mask

        q = m_hat^2 / (m_hat^2 + lam_pop * alpha * s_hat + eps)

    that shrinks parameters whose batch-mean gradient is below the
    leave-one-out noise estimate. With alpha=1 (batch boundary, online
    minibatches) and lam_pop=1 the mask is exactly 1/2 on the LOO
    threshold m_hat^2 = s_hat; above the threshold q rises toward 1,
    below it falls toward 0.

    The Adam direction is then passed through the Cautious mask from
    Liang et al. 2024 (cf. CautiousAdamW): coords where momentum and
    current grad disagree are zeroed and the mask is mean-normalized to
    preserve the effective LR. SNR shrinkage alone left big overshoot
    on QAT runs, this kills the sign-flip component of it.

    State is fp32 for the same underflow reason as AdamW32: (1-rho)*g^2
    sends squared grads through a ~1e-2 multiplier, which kills v on
    fp16 latents near a minimum.
    """

    def __init__(self, params, lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999),
                 rho: float = 0.99, eps: float = 1e-8,
                 weight_decay: float = 0.01,
                 softness: float = 1.0) -> None:
        defaults = dict(lr=lr, betas=betas, rho=rho, eps=eps,
                        weight_decay=weight_decay, softness=softness)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            rho = group["rho"]
            eps = group["eps"]
            wd = group["weight_decay"]
            lam = group["softness"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.to(torch.float32)
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    state["grad_var"] = torch.zeros_like(p, dtype=torch.float32)
                state["step"] += 1
                step = state["step"]
                m, v, s = state["exp_avg"], state["exp_avg_sq"], state["grad_var"]
                if wd != 0:
                    p.mul_(1 - lr * wd)
                # Welford centering: variance update uses pre-update m so
                # s_t is an unbiased estimator of Sigma_B / (b-1) with no
                # beta1 bias from the post-update form.
                diff = grad - m
                s.mul_(rho).addcmul_(diff, diff, value=1 - rho)
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bc1 = 1 - beta1 ** step
                bc2 = 1 - beta2 ** step
                bc_rho = 1 - rho ** step
                m_hat = m / bc1
                v_hat = v / bc2
                s_hat = s / bc_rho
                m_sq = m_hat * m_hat
                # alpha = 1 (batch boundary, online minibatches).
                q = m_sq / (m_sq + lam * s_hat + eps)
                # Cautious mask (Liang et al. 2024): zero coords where the
                # momentum direction disagrees with the current grad, then
                # mean-normalize to preserve the effective LR. Curbs the
                # post-shrink overshoot we saw on PRM runs.
                mask = (m * grad > 0).to(m.dtype)
                mask.div_(mask.mean().clamp_min(1e-3))
                upd = q * mask * m_hat / v_hat.sqrt().add_(eps)
                p.add_(upd.to(p.dtype), alpha=-lr)
        return loss


@torch.no_grad()
def quantized_codes(m: QLinear) -> torch.Tensor:
    """Integer code per weight, used for bin-occupancy + step-to-step
    flip-rate metrics. In `soft` mode we use the same c(w) the attractor
    uses (quantile cutoff if m.target_zero_frac is set, else fixed ±1/3),
    so flip-rate tracks churn against the deployable alphabet."""
    if m.mode == "soft":
        return module_ternary(m).to(torch.int8)
    half = (m.levels - 1) // 2
    q = quantize_levels(m.weight, m.levels)
    return (q * half).round().to(torch.int8)


@torch.no_grad()
def collect_qlinear_metrics(
    model: torch.nn.Module,
    prev_codes: dict[str, torch.Tensor],
    prev_codes_fixed: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, float],
           dict[str, torch.Tensor],
           dict[str, torch.Tensor]]:
    """Aggregate ternary-QAT health metrics across every QLinear module.

    Returns (metrics, new_codes, new_codes_fixed). The two code dicts are
    fed back as `prev_codes` and `prev_codes_fixed` on the next call:
      * `flip_rate`: against the moving (quantile or ±1/3) classifier —
        same alphabet the L2 forward / soft-stage bin counts use. In
        well mode this is largely diagnostic since the forward isn't
        contracted; under cycling it goes to ~0 because the cutoff
        rescales with the |w| distribution.
      * `flip_rate_fixed`: against a frozen per-(row, group) saddle at
        well_a/√3. Catches actual basin migrations that the moving
        classifier misses. Equivalent to the deploy-form classifier in
        pre-rescale coordinates.
    """
    new_codes: dict[str, torch.Tensor] = {}
    new_codes_fixed: dict[str, torch.Tensor] = {}
    total = 0
    n_zero = 0
    n_extreme = 0
    n_flip = 0
    n_compared = 0
    n_flip_fixed = 0
    n_compared_fixed = 0
    scale_max = 0.0
    sum_scale_mean = 0.0
    n_layers = 0
    prev_codes_fixed = prev_codes_fixed or {}
    for name, m in model.named_modules():
        if not isinstance(m, QLinear):
            continue
        # codes in {-1, 0, +1} for soft mode; in [-half, half] for levels mode.
        half = 1 if m.mode == "soft" else (m.levels - 1) // 2
        codes = quantized_codes(m)
        total += codes.numel()
        n_zero += int((codes == 0).sum())
        n_extreme += int(((codes == half) | (codes == -half)).sum())
        prev = prev_codes.get(name)
        if prev is not None and prev.shape == codes.shape:
            n_flip += int((codes != prev).sum())
            n_compared += codes.numel()
        new_codes[name] = codes
        if m.mode == "soft":
            codes_fixed = module_ternary_fixed(m).to(torch.int8)
            prev_f = prev_codes_fixed.get(name)
            if prev_f is not None and prev_f.shape == codes_fixed.shape:
                n_flip_fixed += int((codes_fixed != prev_f).sum())
                n_compared_fixed += codes_fixed.numel()
            new_codes_fixed[name] = codes_fixed
        sm = m.scales.detach().abs()
        scale_max = max(scale_max, sm.max().item())
        sum_scale_mean += sm.mean().item()
        n_layers += 1
    metrics = {
        "bins/frac_zero": n_zero / max(1, total),
        "bins/frac_extreme": n_extreme / max(1, total),
        "scales/max": scale_max,
        "scales/mean": sum_scale_mean / max(1, n_layers),
    }
    if n_compared:
        metrics["weights/flip_rate"] = n_flip / n_compared
    if n_compared_fixed:
        metrics["weights/flip_rate_fixed"] = n_flip_fixed / n_compared_fixed
    return metrics, new_codes, new_codes_fixed


@torch.no_grad()
def collect_soft_metrics(
    model: torch.nn.Module,
    near_boundary_band: float = 0.05,
) -> tuple[dict[str, float], list[tuple[str, torch.Tensor]]]:
    """Soft-stage-specific health metrics + one flat weight sample for TB
    histograms. Aggregates across every QLinear in `model`.

    Scalars:
      `soft/bins/frac_neg|zero|pos`     — full ternary occupancy split
      `soft/latent/l1_to_attractor`     — mean |w - c(w)|; how close latents
                                          sit to their nearest attractor
      `soft/latent/saturation_frac`     — frac of latents at the clamp wall
                                          (|w| ≥ 0.99); high value = clamp
                                          is doing real work, suggests the
                                          attractor force may be overshooting
      `soft/latent/near_boundary_frac`  — frac of latents within ±band of
                                          a ±1/3 decision boundary; these
                                          are the latents most at risk of
                                          flipping which attractor they
                                          serve. (For triple-well mode the
                                          natural saddle is ±1/√3 instead;
                                          this metric still tracks the
                                          deployment-relevant ±1/3 cutoff.)

    Histogram sample: the FIRST QLinear's flattened latent weight, on CPU.
    Caller decides cadence — TB handles binning. One module per call keeps
    the writer's storage bounded; the first module is a stable choice
    across runs (typically `model.layers.0.self_attn.q_proj`).
    """
    total = 0
    n_neg = n_zero = n_pos = 0
    sum_l1 = 0.0
    sum_sat = 0
    sum_near = 0
    sample: list[tuple[str, torch.Tensor]] = []
    for name, m in model.named_modules():
        if not isinstance(m, QLinear):
            continue
        w = m.weight.detach()
        c = module_ternary(m)
        total += w.numel()
        n_neg += int((c == -1).sum())
        n_zero += int((c == 0).sum())
        n_pos += int((c == 1).sum())
        sum_l1 += (w - c).abs().sum().item()
        sum_sat += int((w.abs() >= 0.99).sum())
        d_b = torch.minimum((w - (1.0 / 3.0)).abs(),
                            (w + (1.0 / 3.0)).abs())
        sum_near += int((d_b < near_boundary_band).sum())
        if not sample:
            sample.append((name, w.flatten().to("cpu")))
    if total == 0:
        return {}, []
    return {
        "soft/bins/frac_neg": n_neg / total,
        "soft/bins/frac_zero": n_zero / total,
        "soft/bins/frac_pos": n_pos / total,
        "soft/latent/l1_to_attractor": sum_l1 / total,
        "soft/latent/saturation_frac": sum_sat / total,
        "soft/latent/near_boundary_frac": sum_near / total,
    }, sample


@torch.no_grad()
def first_qlinear_forward_sample(
    model: torch.nn.Module,
) -> tuple[str, torch.Tensor] | None:
    """Return (name, flat_cpu_tensor) for the first QLinear's *forward*
    weight — i.e. quantized_weight() with the attractor applied. In l2 soft
    mode this is c(w) + (1-α)(w - c(w)); in well soft mode α stays 0 so it's
    the raw latent; in levels mode it's the STE-quantized weight at the
    current level. Used for the soft-stage TB histogram so we can watch the
    three-peak structure emerge as α grows."""
    for name, m in model.named_modules():
        if isinstance(m, QLinear):
            qw = m.quantized_weight()
            return name, qw.detach().flatten().to("cpu")
    return None


@torch.no_grad()
def grad_l2_norm(model: torch.nn.Module) -> float:
    """Global L2 norm over all .grad tensors. Equivalent to what
    clip_grad_norm_ measures, without the clipping side effect — used to
    snapshot the KL-only gradient norm before the attractor penalty
    backward adds to it."""
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += float(p.grad.detach().pow(2).sum())
    return total_sq ** 0.5


@torch.no_grad()
def embed_drift_l2(model: torch.nn.Module,
                   embed_init: torch.Tensor | None,
                   embed_stage_init: torch.Tensor | None = None,
                   ) -> tuple[float | None, float | None]:
    """L2 distance of the input-embedding from two reference points:

    * `embed_init` — the FP teacher's initial embedding (cumulative drift).
      A runaway here is a likely culprit when ternary distillation diverges,
      since embeddings aren't quantized or clamped.
    * `embed_stage_init` — the embedding at the start of the current stage
      (per-stage drift). Useful for attributing drift to a particular
      curriculum stage. May be the current weights right after a mid-stage
      resume, in which case the per-stage value will underreport for that
      stage; subsequent clean stage entries are accurate.

    Returns (cumulative, stage). Either entry can be None if its reference
    is unset.
    """
    embed = model.get_input_embeddings()
    if embed is None:
        return None, None
    w = embed.weight.detach()
    cum = (w - embed_init).norm().item() if embed_init is not None else None
    stg = (w - embed_stage_init).norm().item() if embed_stage_init is not None else None
    return cum, stg


class ShardedDataset(IterableDataset):
    def __init__(self, shard_dir: Path, seed: int = 0,
                 start_skip: int = 0) -> None:
        self.paths = sorted(Path(shard_dir).glob("shard_*.safetensors"))
        if not self.paths:
            raise FileNotFoundError(f"no shard_*.safetensors under {shard_dir}")
        self.seed = seed
        # Samples to fast-forward through on the next __iter__. Since the
        # rng is seeded deterministically (seed + wid*7919), walking the
        # inner loops without yielding reproduces the exact pre-skip
        # ordering. DataLoader pickles `self` to the worker, so this
        # survives the fork.
        self.start_skip = int(start_skip)

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        wid = worker.id if worker else 0
        nworkers = worker.num_workers if worker else 1
        rng = random.Random(self.seed + wid * 7919)
        my_paths = self.paths[wid::nworkers] or self.paths
        # DataLoader round-robins workers, so each worker emits ~1/nworkers
        # of the global stream. Split the skip across workers using the
        # round-robin remainder so total skipped == start_skip.
        my_skip = (self.start_skip // nworkers
                   + (1 if wid < (self.start_skip % nworkers) else 0))
        produced = 0
        while True:
            order = list(my_paths)
            rng.shuffle(order)
            for p in order:
                shard = load_file(str(p))
                S = shard["tokens"].shape[0]
                idx_order = list(range(S))
                rng.shuffle(idx_order)
                for i in idx_order:
                    if produced < my_skip:
                        produced += 1
                        continue
                    produced += 1
                    yield {
                        "tokens": shard["tokens"][i].long(),
                        "topk_idx": shard["topk_idx"][i].long(),
                        "topk_prob": shard["topk_prob"][i].float(),
                        "rest_mass": shard["rest_mass"][i].float(),
                    }


def kl_with_rest(student_logits: torch.Tensor,
                 topk_idx: torch.Tensor,
                 topk_prob: torch.Tensor,
                 rest_mass: torch.Tensor,
                 eps: float = 1e-7) -> torch.Tensor:
    # Avoid materializing the full [B,T,V] log_softmax tensor:
    # log_p[i] = logits[i] - logsumexp(logits).  We only need the K
    # gathered positions plus the lse scalar per (B,T).
    lse = torch.logsumexp(student_logits, dim=-1, keepdim=True)        # [B,T,1]
    selected = torch.gather(student_logits, -1, topk_idx)              # [B,T,K]
    log_p_topk = (selected - lse).float()                              # [B,T,K]
    p_topk = log_p_topk.exp()
    p_rest = (1.0 - p_topk.sum(dim=-1)).clamp_min(eps)                 # [B,T]
    log_p_rest = p_rest.log()
    loss_topk = -(topk_prob * log_p_topk).sum(dim=-1)                  # [B,T]
    loss_rest = -(rest_mass * log_p_rest)                              # [B,T]
    return (loss_topk + loss_rest).mean()


def lr_at(step: int, total: int, base_lr: float, warmup: int,
          floor: float = 0.1) -> float:
    """Linear warmup → cosine decay to `floor × base_lr`. floor=0.1 is the
    classic Bonsai recipe; bump it up (e.g. 0.5) when the late-stage flip-rate
    cliff is locking the codebook into a suboptimal configuration. floor=1.0
    disables the cosine entirely (flat LR after warmup)."""
    if warmup and step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    import math
    progress = (step - warmup) / max(1, total - warmup)
    progress = max(0.0, min(1.0, progress))
    return base_lr * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def save_checkpoint(model, path: Path, model_id: str,
                    group_size: int,
                    alpha: float | None = None,
                    target_zero_frac: float | None = None) -> None:
    """Soft-stage safetensors checkpoint. `levels=3` and `mode=soft` are
    always recorded (this is the only training mode now); finalize.py and
    chat.py read `mode`, `alpha`, and `target_zero_frac` to apply the
    matching ternary classifier at deployment."""
    # SmolLM2 ties lm_head.weight to embed_tokens.weight; safetensors
    # refuses shared storage, so clone the second occurrence per data_ptr.
    sd: dict[str, torch.Tensor] = {}
    seen: dict[int, str] = {}
    for k, v in model.state_dict().items():
        t = v.detach().cpu().contiguous()
        ptr = t.data_ptr()
        if ptr in seen:
            t = t.clone()
        else:
            seen[ptr] = k
        sd[k] = t
    metadata = {
        "levels": "3",
        "mode": "soft",
        "model_id": model_id,
        "group_size": str(group_size),
    }
    if alpha is not None:
        metadata["alpha"] = f"{float(alpha):.6f}"
    if target_zero_frac is not None:
        metadata["target_zero_frac"] = f"{float(target_zero_frac):.6f}"
    save_file(sd, str(path), metadata=metadata)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None,
                    help="Optional safetensors checkpoint to warm-start the "
                         "model from. Distinct from interrupted.pt — "
                         "interrupted.pt (auto-checkpointed every "
                         "--soft-checkpoint-every steps and on SIGINT) is "
                         "loaded automatically when present.")
    ap.add_argument("--resume-from-best", action="store_true",
                    help="On resume, restore model weights from the "
                         "best_snapshot stored in interrupted.pt (the lowest "
                         "ctrl-EMA point so far) instead of the latest "
                         "weights. Optimizer momentum is kept (don't combine "
                         "with --reset-opt-on-resume; that gives a "
                         "zero-momentum Lion, which is noisy on first "
                         "steps). Use when training has regressed past a "
                         "known-good checkpoint within the same run. "
                         "(The schedule's plateau detector is reset on any "
                         "resume regardless of this flag.)")
    ap.add_argument("--reset-opt-on-resume", action="store_true",
                    help="When loading interrupted.pt, skip "
                         "opt.load_state_dict and start with a fresh "
                         "optimizer. Use with AdamW-family optimizers when "
                         "resume produces NaN — stale (m, v) buffers + a "
                         "shifted post-resume gradient direction can cause "
                         "the cautious mask to amplify or v underflow to "
                         "spike updates past fp16 range. Costs one cosine-"
                         "warmup's worth of slow steps as momentum re-warms. "
                         "Lion is generally robust to resume and doesn't "
                         "need this.")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--optimizer", default="lion",
                    choices=["lion", "adamw", "cautious-adamw", "prm"],
                    help="Lion (sign-momentum, sweet spot 3e-4..1e-3 for ternary). "
                         "AdamW (standard). Cautious-AdamW (Bonsai recipe; one-line "
                         "mod from Liang et al. 2024 arXiv:2411.16085, lr~1e-2). "
                         "PRM (Population Risk Minimization, Litman & Guo 2026): "
                         "AdamW + SNR mask that downweights coords whose batch-mean "
                         "grad is below the LOO noise floor.")
    ap.add_argument("--prm-softness", type=float, default=1.0,
                    help="PRM lam_pop. Mask q=1/2 sits on the LOO boundary when "
                         "softness=1; larger → more conservative (closes faster), "
                         "smaller → more aggressive. Practical range [0.3, 3].")
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="Base learning rate. Lion sweet spot for ternary "
                         "QAT is 3e-4..1e-3; AdamW family ~5e-4..1e-3. "
                         "Cosine decays to --lr-floor·lr over --soft-steps.")
    ap.add_argument("--lr-floor", type=float, default=0.1,
                    help="Cosine LR decays to floor·base_lr by --soft-steps. "
                         "Default 0.1; 1.0 disables decay (flat LR).")
    ap.add_argument("--opt-warmup-passes", type=int, default=0,
                    help="Run N forward+backward+opt.step() passes with lr=0 "
                         "BEFORE the real training loop, so (m, v) start from "
                         "averaged-grad statistics instead of cold zeros. "
                         "Params don't move (lr=0). For AdamW this mostly "
                         "smooths v_hat (bias correction handles cold v in "
                         "1 step, but averaging dampens noisy-direction "
                         "outliers); for Lion this warms the momentum buffer "
                         "toward grad-mean direction. Each pass costs one "
                         "training step's compute. Default 0 (off).")
    ap.add_argument("--scale-lr-mult", type=float, default=None,
                    help="LR multiplier for QLinear .scales parameters "
                         "(separate optimizer group). Scales receive full "
                         "KL grad (no (1-α) attenuation) and multiply the "
                         "output directly, so a sign-momentum or AdamW step "
                         "at the latent LR can dominate the loss trajectory "
                         "and walk weights out of a good basin on resume. "
                         "Default: 1.0/scale_group_size (~0.016 at gs=64), "
                         "matching the intuition that each scale spans N "
                         "ternary columns so a single scale update has "
                         "~N× the leverage of one latent update.")
    ap.add_argument("--scale-group-size", type=int, default=128,
                    help="Per-(row, column-group) scale granularity for "
                         "QLinear. Each group of N consecutive input columns "
                         "shares one fp32 scale. Bonsai uses 128. Must "
                         "divide every projection's in_features (SmolLM2-135M "
                         "needs 32 or 64; Qwen3 supports 128).")
    ap.add_argument("--permute", action=argparse.BooleanOptionalAction, default=True,
                    help="Permute each transformer block's free input dims "
                         "(MLP intermediate, per-KV-head head_dim) so columns "
                         "are sorted by descending magnitude before "
                         "quantize_in_place. Math-preserving init-time "
                         "transformation that aligns column magnitude with "
                         "scale-group boundaries — each group's max-abs "
                         "scale fits its members tightly, so fewer columns "
                         "waste capacity rounding to 0. Skipped on --resume "
                         "(the saved ckpt already encodes the permutation).")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--wd-cycle-amp", type=float, default=0.0,
                    help="Sinusoidal cycle amplitude on weight decay. "
                         "effective_wd = wd · (1 + amp · sin(2π·step/period)), "
                         "clamped ≥0. amp=1.0 swings 0↔2·wd; amp=1.5 swings "
                         "0↔2.5·wd (with a clamp-to-zero plateau for the lower "
                         "lobe). Default 0 = constant wd. Periodic squeeze "
                         "toward zero gives the zero-basin a tighter peak "
                         "without permanent over-decay.")
    ap.add_argument("--wd-cycle-period", type=int, default=1200,
                    help="Period of the wd cycle in opt steps. Only used when "
                         "--wd-cycle-amp > 0. Default 1200 — much slower than "
                         "the α cycle so the two go in/out of phase rather "
                         "than reinforcing.")
    ap.add_argument("--warmup-steps", type=int, default=30)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--plateau-threshold", type=float, default=1e-3,
                    help="Minimum relative EMA drop counted as a 'best' "
                         "improvement (for best_snapshot tracking).")
    ap.add_argument("--ema-warmup", type=int, default=500,
                    help="Skip best-EMA tracking for the first N steps. "
                         "The LR-warmup loss is often U-shaped past the "
                         "LR-warmup window, so set this past the typical "
                         "peak — otherwise the first post-warmup EMA "
                         "value, which is still on the way up, locks in "
                         "as best.")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--autocast-dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "none"])
    ap.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=True,
                    help="Trade some compute for activation memory (recommended on <=8GB).")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--soft-steps", type=int, default=100000,
                    help="Hard max steps in the soft-ternary stage. The "
                         "natural exit is α-saturation (--soft-saturation-tail) "
                         "— this is a safety cap.")
    ap.add_argument("--soft-saturation-eps", type=float, default=1e-3,
                    help="Tolerance for the α-saturation exit gate. The "
                         "stage is considered 'fully ramped' once schedule.α "
                         "≥ α_max - eps. With α_max=0.95 and eps=1e-3 this "
                         "fires when α reaches ~0.949 (effective λ has "
                         "reached λ_max).")
    ap.add_argument("--soft-hist-every", type=int, default=200,
                    help="TB histogram cadence (in opt.steps) for the "
                         "first QLinear's forward weight. The forward "
                         "weight has the attractor applied — in l2 mode "
                         "this is T_α(w) so you can watch the three-peak "
                         "structure emerge as α grows; in well mode the "
                         "forward is identity so this matches the raw "
                         "latent. Set 0 to disable histograms entirely.")
    ap.add_argument("--soft-saturation-tail", type=int, default=2000,
                    help="Steps to keep training after α-saturation before "
                         "exiting. Once the schedule has stopped moving, "
                         "the latent weights still need time to settle into "
                         "the fully-attracted basins — this tail is that "
                         "anneal window. Set 0 to exit immediately at "
                         "saturation.")
    ap.add_argument("--soft-zero-frac", type=float, default=0.33,
                    help="Target fraction of weights that the ternary "
                         "classifier maps to 0. Per-(row, group) |w|-"
                         "quantile at this value becomes the zero/non-"
                         "zero cutoff: weights below → 0, above → "
                         "±sign(w). Bonsai's deployed model is ~62%% non-"
                         "zero (~38%% zero), so 0.33-0.40 is a reasonable "
                         "range; the default 0.33 matches the uniform "
                         "[-1, 1] interpretation of the fixed ±1/3 "
                         "boundary. Set ≤0 or ≥1 to disable (use fixed "
                         "±1/3 boundary instead, which over-zeros for "
                         "real Gaussian-ish weight distributions). With "
                         "--soft-attractor=well this flag also calibrates "
                         "each QLinear's per-(row, group) well_a at init "
                         "(saddle a/√3 lands at the target |w| quantile) "
                         "so the well's natural basin split matches the "
                         "target — analogue of the L2 quantile cutoff but "
                         "frozen at init.")
    ap.add_argument("--soft-attractor", choices=["l2", "well"], default="l2",
                    help="Form of the soft-stage attractor. 'l2' (default) "
                         "pairs the residual-contraction reparam "
                         "T_α(w)=c(w)+(1-α)(w-c(w)) — c hard nearest of "
                         "{-1,0,+1}, boundaries ±1/3 — with an L2 penalty "
                         "‖w-c(w)‖²; the forward smoothly approaches "
                         "ternary as α grows. 'well' replaces both with the "
                         "triple-well potential U(w)=w²(w²-1)² as a single "
                         "C^∞ regularizer (forward stays identity, no "
                         "piecewise rule); the optimizer sees natural GD "
                         "on L_kl + α·U(W) so α·U'(w) is the attractor "
                         "force in every step's gradient. Saddles at "
                         "±1/√3, so the 0-basin is wider than under l2.")
    ap.add_argument("--soft-l2-coef", type=float, default=1e-2,
                    help="Attractor coefficient λ_max. Effective "
                         "λ(α) = λ_max·α — linear ramp so the basin force "
                         "starts at zero and reaches λ_max at α≈1. Applies "
                         "to both attractor forms (--soft-attractor); the "
                         "well potential's per-element values are roughly "
                         "comparable in magnitude to the L2 form so the "
                         "same coefficient is a reasonable starting point.")
    ap.add_argument("--soft-alpha-init", type=float, default=0.0,
                    help="Initial α (and so initial λ=λ_max·α) at the start "
                         "of soft training. Default 0 means the well/L2 "
                         "penalty is off at step 1 — but with KL≈teacher-"
                         "floor at init, Lion's sign update is dominated "
                         "by gradient noise and produces a warmup loss-"
                         "overshoot. Setting alpha_init>0 (e.g. 0.03) gives "
                         "Lion a coherent signal toward basin minima from "
                         "step 1, eliminating that overshoot at the cost of "
                         "slightly premature basin capture (which the per-"
                         "group well_a calibration makes nearly free since "
                         "init quantiles already define the 'natural' "
                         "assignment). Ignored on resume — schedule state "
                         "in interrupted.pt takes precedence.")
    ap.add_argument("--soft-cycle-amp", type=float, default=0.0,
                    help="Base sinusoidal cycle amplitude on the well-penalty "
                         "α (period=--soft-cycle-period steps). Default 0 = "
                         "disabled (monotonic schedule). Set e.g. 0.05 for a "
                         "small base swing — gives boundary weights a periodic "
                         "chance to re-cross saddles under KL guidance during "
                         "dips, then re-locks them in their (possibly new) "
                         "basins during peaks. Cycle modifies only the penalty "
                         "coefficient (and the soft-forward via effective_α), "
                         "not the schedule's α itself. Effective amplitude is "
                         "scaled by --soft-cycle-grow as the schedule "
                         "progresses.")
    ap.add_argument("--soft-cycle-period", type=int, default=200,
                    help="Period of the α cycle in opt steps. Only used "
                         "when --soft-cycle-amp > 0. Default 200 — slow "
                         "enough that Lion momentum (~10-step window) "
                         "tracks the cycle, fast enough that the cycle "
                         "completes between schedule bumps (which fire "
                         "every patience × bump-count steady steps).")
    ap.add_argument("--latent-noise-amp", type=float, default=0.0,
                    help="Per-step Gaussian noise amplitude on QLinear "
                         "latents (Langevin-style annealing). Per-(row, group) "
                         "noise scale = amp · (1−α)^pow · well_a/√3 — i.e. "
                         "noise is large early when α is small, decays to 0 "
                         "at saturation, and per-group amplitude tracks the "
                         "natural saddle-distance so all groups feel "
                         "comparable saddle-crossing pressure. Default 0 = "
                         "disabled. With this on, the cycle knobs should "
                         "typically be 0 (noise is the annealing mechanism).")
    ap.add_argument("--latent-noise-pow", type=float, default=2.0,
                    help="Exponent on the (1−α) noise decay schedule. 1.0 = "
                         "linear (noise drops smoothly through training). "
                         "2.0 = quadratic (noise stays high through mid-α, "
                         "drops sharply near saturation — more annealing "
                         "time). Higher = more abrupt cooldown.")
    ap.add_argument("--soft-cycle-grow", type=float, default=0.0,
                    help="Quadratic growth factor on the cycle amplitude as "
                         "the schedule progresses: cycle_amp = base · "
                         "(1 + grow · (α/α_max)²). grow=0 → constant base "
                         "amplitude (no taper, no growth). grow=4 → at α=α_max "
                         "the cycle amplitude is 5× base (e.g. base 0.05 → "
                         "0.25 swing → effective_α reaches ±0.25 around the "
                         "schedule, clamped to [0,1]; near saturation this is "
                         "full hard-ternary at peaks, which forces latent "
                         "flips during the next trough). Quadratic so the "
                         "early/mid α range stays calm and the aggression "
                         "concentrates at the polish end where flips matter.")
    ap.add_argument("--soft-alpha-max", type=float, default=0.95,
                    help="α ceiling during distill. <1 keeps the (1-α) "
                         "slope on KL gradient flowing to the latent. The "
                         "final hard rounding to {-1,0,+1} is finalize.py's "
                         "set_levels(model, 3) — no need to drive α to 1 "
                         "here.")
    ap.add_argument("--soft-patience", type=int, default=200,
                    help="Steps of stable loss-EMA required before each α "
                         "bump. With default bump=0.02 and α_max=0.95, the "
                         "schedule needs 47 bumps × 200 stable steps ≈ 9400 "
                         "ideal steps to saturate. In practice the post-"
                         "bump adaptation phase adds steps where stability "
                         "isn't yet reestablished — total run is data-"
                         "dependent.")
    ap.add_argument("--soft-bump", type=float, default=0.02,
                    help="α increment per bump. Smaller = more granular "
                         "schedule with more checkpoints; larger = faster "
                         "but riskier (each step is bigger for the student "
                         "to absorb).")
    ap.add_argument("--soft-tolerance", type=float, default=0.02,
                    help="Loss-EMA rise that still counts as 'stable' "
                         "(in cross-entropy nats). EMA <= baseline + "
                         "tolerance means we keep accumulating steady "
                         "steps; above means steady_count resets to 0.")
    ap.add_argument("--soft-ema", type=float, default=0.05,
                    help="Fast EMA smoothing factor for the loss signal "
                         "that the schedule operates on. Higher = more "
                         "reactive to per-step loss noise (faster ramp, "
                         "more thrashing); lower = smoother but slower. "
                         "Effective fast window ≈ 1/ema_alpha steps.")
    ap.add_argument("--soft-slow-ratio", type=float, default=10.0,
                    help="Slow-EMA window is this many times longer than "
                         "the fast EMA. The schedule's stability check is "
                         "(fast_ema ≤ slow_ema + tolerance), i.e. 'loss "
                         "isn't trending up'. With ema_alpha=0.05 and "
                         "slow_ratio=10, the slow window is ~200 steps. "
                         "Larger ratio = more conservative bumps (waits "
                         "for longer-term stability).")
    ap.add_argument("--soft-checkpoint-every", type=int, default=2000,
                    help="Auto-write interrupted.pt every N opt.steps so a "
                         "crash leaves a recent resume point. Same path "
                         "and format as the SIGINT save; on next run, the "
                         "loop picks up at the latest auto-checkpoint. "
                         "Set 0 to disable (only SIGINT writes).")
    ap.add_argument("--tb-dir", type=Path, default=None,
                    help="TensorBoard root. Default: <out>/tb. "
                         "View with `tensorboard --logdir <tb-dir>`.")
    ap.add_argument("--run-name", type=str, default=None,
                    help="Subdirectory under --tb-dir for this run. "
                         "Default: timestamped. Resumed runs reuse the saved name.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    _install_sigint_handler()

    print(f"[build] loading {args.model} and quantizing projections")
    # Latents stored as fp16 when autocast handles activation/weight dtype
    # mismatch in F.linear (cuBLAS does the cast in-kernel, no alloc); falls
    # back to fp32 when --autocast-dtype none, otherwise F.linear would
    # error on x_fp32 @ w_fp16. Bounded to [-1,1] by init+clamp_qlinear_weights,
    # so fp16's tapered ULP wins over bf16 by ~8x near zero. Optimizer state
    # stays fp32 via Lion32/AdamW32 to avoid v underflow on small late-stage grads.
    latent_dtype = torch.float32 if args.autocast_dtype == "none" else torch.float16
    # Permute is fresh-start only — it operates on the FP teacher, BEFORE
    # quantize_in_place. On --resume the saved ckpt already contains the
    # permuted weights, so re-permuting the teacher is harmless (load_state_dict
    # will overwrite anyway) but redundant; skip to save the work.
    fresh_start = args.resume is None and not (args.out / "interrupted.pt").exists()
    do_permute = args.permute and fresh_start
    model, _tok, n_replaced = load_student(args.model, dtype=torch.float32,
                                           levels=257,
                                           latent_dtype=latent_dtype,
                                           group_size=args.scale_group_size,
                                           permute=do_permute)
    print(f"[build] {n_replaced} QLinear modules "
          f"(latent dtype: {latent_dtype}, group_size: {args.scale_group_size}, "
          f"permute: {do_permute})")
    model = model.to(args.device)
    if hasattr(model, "config"):
        model.config.use_cache = False  # never needed for training fwd/bwd
    if args.grad_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        # use_reentrant=False: reentrant checkpointing doesn't propagate the
        # autocast context across the recomputation pass, which corrupts
        # activations under fp16 latent + bf16 autocast (the recompute uses
        # whatever dtype the saved tensors landed in, not what autocast
        # originally produced). Manifests as NaN loss a few hundred steps in.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        print("[build] gradient checkpointing enabled (use_reentrant=False)")
    # Snapshot the teacher embed BEFORE any resume/interrupted state_dict load
    # overwrites it; that's the reference for cumulative embed/drift_l2.
    # On the device already so the per-step subtraction in embed_drift_l2
    # doesn't trip a device mismatch. The teacher is deterministic from
    # args.model so a fresh load gives the same tensor on every resume —
    # no need to persist this snapshot.
    _embed = model.get_input_embeddings()
    embed_init = _embed.weight.detach().clone() if _embed is not None else None

    interrupted_path = args.out / "interrupted.pt"
    interrupted_state = None
    resume_meta: dict[str, str] = {}
    if args.resume is not None:
        with safe_open(str(args.resume), framework="pt") as f:
            resume_meta = f.metadata() or {}
        sd = load_file(str(args.resume))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[resume] {args.resume.name} (meta={resume_meta})")
        if missing:
            print(f"[resume] missing keys: {len(missing)} (showing 5): {missing[:5]}")
        if unexpected:
            print(f"[resume] unexpected keys: {len(unexpected)} (showing 5): {unexpected[:5]}")
        ckpt_group = resume_meta.get("group_size")
        if ckpt_group and int(ckpt_group) != args.scale_group_size:
            print(f"[resume] WARNING: --scale-group-size={args.scale_group_size} "
                  f"but ckpt was trained with group_size={ckpt_group}")
    elif interrupted_path.exists():
        print(f"[resume] found interrupted snapshot at {interrupted_path}")
        interrupted_state = torch.load(str(interrupted_path),
                                       map_location=args.device,
                                       weights_only=False)
        sd_to_load = interrupted_state["model"]
        if args.resume_from_best:
            best_sd = interrupted_state.get("best_snapshot")
            if best_sd is None:
                print("[resume] --resume-from-best: no best_snapshot in "
                      "interrupted.pt; falling back to current model")
            else:
                cs = interrupted_state.get("ctrl_state") or {}
                print(f"[resume] --resume-from-best: loading model from "
                      f"best_snapshot (best_step={cs.get('best_step')}, "
                      f"best_ema={cs.get('best_ema')})")
                sd_to_load = best_sd
        miss_i, unexp_i = model.load_state_dict(sd_to_load, strict=False)
        if miss_i:
            print(f"[resume] interrupted snapshot missing {len(miss_i)} model "
                  f"keys (showing 5): {miss_i[:5]}")
        if unexp_i:
            print(f"[resume] interrupted snapshot has {len(unexp_i)} unexpected "
                  f"model keys (showing 5): {unexp_i[:5]}")
        ss = interrupted_state.get("soft_state")
        print(f"[resume]   next_step={interrupted_state.get('next_step')}")
        if ss:
            sat = ss.get("saturation_step")
            sched = ss.get("schedule", {})
            print(f"[resume]   soft: α={sched.get('alpha')} "
                  f"bumps={sched.get('bumps')} "
                  f"steady_count={sched.get('steady_count')} "
                  f"slow_ema={sched.get('slow_ema', sched.get('baseline'))} "
                  f"saturation_step={sat}")

    if args.compile:
        model = torch.compile(model)

    scale_lr_mult = (args.scale_lr_mult if args.scale_lr_mult is not None
                     else 1.0 / float(args.scale_group_size))
    scale_param_ids = {id(m.scales) for m in model.modules()
                       if isinstance(m, QLinear)}
    scale_params, other_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (scale_params if id(p) in scale_param_ids else other_params).append(p)
    param_groups = [
        {"params": other_params, "lr": args.lr, "lr_mult": 1.0,
         "name": "latents"},
        {"params": scale_params, "lr": args.lr * scale_lr_mult,
         "lr_mult": scale_lr_mult, "name": "scales"},
    ]
    if args.optimizer == "lion":
        opt = Lion32(param_groups, lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == "adamw":
        opt = AdamW32(param_groups, lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == "cautious-adamw":
        opt = CautiousAdamW(param_groups, lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == "prm":
        opt = PRM32(param_groups, lr=args.lr, weight_decay=args.wd,
                    softness=args.prm_softness)
    else:
        raise ValueError(f"unknown optimizer: {args.optimizer}")
    print(f"[opt] {args.optimizer} lr={args.lr} wd={args.wd} (fp32 state)")
    print(f"[opt]   latents : {len(other_params)} tensors @ lr={args.lr:g}")
    print(f"[opt]   scales  : {len(scale_params)} tensors @ "
          f"lr={args.lr * scale_lr_mult:g} (mult={scale_lr_mult:g})")

    if interrupted_state is not None:
        if args.reset_opt_on_resume:
            print("[resume] --reset-opt-on-resume: skipping opt.load_state_dict; "
                  "starting with fresh optimizer momentum")
        else:
            opt.load_state_dict(interrupted_state["opt"])
        global_step = int(interrupted_state.get("next_step", 0))
        # Older checkpoints (pre-data-position) lack samples_consumed; fall
        # back to inferring from step × grad_accum × batch_size, which is
        # exact when grad_accum/batch_size haven't changed across resume.
        samples_consumed = int(interrupted_state.get(
            "samples_consumed",
            global_step * args.grad_accum * args.batch_size))
        print(f"[resume] continuing at step {global_step}, "
              f"data position = {samples_consumed} samples")
    else:
        global_step = 0
        samples_consumed = 0

    tb_root = args.tb_dir if args.tb_dir is not None else (args.out / "tb")
    if interrupted_state is not None and interrupted_state.get("run_name"):
        run_name = interrupted_state["run_name"]
    elif args.run_name:
        run_name = args.run_name
    else:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = tb_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[tb] run_dir={run_dir} (view: tensorboard --logdir {tb_root})")

    ds = ShardedDataset(args.cache_dir, seed=args.seed,
                        start_skip=samples_consumed)
    if samples_consumed:
        print(f"[data] fast-forwarding {samples_consumed} samples to resume "
              f"position (worker will load shards but skip until counter "
              f"catches up)")
    dl = DataLoader(ds, batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=(args.device.startswith("cuda")),
                    drop_last=True)
    it = iter(dl)

    autocast_dtype = {"bfloat16": torch.bfloat16,
                      "float16": torch.float16,
                      "none": None}[args.autocast_dtype]


    # ============ Soft-ternary training =====================================
    # L_T is purely diagnostic now (the schedule uses fast/slow EMA, not
    # gap to a reference floor). Computed once from the cache as the lumped
    # cross-entropy floor — what `kl_with_rest` actually floors at when the
    # student matches the teacher's K+1 lumped distribution. Logged in TB
    # so you can eyeball "are we near what the teacher achieves" without
    # any flag.
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(args.model)
    floor_data = load_teacher_floor(args.cache_dir, len(_tok))
    L_T = float(floor_data["floor"])
    print(f"\n[soft] L_T (diagnostic) = {L_T:.4f} (lumped floor from cache)")

    # Schedule: load from interrupted snapshot if present, else fresh.
    schedule = AlphaSchedule(alpha_max=args.soft_alpha_max,
                             patience=args.soft_patience,
                             bump=args.soft_bump,
                             tolerance=args.soft_tolerance,
                             ema_alpha=args.soft_ema,
                             slow_ratio=args.soft_slow_ratio,
                             alpha_init=args.soft_alpha_init)
    if (interrupted_state is not None
            and interrupted_state.get("soft_state")):
        soft_st = interrupted_state["soft_state"]
        schedule.load_state_dict(soft_st.get("schedule", {}))
        # Always rebuild the plateau detector from current losses on resume.
        # The saved slow_ema reflects the pre-SIGINT loss range; if the
        # post-resume EMA happens to start below it, every step counts as
        # "steady" and bumps fire continuously until slow_ema catches up.
        # Resetting ensures the schedule re-anchors to current loss reality
        # and only resumes bumping after slow_ema has stabilized.
        print(f"[soft] resetting schedule ema/slow_ema/steady_count on "
              f"resume (had slow_ema={schedule.slow_ema}, "
              f"steady_count={schedule.steady_count})")
        schedule.ema = None
        schedule.slow_ema = None
        schedule.steady_count = 0
        soft_step_offset = int(soft_st.get("next_step", 0))
        print(f"[soft] resuming at step {soft_step_offset} "
              f"α={schedule.alpha:.4f} bumps={schedule.bumps} "
              f"steady={schedule.steady_count}")
    else:
        soft_step_offset = 0

    # In `well` mode the forward is identity (U penalty does the work) so
    # we hold QLinear.alpha at 0; the schedule's α drives only the penalty
    # coefficient. In `l2` mode the forward IS T_α(w).
    forward_alpha = schedule.alpha if args.soft_attractor == "l2" else 0.0
    target_zero_frac = (float(args.soft_zero_frac)
                        if 0.0 < args.soft_zero_frac < 1.0 else None)
    n_set = set_soft_mode(model, alpha=forward_alpha,
                          target_zero_frac=target_zero_frac)
    print(f"[soft] {n_set} QLinear modules → soft mode")
    print(f"[soft]   attractor    = {args.soft_attractor}")
    # Calibrate per-(row, group) well_a from init |w| quantile so the
    # well's 0-basin lines up with target_zero_frac. Only on fresh start —
    # on resume the saved well_a in state_dict is the right one to use,
    # and re-init from current (drifted) weights would shift the basins
    # mid-run.
    if (args.soft_attractor == "well" and target_zero_frac is not None
            and interrupted_state is None):
        n_init = init_well_a(model, target_zero_frac)
        print(f"[soft]   well_a       = init'd per-(row, group) for "
              f"{n_init} modules from √3·|w|.quantile({target_zero_frac}); "
              f"deploy rescale at end")
    elif args.soft_attractor == "well":
        print(f"[soft]   well_a       = "
              + ("loaded from resume snapshot"
                 if interrupted_state is not None
                 else "1.0 (canonical ±1 wells; no zero-frac calibration)"))
    if target_zero_frac is not None:
        print(f"[soft]   zero_frac    = {target_zero_frac} "
              f"(per-(row, group) |w|-quantile cutoff)")
    else:
        print(f"[soft]   zero_frac    = disabled (fixed ±1/3 boundary)")
    print(f"[soft]   L_T          = {L_T:.4f} (diagnostic; lumped from cache)")
    print(f"[soft]   λ_max        = {args.soft_l2_coef}")
    print(f"[soft]   α_max        = {args.soft_alpha_max}")
    print(f"[soft]   patience     = {args.soft_patience} steady steps/bump")
    print(f"[soft]   bump         = {args.soft_bump} (≈"
          f"{int(args.soft_alpha_max / max(args.soft_bump, 1e-9))} bumps "
          f"to saturate)")
    print(f"[soft]   tolerance    = {args.soft_tolerance} "
          f"(loss-EMA rise still counted as stable)")
    print(f"[soft]   ema_alpha    = {args.soft_ema} (loss-EMA smoothing)")
    print(f"[soft]   saturation   = exit when α ≥ α_max - "
          f"{args.soft_saturation_eps:g}, then "
          f"+{args.soft_saturation_tail} steps")
    print(f"[soft]   max_steps    = {args.soft_steps} (safety cap)")
    print(f"[soft]   ckpt_every   = {args.soft_checkpoint_every} steps "
          f"(auto-write {args.out / 'interrupted.pt'})")
    print(f"[soft]   lr           = {args.lr} "
          f"(cosine to {args.lr_floor}·lr, warmup={args.warmup_steps})")
    print(f"[soft]   optimizer    = {args.optimizer}")
    if soft_step_offset:
        print(f"[soft]   resuming     = step {soft_step_offset}, "
              f"α={schedule.alpha:.4f}, bumps={schedule.bumps}, "
              f"steady={schedule.steady_count}, "
              f"slow_ema={schedule.slow_ema}")

    writer = SummaryWriter(log_dir=str(run_dir),
                           purge_step=soft_step_offset if soft_step_offset else None)
    writer.add_text(
        "stage",
        "  \n".join([
            f"**soft** training (resume @ step {soft_step_offset})",
            f"- attractor: `{args.soft_attractor}`",
            (f"- zero_frac: {target_zero_frac} (per-group quantile)"
             if target_zero_frac is not None else
             "- zero_frac: disabled (fixed ±1/3 boundary)"),
            f"- L_T (diagnostic): {L_T:.4f}",
            f"- λ_max (soft_l2_coef): {args.soft_l2_coef}",
            f"- α_max: {args.soft_alpha_max}",
            f"- patience: {args.soft_patience}",
            f"- bump: {args.soft_bump}",
            f"- tolerance: {args.soft_tolerance}",
            f"- ema_alpha: {args.soft_ema}",
            f"- slow_ratio: {args.soft_slow_ratio}",
            f"- saturation eps: {args.soft_saturation_eps}",
            f"- saturation tail: {args.soft_saturation_tail}",
            f"- max_steps: {args.soft_steps}",
            f"- ckpt_every: {args.soft_checkpoint_every}",
            f"- lr: {args.lr} (cosine to floor={args.lr_floor})",
            f"- optimizer: {args.optimizer}",
        ]),
        soft_step_offset)
    writer.add_scalar("soft/cfg/L_T", L_T, soft_step_offset)
    writer.add_scalar("soft/cfg/zero_frac",
                      target_zero_frac if target_zero_frac is not None
                      else -1.0, soft_step_offset)
    writer.add_scalar("soft/cfg/lambda_max", args.soft_l2_coef,
                      soft_step_offset)
    writer.add_scalar("soft/cfg/alpha_max", args.soft_alpha_max,
                      soft_step_offset)
    writer.add_scalar("soft/cfg/patience",
                      float(args.soft_patience), soft_step_offset)
    writer.add_scalar("soft/cfg/bump", args.soft_bump,
                      soft_step_offset)
    writer.add_scalar("soft/cfg/tolerance", args.soft_tolerance,
                      soft_step_offset)
    writer.add_scalar("soft/cfg/ema_alpha", args.soft_ema,
                      soft_step_offset)
    writer.add_scalar("soft/cfg/max_steps", float(args.soft_steps),
                      soft_step_offset)
    writer.add_scalar("soft/cfg/saturation_eps",
                      args.soft_saturation_eps, soft_step_offset)
    writer.add_scalar("soft/cfg/saturation_tail",
                      float(args.soft_saturation_tail),
                      soft_step_offset)
    # 0=l2, 1=well — TB only renders numeric scalars.
    writer.add_scalar("soft/cfg/attractor_id",
                      0.0 if args.soft_attractor == "l2" else 1.0,
                      soft_step_offset)

    _embed = model.get_input_embeddings()
    embed_stage_init = (_embed.weight.detach().clone()
                        if _embed is not None else None)

    ctrl = BestEmaTracker(rel_threshold=args.plateau_threshold,
                          ema_warmup=args.ema_warmup)
    if (interrupted_state is not None
            and interrupted_state.get("ctrl_state")):
        ctrl.load_state_dict(interrupted_state["ctrl_state"])
    if (interrupted_state is not None
            and interrupted_state.get("best_snapshot") is not None):
        best_snapshot = interrupted_state["best_snapshot"]
    else:
        best_snapshot = snapshot_to_cpu(model)
    # Baseline for flip-rate tracking against both the soft alphabet
    # (moving quantile classifier) and the frozen well-saddle classifier.
    _, prev_codes, prev_codes_fixed = collect_qlinear_metrics(model, {})

    soft_lr = args.lr
    model.train()
    opt.zero_grad(set_to_none=True)
    running = 0.0
    running_n = 0
    pbar = tqdm(range(soft_step_offset, args.soft_steps),
                initial=soft_step_offset, total=args.soft_steps,
                desc="soft", dynamic_ncols=True)
    advance_reason = "max_steps"
    last_step = soft_step_offset
    # Saturation gate: fire when α first reaches α_max (within eps), then
    # run `tail` more steps for anneal/cleanup. The schedule is monotone
    # so once saturated it stays — no debouncing needed.
    _ss = (interrupted_state.get("soft_state", {})
           if interrupted_state else {}).get("saturation_step")
    saturation_step: int | None = int(_ss) if _ss is not None else None
    if saturation_step is not None:
        print(f"[soft]   α already saturated at step {saturation_step + 1}")
    # Initial histogram at step soft_step_offset so the TB plot has a
    # baseline tick before any training has happened (the "before"
    # snapshot for the three-peak emergence story). On a fresh run
    # soft_step_offset is 0; on resume it's wherever we picked up.
    if args.soft_hist_every > 0:
        hist_sample = first_qlinear_forward_sample(model)
        if hist_sample is not None:
            name, w_flat = hist_sample
            if torch.isfinite(w_flat).all():
                writer.add_histogram(f"soft/hist/{name}", w_flat,
                                     soft_step_offset, bins=64)
    # Optimizer (m, v) pre-warm: run K forward+backward+opt.step() passes
    # with lr=0 so the optimizer's moment buffers start from averaged
    # gradient statistics. Params don't move (lr=0); only opt state updates.
    # Skipped on resume since (m, v) are restored from the checkpoint.
    if args.opt_warmup_passes > 0 and interrupted_state is None:
        print(f"[opt-warmup] {args.opt_warmup_passes} forward+backward "
              f"passes at lr=0 to populate (m, v)")
        saved_lrs = [g["lr"] for g in opt.param_groups]
        for g in opt.param_groups:
            g["lr"] = 0.0
        warm_bar = tqdm(range(args.opt_warmup_passes),
                        desc="opt-warmup", dynamic_ncols=True, leave=False)
        for _ in warm_bar:
            opt.zero_grad(set_to_none=True)
            for _ in range(args.grad_accum):
                batch = next(it)
                tokens = batch["tokens"].to(args.device, non_blocking=True)
                topk_idx = batch["topk_idx"].to(args.device, non_blocking=True)
                topk_prob = batch["topk_prob"].to(args.device, non_blocking=True)
                rest_mass = batch["rest_mass"].to(args.device, non_blocking=True)
                ctx = (torch.amp.autocast(args.device.split(":")[0],
                                          dtype=autocast_dtype)
                       if autocast_dtype is not None
                       else torch.amp.autocast(args.device.split(":")[0],
                                               enabled=False))
                with ctx:
                    out = model(tokens)
                    loss = kl_with_rest(out.logits, topk_idx, topk_prob,
                                        rest_mass)
                (loss / args.grad_accum).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        for g, lr in zip(opt.param_groups, saved_lrs):
            g["lr"] = lr
        print(f"[opt-warmup] done; (m, v) populated, params unchanged")
    for step in pbar:
        last_step = step
        cur_lr = lr_at(step, args.soft_steps, soft_lr, args.warmup_steps,
                       floor=args.lr_floor)
        if args.wd_cycle_amp > 0 and args.wd_cycle_period > 0:
            wd_cycle_factor = max(0.0, 1.0 + args.wd_cycle_amp * math.sin(
                2.0 * math.pi * global_step / args.wd_cycle_period))
        else:
            wd_cycle_factor = 1.0
        effective_wd = args.wd * wd_cycle_factor
        for g in opt.param_groups:
            g["lr"] = cur_lr * g.get("lr_mult", 1.0)
            g["weight_decay"] = effective_wd
        for _ in range(args.grad_accum):
            batch = next(it)
            tokens = batch["tokens"].to(args.device, non_blocking=True)
            topk_idx = batch["topk_idx"].to(args.device, non_blocking=True)
            topk_prob = batch["topk_prob"].to(args.device, non_blocking=True)
            rest_mass = batch["rest_mass"].to(args.device, non_blocking=True)
            ctx = (torch.amp.autocast(args.device.split(":")[0],
                                      dtype=autocast_dtype)
                   if autocast_dtype is not None
                   else torch.amp.autocast(args.device.split(":")[0],
                                           enabled=False))
            with ctx:
                out = model(tokens)
                loss = kl_with_rest(out.logits, topk_idx, topk_prob, rest_mass)
            if not torch.isfinite(loss):
                # Fail fast in the soft loop too — silent NaN propagation
                # corrupts the loss-EMA, the schedule freezes (steady_count
                # stuck at 0 because NaN comparisons are False), and all
                # downstream TB scalars/histograms turn into NaN.
                logits_max = float(out.logits.detach().abs().max())
                logits_finite = bool(torch.isfinite(out.logits).all())
                w_finite = all(torch.isfinite(p).all().item()
                               for p in model.parameters())
                raise RuntimeError(
                    f"non-finite loss in soft stage at step {step} "
                    f"(α={schedule.alpha:.4f}, loss={loss.item()}); "
                    f"logits |max|={logits_max:.3e} "
                    f"finite={logits_finite}; all-params-finite={w_finite}")
            (loss / args.grad_accum).backward()
            running += loss.item()
            running_n += 1
        # Snapshot the KL-only gradient norm before the attractor penalty
        # adds to it (only on log steps to avoid the per-param iteration).
        log_this_step = (step + 1) % args.log_every == 0
        kl_grad_norm: float | None = None
        if log_this_step:
            kl_grad_norm = grad_l2_norm(model)
        # Attractor penalty: one backward per opt.step (regularization on
        # weights, not data, so no grad_accum scaling). λ(α) = λ_max·α —
        # zero at the start (no need to pull) and ramps up to λ_max at
        # α≈1, matching the schedule.
        # Optional sinusoidal cycle on top of the schedule's α: gives
        # boundary weights periodic re-assignment opportunities during
        # dips. Amplitude scales quadratically with α/α_max via
        # --soft-cycle-grow so polish-phase swings can reach hard
        # ternary at peaks while early-phase swings stay small.
        if args.soft_cycle_amp > 0 and args.soft_cycle_period > 0:
            alpha_frac = min(1.0, schedule.alpha
                             / max(args.soft_alpha_max, 1e-9))
            grow_factor = 1.0 + args.soft_cycle_grow * (alpha_frac ** 2)
            cycle_amp = args.soft_cycle_amp * grow_factor
            cycle_offset = cycle_amp * math.sin(
                2.0 * math.pi * global_step / args.soft_cycle_period)
            effective_alpha = max(0.0, min(1.0,
                                          schedule.alpha + cycle_offset))
        else:
            cycle_amp = 0.0
            cycle_offset = 0.0
            effective_alpha = schedule.alpha
        cur_lambda = args.soft_l2_coef * effective_alpha
        penalty_val: float | None = None
        if cur_lambda > 0:
            if args.soft_attractor == "l2":
                penalty = attractor_l2(model)
            else:
                penalty = triple_well_loss(model)
            (cur_lambda * penalty).backward()
            penalty_val = float(penalty.detach())
        grad_norm: float | None = None
        if args.max_grad_norm:
            g = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.max_grad_norm)
            grad_norm = float(g)
        opt.step()
        clamp_qlinear_weights(model)
        opt.zero_grad(set_to_none=True)
        # Langevin-style latent noise: Gaussian kick scaled per-(row, group)
        # by well_a/√3 (the basin's saddle distance), modulated by
        # (1−α)^pow so it anneals to 0 at saturation. Applied after opt.step
        # so it's a true position perturbation, not a gradient injection.
        noise_sigma_base = 0.0
        if args.latent_noise_amp > 0 and schedule.alpha < 1.0:
            noise_sigma_base = (args.latent_noise_amp
                                * (1.0 - schedule.alpha)
                                ** args.latent_noise_pow)
        if noise_sigma_base > 0:
            sqrt3 = math.sqrt(3.0)
            with torch.no_grad():
                for m in model.modules():
                    if not isinstance(m, QLinear):
                        continue
                    # well_a: [out_features, n_groups]; expand along the
                    # in_features axis to per-element noise scale.
                    a_per_group = (m.well_a / sqrt3).to(m.weight.dtype)
                    scale = a_per_group.repeat_interleave(
                        m.group_size, dim=-1)
                    m.weight.add_(torch.randn_like(m.weight)
                                  * (noise_sigma_base * scale))
            clamp_qlinear_weights(model)
        global_step += 1
        step_loss = running / max(1, running_n)
        cur_alpha, steady_count, bumped = schedule.step(step_loss)
        gap = max(0.0, step_loss - L_T) if L_T is not None else None
        # Only the l2 form uses α in the forward; well-mode forward is
        # always identity (QLinear.alpha stays 0).
        if args.soft_attractor == "l2":
            set_soft_alpha(model, cur_alpha)
        # Saturation gate: latch the step at which α first reaches the
        # ceiling (within --soft-saturation-eps). Schedule is monotone,
        # so once latched we don't unset it.
        if (saturation_step is None
                and cur_alpha >= args.soft_alpha_max
                                 - args.soft_saturation_eps):
            saturation_step = step
            tqdm.write(f"[soft] α saturated at step {step + 1} "
                       f"(α={cur_alpha:.4f} ≥ α_max - eps); will exit "
                       f"after {args.soft_saturation_tail}-step tail.")
        # Best-snapshot tracking is on from the start so a derailed run
        # has a recoverable checkpoint at the lowest loss seen — useful
        # for restarting with new schedule params from a known-good
        # state. End-of-stage rollback only fires if the schedule
        # actually completed (saturation reached); a partial run leaves
        # the model at its current state to preserve schedule progress.
        improved = ctrl.update(step, step_loss)
        if improved:
            best_snapshot = snapshot_to_cpu(model)
        if log_this_step:
            postfix = {
                "loss": f"{step_loss:.4f}",
                "ema": f"{ctrl.ema:.4f}",
                "α": f"{cur_alpha:.3f}",
                "λ": f"{cur_lambda:.2e}",
                "stdy": f"{steady_count}/{args.soft_patience}",
                "lr": f"{cur_lr:.2e}",
            }
            if gap is not None:
                postfix["gap"] = f"{gap:.3f}"
            if penalty_val is not None:
                postfix["pen"] = f"{penalty_val:.3f}"
            pbar.set_postfix(postfix)
            tb_step = step + 1
            writer.add_scalar("loss/step", step_loss, tb_step)
            if ctrl.ema is not None:
                writer.add_scalar("loss/ema", ctrl.ema, tb_step)
            writer.add_scalar("lr", cur_lr, tb_step)
            writer.add_scalar("lr/scales", cur_lr * scale_lr_mult, tb_step)
            writer.add_scalar("global_step", float(global_step), tb_step)
            writer.add_scalar("soft/alpha", cur_alpha, tb_step)
            writer.add_scalar("soft/steady_count",
                              float(schedule.steady_count), tb_step)
            writer.add_scalar("soft/bumps", float(schedule.bumps),
                              tb_step)
            if schedule.ema is not None:
                writer.add_scalar("soft/loss_ema_fast",
                                  schedule.ema, tb_step)
            if schedule.slow_ema is not None:
                writer.add_scalar("soft/loss_ema_slow",
                                  schedule.slow_ema, tb_step)
                if schedule.ema is not None:
                    writer.add_scalar(
                        "soft/loss_ema_diff",
                        schedule.ema - schedule.slow_ema, tb_step)
            if gap is not None:
                writer.add_scalar("soft/gap", gap, tb_step)
            writer.add_scalar("soft/lambda", cur_lambda, tb_step)
            if args.soft_cycle_amp > 0:
                writer.add_scalar("soft/effective_alpha",
                                  effective_alpha, tb_step)
                writer.add_scalar("soft/cycle_offset",
                                  cycle_offset, tb_step)
                writer.add_scalar("soft/cycle_amp", cycle_amp, tb_step)
            if args.wd_cycle_amp > 0:
                writer.add_scalar("wd/cycle_factor", wd_cycle_factor, tb_step)
                writer.add_scalar("wd/effective", effective_wd, tb_step)
            if args.latent_noise_amp > 0:
                writer.add_scalar("noise/sigma_base", noise_sigma_base,
                                  tb_step)
            if L_T is not None:
                writer.add_scalar("soft/L_T", L_T, tb_step)
            if penalty_val is not None:
                writer.add_scalar("soft/penalty", penalty_val, tb_step)
            if grad_norm is not None:
                writer.add_scalar("grad_norm", grad_norm, tb_step)
            if kl_grad_norm is not None:
                writer.add_scalar("soft/grad_norm_kl", kl_grad_norm,
                                  tb_step)
                if grad_norm is not None:
                    # Penalty's net contribution to the gradient (proxy:
                    # vector-sum norms aren't strictly additive but the
                    # difference tracks the order of magnitude).
                    writer.add_scalar(
                        "soft/grad_norm_penalty",
                        max(0.0, grad_norm - kl_grad_norm), tb_step)
            qm, prev_codes, prev_codes_fixed = collect_qlinear_metrics(
                model, prev_codes, prev_codes_fixed)
            for k, v in qm.items():
                writer.add_scalar(k, v, tb_step)
            soft_qm, _ = collect_soft_metrics(model)
            for k, v in soft_qm.items():
                writer.add_scalar(k, v, tb_step)
            drift, drift_stage = embed_drift_l2(model, embed_init,
                                                embed_stage_init)
            if drift is not None:
                writer.add_scalar("embed/drift_l2", drift, tb_step)
            if drift_stage is not None:
                writer.add_scalar("embed/drift_l2_stage", drift_stage,
                                  tb_step)
            running = 0.0
            running_n = 0
        # Forward-weight histogram on its own cadence (independent of
        # log_every). Shows the attractor-applied weight: in l2 mode the
        # soft-blend T_α(w) — three-peak structure emerges as α grows;
        # in well mode at α=0 forward = identity = raw latent.
        if (args.soft_hist_every > 0
                and (step + 1) % args.soft_hist_every == 0):
            hist_sample = first_qlinear_forward_sample(model)
            if hist_sample is not None:
                name, w_flat = hist_sample
                if torch.isfinite(w_flat).all():
                    writer.add_histogram(f"soft/hist/{name}", w_flat,
                                         step + 1, bins=64)
                else:
                    tqdm.write(f"[soft] skipping histogram at step "
                               f"{step + 1}: non-finite values in "
                               f"{name}")
        # Auto-checkpoint: write the same interrupted.pt every N steps so a
        # crash leaves a recent resume point. Same path as the SIGINT save —
        # latest wins; the atomic .tmp/rename in save_resume means a crash
        # mid-write leaves the previous good snapshot intact.
        samples_at_save = (step + 1) * args.grad_accum * args.batch_size
        if (args.soft_checkpoint_every > 0
                and (step + 1) % args.soft_checkpoint_every == 0):
            soft_state = {"schedule": schedule.state_dict(),
                          "next_step": step + 1,
                          "saturation_step": saturation_step}
            save_resume(interrupted_path, model, opt, step + 1,
                        best_snapshot, ctrl.state_dict(), run_name,
                        samples_consumed=samples_at_save,
                        soft_state=soft_state)
            tqdm.write(f"[ckpt] auto-wrote {interrupted_path} at step "
                       f"{step + 1}")
        if _INTERRUPT["flag"]:
            pbar.close()
            soft_state = {"schedule": schedule.state_dict(),
                          "next_step": step + 1,
                          "saturation_step": saturation_step}
            save_resume(interrupted_path, model, opt, step + 1,
                        best_snapshot, ctrl.state_dict(), run_name,
                        samples_consumed=samples_at_save,
                        soft_state=soft_state)
            writer.flush()
            writer.close()
            print(f"[!] saved soft resume snapshot to {interrupted_path}")
            sys.exit(0)
        # Primary exit: α-saturation tail elapsed. The schedule has
        # finished ramping and the latents have had `tail` steps to
        # settle; further training is diminishing returns and risks
        # over-fitting the attractor against the data.
        if (saturation_step is not None
                and (step - saturation_step) >= args.soft_saturation_tail):
            advance_reason = (f"α saturated at step {saturation_step + 1} "
                              f"+ tail={args.soft_saturation_tail}")
            pbar.close()
            break
        # No plateau exit in soft: a flat loss is the IDEAL state — it
        # means the model is settled at the current α, the schedule's
        # `steady_count` is accumulating, and the next bump is on its
        # way. Exiting on flat loss here would terminate before α
        # reaches α_max, leaving the model under-attracted. The only
        # exits are saturation+tail (above) and --soft-steps (the
        # outer for-loop's range cap).
    regressed = (ctrl.ema is not None
                 and ctrl.best_ema != float("inf")
                 and ctrl.ema > ctrl.best_ema * 1.005)
    print(f"[soft] complete after {last_step + 1} steps "
          f"(reason: {advance_reason}; ema={ctrl.ema}, "
          f"best={ctrl.best_ema}@{ctrl.best_step}, "
          f"final α={schedule.alpha:.4f})")
    writer.add_text("stage_end",
                    f"soft steps={last_step + 1} reason={advance_reason} "
                    f"ema={ctrl.ema} best={ctrl.best_ema}@{ctrl.best_step} "
                    f"alpha_final={schedule.alpha:.4f} regressed={regressed}",
                    last_step + 1)
    writer.flush()
    writer.close()
    # Rollback only if the schedule actually completed — a partial run
    # rolling back to a low-α best would erase all the schedule progress.
    # If the user wants a low-α best for restart-with-new-params, they
    # have it in interrupted_state.best_snapshot via SIGINT.
    if regressed and saturation_step is not None:
        print(f"[soft] restoring best snapshot from step {ctrl.best_step}")
        restore_from_snapshot(model, best_snapshot)
    elif regressed:
        print(f"[soft] regressed but schedule did not saturate "
              f"(ema={ctrl.ema:.4f} > best={ctrl.best_ema:.4f}); "
              f"NOT restoring — partial-α progress preserved. The "
              f"best snapshot is in best_snapshot in memory; SIGINT "
              f"now to persist it via interrupted.pt.")
    if args.soft_attractor == "well":
        n_rescaled = rescale_well_for_deploy(model)
        if n_rescaled > 0:
            print(f"[soft] rescaled {n_rescaled} QLinear modules from "
                  f"per-group well minima to deploy codebook ±1 "
                  f"(latent /= a, scales *= a; well_a reset to 1.0)")
    soft_ckpt = args.out / "stage_soft.safetensors"
    save_checkpoint(model, soft_ckpt, args.model, args.scale_group_size,
                    alpha=schedule.alpha,
                    target_zero_frac=target_zero_frac)
    print(f"[soft] saved {soft_ckpt}")

    if interrupted_path.exists():
        interrupted_path.unlink()
        print(f"[done] removed {interrupted_path}")
    print("[done] soft training complete.")


if __name__ == "__main__":
    main()
