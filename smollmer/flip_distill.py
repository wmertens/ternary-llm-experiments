"""flip_distill.py — Stage 0 ternary-Bop QAT distillation.

Latent-free, flip-based ternary QAT against the cached teacher's top-K + rest
distribution. Each QLinear's `weight` stores discrete trits `t ∈ {-1, 0, +1}`
directly (not a latent in [-1, 1]). A per-element EMA `m` of the trit-space
gradient `g_t = s_g · dL/dW` decides flips: when `|m| > tau` and the desired
direction is a valid transition, the trit moves one step toward 0 or one rail.

This is Stage 0 from `ternary_laf_flipping_spec`: faithful Bop, extended to
the ternary state set by a rail clamp on `m`. No second moment, no reset,
no refractory, no throttle, no learned scale. See `flip_research.md` for
the ranked research bets to layer on top of this baseline.

Forward path: every QLinear runs in `mode="levels"` with `levels=3`. The
existing `quantize_levels(w, 3) = round(clamp(w, -1, 1))` rounds {-1,0,+1}
to themselves, the STE residual is zero, and `weight.grad` ends up equal to
`scales_broadcast · dL/dW` — which is exactly the spec's `g_t`. We mutate
trits in `BopTernary.step()` outside autograd and invalidate the cache.

Scales: per spec, **not learned** in Stage 0. Initialized by iterative
least-squares against the teacher's pre-quant projection weights `W_ref`
(stored once at fresh start in `w_ref.safetensors`), then optionally
recomputed every `--scale-recompute-every` steps using the same closed-form
`s_g = <W_ref, t> / <t, t>`.

Non-QLinear params (embeddings, norms, biases) are frozen. The only state
that moves in training is the trit pattern. Add an AdamW for them as a
future bet if the floor needs more.
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .build_student import load_student
from .distill import (
    BestEmaTracker, ShardedDataset, _INTERRUPT, _install_sigint_handler,
    kl_with_rest, save_checkpoint, save_resume, snapshot_to_cpu,
)
from .qlinear import QLinear, set_levels
from .teacher_floor import load_or_compute as load_teacher_floor


# ---------------------------------------------------------------- optimizer


class BopTernary(torch.optim.Optimizer):
    """Bop-style flip optimizer for ternary trits in {-1, 0, +1}.

    Stage 0: per parameter state is `m` — fp32 EMA of the trit-space
    gradient (the weight's `.grad`, which is already `s_g · dL/dW` after
    QLinear's STE). Flip when `|m| > tau` and the direction is valid.

    Bet 1 (use_2nd_moment=True): also maintain `v` = EMA(g_t²), and use
    `|m| / (sqrt(v) + eps) > tau_norm` as the criterion. The ratio is
    unitless and roughly "signal-to-noise" per trit — `tau_norm = 1` means
    "EMA magnitude exceeds the per-trit gradient rms". Much more robust
    near a smooth-QAT optimum where the raw |m| distribution is
    dominated by noise outliers.

    `step()` returns (n_flips, n_elems) summed across all param groups.
    """

    def __init__(self, params, gamma: float, tau: float,
                 use_2nd_moment: bool = False,
                 gamma_v: float | None = None,
                 tau_norm: float = 1.0,
                 eps: float = 1e-12,
                 reset_on_flip: bool = False,
                 refractory: int = 0,
                 cautious: bool = False) -> None:
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        if tau <= 0.0:
            raise ValueError(f"tau must be > 0, got {tau}")
        if use_2nd_moment:
            if gamma_v is None:
                gamma_v = gamma
            if not (0.0 < gamma_v <= 1.0):
                raise ValueError(f"gamma_v must be in (0, 1], got {gamma_v}")
            if tau_norm <= 0.0:
                raise ValueError(f"tau_norm must be > 0, got {tau_norm}")
        if refractory < 0:
            raise ValueError(f"refractory must be >= 0, got {refractory}")
        defaults = dict(gamma=float(gamma), tau=float(tau),
                        use_2nd_moment=bool(use_2nd_moment),
                        gamma_v=float(gamma_v) if gamma_v is not None else float(gamma),
                        tau_norm=float(tau_norm), eps=float(eps),
                        reset_on_flip=bool(reset_on_flip),
                        refractory=int(refractory),
                        cautious=bool(cautious))
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> tuple[int, int]:
        n_flips = 0
        n_elems = 0
        for group in self.param_groups:
            gamma = float(group["gamma"])
            tau = float(group["tau"])
            use_v = bool(group["use_2nd_moment"])
            gamma_v = float(group["gamma_v"])
            tau_norm = float(group["tau_norm"])
            eps = float(group["eps"])
            reset_on_flip = bool(group["reset_on_flip"])
            refractory = int(group["refractory"])
            cautious = bool(group.get("cautious", False))
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p.data, dtype=torch.float32)
                m = state["m"]
                g_t = p.grad.detach().float()

                # Bop EMA: m ← (1-γ)·m + γ·g_t
                m.mul_(1.0 - gamma).add_(g_t, alpha=gamma)

                # Bet 1: second-moment EMA, v ← (1-γ_v)·v + γ_v·g_t²
                if use_v:
                    if "v" not in state:
                        state["v"] = torch.zeros_like(p.data, dtype=torch.float32)
                    v = state["v"]
                    v.mul_(1.0 - gamma_v).addcmul_(g_t, g_t, value=gamma_v)

                # Rail clamp (ternary-specific, no Bop precedent): at a rail,
                # zero the off-rail side so a reversed gradient need not be
                # unwound first.
                t = p.data
                pos_rail = t > 0.5
                neg_rail = t < -0.5
                m[pos_rail & (m < 0)] = 0.0
                m[neg_rail & (m > 0)] = 0.0

                # Flip rule.
                #   Stage 0: |m| > tau
                #   Bet 1  : |m| / (sqrt(v) + eps) > tau_norm
                # AND direction is a valid transition:
                #   t == 0 → either direction valid
                #   t == +1 → only direction = -1 valid
                #   t == -1 → only direction = +1 valid
                direction = -m.sign()
                at_zero = t.abs() < 0.5
                valid = (
                    at_zero
                    | (pos_rail & (direction < 0))
                    | (neg_rail & (direction > 0))
                )
                if use_v:
                    score = m.abs() / (state["v"].sqrt() + eps)
                    flip = (score > tau_norm) & valid
                else:
                    flip = (m.abs() > tau) & valid

                # Cautious (Liang et al. 2024, arXiv:2411.16085, applied to
                # Bop): only flip when this step's gradient direction agrees
                # with the EMA's direction, i.e. the current g_t reinforces
                # the accumulated signal. `direction = -sign(m)`; want
                # direction == -sign(g_t), i.e. sign(m) == sign(g_t), i.e.
                # m·g_t > 0. Filters trits where m has saturated but the
                # current step disagrees — exactly the oscillation regime.
                if cautious:
                    flip = flip & ((m * g_t) > 0)

                # Bet 5: refractory lockout. Decrement counter every step;
                # while > 0, the trit can't flip. Applied BEFORE candidate
                # selection so locked trits never enter the flip set.
                if refractory > 0:
                    if "lockout" not in state:
                        state["lockout"] = torch.zeros_like(
                            p.data, dtype=torch.int16)
                    lock = state["lockout"]
                    lock.clamp_min_(0).sub_(1).clamp_min_(0)
                    flip = flip & (lock == 0)

                # Apply: each flipped trit moves by `direction` (±1). Lands
                # exactly in {-1, 0, +1} given the validity mask.
                t.add_(flip.to(t.dtype) * direction.to(t.dtype))

                # Bet 5: reset-on-flip. Zero m (and v) for trits that just
                # flipped, so the same accumulated EMA can't immediately
                # push another flip ("rail-to-rail sweep" / oscillation).
                if reset_on_flip:
                    m[flip] = 0.0
                    if use_v:
                        state["v"][flip] = 0.0
                if refractory > 0:
                    state["lockout"][flip] = refractory

                n_flips += int(flip.sum().item())
                n_elems += t.numel()
        return n_flips, n_elems


# ---------------------------------------------------------------- W_ref


@torch.no_grad()
def capture_w_ref(model: nn.Module) -> dict[str, torch.Tensor]:
    """Reconstruct pre-quant teacher projection weights as `weight · scales`.

    Right after `load_student`, each QLinear's `weight` is the teacher's
    weight divided by `scales = abs_max_per_group`. So `weight * scales`
    (broadcast across each group) recovers the (possibly permuted) teacher
    weight exactly up to fp roundoff. Captured fp32 on CPU.
    """
    refs: dict[str, torch.Tensor] = {}
    for name, m in model.named_modules():
        if not isinstance(m, QLinear):
            continue
        w = m.weight.detach().float()
        out_f, in_f = w.shape
        s = m.scales.detach().float()
        s_full = (s.unsqueeze(-1)
                  .expand(out_f, m.n_groups, m.group_size)
                  .reshape(out_f, in_f))
        refs[name] = (w * s_full).cpu().contiguous()
    return refs


def save_w_ref(refs: dict[str, torch.Tensor], path: Path) -> None:
    save_file(refs, str(path))


def load_w_ref(path: Path) -> dict[str, torch.Tensor]:
    return load_file(str(path))


# ---------------------------------------------------------------- init


@torch.no_grad()
def init_trits_and_scales_ls(model: nn.Module,
                             w_refs: dict[str, torch.Tensor],
                             n_iters: int = 5) -> tuple[int, float]:
    """Coordinate-descent init of (trits, scales) for each QLinear.

    Starts from per-group abs-max scale, then alternates:
      t = snap(W_ref / s, threshold=0.5)   # round to {-1, 0, +1}
      s = <W_ref, t> / <t, t>              # closed-form per group

    A few iterations converge to a local minimum of the per-group SSE.
    Returns (n_layers, mean_nonzero_fraction).
    """
    n_layers = 0
    nz_frac_sum = 0.0
    nz_groups = 0
    for name, m in model.named_modules():
        if not isinstance(m, QLinear):
            continue
        W_ref = w_refs[name].to(m.weight.device, dtype=torch.float32)
        out_f, in_f = W_ref.shape
        W_blk = W_ref.view(out_f, m.n_groups, m.group_size)
        # Seed scale from per-group abs-max.
        s = W_blk.abs().amax(-1).clamp_min(1e-8)  # [out, n_groups]
        t = torch.zeros_like(W_blk)
        for _ in range(n_iters):
            r = W_blk / s.unsqueeze(-1)
            t = torch.where(r.abs() > 0.5,
                            torch.sign(r),
                            torch.zeros_like(r))
            denom = (t * t).sum(-1)                    # = nonzero count
            numer = (W_blk * t).sum(-1)
            has_nz = denom > 0
            s = torch.where(has_nz, numer / denom.clamp_min(1.0), s)
            s = s.clamp_min(1e-8)
        m.weight.data.copy_(t.view(out_f, in_f).to(m.weight.dtype))
        m.scales.data.copy_(s.to(m.scales.dtype))
        m.invalidate_q_cache()
        nz_frac_sum += float((t != 0).float().mean().item())
        nz_groups += 1
        n_layers += 1
    mean_nz = nz_frac_sum / max(1, nz_groups)
    return n_layers, mean_nz


@torch.no_grad()
def recompute_scales_ls(model: nn.Module,
                        w_refs: dict[str, torch.Tensor]) -> int:
    """Recompute `s_g = <W_ref, t> / <t, t>` for every QLinear group.
    Groups with all-zero trits keep their previous scale."""
    n = 0
    for name, m in model.named_modules():
        if not isinstance(m, QLinear):
            continue
        W_ref = w_refs[name].to(m.weight.device, dtype=torch.float32)
        out_f, in_f = W_ref.shape
        t = m.weight.data.float().view(out_f, m.n_groups, m.group_size)
        W_blk = W_ref.view(out_f, m.n_groups, m.group_size)
        denom = (t * t).sum(-1)
        numer = (W_blk * t).sum(-1)
        has_nz = denom > 0
        s_new = numer / denom.clamp_min(1.0)
        s_new = s_new.clamp_min(1e-8)
        m.scales.data.copy_(torch.where(
            has_nz, s_new.to(m.scales.dtype), m.scales.data))
        m.invalidate_q_cache()
        n += 1
    return n


# ---------------------------------------------------------------- helpers


def freeze_non_trit_params(model: nn.Module) -> tuple[int, int]:
    """Freeze every parameter except QLinear `weight` (trits). Scales are
    also frozen — Stage 0 keeps them static. Returns (n_trainable, n_frozen)."""
    trit_ids = {id(m.weight) for m in model.modules() if isinstance(m, QLinear)}
    n_t, n_f = 0, 0
    for p in model.parameters():
        if id(p) in trit_ids:
            p.requires_grad_(True)
            n_t += 1
        else:
            p.requires_grad_(False)
            n_f += 1
    return n_t, n_f


def invalidate_all_q_caches(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, QLinear):
            m.invalidate_q_cache()


@torch.no_grad()
def trit_stats(model: nn.Module) -> dict[str, float]:
    """Aggregate {fraction at 0, fraction at +1, fraction at -1, total}."""
    n = 0
    z = pos = neg = 0
    for m in model.modules():
        if not isinstance(m, QLinear):
            continue
        t = m.weight.data
        n += t.numel()
        z += int((t == 0).sum().item())
        pos += int((t > 0.5).sum().item())
        neg += int((t < -0.5).sum().item())
    n = max(1, n)
    return {"frac_zero": z / n, "frac_pos": pos / n, "frac_neg": neg / n,
            "n_trits": n}


@torch.no_grad()
def m_stats(opt: BopTernary) -> dict[str, float]:
    """RMS and max|m| over all trits, for tuning tau. If 2nd-moment is
    enabled, also reports rms / max of the score = |m|/sqrt(v)."""
    sumsq = 0.0
    mx = 0.0
    n = 0
    score_sumsq = 0.0
    score_max = 0.0
    score_n = 0
    use_v = False
    for group in opt.param_groups:
        if bool(group.get("use_2nd_moment", False)):
            use_v = True
            eps = float(group["eps"])
        for p in group["params"]:
            st = opt.state.get(p)
            if not st or "m" not in st:
                continue
            m = st["m"]
            sumsq += float(m.float().pow(2).sum().item())
            mx = max(mx, float(m.abs().max().item()))
            n += m.numel()
            if use_v and "v" in st:
                v = st["v"]
                score = m.abs() / (v.sqrt() + eps)
                score_sumsq += float(score.pow(2).sum().item())
                score_max = max(score_max, float(score.max().item()))
                score_n += score.numel()
    out: dict[str, float] = {}
    out["m_rms"] = (sumsq / n) ** 0.5 if n else 0.0
    out["m_max"] = mx
    if use_v and score_n:
        out["score_rms"] = (score_sumsq / score_n) ** 0.5
        out["score_max"] = score_max
    return out


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None,
                    help="Warm-start weights only from a .safetensors. Fresh "
                         "optimizer (zero EMA) and fresh W_ref are derived.")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    # Bop hyperparams
    ap.add_argument("--gamma", type=float, default=1e-3,
                    help="EMA rate for the trit-space gradient (Bop γ). "
                         "Bop CIFAR used 1e-3 to 1e-5. Default 1e-3.")
    ap.add_argument("--tau", type=float, default=1e-6,
                    help="Flip threshold (Bop τ). Bop CIFAR used 1e-6 to "
                         "1e-8. Tune by watching `m/rms` in TB: τ should be "
                         "well below `m/max` but above `m/rms` so only "
                         "outlier-EMA trits flip per step. Default 1e-6. "
                         "Ignored if --use-2nd-moment is set.")
    # Bet 1: Bop2ndOrder
    ap.add_argument("--use-2nd-moment", action="store_true", default=False,
                    help="Bet 1 (Bop2ndOrder): also maintain v=EMA(g_t²) and "
                         "compare |m|/sqrt(v) vs --tau-norm. Makes the "
                         "criterion unitless and scale-invariant — required "
                         "when refining a near-optimal checkpoint where "
                         "raw |m| outliers are noise, not signal.")
    ap.add_argument("--tau-norm", type=float, default=1.0,
                    help="Normalized threshold for Bet 1: flip when "
                         "|m|/sqrt(v) > tau_norm. =1 means EMA exceeds "
                         "per-trit gradient rms (1-σ); =3 is 3-σ. Default 1.")
    ap.add_argument("--gamma-v", type=float, default=None,
                    help="EMA rate for v (defaults to --gamma).")
    ap.add_argument("--eps", type=float, default=1e-12,
                    help="Epsilon for sqrt(v)+eps in Bet 1.")
    # Bet 5: reset-on-flip / refractory
    ap.add_argument("--reset-on-flip", action="store_true", default=False,
                    help="Bet 5: zero m (and v, if 2nd-moment) for trits "
                         "that just flipped. Suppresses the rail-to-rail "
                         "sweep and back-and-forth oscillation that come "
                         "from Bop's no-reset rule. Required for stable "
                         "refinement near an optimum.")
    ap.add_argument("--refractory", type=int, default=0,
                    help="Bet 5: lockout period in steps. Flipped trits "
                         "cannot flip again for this many steps. 0 = off. "
                         "Use with --reset-on-flip when EMA window 1/γ is "
                         "still short enough for new noise to push another "
                         "flip too soon.")
    # Scale recompute
    ap.add_argument("--scale-init-iters", type=int, default=5,
                    help="LS init iterations for (t, s) at fresh start.")
    ap.add_argument("--scale-recompute-every", type=int, default=200,
                    help="Recompute s_g = <W_ref, t>/<t, t> every N steps. "
                         "Set to 0 to freeze scales after init.")
    # Misc
    ap.add_argument("--max-grad-norm", type=float, default=0.0,
                    help="Clip global grad norm before forming g_t. 0 = off "
                         "(Bop-faithful). Enable cautiously — clipping changes "
                         "the m magnitude distribution and re-tuning τ.")
    ap.add_argument("--permute", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--autocast-dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "none"])
    ap.add_argument("--latent-dtype", default="float32",
                    choices=["float32", "float16", "bfloat16"],
                    help="Storage dtype for trits. Spec recommends fp32 for "
                         "first correctness; switch to fp16/int8 after Stage "
                         "0 is stable.")
    ap.add_argument("--scale-group-size", type=int, default=64)
    ap.add_argument("--grad-checkpointing",
                    action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total-steps", type=int, default=40000)
    ap.add_argument("--ema-warmup", type=int, default=500)
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    ap.add_argument("--tb-dir", type=Path, default=None)
    ap.add_argument("--run-name", type=str, default=None)

    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    _install_sigint_handler()

    latent_dtype = {"float32": torch.float32, "float16": torch.float16,
                    "bfloat16": torch.bfloat16}[args.latent_dtype]
    autocast_dtype = {"bfloat16": torch.bfloat16,
                      "float16": torch.float16,
                      "none": None}[args.autocast_dtype]

    interrupted_path = args.out / "interrupted.pt"
    w_ref_path = args.out / "w_ref.safetensors"
    fresh_start = (args.resume is None and not interrupted_path.exists())
    do_permute = args.permute and fresh_start

    # ---- Build student ----
    print(f"[build] loading {args.model}, "
          f"group_size={args.scale_group_size}, permute={do_permute}")
    model, _, n_replaced = load_student(
        args.model, dtype=torch.float32, levels=3,
        latent_dtype=latent_dtype, group_size=args.scale_group_size,
        permute=do_permute,
    )
    model.to(args.device)
    set_levels(model, 3)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        print("[build] gradient checkpointing enabled")
    print(f"[build] {n_replaced} QLinear modules (latent dtype {latent_dtype})")

    # ---- W_ref: capture on fresh start, load on resume ----
    if fresh_start:
        w_refs = capture_w_ref(model)
        save_w_ref(w_refs, w_ref_path)
        print(f"[init] captured W_ref from teacher init, saved → {w_ref_path}")
    else:
        if not w_ref_path.exists():
            print(f"[init] {w_ref_path} missing; reconstructing from a fresh "
                  f"teacher load (only safe if --permute matches the original "
                  f"run).", flush=True)
            ref_model, _, _ = load_student(
                args.model, dtype=torch.float32, levels=3,
                latent_dtype=torch.float32,
                group_size=args.scale_group_size, permute=args.permute,
            )
            w_refs = capture_w_ref(ref_model)
            save_w_ref(w_refs, w_ref_path)
            del ref_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            w_refs = load_w_ref(w_ref_path)
            print(f"[init] loaded W_ref ({len(w_refs)} tensors) ← {w_ref_path}")

    # ---- Freeze everything except trits ----
    n_t, n_f = freeze_non_trit_params(model)
    print(f"[freeze] trainable trit-tensors={n_t}, frozen={n_f}")

    # ---- Init trits + scales via iterative LS (fresh start only) ----
    if fresh_start:
        n_layers, mean_nz = init_trits_and_scales_ls(
            model, w_refs, n_iters=args.scale_init_iters)
        print(f"[init] LS init: {n_layers} QLinears, mean nonzero frac="
              f"{mean_nz:.3f}")

    # ---- Optimizer ----
    trit_params = [m.weight for m in model.modules() if isinstance(m, QLinear)]
    opt = BopTernary(trit_params, gamma=args.gamma, tau=args.tau,
                     use_2nd_moment=args.use_2nd_moment,
                     gamma_v=args.gamma_v, tau_norm=args.tau_norm,
                     eps=args.eps,
                     reset_on_flip=args.reset_on_flip,
                     refractory=args.refractory)
    bets = []
    if args.use_2nd_moment:
        bets.append("Bet1[2ndOrder]")
    if args.reset_on_flip:
        bets.append("Bet5[reset]")
    if args.refractory > 0:
        bets.append(f"Bet5[refractory={args.refractory}]")
    bet_tag = "+".join(bets) if bets else "Stage0"
    if args.use_2nd_moment:
        gv = args.gamma_v if args.gamma_v is not None else args.gamma
        print(f"[opt] BopTernary [{bet_tag}] γ={args.gamma:g} γ_v={gv:g} "
              f"τ_norm={args.tau_norm:g}")
    else:
        print(f"[opt] BopTernary [{bet_tag}] γ={args.gamma:g} τ={args.tau:g}")

    # ---- Resume ----
    interrupted_state = None
    global_step = 0
    samples_consumed = 0

    if args.resume is not None:
        with safe_open(str(args.resume), framework="pt") as f:
            resume_meta = f.metadata() or {}
        sd = load_file(str(args.resume))
        miss, unexp = model.load_state_dict(sd, strict=False)
        invalidate_all_q_caches(model)
        print(f"[resume] warm-start from {args.resume.name} "
              f"(meta={resume_meta}, missing={len(miss)}, unexpected={len(unexp)})")
    elif interrupted_path.exists():
        interrupted_state = torch.load(str(interrupted_path),
                                       map_location="cpu",
                                       weights_only=False)
        model.load_state_dict(interrupted_state["model"], strict=False)
        del interrupted_state["model"]
        opt.load_state_dict(interrupted_state["opt"])
        del interrupted_state["opt"]
        global_step = int(interrupted_state.get("next_step", 0))
        samples_consumed = int(interrupted_state.get(
            "samples_consumed",
            global_step * args.grad_accum * args.batch_size))
        invalidate_all_q_caches(model)
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[resume] {interrupted_path} at step {global_step}")

    # ---- Teacher floor, data, TB ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    floor_data = load_teacher_floor(args.cache_dir, len(tok))
    L_T = float(floor_data["floor"])
    print(f"[flip] L_T = {L_T:.4f}")

    def _worker_init(_worker_id: int) -> None:
        import signal as _sig
        _sig.signal(_sig.SIGINT, _sig.SIG_IGN)

    ds = ShardedDataset(args.cache_dir, seed=args.seed,
                        start_skip=samples_consumed)
    dl = DataLoader(ds, batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=args.device.startswith("cuda"),
                    drop_last=True, worker_init_fn=_worker_init)
    it = iter(dl)

    tb_root = args.tb_dir if args.tb_dir is not None else (args.out / "tb")
    if interrupted_state and interrupted_state.get("run_name"):
        run_name = interrupted_state["run_name"]
    elif args.run_name:
        run_name = args.run_name
    else:
        run_name = datetime.now().strftime("flip_%Y%m%d_%H%M%S")
    run_dir = tb_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir),
                           purge_step=global_step if global_step else None)
    print(f"[tb] {run_dir}")
    writer.add_text("stage", "  \n".join([
        "**flip_distill** Stage 0 — ternary-Bop, no latent",
        f"- γ = {args.gamma:g}, τ = {args.tau:g}",
        f"- scale_init_iters = {args.scale_init_iters}, "
        f"recompute_every = {args.scale_recompute_every}",
        f"- group_size = {args.scale_group_size}",
        f"- max_grad_norm = {args.max_grad_norm} (0 = off)",
        f"- L_T = {L_T:.4f}",
    ]), global_step)

    ctrl = BestEmaTracker(ema_alpha=0.05, ema_warmup=args.ema_warmup)
    if interrupted_state and interrupted_state.get("ctrl_state"):
        ctrl.load_state_dict(interrupted_state["ctrl_state"])
    best_snapshot = (interrupted_state.get("best_snapshot")
                     if interrupted_state else None)
    interrupted_state = None

    # ---- Train ----
    running = 0.0
    running_n = 0
    flips_window = 0
    elems_window = 0
    pbar = tqdm(desc="flip", dynamic_ncols=True,
                initial=global_step, total=args.total_steps)
    opt.zero_grad(set_to_none=True)
    model.train()

    try:
        while global_step < args.total_steps:
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
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"non-finite loss at step {global_step}: {loss.item()}")
                (loss / args.grad_accum).backward()
                running += loss.item()
                running_n += 1

            grad_norm = None
            if args.max_grad_norm and args.max_grad_norm > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.max_grad_norm))

            # Flip step (manages its own state; no LR).
            n_flips, n_elems = opt.step()
            flips_window += n_flips
            elems_window += n_elems
            # Cache invalidation: every trit may have changed.
            invalidate_all_q_caches(model)
            opt.zero_grad(set_to_none=True)
            global_step += 1

            step_loss = running / max(1, running_n)
            improved = ctrl.update(global_step, step_loss)
            if improved:
                best_snapshot = snapshot_to_cpu(model)

            # Periodic scale recompute (Stage 0: closed-form LS vs W_ref).
            if (args.scale_recompute_every > 0
                    and global_step % args.scale_recompute_every == 0):
                recompute_scales_ls(model, w_refs)

            if global_step % args.log_every == 0:
                rate = flips_window / max(1, elems_window)
                postfix = {
                    "step": global_step,
                    "loss": f"{step_loss:.4f}",
                    "ema": f"{ctrl.ema:.4f}" if ctrl.ema else "—",
                    "flip%": f"{rate * 100:.3f}",
                }
                pbar.set_postfix(postfix)
                pbar.update(args.log_every)
                writer.add_scalar("loss/step", step_loss, global_step)
                if ctrl.ema is not None:
                    writer.add_scalar("loss/ema", ctrl.ema, global_step)
                    writer.add_scalar("loss/gap", ctrl.ema - L_T, global_step)
                writer.add_scalar("flip/rate", rate, global_step)
                writer.add_scalar("flip/count", flips_window, global_step)
                if grad_norm is not None:
                    writer.add_scalar("grad_norm", grad_norm, global_step)

                ms = m_stats(opt)
                writer.add_scalar("m/rms", ms["m_rms"], global_step)
                writer.add_scalar("m/max", ms["m_max"], global_step)
                writer.add_scalar("m/tau_ratio_rms",
                                  ms["m_rms"] / max(1e-30, args.tau),
                                  global_step)
                if "score_rms" in ms:
                    writer.add_scalar("score/rms", ms["score_rms"],
                                      global_step)
                    writer.add_scalar("score/max", ms["score_max"],
                                      global_step)

                ts = trit_stats(model)
                writer.add_scalar("trits/frac_zero", ts["frac_zero"],
                                  global_step)
                writer.add_scalar("trits/frac_pos", ts["frac_pos"],
                                  global_step)
                writer.add_scalar("trits/frac_neg", ts["frac_neg"],
                                  global_step)

                with torch.no_grad():
                    all_s = torch.cat([m.scales.data.flatten()
                                       for m in model.modules()
                                       if isinstance(m, QLinear)])
                    writer.add_scalar("s/mean", float(all_s.mean()), global_step)
                    writer.add_scalar("s/min", float(all_s.min()), global_step)
                    writer.add_scalar("s/max", float(all_s.max()), global_step)
                    writer.add_scalar("s/p50", float(all_s.median()), global_step)

                running = 0.0
                running_n = 0
                flips_window = 0
                elems_window = 0

            samples_at_save = global_step * args.grad_accum * args.batch_size
            if (args.checkpoint_every > 0
                    and global_step % args.checkpoint_every == 0):
                save_resume(interrupted_path, model, opt, global_step,
                            best_snapshot, ctrl.state_dict(), run_name,
                            samples_consumed=samples_at_save,
                            soft_state=None)
                tqdm.write(f"[ckpt] {interrupted_path} @ step {global_step}")

            if _INTERRUPT["flag"]:
                save_resume(interrupted_path, model, opt, global_step,
                            best_snapshot, ctrl.state_dict(), run_name,
                            samples_consumed=samples_at_save,
                            soft_state=None)
                writer.flush()
                writer.close()
                pbar.close()
                print(f"[!] saved {interrupted_path}")
                sys.exit(0)

    except SystemExit:
        raise
    except BaseException as e:
        try:
            samples_at_save = global_step * args.grad_accum * args.batch_size
            save_resume(interrupted_path, model, opt, global_step,
                        best_snapshot, ctrl.state_dict(), run_name,
                        samples_consumed=samples_at_save,
                        soft_state=None)
            print(f"[!] emergency save → {interrupted_path} "
                  f"(reason: {type(e).__name__})", flush=True)
        except Exception as save_err:
            print(f"[!!] emergency save failed: {save_err}", flush=True)
        raise
    finally:
        pbar.close()

    # ---- Deploy fold: trits are already exactly {-1,0,+1}; scales absorb
    # no codepoint_c here (we never attached one). Just save the safetensors.
    print(f"[flip] complete after {global_step} steps")
    out_ckpt = args.out / "stage_flip.safetensors"
    save_checkpoint(model, out_ckpt, args.model, args.scale_group_size,
                    alpha=0.0, target_zero_frac=None)
    print(f"[flip] saved {out_ckpt}")

    if best_snapshot is not None and ctrl.best_step != global_step:
        model.load_state_dict(best_snapshot, strict=False)
        invalidate_all_q_caches(model)
        best_ckpt = args.out / "stage_flip_best.safetensors"
        save_checkpoint(model, best_ckpt, args.model, args.scale_group_size,
                        alpha=0.0, target_zero_frac=None)
        print(f"[flip] saved best-snapshot → {best_ckpt} "
              f"(step {ctrl.best_step}, EMA {ctrl.best_ema:.4f})")

    writer.add_text("stage_end",
                    f"flip_distill complete: {global_step} steps",
                    global_step)
    writer.flush()
    writer.close()
    if interrupted_path.exists():
        interrupted_path.unlink()
    print("[done]")


if __name__ == "__main__":
    main()
