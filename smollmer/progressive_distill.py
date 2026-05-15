"""Progressive ternary clamping (alternative to distill.py's α-anneal).

Train SmolLM2 with a soft forward (no quantizer; identity) and progressively
*commit* one weight per (row, column-group) at a time to its nearest
leeway-codebook point in {-c, 0, +c}. Once committed:
  - the latent is forced back to its target value after every opt.step
  - the slot's gradient is zeroed pre-step
  - the slot's optimizer momentum (exp_avg / exp_avg_sq) is wiped at commit

After `group_size` rounds every weight is committed; a math-preserving 1/c
rescale folds c into the per-(row, group) scales so the deployed codebook is
{-1, 0, +1} — identical to the soft-stage output of distill.py.

The selection criterion per group is the weight minimizing
    |q_error(w)|  +  λ_m · |momentum(w)| / median(|momentum|)
where q_error is the distance to the nearest of {-c, 0, +c}. The momentum
term avoids committing weights the optimizer is actively trying to move.

Reuses scaffolding (data, KL loss, optimizers, lr schedule, BestEmaTracker,
checkpoint I/O, sigint, metrics) from distill.py.
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .build_student import load_student
from .distill import (
    _INTERRUPT,
    AdamW32,
    BestEmaTracker,
    CautiousAdamW,
    Lion32,
    ShardedDataset,
    _install_sigint_handler,
    collect_qlinear_metrics,
    collect_soft_metrics,
    embed_drift_l2,
    first_qlinear_forward_sample,
    kl_with_rest,
    lr_at,
    save_checkpoint,
    save_resume,
    snapshot_to_cpu,
)
from .qlinear import QLinear, clamp_qlinear_weights, set_soft_mode
from .teacher_floor import load_or_compute as load_teacher_floor


# ============================================================================
# Per-QLinear progressive state (progressive QAT)
# ============================================================================
# Two persistent buffers per QLinear (so they ride along in state_dict and
# resume Just Works):
#   qat_mask    : bool[out, in]            — True for promoted (ternarized) slots
#   codepoint_c : fp32[out, n_groups]      — codepoint magnitude per (row, group)
#
# Promoted slots use STE-ternarization in the forward: the effective weight at
# slot (r, c) is sign(w)·c_{r,g} when |w| > c_{r,g}/2, else 0 — but the
# gradient still flows back to the latent as identity. The latent can drift,
# and as it crosses c_{r,g}/2 the ternary value flips. This is the key
# difference from the prior "freeze" design: promotion is not a one-way trap.
#
# c_{r,g} is data-driven at init (mean of |w| over the band per group), which
# minimizes commit-time q_err on average — band weights are centered on the
# codepoint, not crowded against the |w|=1 edge.

def attach_progressive_buffers(model: torch.nn.Module,
                               default_c: float = 2.0 / 3.0) -> int:
    n = 0
    for m in model.modules():
        if not isinstance(m, QLinear):
            continue
        if not hasattr(m, "qat_mask"):
            m.register_buffer(
                "qat_mask",
                torch.zeros_like(m.weight, dtype=torch.bool),
                persistent=True,
            )
        if not hasattr(m, "codepoint_c"):
            out_f = m.weight.shape[0]
            m.register_buffer(
                "codepoint_c",
                torch.full((out_f, m.n_groups), default_c,
                           dtype=torch.float32, device=m.weight.device),
                persistent=True,
            )
        n += 1
    return n


@torch.no_grad()
def compute_per_group_c(model: torch.nn.Module,
                        target_zero_frac: float | None,
                        fallback_c: float) -> dict[str, float]:
    """Compute c_{r,g} = mean(|w| over band) per (row, group), where the band
    is defined by |w| > quantile(target_zero_frac). Writes into m.codepoint_c.
    Returns aggregate stats."""
    all_c = []
    for m in model.modules():
        if not isinstance(m, QLinear):
            continue
        out_f = m.weight.shape[0]
        gs, ng = m.group_size, m.n_groups
        w = m.weight.detach().float()
        abs_wb = w.view(out_f, ng, gs).abs()
        if target_zero_frac is not None and 0.0 < target_zero_frac < 1.0:
            cutoff = abs_wb.quantile(float(target_zero_frac),
                                     dim=-1, keepdim=True)
            is_band = abs_wb > cutoff
        else:
            is_band = abs_wb >= 0.5
        band_sum = (abs_wb * is_band.float()).sum(dim=-1)        # [out, ng]
        band_count = is_band.float().sum(dim=-1)                  # [out, ng]
        c_rg = band_sum / band_count.clamp_min(1.0)
        c_rg = torch.where(band_count > 0, c_rg,
                           torch.full_like(c_rg, fallback_c))
        m.codepoint_c.copy_(c_rg.to(m.codepoint_c.dtype))
        all_c.append(c_rg.flatten())
    if not all_c:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0}
    cat = torch.cat(all_c)
    return {
        "mean": float(cat.mean()),
        "min": float(cat.min()),
        "max": float(cat.max()),
        "p50": float(cat.median()),
    }


def first_qlinear_qat_sample(
    model: torch.nn.Module,
) -> tuple[str, torch.Tensor] | None:
    """Like distill.first_qlinear_forward_sample but uses the QAT-effective
    weight: STE-ternary (±c_{r,g} or 0) at promoted slots, latent
    elsewhere. The histogram of this is much more interpretable than the
    raw latent — promoted weights show up as spikes near ±c_{r,g} and 0,
    so progress through the schedule is visible as the spikes growing."""
    for name, m in model.named_modules():
        if not isinstance(m, QLinear):
            continue
        if hasattr(m, "qat_mask") and hasattr(m, "codepoint_c"):
            qw = m._qat_effective_weight()
        else:
            qw = m.quantized_weight()
        return name, qw.detach().flatten().to("cpu")
    return None


def invalidate_c_elem_cache(model: torch.nn.Module) -> None:
    """Force QLinears to rebuild their `_c_elem` cache (codepoint_c
    broadcast to weight shape) on next forward. Call after init's
    compute_per_group_c, or after a state_dict load that touched
    codepoint_c — values changed but shape/dtype didn't, so the lazy
    cache check inside QLinear.forward wouldn't otherwise rebuild."""
    for m in model.modules():
        if not isinstance(m, QLinear):
            continue
        if hasattr(m, "_c_elem"):
            try:
                del m._c_elem
            except AttributeError:
                pass


# ============================================================================
# Selection + promotion (progressive QAT)
# ============================================================================

@torch.no_grad()
def compute_commit_targets(m: QLinear,
                           target_zero_frac: float | None) -> torch.Tensor:
    """Per-element ternary target in latent space ∈ {-c_{r,g}, 0, +c_{r,g}}.
    Used by selection (find weight nearest its target) and by the QAT
    forward (forward-with-STE at promoted slots).

    target_zero_frac=None → fixed 0.5 boundary on |w|.
    target_zero_frac ∈ (0, 1) → per-(row, group) |w|-quantile cutoff.
    Below cutoff → 0, above → sign(w)·c_{r,g}.
    """
    out_f = m.weight.shape[0]
    gs, ng = m.group_size, m.n_groups
    w = m.weight.detach().float()
    wb = w.view(out_f, ng, gs)
    abs_wb = wb.abs()
    if target_zero_frac is not None and 0.0 < target_zero_frac < 1.0:
        cutoff = abs_wb.quantile(float(target_zero_frac),
                                 dim=-1, keepdim=True)
        is_zero = abs_wb <= cutoff
    else:
        is_zero = abs_wb < 0.5
    c_b = m.codepoint_c.unsqueeze(-1).expand(out_f, ng, gs).float()
    target_b = torch.where(is_zero, torch.zeros_like(wb),
                           torch.sign(wb) * c_b)
    return target_b.view(out_f, -1).to(m.weight.dtype)


@torch.no_grad()
def select_and_promote_one_per_group(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    momentum_weight: float,
    target_zero_frac: float | None,
) -> tuple[int, dict[str, float]]:
    """For each (row, group) that still has at least one un-promoted weight,
    pick the weight minimizing
        |w − target(w)|  +  λ_m · |exp_avg| / median(|exp_avg|)
    and add it to qat_mask. The latent is NOT snapped: from now on its
    forward goes through STE-ternarization (forward = target, backward
    = identity) so the latent retains gradient flow. As the latent
    drifts across c_{r,g}/2, the ternary value can flip.

    Optimizer momentum at the promoted slot is zeroed: the slot's
    gradient is about to be redirected through STE; old momentum from
    pre-promotion is no longer meaningful.

    Returns (n_promoted, stats). One promotion per non-fully-promoted
    group per call.
    """
    state = opt.state
    n_promoted = 0
    sum_q_err = 0.0
    sum_q_err_count = 0
    n_neg = n_zero_ = n_pos = 0
    for m in model.modules():
        if not isinstance(m, QLinear):
            continue
        out_f, in_f = m.weight.shape
        gs, ng = m.group_size, m.n_groups
        w = m.weight.detach().float()
        fm = m.qat_mask
        target = compute_commit_targets(m, target_zero_frac).float()
        q_err = (w - target).abs()
        mst = state.get(m.weight, None)
        if (momentum_weight != 0.0
                and mst is not None and "exp_avg" in mst):
            mom = mst["exp_avg"].abs().float()
            mscale = mom.median().clamp_min(1e-8)
            score = q_err + momentum_weight * mom / mscale
        else:
            score = q_err
        score = torch.where(fm, torch.full_like(score, float("inf")), score)
        score_g = score.view(out_f, ng, gs)
        finite = torch.isfinite(score_g).any(dim=-1)
        idx_in_group = score_g.argmin(dim=-1)
        group_idx = (torch.arange(ng, device=w.device)
                     .view(1, -1).expand(out_f, -1))
        flat_in = group_idx * gs + idx_in_group
        row_idx = (torch.arange(out_f, device=w.device)
                   .view(-1, 1).expand(-1, ng))
        rr = row_idx[finite]
        cc = flat_in[finite]
        if rr.numel() == 0:
            continue
        targets_chosen = target[rr, cc]
        q_err_chosen = q_err[rr, cc]
        # Promote: set the mask. Do NOT touch m.weight — the latent stays
        # where it is and continues to receive gradient via STE.
        m.qat_mask[rr, cc] = True
        m.invalidate_q_cache()
        # Reset opt-state momentum at the promoted slots: the gradient
        # at this slot is now identity-via-STE rather than the soft-mode
        # identity, so the old running averages aren't appropriate.
        if mst is not None:
            for key in ("exp_avg", "exp_avg_sq"):
                if key in mst:
                    mst[key][rr, cc] = 0.0
        # Stats: q_err is the "would-be commit damage" — under freezing
        # this would be a permanent loss tax; under progressive QAT the
        # latent can drift to reduce it.
        signs = torch.zeros_like(targets_chosen, dtype=torch.int8)
        signs[targets_chosen > 0] = 1
        signs[targets_chosen < 0] = -1
        n_promoted += int(rr.numel())
        sum_q_err += float(q_err_chosen.sum())
        sum_q_err_count += int(q_err_chosen.numel())
        n_neg += int((signs == -1).sum())
        n_zero_ += int((signs == 0).sum())
        n_pos += int((signs == 1).sum())
        del w, target, q_err, score, score_g, finite, idx_in_group
        del group_idx, flat_in, row_idx, rr, cc, targets_chosen, q_err_chosen
        del signs
        if torch.cuda.is_available() and m.weight.is_cuda:
            torch.cuda.empty_cache()
    stats = {
        "q_err_mean": (sum_q_err / max(1, sum_q_err_count)),
        "frac_neg": n_neg / max(1, n_promoted),
        "frac_zero": n_zero_ / max(1, n_promoted),
        "frac_pos": n_pos / max(1, n_promoted),
    }
    return n_promoted, stats


@torch.no_grad()
def total_committed_fraction(model: torch.nn.Module) -> float:
    """Fraction of QLinear weights promoted to ternary forward (QAT)."""
    n_promoted = 0
    n_total = 0
    for m in model.modules():
        if not isinstance(m, QLinear):
            continue
        n_promoted += int(m.qat_mask.sum())
        n_total += m.weight.numel()
    return n_promoted / max(1, n_total)


@torch.no_grad()
def max_committed_per_group(model: torch.nn.Module) -> int:
    """Largest count of promoted weights in any single (row, group). When
    this equals group_size for every module, the schedule is done."""
    worst = 0
    for m in model.modules():
        if not isinstance(m, QLinear):
            continue
        out_f = m.weight.shape[0]
        gs, ng = m.group_size, m.n_groups
        cnt = m.qat_mask.view(out_f, ng, gs).sum(dim=-1).max()
        worst = max(worst, int(cnt))
    return worst


# ============================================================================
# Commit gate: when to fire the next round
# ============================================================================

class CommitGate:
    """Decide when to commit the next weight per group.

    Fires when (a) the fast and slow EMAs have converged within `tolerance`
    for `patience` consecutive steps (loss is genuinely flat — neither
    actively dropping nor rising) AND (b) fast_ema − L_T < gap_threshold;
    OR when step_in_round >= max_round_steps (safety cap). Note: a
    *dropping* loss has fast EMA below slow EMA, which fails the two-sided
    convergence check — we don't commit while the model is still
    recovering.

    When (a) holds but (b) doesn't: keep training. We don't want to commit
    while the loss gap is still large — that's the user's explicit preference,
    on the theory that committing a weight with the model still mid-recovery
    locks in a sub-optimal latent.
    """
    def __init__(self, patience: int, max_round_steps: int,
                 gap_threshold: float,
                 min_steps_per_round: int = 50,
                 ema_alpha: float = 0.05,
                 slow_ratio: float = 10.0,
                 tolerance: float = 0.02) -> None:
        self.patience = int(patience)
        self.max_round_steps = int(max_round_steps)
        self.min_steps_per_round = int(min_steps_per_round)
        self.gap_threshold = float(gap_threshold)
        self.tolerance = float(tolerance)
        self.ema_alpha = float(ema_alpha)
        self.slow_ema_alpha = self.ema_alpha / max(1.0, slow_ratio)
        self.fast: float | None = None
        self.slow: float | None = None
        self.steady = 0
        self.step_in_round = 0

    def reset(self) -> None:
        self.fast = None
        self.slow = None
        self.steady = 0
        self.step_in_round = 0

    def step(self, loss: float, L_T: float) -> tuple[bool, str | None]:
        x = float(loss)
        self.step_in_round += 1
        if not (x == x and x != float("inf") and x != float("-inf")):
            self.steady = 0
            return False, None
        if self.fast is None:
            self.fast = self.slow = x
        else:
            self.fast = (1 - self.ema_alpha) * self.fast + self.ema_alpha * x
            self.slow = ((1 - self.slow_ema_alpha) * self.slow
                         + self.slow_ema_alpha * x)
        if abs(self.fast - self.slow) <= self.tolerance:
            self.steady += 1
        else:
            self.steady = 0
        if self.step_in_round < self.min_steps_per_round:
            # Hard floor: never commit within `min_steps_per_round` of the
            # previous commit. Prevents back-to-back commits when loss
            # happens to look stable right after a commit (e.g. because
            # the perturbation barely moved the model).
            return False, None
        if self.step_in_round >= self.max_round_steps:
            return True, f"max_round_steps={self.max_round_steps}"
        if self.steady >= self.patience:
            gap = self.fast - L_T
            if gap < self.gap_threshold:
                return True, f"stable + gap={gap:.4f}<{self.gap_threshold}"
        return False, None

    def state_dict(self) -> dict:
        return {"fast": self.fast, "slow": self.slow,
                "steady": self.steady, "step_in_round": self.step_in_round}

    def load_state_dict(self, state: dict) -> None:
        self.fast = state.get("fast")
        self.slow = state.get("slow")
        self.steady = int(state.get("steady", 0))
        self.step_in_round = int(state.get("step_in_round", 0))


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    # ---- shared with distill.py ----
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-floor", type=float, default=1.0,
                    help="Default 1.0 (flat LR) since the run length isn't "
                         "known a priori. Set <1.0 with a nominal horizon "
                         "to engage cosine decay.")
    ap.add_argument("--warmup-steps", type=int, default=30)
    ap.add_argument("--wd", type=float, default=0.001)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--optimizer", default="cautious-adamw",
                    choices=["lion", "adamw", "cautious-adamw"])
    ap.add_argument("--scale-group-size", type=int, default=64,
                    help="64 for SmolLM2-135M (default model); 128 for "
                         "Qwen3-1.7B. Must divide every projection's "
                         "in_features.")
    ap.add_argument("--scale-lr-mult", type=float, default=None)
    ap.add_argument("--permute", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Match distill.py: permute free input dims at init "
                         "to cluster like magnitudes. May want to ablate — "
                         "permute=False gives heterogeneous groups which "
                         "could be easier for greedy commit ordering.")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--autocast-dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "none"])
    ap.add_argument("--latent-dtype", default="auto",
                    choices=["auto", "float32", "float16", "bfloat16"],
                    help="QLinear latent weight storage dtype. 'auto' "
                         "couples to --autocast-dtype: fp32 when autocast "
                         "is 'none', else fp16. Set explicitly to decouple "
                         "(e.g. --autocast-dtype none --latent-dtype "
                         "bfloat16 = fp32 forward for stability with bf16 "
                         "latent storage to save ~270 MB).")
    ap.add_argument("--grad-checkpointing",
                    action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ema-warmup", type=int, default=500)
    ap.add_argument("--plateau-threshold", type=float, default=1e-3)
    ap.add_argument("--soft-hist-every", type=int, default=200)
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    ap.add_argument("--tb-dir", type=Path, default=None)
    ap.add_argument("--run-name", type=str, default=None)

    # ---- progressive-specific ----
    ap.add_argument("--c", type=float, default=2.0 / 3.0,
                    help="Fallback scalar codepoint magnitude. Actual "
                         "codepoint c_{r,g} is per-(row, group), computed "
                         "at init as mean(|w| over band) — minimizes commit "
                         "q_err on average. This flag only matters as a "
                         "fallback for empty bands (target_zero_frac=1.0) "
                         "or as the buffer's pre-load default before resume "
                         "state_dict overwrites it. 2/3 ≈ 0.667 is roughly "
                         "where mean(|w| over band) lands for SmolLM2-shape "
                         "Gaussian-ish weight distributions.")
    ap.add_argument("--barrier-coef", type=float, default=0.0,
                    help="Deprecated. Was: relu(|w|−1)² soft barrier "
                         "coefficient. Now the hard clamp_(-1, 1) after "
                         "each opt.step does the job; this flag is a no-op "
                         "but kept for backward-compat with older run.sh.")
    ap.add_argument("--target-zero-frac", type=float, default=0.38,
                    help="Fraction of weights per (row, group) targeting 0 "
                         "(the rest target ±c). Per-group |w|-quantile cutoff. "
                         "Bonsai's deployed model is ~38%% zero (62%% non-zero); "
                         "default matches. Set ≤0 or ≥1 to disable (fall back "
                         "to fixed c/2 boundary, which over-zeros heavily for "
                         "Gaussian-ish weight distributions — Run A produced "
                         "~91%% zeros with c/2).")
    ap.add_argument("--momentum-weight", type=float, default=0.0,
                    help="Selection penalty on |exp_avg|/median(|exp_avg|). "
                         "0 = pure quantization-error greedy. Bump to ~0.1-1 "
                         "to prefer committing weights the optimizer isn't "
                         "actively pushing.")
    ap.add_argument("--warmup-rounds-steps", type=int, default=200,
                    help="Steps to train before the FIRST commit. Lets "
                         "momentum populate so the selection score's "
                         "momentum term is meaningful. Reduces 'commit a "
                         "weight mid-stride' risk.")
    ap.add_argument("--commit-patience", type=int, default=50,
                    help="Consecutive steady steps (fast EMA ≤ slow EMA + "
                         "tolerance) required per commit round.")
    ap.add_argument("--commit-tolerance", type=float, default=0.02,
                    help="Loss-EMA rise still counted as 'stable' (nats).")
    ap.add_argument("--commit-gap-threshold", type=float, default=0.05,
                    help="Required gap (fast_ema − L_T) below this to "
                         "permit a commit. If stable but gap is wider, "
                         "keep training (don't lock in a bad position).")
    ap.add_argument("--commit-max-round-steps", type=int, default=2000,
                    help="Hard cap on steps per commit round; safety net "
                         "if the gap never closes.")
    ap.add_argument("--min-steps-per-round", type=int, default=50,
                    help="Hard floor on steps per commit round. The gate "
                         "won't fire within this many steps of a previous "
                         "commit even if loss looks stable. Prevents "
                         "rapid-fire commits when the perturbation barely "
                         "moves the model.")
    ap.add_argument("--post-commit-momentum-damp", type=float, default=1.0,
                    help="Multiply ALL exp_avg by this after each commit. "
                         "1.0 = off. <1 cools momentum to prevent "
                         "overshoot after the perturbation.")
    ap.add_argument("--settle-max-steps", type=int, default=5000,
                    help="After the final commit round, keep training "
                         "until the EMAs converge (two-sided), capped at "
                         "this many steps. Without this, the last commit "
                         "(usually the biggest perturbation — every "
                         "remaining weight gets snapped at once across "
                         "all groups) leaves the model in a degraded "
                         "post-perturbation state. Run F's final round "
                         "fired at max_round_steps with q_err=0.4 and "
                         "the deploy fold captured that bad state.")

    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    _install_sigint_handler()

    # ---- Build student ----
    latent_dtype = {
        "auto": (torch.float32 if args.autocast_dtype == "none"
                 else torch.float16),
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.latent_dtype]
    interrupted_path = args.out / "interrupted.pt"
    fresh_start = args.resume is None and not interrupted_path.exists()
    do_permute = args.permute and fresh_start
    print(f"[build] loading {args.model}, group_size={args.scale_group_size}, "
          f"permute={do_permute}")
    model, _tok_unused, n_replaced = load_student(
        args.model, dtype=torch.float32, levels=257,
        latent_dtype=latent_dtype, group_size=args.scale_group_size,
        permute=do_permute)
    print(f"[build] {n_replaced} QLinear modules (latent dtype {latent_dtype})")
    model = model.to(args.device)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if (args.grad_checkpointing
            and hasattr(model, "gradient_checkpointing_enable")):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        print("[build] gradient checkpointing enabled")
    _embed = model.get_input_embeddings()
    embed_init = (_embed.weight.detach().clone()
                  if _embed is not None else None)

    # ---- Soft mode at α=0 (forward = identity), attach buffers, fit c ----
    # codepoint_c is per (row, group). Fresh start: compute from data.
    # Resume: state_dict load below restores the saved tensor.
    set_soft_mode(model, alpha=0.0, target_zero_frac=None)
    n_buf = attach_progressive_buffers(model, default_c=args.c)
    if fresh_start:
        tzf = (args.target_zero_frac
               if 0.0 < args.target_zero_frac < 1.0 else None)
        cstats = compute_per_group_c(model, tzf, fallback_c=args.c)
        invalidate_c_elem_cache(model)
        print(f"[progressive] codepoint_c: mean={cstats['mean']:.4f} "
              f"min={cstats['min']:.4f} max={cstats['max']:.4f} "
              f"p50={cstats['p50']:.4f}")
    print(f"[progressive] soft mode α=0; {n_buf} QLinears w/ progressive "
          f"buffers; fresh_start={fresh_start}")

    # ---- Optimizer (same param-group split as distill.py) ----
    scale_lr_mult = (args.scale_lr_mult if args.scale_lr_mult is not None
                     else 1.0 / float(args.scale_group_size))
    scale_param_ids = {id(m.scales) for m in model.modules()
                       if isinstance(m, QLinear)}
    scale_params, other_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (scale_params if id(p) in scale_param_ids
         else other_params).append(p)
    param_groups = [
        {"params": other_params, "lr": args.lr, "lr_mult": 1.0,
         "name": "latents"},
        {"params": scale_params, "lr": args.lr * scale_lr_mult,
         "lr_mult": scale_lr_mult, "name": "scales"},
    ]
    OptCls = {"lion": Lion32, "adamw": AdamW32,
              "cautious-adamw": CautiousAdamW}[args.optimizer]
    opt = OptCls(param_groups, lr=args.lr, weight_decay=args.wd)
    print(f"[opt] {args.optimizer} lr={args.lr} wd={args.wd} "
          f"scale_lr_mult={scale_lr_mult:g}")

    # ---- Resume ----
    interrupted_state = None
    global_step = 0
    samples_consumed = 0
    round_idx = 0
    if args.resume is not None:
        with safe_open(str(args.resume), framework="pt") as f:
            resume_meta = f.metadata() or {}
        sd = load_file(str(args.resume))
        miss, unexp = model.load_state_dict(sd, strict=False)
        invalidate_c_elem_cache(model)
        print(f"[resume] warm-start from {args.resume.name} "
              f"(meta={resume_meta}, missing={len(miss)}, "
              f"unexpected={len(unexp)})")
    elif interrupted_path.exists():
        # Load to CPU first to avoid ~2 GB of redundant GPU allocations:
        # the loaded state dict and best_snapshot get held as long as
        # `interrupted_state` is in scope. model.load_state_dict and
        # opt.load_state_dict both move tensors to the right device
        # (matching the model's params), so the GPU-resident copy is
        # only created where needed.
        interrupted_state = torch.load(str(interrupted_path),
                                       map_location="cpu",
                                       weights_only=False)
        model.load_state_dict(interrupted_state["model"], strict=False)
        invalidate_c_elem_cache(model)
        # Drop the loaded model dict — its data is now in the model's
        # parameters; no need to keep a CPU duplicate.
        del interrupted_state["model"]
        opt.load_state_dict(interrupted_state["opt"])
        del interrupted_state["opt"]
        global_step = int(interrupted_state.get("next_step", 0))
        samples_consumed = int(interrupted_state.get(
            "samples_consumed",
            global_step * args.grad_accum * args.batch_size))
        prog = interrupted_state.get("soft_state") or {}
        round_idx = int(prog.get("round_idx", 0))
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[resume] {interrupted_path} at step {global_step}, "
              f"round {round_idx}, "
              f"committed_frac={total_committed_fraction(model):.4f}")

    # ---- Teacher floor, data, TB ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    floor_data = load_teacher_floor(args.cache_dir, len(tok))
    L_T = float(floor_data["floor"])
    print(f"[progressive] L_T = {L_T:.4f}")

    ds = ShardedDataset(args.cache_dir, seed=args.seed,
                        start_skip=samples_consumed)
    def _worker_init(_worker_id: int) -> None:
        # Workers inherit the main process's SIGINT handler via fork, which
        # causes the handler's print to fire once per worker on Ctrl-C.
        # IGNORE (not DEFAULT) in workers: SIG_DFL would kill the worker on
        # Ctrl-C, and the main process's next loss.item()/batch fetch would
        # raise "DataLoader worker is killed by signal" before reaching the
        # save check — losing the resume save we wanted. SIG_IGN keeps the
        # worker alive and silent; main shuts it down cleanly via the
        # gate-flag save path.
        import signal as _sig
        _sig.signal(_sig.SIGINT, _sig.SIG_IGN)

    dl = DataLoader(ds, batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=(args.device.startswith("cuda")),
                    drop_last=True,
                    worker_init_fn=_worker_init)
    it = iter(dl)
    autocast_dtype = {"bfloat16": torch.bfloat16,
                      "float16": torch.float16,
                      "none": None}[args.autocast_dtype]
    tb_root = args.tb_dir if args.tb_dir is not None else (args.out / "tb")
    if interrupted_state and interrupted_state.get("run_name"):
        run_name = interrupted_state["run_name"]
    elif args.run_name:
        run_name = args.run_name
    else:
        run_name = datetime.now().strftime("prog_%Y%m%d_%H%M%S")
    run_dir = tb_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir),
                           purge_step=global_step if global_step else None)
    print(f"[tb] {run_dir}")
    writer.add_text("stage", "  \n".join([
        "**progressive** clamping",
        f"- c = {args.c}",
        f"- barrier_coef = {args.barrier_coef}",
        f"- momentum_weight = {args.momentum_weight}",
        f"- warmup_rounds_steps = {args.warmup_rounds_steps}",
        f"- commit_patience = {args.commit_patience}",
        f"- commit_tolerance = {args.commit_tolerance}",
        f"- commit_gap_threshold = {args.commit_gap_threshold}",
        f"- commit_max_round_steps = {args.commit_max_round_steps}",
        f"- post_commit_momentum_damp = {args.post_commit_momentum_damp}",
        f"- group_size = {args.scale_group_size}",
        f"- optimizer = {args.optimizer}, lr = {args.lr}, wd = {args.wd}",
        f"- L_T = {L_T:.4f}",
    ]), global_step)
    writer.add_scalar("cfg/c", args.c, global_step)
    writer.add_scalar("cfg/L_T", L_T, global_step)
    writer.add_scalar("cfg/group_size", float(args.scale_group_size),
                      global_step)

    # ---- Training loop state ----
    ctrl = BestEmaTracker(rel_threshold=args.plateau_threshold,
                          ema_warmup=args.ema_warmup)
    if interrupted_state and interrupted_state.get("ctrl_state"):
        ctrl.load_state_dict(interrupted_state["ctrl_state"])
    if (interrupted_state
            and interrupted_state.get("best_snapshot") is not None):
        best_snapshot = interrupted_state["best_snapshot"]
    else:
        best_snapshot = snapshot_to_cpu(model)

    gate = CommitGate(patience=args.commit_patience,
                      max_round_steps=args.commit_max_round_steps,
                      gap_threshold=args.commit_gap_threshold,
                      min_steps_per_round=args.min_steps_per_round,
                      tolerance=args.commit_tolerance)
    if interrupted_state and (interrupted_state.get("soft_state") or {}).get("gate"):
        gate.load_state_dict(interrupted_state["soft_state"]["gate"])

    embed_stage_init = (model.get_input_embeddings().weight.detach().clone()
                        if model.get_input_embeddings() is not None else None)
    _, prev_codes, prev_codes_fixed = collect_qlinear_metrics(model, {})

    group_size = args.scale_group_size
    is_warmup = (round_idx == 0
                 and global_step < args.warmup_rounds_steps)
    pbar = tqdm(desc="progressive", dynamic_ncols=True)
    running = 0.0
    running_n = 0
    model.train()
    opt.zero_grad(set_to_none=True)

    # Cosine horizon for lr_at: long nominal total since real total is
    # data-dependent. With default --lr-floor=1.0 this is moot (flat LR).
    nominal_total = max(1, group_size * args.commit_max_round_steps)

    in_settle = False
    settle_gate: CommitGate | None = None
    try:
        while round_idx < group_size or in_settle:
            cur_lr = lr_at(global_step, nominal_total, args.lr,
                           args.warmup_steps, floor=args.lr_floor)
            for g in opt.param_groups:
                g["lr"] = cur_lr * g.get("lr_mult", 1.0)
                g["weight_decay"] = args.wd

            # --- KL grad accumulation ---
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
                        f"non-finite loss at step {global_step}, "
                        f"round {round_idx}, loss={loss.item()}")
                (loss / args.grad_accum).backward()
                running += loss.item()
                running_n += 1

            barrier_val = 0.0  # barrier deprecated under hard-clamp scheme

            # Under progressive QAT, promoted slots use STE — gradient flows
            # to the latent so the optimizer can keep moving it. No
            # zero-grad masking at promoted slots; that would defeat the
            # whole point.

            grad_norm = None
            if args.max_grad_norm:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm))
            opt.step()
            clamp_qlinear_weights(model)  # keep latents in [-1, 1]
            opt.zero_grad(set_to_none=True)
            global_step += 1

            step_loss = running / max(1, running_n)

            # --- warmup vs commit gate vs settle ---
            if is_warmup:
                if global_step >= args.warmup_rounds_steps:
                    tqdm.write(f"[warmup] done at step {global_step}; "
                               f"starting commit rounds")
                    gate.reset()
                    is_warmup = False
            elif in_settle:
                # Settle phase: train until EMAs converge (two-sided check
                # in the gate; we pass gap_threshold=inf so any stable
                # state fires the exit), capped at --settle-max-steps.
                done, settle_reason = settle_gate.step(step_loss, L_T)
                if done:
                    tqdm.write(f"[settle] done at step {global_step}; "
                               f"reason={settle_reason}")
                    break
            else:
                should_commit, reason = gate.step(step_loss, L_T)
                if should_commit:
                    tzf = (args.target_zero_frac
                           if 0.0 < args.target_zero_frac < 1.0 else None)
                    n_c, sel_stats = select_and_promote_one_per_group(
                        model, opt, args.momentum_weight, tzf)
                    if args.post_commit_momentum_damp != 1.0:
                        for p in model.parameters():
                            st = opt.state.get(p, {})
                            if "exp_avg" in st:
                                st["exp_avg"].mul_(args.post_commit_momentum_damp)
                    if args.device.startswith("cuda"):
                        torch.cuda.empty_cache()
                    round_idx += 1
                    if round_idx == group_size:
                        # Last commit done — enter settle phase to let the
                        # post-perturbation loss recover before the deploy
                        # fold captures it. Gate fires on stable EMAs
                        # regardless of gap (gap_threshold=inf).
                        in_settle = True
                        settle_gate = CommitGate(
                            patience=args.commit_patience,
                            max_round_steps=args.settle_max_steps,
                            gap_threshold=float("inf"),
                            min_steps_per_round=args.min_steps_per_round,
                            tolerance=args.commit_tolerance,
                        )
                        tqdm.write(
                            f"[settle] entering settle phase at step "
                            f"{global_step}; cap={args.settle_max_steps} steps")
                    cf = total_committed_fraction(model)
                    mx = max_committed_per_group(model)
                    tqdm.write(
                        f"[commit] round {round_idx}/{group_size} "
                        f"reason={reason}; committed {n_c}; "
                        f"total_frac={cf:.4f}; max_per_group={mx}; "
                        f"q_err_mean={sel_stats['q_err_mean']:.4f}; "
                        f"targets neg/zero/pos="
                        f"{sel_stats['frac_neg']:.2f}/"
                        f"{sel_stats['frac_zero']:.2f}/"
                        f"{sel_stats['frac_pos']:.2f}")
                    writer.add_scalar("progressive/round",
                                      float(round_idx), global_step)
                    writer.add_scalar("progressive/committed_frac", cf,
                                      global_step)
                    writer.add_scalar("progressive/max_per_group",
                                      float(mx), global_step)
                    writer.add_scalar("progressive/last_commit_q_err",
                                      sel_stats["q_err_mean"], global_step)
                    for k in ("frac_neg", "frac_zero", "frac_pos"):
                        writer.add_scalar(
                            f"progressive/last_commit_target_{k}",
                            sel_stats[k], global_step)
                    gate.reset()

            # --- best-EMA tracking (for SIGINT recovery) ---
            improved = ctrl.update(global_step, step_loss)
            if improved:
                best_snapshot = snapshot_to_cpu(model)

            # --- log ---
            if global_step % args.log_every == 0:
                cf = total_committed_fraction(model)
                pbar.set_postfix({
                    "step": global_step,
                    "loss": f"{step_loss:.4f}",
                    "ema": f"{ctrl.ema:.4f}" if ctrl.ema else "—",
                    "round": f"{round_idx}/{group_size}",
                    "cf": f"{cf:.3f}",
                    "in_round": gate.step_in_round,
                    "lr": f"{cur_lr:.2e}",
                })
                writer.add_scalar("loss/step", step_loss, global_step)
                if ctrl.ema is not None:
                    writer.add_scalar("loss/ema", ctrl.ema, global_step)
                writer.add_scalar("loss/gap", step_loss - L_T, global_step)
                writer.add_scalar("lr", cur_lr, global_step)
                writer.add_scalar("progressive/committed_frac", cf,
                                  global_step)
                writer.add_scalar("progressive/round", float(round_idx),
                                  global_step)
                writer.add_scalar("progressive/step_in_round",
                                  float(gate.step_in_round), global_step)
                writer.add_scalar("progressive/barrier", barrier_val,
                                  global_step)
                if gate.fast is not None:
                    writer.add_scalar("progressive/loss_ema_fast",
                                      gate.fast, global_step)
                if gate.slow is not None:
                    writer.add_scalar("progressive/loss_ema_slow",
                                      gate.slow, global_step)
                writer.add_scalar("progressive/steady",
                                  float(gate.steady), global_step)
                if grad_norm is not None:
                    writer.add_scalar("grad_norm", grad_norm, global_step)
                qm, prev_codes, prev_codes_fixed = collect_qlinear_metrics(
                    model, prev_codes, prev_codes_fixed)
                for k, v in qm.items():
                    writer.add_scalar(k, v, global_step)
                # collect_soft_metrics's `frac_zero/pos/neg` here uses
                # the ±1/3 c(w) classifier, NOT our ±c codepoints.
                # Treat as a coarse "which way is this weight leaning"
                # rather than as our codebook occupancy.
                soft_qm, _ = collect_soft_metrics(model)
                for k, v in soft_qm.items():
                    writer.add_scalar(k, v, global_step)
                drift, drift_stage = embed_drift_l2(
                    model, embed_init, embed_stage_init)
                if drift is not None:
                    writer.add_scalar("embed/drift_l2", drift, global_step)
                if drift_stage is not None:
                    writer.add_scalar("embed/drift_l2_stage",
                                      drift_stage, global_step)
                running = 0.0
                running_n = 0
                pbar.update(args.log_every)

            # --- weight histogram (QAT-effective weight, not raw latent) ---
            if (args.soft_hist_every > 0
                    and global_step % args.soft_hist_every == 0):
                hs = first_qlinear_qat_sample(model)
                if hs is not None:
                    name, w_flat = hs
                    if torch.isfinite(w_flat).all():
                        writer.add_histogram(f"progressive/hist/{name}",
                                             w_flat, global_step, bins=64)

            # --- auto-checkpoint ---
            samples_at_save = (global_step
                               * args.grad_accum * args.batch_size)
            if (args.checkpoint_every > 0
                    and global_step % args.checkpoint_every == 0):
                prog_state = {"round_idx": round_idx,
                              "gate": gate.state_dict(),
                              "type": "progressive"}
                save_resume(interrupted_path, model, opt, global_step,
                            best_snapshot, ctrl.state_dict(), run_name,
                            samples_consumed=samples_at_save,
                            soft_state=prog_state)
                tqdm.write(f"[ckpt] {interrupted_path} @ step {global_step}")

            if _INTERRUPT["flag"]:
                prog_state = {"round_idx": round_idx,
                              "gate": gate.state_dict(),
                              "type": "progressive"}
                save_resume(interrupted_path, model, opt, global_step,
                            best_snapshot, ctrl.state_dict(), run_name,
                            samples_consumed=samples_at_save,
                            soft_state=prog_state)
                writer.flush()
                writer.close()
                pbar.close()
                print(f"[!] saved {interrupted_path}")
                sys.exit(0)
    except SystemExit:
        # The gate-flag SIGINT path calls sys.exit(0) AFTER its own save
        # completes. Skip the emergency-save branch so we don't save twice.
        raise
    except BaseException as e:
        # Best-effort save on any unexpected exit (KeyboardInterrupt,
        # DataLoader worker death, OOM, etc.) so we don't lose progress.
        try:
            prog_state = {"round_idx": round_idx,
                          "gate": gate.state_dict(),
                          "type": "progressive"}
            samples_at_save = (global_step
                               * args.grad_accum * args.batch_size)
            save_resume(interrupted_path, model, opt, global_step,
                        best_snapshot, ctrl.state_dict(), run_name,
                        samples_consumed=samples_at_save,
                        soft_state=prog_state)
            print(f"[!] emergency save → {interrupted_path} "
                  f"(reason: {type(e).__name__})", flush=True)
        except Exception as save_err:
            print(f"[!!] emergency save failed: {save_err}", flush=True)
        raise
    finally:
        pbar.close()

    # ---- Done: snap latents to ternary + fold c into scales → {-1, 0, +1} ----
    # Per element: target = sign(w)·c_{r,g} if |w| > c_{r,g}/2, else 0.
    # This matches the STE-ternarization the QAT forward was already using
    # at promoted slots — math-preserving when every slot was promoted.
    # Then fold c_{r,g} into the scales: w /= c, s *= c. After the fold
    # the latent is in {-1, 0, +1} and the scales absorb the codepoint
    # magnitude — Bonsai's deployment form.
    print(f"[progressive] complete after {global_step} steps; "
          f"committed_frac={total_committed_fraction(model):.4f}")
    with torch.no_grad():
        for m in model.modules():
            if not isinstance(m, QLinear):
                continue
            out_f, in_f = m.weight.shape
            c_elem = (m.codepoint_c.unsqueeze(-1)
                      .expand(out_f, m.n_groups, m.group_size)
                      .reshape(out_f, in_f)
                      .to(m.weight.dtype))
            thresh = c_elem * 0.5
            target = torch.where(m.weight.abs() > thresh,
                                 torch.sign(m.weight) * c_elem,
                                 torch.zeros_like(m.weight))
            m.weight.data.copy_(target)
            m.weight.data.div_(c_elem)
            m.scales.data.mul_(m.codepoint_c.to(m.scales.dtype))
            m.invalidate_q_cache()
    out_ckpt = args.out / "stage_progressive.safetensors"
    save_checkpoint(model, out_ckpt, args.model, args.scale_group_size,
                    alpha=0.0, target_zero_frac=None)
    print(f"[progressive] saved {out_ckpt}")
    writer.add_text(
        "stage_end",
        f"progressive complete: {global_step} steps, "
        f"{round_idx}/{group_size} rounds, "
        f"committed_frac={total_committed_fraction(model):.4f}",
        global_step)
    writer.flush()
    writer.close()
    if interrupted_path.exists():
        interrupted_path.unlink()
    print("[done] progressive training complete.")


if __name__ == "__main__":
    main()
