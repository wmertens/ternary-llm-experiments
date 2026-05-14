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
# Per-QLinear progressive state
# ============================================================================
# Three persistent buffers per QLinear (so they ride along in state_dict and
# resume Just Works):
#   frozen_mask   : bool[out, in]            — True for committed slots
#   frozen_target : int8[out, in]            — sign of committed value (-1/0/+1)
#   codepoint_c   : fp32[out, n_groups]      — codepoint magnitude per (row,group)
#
# The latent value at a frozen slot is `frozen_target[r, c] * codepoint_c[r, g]`
# where g = c // group_size. c_{r,g} is data-driven: at init we set it to the
# mean of |w| over the band weights (|w| above the quantile boundary) in each
# (row, group). That choice minimizes commit q_err on average — the band
# weights are centered on the codepoint, not crowded against the |w|=1 edge.

def attach_progressive_buffers(model: torch.nn.Module,
                               default_c: float = 2.0 / 3.0) -> int:
    n = 0
    for m in model.modules():
        if not isinstance(m, QLinear):
            continue
        if not hasattr(m, "frozen_mask"):
            m.register_buffer(
                "frozen_mask",
                torch.zeros_like(m.weight, dtype=torch.bool),
                persistent=True,
            )
            m.register_buffer(
                "frozen_target",
                torch.zeros_like(m.weight, dtype=torch.int8),
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


def _c_per_element(m: QLinear) -> torch.Tensor:
    """Broadcast codepoint_c [out, ng] → [out, in] for per-element ops."""
    out_f, in_f = m.weight.shape
    gs, ng = m.group_size, m.n_groups
    return (m.codepoint_c.unsqueeze(-1)
            .expand(out_f, ng, gs)
            .reshape(out_f, in_f)
            .to(m.weight.dtype))


@torch.no_grad()
def _build_frozen_value(m: QLinear) -> torch.Tensor:
    """Materialize the full-shape `frozen_target * codepoint_c[r, g]`
    tensor in weight dtype. Zero at unfrozen slots (since
    frozen_target=0 there). Used at init/resume to populate the cache;
    subsequent updates happen incrementally at commit time."""
    out_f, in_f = m.weight.shape
    c_elem = (m.codepoint_c.unsqueeze(-1)
              .expand(out_f, m.n_groups, m.group_size)
              .reshape(out_f, in_f)
              .to(m.weight.dtype))
    return (m.frozen_target.to(m.weight.dtype) * c_elem).contiguous()


@torch.no_grad()
def enforce_frozen(model: torch.nn.Module) -> None:
    """Overwrite committed slots with their precomputed target value.

    `_frozen_value` is a non-persistent per-QLinear cache of
    `frozen_target * codepoint_c[r, g]`, populated at init/resume and
    incrementally updated at commit time. Per step this is a single
    indexed copy — no broadcast, no multiply, no per-step allocations.
    Rebuilt lazily if shape/dtype/device drifts (handles --latent-dtype
    changes and resume-into-different-dtype).
    """
    for m in model.modules():
        if not isinstance(m, QLinear):
            continue
        if not bool(m.frozen_mask.any()):
            continue
        cache = getattr(m, "_frozen_value", None)
        if (cache is None
                or cache.shape != m.weight.shape
                or cache.dtype != m.weight.dtype
                or cache.device != m.weight.device):
            m._frozen_value = _build_frozen_value(m)
        m.weight.data[m.frozen_mask] = m._frozen_value[m.frozen_mask]
        m.invalidate_q_cache()


def invalidate_frozen_value_cache(model: torch.nn.Module) -> None:
    """Force rebuild of the precomputed frozen-value cache (e.g., after
    compute_per_group_c writes to codepoint_c, or after a state_dict load
    that touched codepoint_c / frozen_target / frozen_mask)."""
    for m in model.modules():
        if not isinstance(m, QLinear):
            continue
        if hasattr(m, "_frozen_value"):
            try:
                del m._frozen_value
            except AttributeError:
                pass


@torch.no_grad()
def zero_frozen_grad(model: torch.nn.Module) -> None:
    """Zero the gradient at frozen slots, after backward, before opt.step.
    Prevents the optimizer from accumulating momentum on locked positions."""
    for m in model.modules():
        if not isinstance(m, QLinear):
            continue
        if m.weight.grad is None:
            continue
        if bool(m.frozen_mask.any()):
            m.weight.grad.masked_fill_(m.frozen_mask, 0.0)


def barrier_loss(model: torch.nn.Module) -> torch.Tensor:
    """Soft barrier ‖relu(|w|−1)‖² (per-element mean) over UNFROZEN latents
    only. Smooth everywhere; lets weights live in roughly [-1, 1] with the
    codepoints ±c sitting comfortably interior. Frozen slots are masked out
    of the sum (they're at ±c ≤ 1 anyway, so their contribution is 0, but
    explicit masking makes the gradient route to them exactly zero)."""
    parts: list[torch.Tensor] = []
    n_total = 0
    device = None
    for m in model.modules():
        if not isinstance(m, QLinear):
            continue
        w = m.weight
        device = w.device
        excess = (w.float().abs() - 1.0).clamp_min(0.0)
        if hasattr(m, "frozen_mask"):
            excess = excess.masked_fill(m.frozen_mask, 0.0)
        parts.append((excess * excess).sum())
        n_total += w.numel()
    if not parts or n_total == 0:
        return torch.tensor(0.0, device=device or "cpu")
    return torch.stack(parts).sum() / n_total


# ============================================================================
# Selection + commit
# ============================================================================

@torch.no_grad()
def compute_commit_targets(m: QLinear,
                           target_zero_frac: float | None) -> torch.Tensor:
    """Per-element target ∈ {-c_{r,g}, 0, +c_{r,g}}, in m.weight's dtype.
    c_{r,g} comes from m.codepoint_c (data-driven, one per (row, group)).

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
def select_and_commit_one_per_group(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    momentum_weight: float,
    target_zero_frac: float | None,
) -> tuple[int, dict[str, float]]:
    """For each (row, group) that still has at least one unfrozen weight,
    pick the weight minimizing
        |w − target(w)|  +  λ_m · |exp_avg| / median(|exp_avg|)
    where target(w) is the nearest of {-c, 0, +c} under the boundary
    rule selected by target_zero_frac (quantile if set, else c/2). Snap
    the chosen weight to its target, mark it frozen, zero its (exp_avg,
    exp_avg_sq) entries.

    Returns (n_committed, stats). One commit per non-fully-frozen group
    per call.
    """
    state = opt.state
    n_committed = 0
    sum_q_err = 0.0
    sum_q_err_count = 0
    n_neg = n_zero_ = n_pos = 0
    for m in model.modules():
        if not isinstance(m, QLinear):
            continue
        out_f, in_f = m.weight.shape
        gs, ng = m.group_size, m.n_groups
        w = m.weight.detach().float()
        fm = m.frozen_mask
        target = compute_commit_targets(m, target_zero_frac).float()
        q_err = (w - target).abs()
        # Momentum penalty (Lion / AdamW both store 'exp_avg').
        mst = state.get(m.weight, None)
        if (momentum_weight != 0.0
                and mst is not None and "exp_avg" in mst):
            mom = mst["exp_avg"].abs().float()
            mscale = mom.median().clamp_min(1e-8)
            score = q_err + momentum_weight * mom / mscale
        else:
            score = q_err
        # Frozen slots can't be re-selected.
        score = torch.where(fm, torch.full_like(score, float("inf")), score)
        score_g = score.view(out_f, ng, gs)
        # Any group with all-inf score is fully frozen; skip it.
        finite = torch.isfinite(score_g).any(dim=-1)  # [out, ng]
        idx_in_group = score_g.argmin(dim=-1)         # [out, ng] in [0, gs)
        # Build the (row, col) coordinates of the chosen weights.
        group_idx = (torch.arange(ng, device=w.device)
                     .view(1, -1).expand(out_f, -1))
        flat_in = group_idx * gs + idx_in_group       # [out, ng] in [0, in_f)
        row_idx = (torch.arange(out_f, device=w.device)
                   .view(-1, 1).expand(-1, ng))
        rr = row_idx[finite]
        cc = flat_in[finite]
        if rr.numel() == 0:
            continue
        targets_chosen = target[rr, cc]
        q_err_chosen = q_err[rr, cc]
        m.weight.data[rr, cc] = targets_chosen.to(m.weight.dtype)
        signs = torch.zeros_like(targets_chosen, dtype=torch.int8)
        signs[targets_chosen > 0] = 1
        signs[targets_chosen < 0] = -1
        m.frozen_target[rr, cc] = signs
        m.frozen_mask[rr, cc] = True
        # Incrementally update the precomputed frozen-value cache at the
        # new commit slots. targets_chosen is already sign(w) * c_{r,g}
        # in float; cast to weight dtype matches _frozen_value's dtype.
        if hasattr(m, "_frozen_value"):
            m._frozen_value[rr, cc] = targets_chosen.to(m._frozen_value.dtype)
        m.invalidate_q_cache()
        if mst is not None:
            for key in ("exp_avg", "exp_avg_sq"):
                if key in mst:
                    mst[key][rr, cc] = 0.0
        n_committed += int(rr.numel())
        sum_q_err += float(q_err_chosen.sum())
        sum_q_err_count += int(q_err_chosen.numel())
        n_neg += int((signs == -1).sum())
        n_zero_ += int((signs == 0).sum())
        n_pos += int((signs == 1).sum())
        # Release this module's fp32 temporaries before moving on; otherwise
        # the caching allocator accumulates mixed-size holes across 210
        # QLinears, leaving no contiguous block for the next backward.
        del w, target, q_err, score, score_g, finite, idx_in_group
        del group_idx, flat_in, row_idx, rr, cc, targets_chosen, q_err_chosen
        del signs
        if torch.cuda.is_available() and m.weight.is_cuda:
            torch.cuda.empty_cache()
    stats = {
        "q_err_mean": (sum_q_err / max(1, sum_q_err_count)),
        "frac_neg": n_neg / max(1, n_committed),
        "frac_zero": n_zero_ / max(1, n_committed),
        "frac_pos": n_pos / max(1, n_committed),
    }
    return n_committed, stats


@torch.no_grad()
def total_committed_fraction(model: torch.nn.Module) -> float:
    n_frozen = 0
    n_total = 0
    for m in model.modules():
        if not isinstance(m, QLinear):
            continue
        n_frozen += int(m.frozen_mask.sum())
        n_total += m.weight.numel()
    return n_frozen / max(1, n_total)


@torch.no_grad()
def max_committed_per_group(model: torch.nn.Module) -> int:
    """Largest count of frozen weights in any single (row, group). When
    this equals group_size for every module, the schedule is done."""
    worst = 0
    for m in model.modules():
        if not isinstance(m, QLinear):
            continue
        out_f = m.weight.shape[0]
        gs, ng = m.group_size, m.n_groups
        cnt = m.frozen_mask.view(out_f, ng, gs).sum(dim=-1).max()
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
                    help="Coefficient on the relu(|w|−1)² soft barrier loss. "
                         "Default 0: we rely on hard clamp_(-1, 1) after each "
                         "opt.step instead — natural weight distribution and "
                         "weight decay pull weights inward; the soft barrier "
                         "added ~1 GB of fp32 autograd graph for negligible "
                         "benefit. Set >0 to re-enable the soft form.")
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
        invalidate_frozen_value_cache(model)
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
        invalidate_frozen_value_cache(model)
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
        invalidate_frozen_value_cache(model)
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

    try:
        while round_idx < group_size:
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

            # --- barrier loss (one backward, no grad_accum scaling) ---
            # Short-circuit when no weight crosses |w|=1: the autograd graph
            # for barrier_loss holds ~1 GB of fp32 copies of every QLinear
            # weight, which is wasted memory when the barrier is inactive.
            barrier_val = 0.0
            if args.barrier_coef > 0:
                with torch.no_grad():
                    excess_max = 0.0
                    for m in model.modules():
                        if not isinstance(m, QLinear):
                            continue
                        excess_max = max(excess_max,
                                         float(m.weight.abs().max()))
                if excess_max > 1.0:
                    bl = barrier_loss(model)
                    (args.barrier_coef * bl).backward()
                    barrier_val = float(bl.detach())

            # --- mask out frozen-slot grads BEFORE clip + step ---
            zero_frozen_grad(model)

            grad_norm = None
            if args.max_grad_norm:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm))
            opt.step()
            enforce_frozen(model)        # undo wd / barrier on frozen
            clamp_qlinear_weights(model)          # keep unfrozen in [-1, 1]
            opt.zero_grad(set_to_none=True)
            global_step += 1

            step_loss = running / max(1, running_n)

            # --- warmup vs commit gate ---
            if is_warmup:
                if global_step >= args.warmup_rounds_steps:
                    tqdm.write(f"[warmup] done at step {global_step}; "
                               f"starting commit rounds")
                    gate.reset()
                    is_warmup = False
            else:
                should_commit, reason = gate.step(step_loss, L_T)
                if should_commit:
                    tzf = (args.target_zero_frac
                           if 0.0 < args.target_zero_frac < 1.0 else None)
                    n_c, sel_stats = select_and_commit_one_per_group(
                        model, opt, args.momentum_weight, tzf)
                    if args.post_commit_momentum_damp != 1.0:
                        for p in model.parameters():
                            st = opt.state.get(p, {})
                            if "exp_avg" in st:
                                st["exp_avg"].mul_(args.post_commit_momentum_damp)
                    enforce_frozen(model)  # re-pin after damp
                    if args.device.startswith("cuda"):
                        torch.cuda.empty_cache()
                    round_idx += 1
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

            # --- weight histogram ---
            if (args.soft_hist_every > 0
                    and global_step % args.soft_hist_every == 0):
                hs = first_qlinear_forward_sample(model)
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

    # ---- Done: fold c_{r,g} into scales (math-preserving) → codebook {-1,0,+1} ----
    # Per-group: w /= c_{r,g}, s *= c_{r,g}. Committed slots at ±c_{r,g}
    # latent become ±1 in deploy form; scales absorb the magnitude.
    print(f"[progressive] complete after {global_step} steps; "
          f"committed_frac={total_committed_fraction(model):.4f}")
    with torch.no_grad():
        for m in model.modules():
            if not isinstance(m, QLinear):
                continue
            c_elem = _c_per_element(m)
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
