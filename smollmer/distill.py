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
from .qlinear import (QLinear, clamp_qlinear_weights, quantize_levels,
                      set_levels, set_sherry, set_sherry_alpha,
                      set_top1, set_top1_alpha)


_INTERRUPT = {"flag": False}


def _install_sigint_handler() -> None:
    def handler(signum, frame):
        if _INTERRUPT["flag"]:
            print("\n[!!] second SIGINT — hard exit, no save", flush=True)
            sys.exit(130)
        _INTERRUPT["flag"] = True
        print("\n[!] SIGINT — finishing current step, then saving resume state. "
              "Press Ctrl-C again to hard-exit.", flush=True)
    signal.signal(signal.SIGINT, handler)


def save_resume(path: Path, model, opt, stage_idx: int, next_step: int,
                curriculum: list[tuple[int, int]],
                global_step: int,
                best_snapshot: dict[str, torch.Tensor] | None,
                ctrl_state: dict | None,
                run_name: str | None) -> None:
    payload = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "opt": opt.state_dict(),
        "stage_idx": int(stage_idx),
        "next_step": int(next_step),
        "curriculum": curriculum,
        "global_step": int(global_step),
        "best_snapshot": best_snapshot,
        "ctrl_state": ctrl_state,
        "run_name": run_name,
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


DEFAULT_CURRICULUM: list[tuple[int, int]] = [
    # (levels, max_steps).  Advance early if EMA loss plateaus
    # (see --patience).  L=257 is the "match the teacher" anchor and
    # gets a generous cap; later stages are perturbations.
    (257, 5000), (129, 2000), (65, 2000), (33, 3000),
    (17, 4000), (9, 5000), (5, 8000), (3, 15000),
]


class PlateauController:
    """Track EMA loss; advance when it stops improving for `patience` steps.

    `ema_warmup` skips best-EMA tracking for the first N steps so the seed
    value (the first step's loss, taken at LR warmup minimum) doesn't lock
    in a forever-best that later, properly-trained losses can't beat. EMA
    still updates during warmup; only the best is frozen until step >= warmup.

    `flip_threshold` (>0) gates patience-based advance on the codebook
    actually being frozen, not just the loss being flat. Patience-fire is
    honored only when the mean `weights/flip_rate` over the patience window
    is below `flip_threshold` (e.g. 1e-4 = 0.01% of trits flipping per
    log-interval, on average). Loss EMA can plateau while the codebook is
    still reorganizing (esp. L=5 → L=3 collapse) — flip_rate goes to zero
    only when there's nothing left to extract at this quantization level.
    Set to 0 to disable. Samples fed via `record_flip(step, flip_rate)`.
    """

    def __init__(self, max_steps: int, patience: int, min_steps: int,
                 ema_alpha: float = 0.05, rel_threshold: float = 1e-3,
                 ema_warmup: int = 0,
                 flip_threshold: float = 0.0) -> None:
        self.max_steps = max_steps
        self.patience = patience
        self.min_steps = min_steps
        self.ema_alpha = ema_alpha
        self.rel_threshold = rel_threshold
        self.ema_warmup = ema_warmup
        self.flip_threshold = flip_threshold
        self.ema: float | None = None
        self.best_ema: float = float("inf")
        self.best_step: int = 0
        # (step, flip_rate) samples; trimmed to the patience window in
        # record_flip to keep memory at O(patience/log_every).
        self.flip_history: list[tuple[int, float]] = []

    def update(self, step: int, loss: float) -> bool:
        """Update EMA. Returns True iff this step set a new best."""
        self.ema = loss if self.ema is None else \
            (1.0 - self.ema_alpha) * self.ema + self.ema_alpha * loss
        if step < self.ema_warmup:
            return False
        if self.ema < self.best_ema * (1.0 - self.rel_threshold):
            self.best_ema = self.ema
            self.best_step = step
            return True
        return False

    def record_flip(self, step: int, flip_rate: float) -> None:
        self.flip_history.append((step, float(flip_rate)))
        if self.patience > 0:
            cutoff = step - self.patience
            self.flip_history = [
                (s, f) for s, f in self.flip_history if s >= cutoff]

    def _flip_gate(self, step: int) -> tuple[bool, str]:
        """Return (gate_open, reason). gate_open=True means flip_rate has
        dropped enough that patience-fire is allowed."""
        if self.flip_threshold <= 0:
            return True, "gate disabled"
        if not self.flip_history:
            return True, "no flip data"
        rates = [f for _, f in self.flip_history]
        mean_rate = sum(rates) / len(rates)
        if mean_rate < self.flip_threshold:
            return True, f"flip plateaued (mean={mean_rate:.2e})"
        return False, f"flip still active (mean={mean_rate:.2e})"

    def should_advance(self, step: int) -> tuple[bool, str]:
        if step + 1 >= self.max_steps:
            return True, "max_steps"
        if step + 1 < self.min_steps:
            return False, ""
        if self.patience > 0 and (step - self.best_step) >= self.patience:
            gate_open, gate_reason = self._flip_gate(step)
            if gate_open:
                return True, (f"plateau (best ema {self.best_ema:.4f} at step "
                              f"{self.best_step}; {gate_reason})")
            # Patience expired but codebook still flipping — keep training so
            # the trits can finish settling.
            return False, ""
        return False, ""

    def state_dict(self) -> dict:
        return {
            "ema": self.ema,
            "best_ema": self.best_ema,
            "best_step": self.best_step,
            "flip_history": list(self.flip_history),
        }

    def load_state_dict(self, state: dict) -> bool:
        """Restore EMA tracking. Returns True if a stale (within-warmup) best
        was discarded; the caller should also drop its best_snapshot in that
        case so a poisoned step-0 model isn't restored at stage end."""
        self.ema = state.get("ema")
        bs = int(state.get("best_step", 0))
        if bs < self.ema_warmup:
            self.best_ema = float("inf")
            self.best_step = 0
            stale = True
        else:
            self.best_ema = state.get("best_ema", float("inf"))
            self.best_step = bs
            stale = False
        self.flip_history = list(state.get("flip_history", []))
        return stale


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


@torch.no_grad()
def quantized_codes(m: QLinear) -> torch.Tensor:
    """Integer code in [-half, half] for each weight at the module's current level.
    Used to compute bin occupancy and step-to-step flip rate."""
    half = (m.levels - 1) // 2
    q = quantize_levels(m.weight, m.levels)
    return (q * half).round().to(torch.int8)


@torch.no_grad()
def collect_qlinear_metrics(
    model: torch.nn.Module,
    prev_codes: dict[str, torch.Tensor],
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Aggregate ternary-QAT health metrics across every QLinear module."""
    new_codes: dict[str, torch.Tensor] = {}
    total = 0
    n_zero = 0
    n_extreme = 0
    n_flip = 0
    n_compared = 0
    scale_max = 0.0
    sum_scale_mean = 0.0
    n_layers = 0
    for name, m in model.named_modules():
        if not isinstance(m, QLinear):
            continue
        half = (m.levels - 1) // 2
        codes = quantized_codes(m)
        total += codes.numel()
        n_zero += int((codes == 0).sum())
        n_extreme += int(((codes == half) | (codes == -half)).sum())
        prev = prev_codes.get(name)
        if prev is not None and prev.shape == codes.shape:
            n_flip += int((codes != prev).sum())
            n_compared += codes.numel()
        new_codes[name] = codes
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
    return metrics, new_codes


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
    def __init__(self, shard_dir: Path, seed: int = 0) -> None:
        self.paths = sorted(Path(shard_dir).glob("shard_*.safetensors"))
        if not self.paths:
            raise FileNotFoundError(f"no shard_*.safetensors under {shard_dir}")
        self.seed = seed

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        wid = worker.id if worker else 0
        nworkers = worker.num_workers if worker else 1
        rng = random.Random(self.seed + wid * 7919)
        my_paths = self.paths[wid::nworkers] or self.paths
        while True:
            order = list(my_paths)
            rng.shuffle(order)
            for p in order:
                shard = load_file(str(p))
                S = shard["tokens"].shape[0]
                idx_order = list(range(S))
                rng.shuffle(idx_order)
                for i in idx_order:
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


def parse_curriculum(spec: str) -> list[tuple[int, int]]:
    if not spec:
        return DEFAULT_CURRICULUM
    out: list[tuple[int, int]] = []
    for chunk in spec.split(","):
        L, n = chunk.split(":")
        out.append((int(L), int(n)))
    return out


def parse_lr_overrides(spec: str) -> dict[int, float]:
    """Per-level LR overrides spec, e.g. '9:2e-4,5:1.5e-4'."""
    if not spec:
        return {}
    out: dict[int, float] = {}
    for chunk in spec.split(","):
        L, lr = chunk.split(":")
        out[int(L)] = float(lr)
    return out


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


def save_checkpoint(model, path: Path, levels: int, stage: int, model_id: str,
                    sherry: bool, top1: bool = False) -> None:
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(sd, str(path), metadata={
        "levels": str(levels),
        "stage": str(stage),
        "model_id": model_id,
        "sherry": "1" if sherry else "0",
        "top1": "1" if top1 else "0",
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None,
                    help="Stage checkpoint (.safetensors) to load before training.")
    ap.add_argument("--start-stage", type=int, default=None,
                    help="Skip curriculum stages before this index. "
                         "If unset and --resume is given, auto-advances to "
                         "ckpt's stage+1 (i.e. starts the NEXT stage). "
                         "Pass the same stage as the ckpt to redo it.")
    ap.add_argument("--curriculum", type=str, default="",
                    help="Override default curriculum, e.g. `33:200,17:200,9:300,5:500,3:1000`")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--optimizer", default="lion",
                    choices=["lion", "adamw", "cautious-adamw"],
                    help="Lion (sign-momentum, sweet spot 3e-4..1e-3 for ternary). "
                         "AdamW (standard). Cautious-AdamW (Bonsai recipe; one-line "
                         "mod from Liang et al. 2024 arXiv:2411.16085, lr~1e-2).")
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="Lion LR. Sweet spot for Lion + ternary QAT is 3e-4 to "
                         "1e-3 (per bitlooplm sweep + Bonsai recipe). At lr=5e-5 "
                         "only ~4%% of weights can flip bins per 100 steps -- "
                         "too slow for L=3. The best-snapshot restore catches "
                         "the case where high LR thrashes a near-optimal init.")
    ap.add_argument("--lr-overrides", type=str, default="",
                    help="Per-stage LR overrides as 'L:lr[,L:lr...]', e.g. "
                         "'9:2e-4,5:1.5e-4'. Stages not listed use --lr.")
    ap.add_argument("--lr-floor", type=float, default=0.1,
                    help="Cosine LR decays to floor*base_lr at end of stage. "
                         "Default 0.1 = classic Bonsai recipe. Bump up (e.g. "
                         "0.5) when the late-stage flip-rate cliff at L=3 is "
                         "locking the codebook into a suboptimal Sherry-valid "
                         "configuration the optimizer can't escape. 1.0 = flat "
                         "LR after warmup (no cosine decay at all).")
    ap.add_argument("--sherry-warmup-steps", type=int, default=0,
                    help="Linearly ramp the Sherry constraint from off "
                         "(alpha=0, plain quantize_levels) to on (alpha=1) "
                         "over the first N steps of each Sherry-enabled "
                         "stage. Lets the network find a good ternary sign "
                         "pattern before being pinned to a Sherry-valid "
                         "configuration. 0 disables (instant full Sherry, "
                         "old behavior). Try 2000 for L=3.")
    ap.add_argument("--permute-staleness-threshold", type=float, default=0.0,
                    help="Adaptively re-permute mid-stage when the column-"
                         "score alignment of down_proj+o_proj drifts: trigger "
                         "permute_for_sherry when permutation_staleness "
                         "exceeds this fraction. 0 disables (default). "
                         "Reasonable values: 0.2-0.4. Suppressed in the last "
                         "10%% of each stage to let the codebook settle.")
    ap.add_argument("--permute-min-gap-steps", type=int, default=200,
                    help="Minimum steps between adaptive re-permutes within "
                         "a stage. Prevents rapid back-to-back permutes when "
                         "staleness oscillates near the threshold.")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup-steps", type=int, default=30)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=750,
                    help="Advance to next stage if EMA loss hasn't improved for "
                         "this many steps. Set 0 to always run full max_steps.")
    ap.add_argument("--min-stage-steps", type=int, default=200,
                    help="Always run at least this many steps in each stage.")
    ap.add_argument("--plateau-threshold", type=float, default=1e-3,
                    help="Minimum relative EMA drop counted as 'improvement'.")
    ap.add_argument("--flip-plateau-threshold", type=float, default=1e-4,
                    help="Gate patience-based stage advance on weights/flip_rate "
                         "(fraction of trits that changed since the previous "
                         "log-interval). Patience-fire is honored only when the "
                         "MEAN flip_rate over the last `patience` steps is below "
                         "this threshold. Default 1e-4 = 0.01%% of trits "
                         "flipping per interval on average. Loss EMA can flatten "
                         "while the codebook is still reorganizing "
                         "(esp. L=5→L=3 collapse) — flip_rate going to zero is "
                         "the direct signal that there's nothing left to extract "
                         "at this level. Set 0 to disable. Does not gate "
                         "max_steps advance.")
    ap.add_argument("--ema-warmup", type=int, default=500,
                    help="Skip best-EMA tracking for the first N steps of each "
                         "stage. The Lion-LR-warmup loss is often U-shaped past "
                         "the LR-warmup window (peak around step ~200 in this "
                         "setup), so set this past the typical peak — otherwise "
                         "the first post-warmup EMA value, which is still on the "
                         "way up, locks in as best.")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--autocast-dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "none"])
    ap.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=True,
                    help="Trade some compute for activation memory (recommended on <=8GB).")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tb-dir", type=Path, default=None,
                    help="TensorBoard root. Default: <out>/tb. "
                         "View with `tensorboard --logdir <tb-dir>`.")
    ap.add_argument("--run-name", type=str, default=None,
                    help="Subdirectory under --tb-dir for this run. "
                         "Default: timestamped. Resumed runs reuse the saved name.")
    ap.add_argument("--sherry", action=argparse.BooleanOptionalAction, default=False,
                    help="Apply the Sherry-encoding constraint to every QLinear: "
                         "in each contiguous block of 4 weights along in_features, "
                         "force the smallest-|w| slot to 0. Forces 75%% of trits to "
                         "be ±1 — far more dense than the natural ~85%%-zero ternary "
                         "distribution. Mutually exclusive with --top1. Pass "
                         "consistently across resumes; saved in ckpt metadata.")
    ap.add_argument("--top1", action=argparse.BooleanOptionalAction, default=False,
                    help="Apply the top1 constraint to every QLinear: in each "
                         "block of 4, keep only the largest-|w| slot at its "
                         "quantized value (±1 at L=3); zero the other 3. Forces "
                         "25%% nonzero density, much closer to natural ~14%% "
                         "nonzero — entropy-optimal at 0.75 bpw vs Sherry's 1.25 "
                         "bpw. Mutually exclusive with --sherry. Pass "
                         "consistently across resumes; saved in ckpt metadata.")
    ap.add_argument("--top1-warmup-steps", type=int, default=0,
                    help="Linearly ramp the top1 constraint from off (alpha=0, "
                         "plain quantize_levels) to on (alpha=1) over the first "
                         "N steps of each top1-enabled stage. Mirror of "
                         "--sherry-warmup-steps. Try 2000 at L=3.")
    ap.add_argument("--permute-for-sherry", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Apply free-dim math-preserving permutations once at "
                         "init: down_proj cols + up_proj/gate_proj rows on the "
                         "MLP intermediate dim, and v_proj rows + o_proj cols on "
                         "per-KV-head head_dim slices. Goal: align block-position "
                         "0 with the constraint's designated slot (smallest-|w| "
                         "for sherry, largest-|w| for top1). The flag name is "
                         "historical; it works for whichever constraint is "
                         "active. Skipped on --resume / interrupted snapshot. "
                         "Implied (and redundant) when --permute-each-stage is "
                         "set.")
    ap.add_argument("--permute-each-stage", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Re-apply the free-dim permutation at the start of "
                         "every curriculum stage, scoring against the current "
                         "trained weights and syncing optimizer state alongside. "
                         "Only fires when entering a stage cleanly (skipped if "
                         "we're resuming inside an in-progress stage from an "
                         "interrupt — the on-disk weights are already post-"
                         "permute for that stage).")
    args = ap.parse_args()
    if args.sherry and args.top1:
        ap.error("--sherry and --top1 are mutually exclusive")

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
    model, _tok, n_replaced = load_student(args.model, dtype=torch.float32,
                                           levels=257,
                                           latent_dtype=latent_dtype)
    print(f"[build] {n_replaced} QLinear modules (latent dtype: {latent_dtype})")
    if args.top1:
        n_top1 = set_top1(model, True)
        print(f"[build] top1 constraint enabled on {n_top1} layers")
    elif args.sherry:
        n_sherry = set_sherry(model, True)
        print(f"[build] sherry constraint enabled on {n_sherry} layers")
    model = model.to(args.device)
    if hasattr(model, "config"):
        model.config.use_cache = False  # never needed for training fwd/bwd
    if args.grad_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("[build] gradient checkpointing enabled")
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
        ckpt_sherry = resume_meta.get("sherry") == "1"
        ckpt_top1 = resume_meta.get("top1") == "1"
        if ckpt_sherry != args.sherry:
            print(f"[resume] WARNING: --sherry={args.sherry} but ckpt has "
                  f"sherry={ckpt_sherry}; using --sherry value")
        if ckpt_top1 != args.top1:
            print(f"[resume] WARNING: --top1={args.top1} but ckpt has "
                  f"top1={ckpt_top1}; using --top1 value")
    elif interrupted_path.exists():
        print(f"[resume] found interrupted snapshot at {interrupted_path}")
        interrupted_state = torch.load(str(interrupted_path),
                                       map_location=args.device,
                                       weights_only=False)
        model.load_state_dict(interrupted_state["model"])

    # The active block-constraint mode dictates which perm to use. Both
    # permute_for_sherry and permute_for_top1 share infrastructure but
    # differ in slot-0 selection (smallest-|w| for sherry, largest-|w| for
    # top1). 'plain' = no permutation makes sense (no constraint to align to).
    constraint_mode = "top1" if args.top1 else ("sherry" if args.sherry else "plain")

    def _do_permute(opt=None):
        from .permute import permute_for_sherry, permute_for_top1
        if constraint_mode == "top1":
            return permute_for_top1(model, block=4, optimizer=opt)
        if constraint_mode == "sherry":
            return permute_for_sherry(model, block=4, optimizer=opt)
        raise RuntimeError("--permute-for-sherry passed but no constraint "
                           "(--sherry / --top1) is active")

    fresh_start = args.resume is None and interrupted_state is None
    # --permute-each-stage handles stage 0's permute itself; only fire the
    # init-time permute when running with --permute-for-sherry alone.
    if args.permute_for_sherry and not args.permute_each_stage:
        if fresh_start:
            n_perm = _do_permute()
            print(f"[build] permute-for-{constraint_mode} applied to {n_perm} "
                  "matrices (MLP intermediate + per-KV-head V/O)")
        else:
            print("[build] WARNING: --permute-for-sherry ignored on resume "
                  "(loaded weights already encode any prior permutation)")

    if args.compile:
        model = torch.compile(model)

    if args.optimizer == "lion":
        opt = Lion32(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == "adamw":
        opt = AdamW32(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == "cautious-adamw":
        opt = CautiousAdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise ValueError(f"unknown optimizer: {args.optimizer}")
    print(f"[opt] {args.optimizer} lr={args.lr} wd={args.wd} (fp32 state)")

    curriculum = parse_curriculum(args.curriculum)
    lr_overrides = parse_lr_overrides(args.lr_overrides)
    print(f"[plan] curriculum: {curriculum}")
    if lr_overrides:
        print(f"[plan] lr_overrides: {lr_overrides}")

    if interrupted_state is not None:
        opt.load_state_dict(interrupted_state["opt"])
        start_stage = interrupted_state["stage_idx"]
        start_step = interrupted_state["next_step"]
        if args.start_stage is not None:
            print(f"[resume] --start-stage {args.start_stage} overrides snapshot's stage {start_stage}")
            start_stage = args.start_stage
            start_step = 0
        else:
            print(f"[resume] continuing at stage {start_stage} step {start_step}")
        if interrupted_state.get("curriculum") and interrupted_state["curriculum"] != curriculum:
            print(f"[resume] WARNING: curriculum changed since interrupt — "
                  f"snapshot had {interrupted_state['curriculum']}")
        global_step = int(interrupted_state.get("global_step", 0))
    else:
        start_step = 0
        global_step = 0
        if args.start_stage is not None:
            start_stage = args.start_stage
            print(f"[plan] starting at stage {start_stage}")
        elif args.resume is not None and "stage" in resume_meta:
            start_stage = int(resume_meta["stage"]) + 1
            print(f"[plan] auto-advancing to stage {start_stage} "
                  f"(checkpoint was saved at end of stage {resume_meta['stage']}; "
                  f"pass --start-stage to override)")
        else:
            start_stage = 0

    tb_root = args.tb_dir if args.tb_dir is not None else (args.out / "tb")
    if interrupted_state is not None and interrupted_state.get("run_name"):
        run_name = interrupted_state["run_name"]
    elif args.run_name:
        run_name = args.run_name
    else:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = tb_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[tb] run_dir={run_dir} (view: tensorboard --logdir {tb_root}); "
          "each stage writes to a separate sub-run.")
    _, prev_codes = collect_qlinear_metrics(model, {})

    ds = ShardedDataset(args.cache_dir, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=(args.device.startswith("cuda")),
                    drop_last=True)
    it = iter(dl)

    autocast_dtype = {"bfloat16": torch.bfloat16,
                      "float16": torch.float16,
                      "none": None}[args.autocast_dtype]

    for stage_idx, (levels, max_steps) in enumerate(curriculum):
        if stage_idx < start_stage:
            continue
        n_set = set_levels(model, levels)
        # Per-stage embed reference. On a mid-stage resume this captures the
        # current (already-drifted) embed rather than the true stage-start
        # value — accept the underreporting for that partial stage; every
        # clean stage entry is accurate.
        _embed = model.get_input_embeddings()
        embed_stage_init = (_embed.weight.detach().clone()
                            if _embed is not None else None)
        stage_lr = lr_overrides.get(levels, args.lr)
        if levels in lr_overrides:
            print(f"[stage {stage_idx}] lr override: {stage_lr} (vs default {args.lr})")
        step_offset = start_step if stage_idx == start_stage else 0
        if args.permute_each_stage:
            # Skip if we're resuming inside this stage from an interrupt — the
            # restored model + opt-state are already post-permute for this
            # stage. (Subsequent stages' interrupted_state is None, so they
            # always permute when entered.)
            is_interrupted_resume = (interrupted_state is not None
                                     and stage_idx == interrupted_state["stage_idx"])
            if not is_interrupted_resume:
                n_perm = _do_permute(opt=opt)
                print(f"[stage {stage_idx}] permuted {n_perm} matrices "
                      "(--permute-each-stage; opt state synced)")
            else:
                print(f"[stage {stage_idx}] skipping permute (resuming "
                      "in-progress stage from interrupt)")
        ctrl = PlateauController(max_steps=max_steps,
                                 patience=args.patience,
                                 min_steps=max(args.min_stage_steps, step_offset + 1),
                                 rel_threshold=args.plateau_threshold,
                                 ema_warmup=args.ema_warmup,
                                 flip_threshold=args.flip_plateau_threshold)
        # Remember the best-so-far model in CPU RAM. Lion can thrash around
        # near a minimum; restoring the best snapshot at stage end avoids
        # carrying regression damage into the next stage.
        stale_best = False
        if (interrupted_state is not None and stage_idx == interrupted_state["stage_idx"]
                and interrupted_state.get("best_snapshot") is not None):
            if interrupted_state.get("ctrl_state"):
                stale_best = ctrl.load_state_dict(interrupted_state["ctrl_state"])
                if stale_best:
                    print(f"[resume] discarded stale best (saved within "
                          f"ema_warmup={args.ema_warmup}); will re-snapshot")
                else:
                    print(f"[resume] restored controller (ema={ctrl.ema}, "
                          f"best_ema={ctrl.best_ema}@{ctrl.best_step}) and best_snapshot")
            best_snapshot = (snapshot_to_cpu(model) if stale_best
                             else interrupted_state["best_snapshot"])
        else:
            best_snapshot = snapshot_to_cpu(model)
        # Per-stage TB writer. Each stage gets its own sub-run so curves are
        # directly comparable across stages (TB's compare-runs view overlays
        # them at matching in-stage step). On mid-stage resume, purge_step
        # drops any events from this run that are at >= step_offset so we
        # don't double-write across the interrupt boundary.
        stage_tb_dir = run_dir / f"stage_{stage_idx:02d}_L{levels}"
        stage_tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(stage_tb_dir),
                               purge_step=step_offset if step_offset else None)
        writer.add_text("stage", f"stage {stage_idx} levels={levels} "
                                  f"max_steps={max_steps} step_offset={step_offset} "
                                  f"global_step={global_step}",
                        step_offset)
        # Re-snapshot the quantized codes so flip-rate measures change *within
        # this stage's level setting*, not across the level transition.
        _, prev_codes = collect_qlinear_metrics(model, {})
        extras = []
        if args.lr_floor != 0.1:
            extras.append(f"lr_floor={args.lr_floor}")
        if args.sherry and args.sherry_warmup_steps > 0:
            extras.append(f"sherry_warmup={args.sherry_warmup_steps}")
        if args.top1 and args.top1_warmup_steps > 0:
            extras.append(f"top1_warmup={args.top1_warmup_steps}")
        extras_str = (" " + ", ".join(extras)) if extras else ""
        if step_offset:
            print(f"\n[stage {stage_idx}] levels={levels} on {n_set} layers, "
                  f"resuming at step {step_offset}/{max_steps} "
                  f"(patience={args.patience}, min={args.min_stage_steps}{extras_str})")
        else:
            print(f"\n[stage {stage_idx}] levels={levels} on {n_set} layers, "
                  f"max {max_steps} steps "
                  f"(patience={args.patience}, min={args.min_stage_steps}{extras_str})")
        model.train()
        opt.zero_grad(set_to_none=True)
        running = 0.0
        running_n = 0
        # Track when we last re-permuted in this stage; used by the adaptive
        # staleness trigger to enforce --permute-min-gap-steps between fires.
        # Initialize to step_offset so a stage-resumed from interrupt won't
        # immediately fire (the perm is still fresh from the recent run).
        last_permute_step = step_offset
        pbar = tqdm(range(step_offset, max_steps), initial=step_offset, total=max_steps,
                    desc=f"L={levels}", dynamic_ncols=True)
        advance_reason = "max_steps"
        last_step = step_offset
        for step in pbar:
            last_step = step
            cur_lr = lr_at(step, max_steps, stage_lr, args.warmup_steps,
                           floor=args.lr_floor)
            for g in opt.param_groups:
                g["lr"] = cur_lr
            # Linear constraint-warmup ramp: 0 → 1 over the first N in-stage
            # steps (uses absolute in-stage step so a mid-warmup resume picks
            # up where it left off). Cache is invalidated each opt.step by
            # clamp_qlinear_weights so the new alpha takes effect on the first
            # forward of this step.
            if args.top1 and args.top1_warmup_steps > 0:
                alpha = min(1.0, (step + 1) / args.top1_warmup_steps)
                set_top1_alpha(model, alpha)
            elif args.sherry and args.sherry_warmup_steps > 0:
                alpha = min(1.0, (step + 1) / args.sherry_warmup_steps)
                set_sherry_alpha(model, alpha)
            for _ in range(args.grad_accum):
                batch = next(it)
                tokens = batch["tokens"].to(args.device, non_blocking=True)
                topk_idx = batch["topk_idx"].to(args.device, non_blocking=True)
                topk_prob = batch["topk_prob"].to(args.device, non_blocking=True)
                rest_mass = batch["rest_mass"].to(args.device, non_blocking=True)
                ctx = (torch.amp.autocast(args.device.split(":")[0], dtype=autocast_dtype)
                       if autocast_dtype is not None else torch.amp.autocast(args.device.split(":")[0], enabled=False))
                with ctx:
                    out = model(tokens)
                    loss = kl_with_rest(out.logits, topk_idx, topk_prob, rest_mass)
                (loss / args.grad_accum).backward()
                running += loss.item()
                running_n += 1
            grad_norm: float | None = None
            if args.max_grad_norm:
                g = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                grad_norm = float(g)
            opt.step()
            clamp_qlinear_weights(model)
            opt.zero_grad(set_to_none=True)
            global_step += 1
            step_loss = running / max(1, running_n)
            improved = ctrl.update(step, step_loss)
            if improved:
                best_snapshot = snapshot_to_cpu(model)
            if (step + 1) % args.log_every == 0:
                pbar.set_postfix(loss=f"{step_loss:.4f}",
                                 ema=f"{ctrl.ema:.4f}",
                                 best=f"{ctrl.best_ema:.4f}@{ctrl.best_step}",
                                 lr=f"{cur_lr:.2e}")
                # Per-stage runs use in-stage step on the x-axis so curves
                # from different stages overlay cleanly in TB's compare view.
                tb_step = step + 1
                writer.add_scalar("loss/step", step_loss, tb_step)
                if ctrl.ema is not None:
                    writer.add_scalar("loss/ema", ctrl.ema, tb_step)
                writer.add_scalar("lr", cur_lr, tb_step)
                writer.add_scalar("levels", float(levels), tb_step)
                writer.add_scalar("global_step", float(global_step), tb_step)
                if grad_norm is not None:
                    writer.add_scalar("grad_norm", grad_norm, tb_step)
                qm, prev_codes = collect_qlinear_metrics(model, prev_codes)
                for k, v in qm.items():
                    writer.add_scalar(k, v, tb_step)
                fr = qm.get("weights/flip_rate")
                if fr is not None:
                    ctrl.record_flip(step, fr)
                drift, drift_stage = embed_drift_l2(model, embed_init,
                                                    embed_stage_init)
                if drift is not None:
                    writer.add_scalar("embed/drift_l2", drift, tb_step)
                if drift_stage is not None:
                    writer.add_scalar("embed/drift_l2_stage", drift_stage,
                                      tb_step)
                # Adaptive re-permute: trigger when the column-score alignment
                # has drifted past threshold, gated on (1) being out of the
                # last 10% of the stage so we don't disrupt convergence, and
                # (2) having waited at least --permute-min-gap-steps since
                # the previous permute. Mode-aware: argmin invariant for
                # sherry (slot 0 = smallest), argmax for top1 (slot 0 =
                # largest); _do_permute already routes to the right function.
                if (args.permute_staleness_threshold > 0
                        and constraint_mode in ("sherry", "top1")):
                    from .permute import permutation_staleness
                    staleness = permutation_staleness(model, block=4,
                                                      mode=constraint_mode)
                    writer.add_scalar("permute/staleness", staleness, tb_step)
                    in_late_stage = step >= int(0.9 * max_steps)
                    enough_gap = (step - last_permute_step) >= args.permute_min_gap_steps
                    if (staleness > args.permute_staleness_threshold
                            and enough_gap and not in_late_stage):
                        n_perm = _do_permute(opt=opt)
                        last_permute_step = step
                        writer.add_scalar("permute/fired", 1.0, tb_step)
                        # Permute is a "soft restart" of stage progress. Reset:
                        # - best_snapshot: commits to the new layout, prevents
                        #   stage-end regress check from silently undoing the
                        #   permute by restoring a pre-permute snapshot.
                        # - best_ema / best_step: pre-permute loss isn't
                        #   directly comparable post-permute (model is briefly
                        #   perturbed); restart the patience clock.
                        # - prev_codes: avoid a spurious 100% flip_rate on the
                        #   next log_every (everything just moved).
                        # - flip_history: post-permute flip_rate isn't
                        #   comparable to pre-permute, so don't average them.
                        best_snapshot = snapshot_to_cpu(model)
                        ctrl.best_ema = float("inf")
                        ctrl.best_step = step
                        ctrl.flip_history = []
                        _, prev_codes = collect_qlinear_metrics(model, {})
                        print(f"\n[stage {stage_idx}] step {step}: "
                              f"re-permuted {n_perm} matrices "
                              f"(staleness={staleness:.3f}); reset "
                              "best_snapshot, best_ema, flip_history, prev_codes")
                    else:
                        writer.add_scalar("permute/fired", 0.0, tb_step)
                running = 0.0
                running_n = 0
            if _INTERRUPT["flag"]:
                pbar.close()
                save_resume(interrupted_path, model, opt, stage_idx, step + 1, curriculum,
                            global_step, best_snapshot, ctrl.state_dict(), run_name)
                writer.flush()
                writer.close()
                print(f"[!] saved resume snapshot to {interrupted_path}; "
                      f"re-run smollmer-distill to continue.")
                sys.exit(0)
            advance, why = ctrl.should_advance(step)
            if advance:
                advance_reason = why
                pbar.close()
                break
        regressed = ctrl.ema is not None and ctrl.ema > ctrl.best_ema * 1.005
        print(f"[stage {stage_idx}] advancing after {last_step + 1} steps "
              f"(reason: {advance_reason}; ema={ctrl.ema:.4f}, best={ctrl.best_ema:.4f}@{ctrl.best_step})")
        writer.add_text("stage_end",
                        f"stage {stage_idx} levels={levels} steps={last_step + 1} "
                        f"reason={advance_reason} ema={ctrl.ema:.4f} "
                        f"best={ctrl.best_ema:.4f}@{ctrl.best_step} regressed={regressed}",
                        last_step + 1)
        writer.flush()
        writer.close()
        if regressed:
            print(f"[stage {stage_idx}] restoring best snapshot from step {ctrl.best_step} "
                  f"(EMA regressed {ctrl.ema:.4f} > best {ctrl.best_ema:.4f})")
            restore_from_snapshot(model, best_snapshot)
        # Stage complete; clear in-stage start_step so subsequent stages start fresh.
        # Drop the interrupted_state reference so we don't reapply it on later
        # stages — its best_snapshot is for the now-finished stage only.
        start_step = 0
        interrupted_state = None
        ckpt_path = args.out / f"stage_{stage_idx:02d}_L{levels}.safetensors"
        save_checkpoint(model, ckpt_path, levels, stage_idx, args.model,
                        args.sherry, args.top1)
        print(f"[stage {stage_idx}] saved {ckpt_path}")

    if interrupted_path.exists():
        interrupted_path.unlink()
        print(f"[done] removed {interrupted_path}")
    print("[done] curriculum complete.")


if __name__ == "__main__":
    main()
