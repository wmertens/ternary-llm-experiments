"""hrm_bop — trainer for the HRM-style ternary recurrent LM.

Random-init trits (50% zero, 25% ±1), per-(row, group) scales initialized to
sqrt(2/in_features), BopTernary (Bet 1, |m|/sqrt(v) > τ_norm) on the trits,
Lion32 with cosine schedule on FP params (embeddings, norms, z_L_init,
scales). CE on the final z_H only; no teacher.

Pretty much the same skeleton as smollmer/flip_distill.py — SIGINT-safe
auto-checkpoint to interrupted.pt, BestEmaTracker, snapshot_to_cpu, and
the TB scalar set called out in hrm_bop_spec.md.

Usage:
    smollmer-hrm-bop --out ckpts/hrm-A --total-steps 40000

Resume on next launch is automatic (interrupted.pt). For warm-start with
fresh optimizers, pass `--resume PATH.safetensors`.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .cmuon import CMuon
from .distill import (BestEmaTracker, Lion32, _INTERRUPT,
                      _install_sigint_handler, lr_at, save_resume,
                      snapshot_to_cpu)
from .flip_distill import BopTernary, m_stats, trit_stats
from .hrm_data import make_train_loader, make_val_loader
from .hrm_model import HrmBopConfig, HrmBopModel, RMSNorm
from .qlinear import QLinear, clamp_qlinear_weights, quantize_levels


# ---------------------------------------------------------------- init


@torch.no_grad()
def init_trits_random(model: nn.Module, zero_frac: float = 0.5,
                      generator: torch.Generator | None = None) -> int:
    """Replace every QLinear.weight with a fresh random ternary draw.

    P(0)=zero_frac, P(+1)=P(-1)=(1-zero_frac)/2. Returns the number of
    QLinear modules initialized.
    """
    n = 0
    for m in model.modules():
        if not isinstance(m, QLinear):
            continue
        device = m.weight.device
        u = torch.rand(m.weight.shape, device=device, generator=generator)
        # u < zero_frac → 0; else ±1 with equal probability.
        rest = (1.0 - zero_frac) / 2.0
        t = torch.where(u < zero_frac, torch.zeros_like(u),
                        torch.where(u < zero_frac + rest,
                                    torch.ones_like(u),
                                    -torch.ones_like(u)))
        m.weight.data.copy_(t.to(m.weight.dtype))
        m.invalidate_q_cache()
        n += 1
    return n


@torch.no_grad()
def init_fp_weights(model: nn.Module,
                    generator: torch.Generator | None = None) -> int:
    """Continuous LeCun-normal init for the FP-weights control: each
    QLinear.weight ~ N(0, 1/fan_in). Replaces the discrete random-ternary
    init (which would be a terrible FP start — 3 values, 50% zeros) and is
    the natural counterpart used with --fp-weights, where the forward skips
    quantize+scales and CMuon trains the raw weight."""
    n = 0
    for m in model.modules():
        if not isinstance(m, QLinear):
            continue
        fan_in = m.weight.shape[1]
        std = fan_in ** -0.5
        noise = torch.randn(m.weight.shape, device=m.weight.device,
                            generator=generator)
        m.weight.data.copy_((noise * std).to(m.weight.dtype))
        m.invalidate_q_cache()
        n += 1
    return n


@torch.no_grad()
def init_scales_fanin(model: nn.Module) -> int:
    """Initialize per-(row, group) scales to sqrt(2/in_features).

    With trits ~50% zero (var(t) ≈ 0.5), this gives unit-variance pre-activations
    for a unit-variance input — the standard fan-in init reasoning, modified
    for the ternary value set.
    """
    n = 0
    for m in model.modules():
        if not isinstance(m, QLinear):
            continue
        s = math.sqrt(2.0 / m.in_features)
        m.scales.data.fill_(s)
        m.invalidate_q_cache()
        n += 1
    return n


@torch.no_grad()
def init_scales_random(model: nn.Module, sigma: float = 0.5,
                       generator: torch.Generator | None = None) -> int:
    """Initialize per-(row, group) scales as `s_init · exp(N(0, sigma))`.

    Same per-layer expected magnitude as the fan-in init, but each (row,
    group) gets its own multiplicative factor (~lognormal, σ=0.5 → ±~50%
    spread). Used in Bop-isolation experiments to remove the "all scales
    equal at init" bias.
    """
    n = 0
    for m in model.modules():
        if not isinstance(m, QLinear):
            continue
        s_init = math.sqrt(2.0 / m.in_features)
        noise = torch.randn(m.scales.shape, device=m.scales.device,
                            generator=generator)
        m.scales.data.copy_(s_init * torch.exp(noise * sigma))
        m.invalidate_q_cache()
        n += 1
    return n


def freeze_scales(model: nn.Module) -> int:
    n = 0
    for m in model.modules():
        if isinstance(m, QLinear):
            m.scales.requires_grad_(False)
            n += 1
    return n


def freeze_non_embed_fp(model: HrmBopModel) -> int:
    """Freeze every FP param except embed_tokens. RMSNorm weights, z_L_init,
    and an untied lm_head all get frozen; embed_tokens stays trainable.
    Used in the Bop-isolation experiment: only the linguistic prior moves
    under Lion, all other FP structure is fixed."""
    n = 0
    for m in model.modules():
        if isinstance(m, RMSNorm):
            m.weight.requires_grad_(False)
            n += 1
    model.z_L_init.requires_grad_(False)
    n += 1
    if model.lm_head is not None:
        model.lm_head.weight.requires_grad_(False)
        n += 1
    return n


# ---------------------------------------------------------------- helpers


def invalidate_all_q_caches(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, QLinear):
            m.invalidate_q_cache()


def split_params(model: HrmBopModel) -> tuple[list[nn.Parameter],
                                              list[nn.Parameter],
                                              list[nn.Parameter]]:
    """Returns (trit_params, scale_params, fp_params). Params with
    `requires_grad=False` are silently dropped from each list, so
    freezing scales / non-embed FP at init makes them invisible here.

    trit_params: every QLinear.weight (managed by Bop). Always trainable
        by convention — never freezing trits, that defeats the point.
    scale_params: every QLinear.scales (managed by Lion).
    fp_params: every other parameter — embeddings, RMSNorm weights,
        z_L_init, optional lm_head (managed by Lion).
    """
    trits: list[nn.Parameter] = []
    scales: list[nn.Parameter] = []
    trit_ids: set[int] = set()
    scale_ids: set[int] = set()
    for m in model.modules():
        if isinstance(m, QLinear):
            trits.append(m.weight)
            trit_ids.add(id(m.weight))
            if m.scales.requires_grad:
                scales.append(m.scales)
            scale_ids.add(id(m.scales))
    fp: list[nn.Parameter] = []
    for p in model.parameters():
        if id(p) in trit_ids or id(p) in scale_ids:
            continue
        if p.requires_grad:
            fp.append(p)
    return trits, scales, fp


@torch.no_grad()
def quantized_codes_snapshot(model: nn.Module) -> dict[int, torch.Tensor]:
    """Snapshot the quantized {-1, 0, +1} code per QLinear weight (by id(m)).
    Used to compute step-to-step code flip rate when --ste-trits is on —
    Bop's flip counter doesn't apply since the latent is continuous."""
    snap: dict[int, torch.Tensor] = {}
    for m in model.modules():
        if isinstance(m, QLinear):
            q = quantize_levels(m.weight.data, m.levels)
            snap[id(m)] = q.to(torch.int8).cpu()
    return snap


@torch.no_grad()
def code_flip_count(model: nn.Module,
                    prev: dict[int, torch.Tensor]) -> tuple[int, int,
                                                            dict[int, torch.Tensor]]:
    """Count trits whose quantized code changed since the prev snapshot.
    Returns (n_flips, n_trits, new_snapshot). If prev is empty (first call),
    returns 0 flips and just records the snapshot."""
    n_flips = 0
    n_trits = 0
    new: dict[int, torch.Tensor] = {}
    for m in model.modules():
        if not isinstance(m, QLinear):
            continue
        q = quantize_levels(m.weight.data, m.levels).to(torch.int8).cpu()
        new[id(m)] = q
        n_trits += q.numel()
        old = prev.get(id(m))
        if old is not None and old.shape == q.shape:
            n_flips += int((q != old).sum().item())
    return n_flips, n_trits, new


@torch.no_grad()
def trit_stats_quantized(model: nn.Module) -> dict[str, float]:
    """trit_stats but reading from the quantized code (works for both Bop
    discrete-weight and STE continuous-latent modes)."""
    n = pos = neg = z = 0
    for m in model.modules():
        if not isinstance(m, QLinear):
            continue
        q = quantize_levels(m.weight.data, m.levels)
        n += q.numel()
        z += int((q == 0).sum().item())
        pos += int((q > 0.5).sum().item())
        neg += int((q < -0.5).sum().item())
    n = max(1, n)
    return {"frac_zero": z / n, "frac_pos": pos / n, "frac_neg": neg / n,
            "n_trits": n}


def trit_stats_per_stack(model: HrmBopModel) -> dict[str, float]:
    """Per-H-vs-L trit fraction-zero (+ overall via flip_distill.trit_stats)."""
    out: dict[str, float] = {}
    for stack_name in ("H_stack", "L_stack"):
        stack = getattr(model, stack_name)
        n_total = n_zero = 0
        for m in stack.modules():
            if not isinstance(m, QLinear):
                continue
            t = m.weight.data
            n_total += t.numel()
            n_zero += int((t == 0).sum().item())
        if n_total:
            out[f"trits/{stack_name}_frac_zero"] = n_zero / n_total
    return out


def scale_stats_per_stack(model: HrmBopModel) -> dict[str, float]:
    out: dict[str, float] = {}
    for stack_name in ("H_stack", "L_stack"):
        stack = getattr(model, stack_name)
        flats = [m.scales.detach().flatten()
                 for m in stack.modules() if isinstance(m, QLinear)]
        if not flats:
            continue
        s = torch.cat(flats).float()
        out[f"scales/{stack_name}_mean"] = float(s.mean())
        out[f"scales/{stack_name}_max"] = float(s.max())
    return out


def fp_grad_norm(params: list[nn.Parameter]) -> float:
    sq = 0.0
    for p in params:
        if p.grad is not None:
            sq += float(p.grad.detach().pow(2).sum())
    return sq ** 0.5


def save_safetensors(model: nn.Module, path: Path,
                     cfg: HrmBopConfig, extra_meta: dict | None = None) -> None:
    """Atomic safetensors save. Clones shared-storage tensors (tied embed/lm_head)."""
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
    meta = {
        "format": "hrm-bop-v1",
        "hidden_size": str(cfg.hidden_size),
        "num_attention_heads": str(cfg.num_attention_heads),
        "num_kv_heads": str(cfg.num_kv_heads),
        "intermediate_size": str(cfg.intermediate_size),
        "H_layers": str(cfg.H_layers),
        "L_layers": str(cfg.L_layers),
        "H_cycles": str(cfg.H_cycles),
        "L_cycles": str(cfg.L_cycles),
        "vocab_size": str(cfg.vocab_size),
        "max_position_embeddings": str(cfg.max_position_embeddings),
        "rope_theta": str(cfg.rope_theta),
        "rms_norm_eps": str(cfg.rms_norm_eps),
        "tie_word_embeddings": "true" if cfg.tie_word_embeddings else "false",
        "scale_group_size": str(cfg.scale_group_size),
        "embedding_scale": str(cfg.embedding_scale),
    }
    if extra_meta:
        meta.update({k: str(v) for k, v in extra_meta.items()})
    tmp = path.with_suffix(path.suffix + ".tmp")
    save_file(sd, str(tmp), metadata=meta)
    tmp.replace(path)


def causal_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Standard shift-by-one CE."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


@torch.no_grad()
def evaluate(model: HrmBopModel, val_batches: list[dict],
             device: str, autocast_dtype) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for b in val_batches:
        ids = b["input_ids"].to(device, non_blocking=True)
        ctx = (torch.amp.autocast(device.split(":")[0], dtype=autocast_dtype)
               if autocast_dtype is not None
               else torch.amp.autocast(device.split(":")[0], enabled=False))
        with ctx:
            logits = model(ids)
            loss = causal_lm_loss(logits, ids)
        total_loss += float(loss.item())
        n += 1
    model.train()
    return total_loss / max(1, n)


@torch.no_grad()
def evaluate_at_cycles(model: HrmBopModel, val_batches: list[dict],
                       device: str, autocast_dtype,
                       cycles: list[int]) -> dict[int, float]:
    """Val loss with H_cycles overridden to each value in `cycles`. Tests
    test-time loop extrapolation: a model trained to a true fixed point
    (variable per-step H_cycles) should keep its loss flat or improving as
    the inference loop count is pushed beyond the training range, whereas a
    fixed-cycle model degrades once it leaves the count it memorised."""
    model.eval()
    out: dict[int, float] = {}
    for nc in cycles:
        total_loss = 0.0
        n = 0
        for b in val_batches:
            ids = b["input_ids"].to(device, non_blocking=True)
            ctx = (torch.amp.autocast(device.split(":")[0], dtype=autocast_dtype)
                   if autocast_dtype is not None
                   else torch.amp.autocast(device.split(":")[0], enabled=False))
            with ctx:
                logits = model(ids, h_cycles=nc)
                loss = causal_lm_loss(logits, ids)
            total_loss += float(loss.item())
            n += 1
        out[nc] = total_loss / max(1, n)
    model.train()
    return out


@torch.no_grad()
def per_loop_ce_diag(model: HrmBopModel, batch_ids: torch.Tensor,
                     device: str, autocast_dtype) -> list[float]:
    """One CE per H-cycle, computed under no_grad. Doubles forward cost while
    running, so call at a low cadence (~every 4 × log_every)."""
    model.eval()
    ctx = (torch.amp.autocast(device.split(":")[0], dtype=autocast_dtype)
           if autocast_dtype is not None
           else torch.amp.autocast(device.split(":")[0], enabled=False))
    with ctx:
        logits_list = model.per_loop_logits(batch_ids)
        ces = [float(causal_lm_loss(l, batch_ids).item()) for l in logits_list]
    model.train()
    return ces


# ---------------------------------------------------------------- main


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    # Model
    ap.add_argument("--hidden-size", type=int, default=1024)
    ap.add_argument("--num-attention-heads", type=int, default=16)
    ap.add_argument("--num-kv-heads", type=int, default=16)
    ap.add_argument("--intermediate-size", type=int, default=2752)
    ap.add_argument("--H-layers", type=int, default=4)
    ap.add_argument("--L-layers", type=int, default=4)
    ap.add_argument("--H-cycles", type=int, default=2)
    ap.add_argument("--L-cycles", type=int, default=3)
    ap.add_argument("--vocab-size", type=int, default=49152)
    ap.add_argument("--max-position-embeddings", type=int, default=1024)
    ap.add_argument("--rope-theta", type=float, default=10000.0)
    ap.add_argument("--scale-group-size", type=int, default=64)
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    # Init
    ap.add_argument("--init-zero-frac", type=float, default=0.5)
    ap.add_argument("--init-seed", type=int, default=0)
    # Bop-isolation knobs
    ap.add_argument("--random-scales", action="store_true", default=False,
                    help="Init per-(row, group) scales as s_init·exp(N(0, "
                         "--random-scales-sigma)) instead of constant fan-in. "
                         "Used in the Bop-isolation experiments to remove "
                         "the per-layer uniform-scale bias.")
    ap.add_argument("--random-scales-sigma", type=float, default=0.5)
    ap.add_argument("--freeze-scales", action="store_true", default=False,
                    help="Don't train per-(row, group) scales — exclude from "
                         "Lion. Used to test whether Bop alone (without "
                         "scale-coupled Lion updates) can drive loss.")
    ap.add_argument("--freeze-non-embed-fp", action="store_true", default=False,
                    help="Freeze RMSNorm weights, z_L_init, lm_head. Embed "
                         "stays trainable (under Lion). Combined with "
                         "--freeze-scales and --random-scales this isolates "
                         "Bop's contribution: only the trits + the linguistic "
                         "prior in embed get to move.")
    ap.add_argument("--freeze-trits", action="store_true", default=False,
                    help="Ablation: freeze the trit weights so Bop does "
                         "nothing. Combined with the rest of the isolation "
                         "flags, this measures the loss reachable by Lion-on-"
                         "embed alone given random frozen ternary structure.")
    # Bop hyperparams
    ap.add_argument("--gamma", type=float, default=1e-3,
                    help="Bop EMA rate for the trit-space gradient.")
    ap.add_argument("--gamma-v", type=float, default=1e-3,
                    help="EMA rate for the second moment v.")
    ap.add_argument("--tau-norm", type=float, default=0.5,
                    help="Flip threshold on |m|/sqrt(v). Bet-1 default.")
    ap.add_argument("--bop-eps", type=float, default=1e-12)
    ap.add_argument("--cautious-bop", action="store_true", default=False,
                    help="Liang et al. 2024 cautious mask applied to Bop: "
                         "flip only when sign(m) == sign(g_t) (m·g_t > 0). "
                         "Filters oscillating trits whose m has saturated "
                         "but whose current step disagrees with the EMA.")
    # STE + alternative trit optimizers (mutually exclusive with --cautious-bop
    # actually applying — when --ste-trits the trit optimizer changes entirely).
    ap.add_argument("--ste-trits", action="store_true", default=False,
                    help="Treat QLinear.weight as a continuous latent in "
                         "[-1, 1] with STE quantization to {-1, 0, +1} at "
                         "forward (mode=levels, levels=3 already does this). "
                         "Skips the discrete random-ternary init; latents "
                         "start from the standard Normal(0, --init-std) Linear "
                         "init. After each opt step, latents are clamped back "
                         "into [-1, 1]. Required for the continuous trit "
                         "optimizers (--c-muon).")
    ap.add_argument("--c-muon", action="store_true", default=False,
                    help="Cautious Muon (Jordan 2024 + Liang 2024) on trit "
                         "latents. Replaces BopTernary for the trit weights; "
                         "non-trit FP params still go to Lion. Requires "
                         "--ste-trits.")
    ap.add_argument("--muon-lr", type=float, default=0.02,
                    help="Learning rate for CMuon (default 0.02 per Jordan's "
                         "blog). The orthogonalized update has Frobenius "
                         "norm ~sqrt(min(m,n)), so effective per-coord step "
                         "is muon-lr · sqrt(min(m,n))/sqrt(m*n) ≈ muon-lr / "
                         "sqrt(max(m,n)).")
    ap.add_argument("--muon-beta", type=float, default=0.95,
                    help="CMuon first-moment EMA.")
    ap.add_argument("--muon-ns-steps", type=int, default=5,
                    help="Newton-Schulz iterations per CMuon step.")
    ap.add_argument("--no-cautious-muon", action="store_true", default=False,
                    help="Disable the cautious mask in CMuon (vanilla Muon). "
                         "Tests whether the 90%-zeroed cautious dynamic on HRM "
                         "is helping or hurting.")
    ap.add_argument("--muon-lr-floor", type=float, default=1.0,
                    help="CMuon LR cosine-decays from --muon-lr to "
                         "--muon-lr * --muon-lr-floor by --total-steps. "
                         "Default 1.0 = no decay (constant lr). 0.1 = decay "
                         "to 10%% of peak (matches Lion's default schedule).")
    ap.add_argument("--muon-warmup-steps", type=int, default=0,
                    help="Linear warmup steps for CMuon. Default 0 (no warmup).")
    ap.add_argument("--cmuon-state-dtype", default="float32",
                    choices=["float32", "float16", "bfloat16"],
                    help="Storage dtype for CMuon's momentum buffer. fp16 "
                         "halves CMuon's per-trit state (~200 MB on the 150M "
                         "model). EMA update done in m's dtype; NS5 always "
                         "fp32.")
    ap.add_argument("--lion-trits", action="store_true", default=False,
                    help="STE+Lion32 on trit latents (bitlooplm-style). "
                         "Mutually exclusive with --c-muon. Requires "
                         "--ste-trits.")
    ap.add_argument("--lion-trit-lr", type=float, default=5e-4,
                    help="LR for the trit-side Lion (separate from FP Lion).")
    ap.add_argument("--int8-activations", action="store_true", default=False,
                    help="BitNet-style per-token absmax int8 STE on the input "
                         "to every QLinear. Approximately halves activation "
                         "memory and adds quantization noise in the gradient.")
    ap.add_argument("--full-bptt-steps", type=int, default=0,
                    help="(Legacy) first N steps use full BPTT; after step N "
                         "the trainer switches to 1-step. Translates to "
                         "--grad-mode full-bptt for first N, one-step after. "
                         "Overridden by --grad-mode if that is not 'one-step'.")
    ap.add_argument("--min-h-cycles", type=int, default=0,
                    help="If > 0, each training step samples H_cycles "
                         "uniformly from [--min-h-cycles, --max-h-cycles]. "
                         "Fixpoint regularization: forces the recurrent "
                         "layers to converge so output is robust to loop "
                         "count. 0 disables (use cfg.H_cycles fixed).")
    ap.add_argument("--max-h-cycles", type=int, default=0,
                    help="Upper bound on the per-step H_cycles sample (incl). "
                         "Only used when --min-h-cycles > 0.")
    ap.add_argument("--grad-mode", default="one-step",
                    choices=["one-step", "last-per-cycle", "full-bptt"],
                    help="Gradient mode through the recurrent core. one-step: "
                         "only the final inner L iter and final H iter are "
                         "differentiable. last-per-cycle: last L iter of each "
                         "H cycle plus all H iters are differentiable. "
                         "full-bptt: every iter is in the graph. See "
                         "HrmBopModel._core for details.")
    ap.add_argument("--eval-cycle-sweep", default="",
                    help="Comma-separated H_cycles values to evaluate val "
                         "loss at, at every val step (e.g. '1,2,4,8,16,32'). "
                         "Logs val/cyc_<n> to TB. Tests test-time loop "
                         "extrapolation of the fixpoint. Empty disables.")
    ap.add_argument("--fp-weights", action="store_true", default=False,
                    help="FP-weights control: QLinear forward uses the raw "
                         "weight (no quantize, no STE, no per-group scales); "
                         "CMuon trains it as an ordinary FP matrix. Isolates "
                         "whether the recurrence/fixpoint behaviour is a "
                         "ternary artefact. Pairs with --c-muon; continuous "
                         "LeCun init replaces the random-ternary init.")
    # Lion hyperparams
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr-floor", type=float, default=0.1,
                    help="Cosine decays to floor*lr by --total-steps.")
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--lion-betas", nargs=2, type=float, default=(0.95, 0.98))
    ap.add_argument("--lion-wd", type=float, default=0.0)
    ap.add_argument("--max-grad-norm", type=float, default=2.0,
                    help="Clip global L2 norm of FP grads (does not touch "
                         "trits' grads / Bop EMA). 0 disables.")
    # Training shape
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None,
                    help="Warm-start weights only from a .safetensors. Fresh "
                         "optimizers (zero Bop EMA, zero Lion momentum).")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--total-steps", type=int, default=40000)
    ap.add_argument("--ema-warmup", type=int, default=500)
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--autocast-dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "none"])
    ap.add_argument("--latent-dtype", default="float16",
                    choices=["float32", "float16", "bfloat16"],
                    help="Storage dtype for trits. fp16 saves 200 MB and the "
                         "value set is exactly {-1,0,+1} so there's no "
                         "rounding-regime concern.")
    ap.add_argument("--grad-checkpointing",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="Activation checkpointing inside the 1-step grad "
                         "slice. Only ~8 layer-grads worth of activations; "
                         "typically not needed.")
    ap.add_argument("--seed", type=int, default=0)
    # Logging
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-batches", type=int, default=16)
    ap.add_argument("--per-loop-every-mult", type=int, default=4,
                    help="Per-loop CE diagnostic cadence = "
                         "per_loop_every_mult * log_every.")
    ap.add_argument("--hist-every", type=int, default=1000)
    ap.add_argument("--tb-dir", type=Path, default=None)
    ap.add_argument("--run-name", type=str, default=None)
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    _install_sigint_handler()

    latent_dtype = {"float32": torch.float32, "float16": torch.float16,
                    "bfloat16": torch.bfloat16}[args.latent_dtype]
    autocast_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                      "none": None}[args.autocast_dtype]

    interrupted_path = args.out / "interrupted.pt"
    fresh_start = (args.resume is None and not interrupted_path.exists())

    # ---- Model ----
    cfg = HrmBopConfig(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_kv_heads=args.num_kv_heads,
        intermediate_size=args.intermediate_size,
        H_layers=args.H_layers,
        L_layers=args.L_layers,
        H_cycles=args.H_cycles,
        L_cycles=args.L_cycles,
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_position_embeddings,
        rope_theta=args.rope_theta,
        scale_group_size=args.scale_group_size,
    )
    print(f"[build] cfg={cfg}", flush=True)
    model = HrmBopModel(cfg)
    # Trit storage: switch each QLinear.weight to latent_dtype.
    for m in model.modules():
        if isinstance(m, QLinear):
            m.weight.data = m.weight.data.to(latent_dtype)
    model.to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trits = sum(m.weight.numel() for m in model.modules()
                  if isinstance(m, QLinear))
    print(f"[build] total params ≈ {n_params/1e6:.1f}M, "
          f"of which trits ≈ {n_trits/1e6:.1f}M", flush=True)

    if args.grad_checkpointing:
        # HF's wrapper expects gradient_checkpointing_enable; not available on
        # our bare nn.Module — best to leave off here unless we add manual
        # checkpoint wrappers in HrmStack.forward. Spec defaults this off.
        print("[build] --grad-checkpointing requested but the model exposes "
              "no enable hook; ignoring.", flush=True)

    # Validate STE/CMuon/Lion-trit flag combination.
    if args.c_muon and args.lion_trits:
        raise ValueError("--c-muon and --lion-trits are mutually exclusive")
    if (args.c_muon or args.lion_trits) and not args.ste_trits:
        print(f"[arg] --c-muon/--lion-trits requires --ste-trits; "
              f"forcing --ste-trits on.", flush=True)
        args.ste_trits = True

    # FP-weights control: flag every QLinear so forward uses the raw weight
    # (no quantize/scales/STE). CMuon then trains it as an ordinary matrix.
    if args.fp_weights:
        if not args.c_muon:
            raise ValueError("--fp-weights pairs with --c-muon "
                             "(CMuon on the raw FP weight)")
        n = 0
        for m in model.modules():
            if isinstance(m, QLinear):
                m.fp_weights = True
                n += 1
        print(f"[init] FP-weights control: {n} QLinears use raw FP weight "
              f"(no quantize/scales/STE)", flush=True)

    # ---- Init (fresh-start only) ----
    if fresh_start and args.fp_weights:
        # FP control: continuous LeCun init; the ternary-specific init and
        # the per-group scales (unused in the FP forward path) are skipped.
        gen = torch.Generator(device=args.device).manual_seed(args.init_seed)
        n_t = init_fp_weights(model, generator=gen)
        print(f"[init] FP mode: {n_t} QLinear weights ~ N(0, 1/fan_in) "
              f"(continuous LeCun init)", flush=True)
    elif fresh_start:
        gen = torch.Generator(device=args.device).manual_seed(args.init_seed)
        # Even in STE mode we want non-trivial quantized output from step 0.
        # The standard Normal(0, 0.02) Linear init puts ALL latents inside
        # the ±1/3 zero-attractor, so `quantize_levels(w, 3)` returns 0
        # everywhere and the forward output is zero → trit gradients are
        # exactly zero → CMuon does nothing. Start from the discrete
        # random-ternary distribution; the STE optimizer then drifts the
        # latents continuously and code flips happen as they cross ±1/3.
        n_t = init_trits_random(model, zero_frac=args.init_zero_frac,
                                generator=gen)
        if args.ste_trits:
            print(f"[init] STE mode: {n_t} QLinear latents start at discrete "
                  f"{{-1, 0, +1}} (random, {args.init_zero_frac:.0%} zero); "
                  f"continuous evolution begins from step 1", flush=True)
        if args.random_scales:
            scale_gen = torch.Generator(device=args.device).manual_seed(
                args.init_seed + 1)
            n_s = init_scales_random(model, sigma=args.random_scales_sigma,
                                     generator=scale_gen)
            print(f"[init] random ternary on {n_t} QLinears (zero_frac="
                  f"{args.init_zero_frac:.2f}); lognormal scales (σ="
                  f"{args.random_scales_sigma}) on {n_s}", flush=True)
        else:
            n_s = init_scales_fanin(model)
            print(f"[init] random ternary on {n_t} QLinears (zero_frac="
                  f"{args.init_zero_frac:.2f}); fan-in scales on {n_s}",
                  flush=True)

    # Enable int8 activation quantization on every QLinear if requested.
    if args.int8_activations:
        n = 0
        for m in model.modules():
            if isinstance(m, QLinear):
                m.int8_activations = True
                n += 1
        print(f"[init] int8 activations enabled on {n} QLinears", flush=True)

    # ---- Freezing ----
    if args.freeze_scales:
        nf = freeze_scales(model)
        print(f"[freeze] scales frozen on {nf} QLinears", flush=True)
    if args.freeze_non_embed_fp:
        nf = freeze_non_embed_fp(model)
        print(f"[freeze] non-embed FP frozen ({nf} tensors: norms + "
              f"z_L_init + lm_head if untied)", flush=True)
    if args.freeze_trits:
        nf = 0
        for m in model.modules():
            if isinstance(m, QLinear):
                m.weight.requires_grad_(False)
                nf += 1
        print(f"[freeze] trits frozen on {nf} QLinears — Bop will be a "
              f"no-op (grad is None on every trit)", flush=True)

    # ---- Parameter split + optimizers ----
    trit_params, scale_params, fp_params = split_params(model)
    print(f"[opt] trainable: trits="
          f"{sum(p.numel() for p in trit_params)/1e6:.1f}M, "
          f"scales={sum(p.numel() for p in scale_params)/1e6:.3f}M, "
          f"fp={sum(p.numel() for p in fp_params)/1e6:.1f}M", flush=True)

    # Trit optimizer: Bop (latent-free) by default, CMuon for --c-muon, or
    # Lion32 for --lion-trits. All three are mutually exclusive.
    opt_bop: BopTernary | None = None
    opt_cmuon: CMuon | None = None
    opt_lion_trits: Lion32 | None = None
    if args.c_muon:
        cmuon_state_dtype = {"float32": torch.float32,
                             "float16": torch.float16,
                             "bfloat16": torch.bfloat16}[args.cmuon_state_dtype]
        cmuon_cautious = not args.no_cautious_muon
        opt_cmuon = CMuon(trit_params, lr=args.muon_lr, beta=args.muon_beta,
                          ns_steps=args.muon_ns_steps,
                          cautious=cmuon_cautious,
                          state_dtype=cmuon_state_dtype)
        print(f"[opt] CMuon trit opt lr={args.muon_lr:g} "
              f"beta={args.muon_beta:g} ns={args.muon_ns_steps} "
              f"cautious={cmuon_cautious} state_dtype={args.cmuon_state_dtype}",
              flush=True)
    elif args.lion_trits:
        opt_lion_trits = Lion32(trit_params, lr=args.lion_trit_lr,
                                betas=tuple(args.lion_betas),
                                weight_decay=0.0)
        print(f"[opt] Lion32 trit opt lr={args.lion_trit_lr:g} "
              f"betas={tuple(args.lion_betas)}", flush=True)
    else:
        opt_bop = BopTernary(trit_params, gamma=args.gamma, tau=1.0,
                             use_2nd_moment=True,
                             gamma_v=args.gamma_v,
                             tau_norm=args.tau_norm,
                             eps=args.bop_eps,
                             reset_on_flip=False,
                             refractory=0,
                             cautious=args.cautious_bop)
    lion_groups: list[dict] = []
    if scale_params:
        lion_groups.append({"params": scale_params, "lr": args.lr})
    if fp_params:
        lion_groups.append({"params": fp_params, "lr": args.lr})
    if not lion_groups:
        raise RuntimeError(
            "Lion has zero parameters to train after --freeze-* flags. "
            "Trainer assumes at least one FP-class param is live; if you "
            "really want pure-Bop, comment out the opt_lion.step() call.")
    opt_lion = Lion32(lion_groups, lr=args.lr, betas=tuple(args.lion_betas),
                      weight_decay=args.lion_wd)
    print(f"[opt] Bop(Bet1) γ={args.gamma:g} γv={args.gamma_v:g} "
          f"τ_norm={args.tau_norm:g}; Lion lr={args.lr:g} "
          f"betas={tuple(args.lion_betas)}", flush=True)

    # ---- Resume ----
    interrupted_state = None
    global_step = 0
    samples_consumed = 0
    best_snapshot = None
    if args.resume is not None:
        sd = load_file(str(args.resume))
        miss, unexp = model.load_state_dict(sd, strict=False)
        invalidate_all_q_caches(model)
        print(f"[resume] warm-start ← {args.resume.name} "
              f"(missing={len(miss)}, unexpected={len(unexp)})", flush=True)
    elif interrupted_path.exists():
        interrupted_state = torch.load(str(interrupted_path),
                                       map_location="cpu",
                                       weights_only=False)
        model.load_state_dict(interrupted_state["model"], strict=False)
        del interrupted_state["model"]
        if opt_bop is not None and "opt_bop" in interrupted_state:
            opt_bop.load_state_dict(interrupted_state["opt_bop"])
            del interrupted_state["opt_bop"]
        if opt_cmuon is not None and "opt_cmuon" in interrupted_state:
            opt_cmuon.load_state_dict(interrupted_state["opt_cmuon"])
            del interrupted_state["opt_cmuon"]
        if opt_lion_trits is not None and "opt_lion_trits" in interrupted_state:
            opt_lion_trits.load_state_dict(interrupted_state["opt_lion_trits"])
            del interrupted_state["opt_lion_trits"]
        opt_lion.load_state_dict(interrupted_state["opt_lion"])
        del interrupted_state["opt_lion"]
        # PyTorch's load_state_dict restores param_groups from the resume
        # file, clobbering the CLI-set hyperparams. Re-apply them so a
        # mid-run retune actually takes effect on the warm-resumed run.
        if opt_bop is not None:
            for g in opt_bop.param_groups:
                g["tau_norm"] = args.tau_norm
                g["gamma"] = args.gamma
                g["gamma_v"] = args.gamma_v
                g["eps"] = args.bop_eps
        if opt_cmuon is not None:
            for g in opt_cmuon.param_groups:
                g["lr"] = args.muon_lr
                g["beta"] = args.muon_beta
        for g in opt_lion.param_groups:
            g["lr"] = args.lr     # cosine sched re-applies each step anyway
        print(f"[resume] reapplied trit-opt + lion hyperparams over loaded "
              f"opt state", flush=True)
        global_step = int(interrupted_state.get("next_step", 0))
        samples_consumed = int(interrupted_state.get(
            "samples_consumed",
            global_step * args.grad_accum * args.batch_size))
        invalidate_all_q_caches(model)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[resume] {interrupted_path} at step {global_step}", flush=True)

    # ---- Tokenizer + data ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.eos_token_id is None and tok.pad_token_id is None:
        # SmolLM2 tokenizer ships eos; fallback only here as belt-and-braces.
        tok.add_special_tokens({"eos_token": "<|endoftext|>"})
    train_loader = make_train_loader(
        tok, seq_len=args.max_position_embeddings,
        batch_size=args.batch_size, seed=args.seed,
        num_workers=args.num_workers, start_skip=samples_consumed)
    print(f"[data] train mix at seq={args.max_position_embeddings} "
          f"bs={args.batch_size} workers={args.num_workers}", flush=True)
    print(f"[data] loading {args.val_batches} val batches "
          f"(fineweb-edu/sample-100BT)…", flush=True)
    val_batches = make_val_loader(tok, seq_len=args.max_position_embeddings,
                                  batch_size=args.batch_size,
                                  n_batches=args.val_batches)
    print(f"[data] val ready: {len(val_batches)} batches", flush=True)

    train_iter = iter(train_loader)

    # ---- TB ----
    tb_root = args.tb_dir if args.tb_dir is not None else (args.out / "tb")
    if interrupted_state and interrupted_state.get("run_name"):
        run_name = interrupted_state["run_name"]
    elif args.run_name:
        run_name = args.run_name
    else:
        run_name = datetime.now().strftime("hrmbop_%Y%m%d_%H%M%S")
    run_dir = tb_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir),
                           purge_step=global_step if global_step else None)
    print(f"[tb] {run_dir}", flush=True)
    writer.add_text("config", "  \n".join([
        "**hrm_bop** — HRM-style ternary recurrent LM",
        f"- model: H={cfg.H_layers}, L={cfg.L_layers}, "
        f"cycles={cfg.H_cycles}×{cfg.L_cycles}, hidden={cfg.hidden_size}",
        f"- params: ~{n_params/1e6:.1f}M (trits {n_trits/1e6:.1f}M)",
        f"- Bop: γ={args.gamma:g} γv={args.gamma_v:g} τ_norm={args.tau_norm:g}",
        f"- Lion: lr={args.lr:g}, warmup={args.warmup_steps}, "
        f"floor={args.lr_floor:g}, total_steps={args.total_steps}",
        f"- init: zero_frac={args.init_zero_frac:.2f}, "
        f"latent_dtype={args.latent_dtype}",
    ]), global_step)

    # ---- Tracker ----
    ctrl = BestEmaTracker(ema_alpha=0.05, rel_threshold=1e-3,
                          ema_warmup=args.ema_warmup)
    if interrupted_state and interrupted_state.get("ctrl_state"):
        ctrl.load_state_dict(interrupted_state["ctrl_state"])
    best_snapshot = (interrupted_state.get("best_snapshot")
                     if interrupted_state else None)
    interrupted_state = None

    # ---- Train ----
    model.train()
    running_loss = 0.0
    running_n = 0
    flips_window = 0
    elems_window = 0
    tokens_window = 0
    window_t0 = time.time()
    pbar = tqdm(desc="hrmbop", dynamic_ncols=True,
                initial=global_step, total=args.total_steps)
    if opt_bop is not None:
        opt_bop.zero_grad(set_to_none=True)
    if opt_cmuon is not None:
        opt_cmuon.zero_grad(set_to_none=True)
    if opt_lion_trits is not None:
        opt_lion_trits.zero_grad(set_to_none=True)
    opt_lion.zero_grad(set_to_none=True)
    # For STE mode: snapshot quantized codes once at fresh start so the
    # first window's code_flip_count has a baseline.
    prev_codes: dict[int, torch.Tensor] = {}
    if args.ste_trits:
        prev_codes = quantized_codes_snapshot(model)

    try:
        while global_step < args.total_steps:
            # --- grad_accum forward+backward passes ---
            for _ in range(args.grad_accum):
                batch = next(train_iter)
                ids = batch["input_ids"].to(args.device, non_blocking=True)
                ctx = (torch.amp.autocast(args.device.split(":")[0],
                                          dtype=autocast_dtype)
                       if autocast_dtype is not None
                       else torch.amp.autocast(args.device.split(":")[0],
                                               enabled=False))
                # Resolve grad mode for THIS step. --grad-mode is the explicit
                # control; --full-bptt-steps stays as a legacy compatibility
                # path (full-bptt for first N steps, then default 'one-step').
                if args.grad_mode != "one-step":
                    grad_mode_now = args.grad_mode
                elif global_step < args.full_bptt_steps:
                    grad_mode_now = "full-bptt"
                else:
                    grad_mode_now = "one-step"
                # Variable H_cycles per step (fixpoint regularization).
                if args.min_h_cycles > 0:
                    import random
                    h_cyc_now = random.randint(
                        args.min_h_cycles,
                        max(args.min_h_cycles, args.max_h_cycles))
                else:
                    h_cyc_now = None
                with ctx:
                    logits = model(ids, grad_mode=grad_mode_now,
                                   h_cycles=h_cyc_now)
                    loss = causal_lm_loss(logits, ids)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"non-finite loss at step {global_step}: {loss.item()}")
                (loss / args.grad_accum).backward()
                running_loss += float(loss.item())
                running_n += 1
                tokens_window += ids.numel()

            # --- LR schedule on Lion ---
            cur_lr = lr_at(global_step, args.total_steps, args.lr,
                           args.warmup_steps, floor=args.lr_floor)
            for g in opt_lion.param_groups:
                g["lr"] = cur_lr
            # --- LR schedule on CMuon (if active) ---
            if opt_cmuon is not None and (args.muon_lr_floor != 1.0
                                          or args.muon_warmup_steps > 0):
                cur_muon_lr = lr_at(global_step, args.total_steps,
                                    args.muon_lr,
                                    args.muon_warmup_steps,
                                    floor=args.muon_lr_floor)
                for g in opt_cmuon.param_groups:
                    g["lr"] = cur_muon_lr

            # --- FP grad clip (skip trits) ---
            fp_gnorm = None
            if args.max_grad_norm and args.max_grad_norm > 0:
                fp_gnorm = float(torch.nn.utils.clip_grad_norm_(
                    fp_params + scale_params, args.max_grad_norm))

            # --- Trit opt step (Bop or CMuon or Lion-STE) + FP Lion step ---
            cmuon_zeroed_frac = 0.0
            if opt_bop is not None:
                n_flips, n_elems = opt_bop.step()
            elif opt_cmuon is not None:
                _, cmuon_zeroed_frac = opt_cmuon.step()
                # In STE mode the "code flip rate" is computed from
                # quantized-code diffs below in the logging block, not here.
                n_flips, n_elems = 0, 0
            elif opt_lion_trits is not None:
                opt_lion_trits.step()
                n_flips, n_elems = 0, 0
            else:
                n_flips, n_elems = 0, 0
            opt_lion.step()
            # Constrain per-(row, group) scales to be strictly positive.
            # Lion's sign update has no positivity bias, so without this clamp
            # scales drift through zero (see hrm_bop_spec.md "negative scales"
            # failure mode).
            with torch.no_grad():
                for s in scale_params:
                    s.data.clamp_min_(1e-6)
            # STE: keep trit latents in box. The forward STE clips to [-1, 1]
            # inside quantize_levels anyway, but the latent itself can drift
            # past the boundaries under Muon's update; out-of-box latents are
            # wasted capacity since the quantized code can't change until the
            # latent re-enters.
            if args.ste_trits and not args.fp_weights:
                clamp_qlinear_weights(model, lo=-1.0, hi=1.0)
            flips_window += n_flips
            elems_window += n_elems
            invalidate_all_q_caches(model)
            if opt_bop is not None:
                opt_bop.zero_grad(set_to_none=True)
            if opt_cmuon is not None:
                opt_cmuon.zero_grad(set_to_none=True)
            if opt_lion_trits is not None:
                opt_lion_trits.zero_grad(set_to_none=True)
            opt_lion.zero_grad(set_to_none=True)
            global_step += 1

            step_loss = running_loss / max(1, running_n)
            improved = ctrl.update(global_step, step_loss)
            if improved:
                best_snapshot = snapshot_to_cpu(model)

            # --- Periodic logging ---
            if global_step % args.log_every == 0:
                t1 = time.time()
                tps = tokens_window / max(1e-6, t1 - window_t0)
                # Flip rate semantics differ between Bop and STE+CMuon:
                #   Bop: count flips the opt's own discrete flip rule did
                #        in the window (flips_window / elems_window).
                #   STE: count trits whose quantized code changed between
                #        the start and end of this window — a real measure
                #        of how much the continuous latent moved relative
                #        to the ±1/3 boundaries.
                if args.ste_trits:
                    n_diff, n_trits, prev_codes = code_flip_count(
                        model, prev_codes)
                    rate = n_diff / max(1, n_trits) / max(1, args.log_every)
                else:
                    rate = flips_window / max(1, elems_window)
                postfix = {
                    "step": global_step,
                    "loss": f"{step_loss:.4f}",
                    "ema": f"{ctrl.ema:.4f}" if ctrl.ema else "—",
                    "flip%": f"{rate*100:.3f}",
                    "tok/s": f"{tps:.0f}",
                }
                pbar.set_postfix(postfix)
                pbar.update(args.log_every)
                writer.add_scalar("loss/step", step_loss, global_step)
                if ctrl.ema is not None:
                    writer.add_scalar("loss/ema", ctrl.ema, global_step)
                    writer.add_scalar("loss/best", ctrl.best_ema, global_step)
                writer.add_scalar("bop/flip_rate", rate, global_step)
                if opt_bop is not None:
                    writer.add_scalar("bop/flip_count", flips_window,
                                      global_step)
                    ms = m_stats(opt_bop)
                    writer.add_scalar("bop/m_rms", ms["m_rms"], global_step)
                    writer.add_scalar("bop/m_max", ms["m_max"], global_step)
                    if "score_rms" in ms:
                        writer.add_scalar("bop/score_rms", ms["score_rms"],
                                          global_step)
                        writer.add_scalar("bop/score_max", ms["score_max"],
                                          global_step)
                if opt_cmuon is not None:
                    writer.add_scalar("cmuon/cautious_zeroed_frac",
                                      cmuon_zeroed_frac, global_step)
                    writer.add_scalar("cmuon/lr",
                                      opt_cmuon.param_groups[0]["lr"],
                                      global_step)
                # In STE mode the latent is continuous; read trit fractions
                # from the quantized code instead of the raw weight.
                ts = (trit_stats_quantized(model) if args.ste_trits
                      else trit_stats(model))
                writer.add_scalar("trits/frac_zero", ts["frac_zero"], global_step)
                writer.add_scalar("trits/frac_pos", ts["frac_pos"], global_step)
                writer.add_scalar("trits/frac_neg", ts["frac_neg"], global_step)
                for k, v in trit_stats_per_stack(model).items():
                    writer.add_scalar(k, v, global_step)
                with torch.no_grad():
                    all_s = torch.cat([m.scales.data.flatten()
                                       for m in model.modules()
                                       if isinstance(m, QLinear)]).float()
                    writer.add_scalar("scales/mean", float(all_s.mean()),
                                      global_step)
                    writer.add_scalar("scales/min", float(all_s.min()),
                                      global_step)
                    writer.add_scalar("scales/max", float(all_s.max()),
                                      global_step)
                    writer.add_scalar("scales/p50", float(all_s.median()),
                                      global_step)
                for k, v in scale_stats_per_stack(model).items():
                    writer.add_scalar(k, v, global_step)
                writer.add_scalar("lion/lr", cur_lr, global_step)
                if fp_gnorm is not None:
                    writer.add_scalar("lion/grad_norm", fp_gnorm, global_step)
                writer.add_scalar("throughput/tokens_per_sec", tps, global_step)
                writer.add_scalar("throughput/steps_per_sec",
                                  args.log_every / max(1e-6, t1 - window_t0),
                                  global_step)
                writer.add_scalar(
                    "diag/zL_init_rms",
                    float(model.z_L_init.detach().float().pow(2).mean().sqrt()),
                    global_step)
                running_loss = 0.0
                running_n = 0
                flips_window = 0
                elems_window = 0
                tokens_window = 0
                window_t0 = t1

            # --- Per-loop CE diagnostic ---
            if (global_step % (args.log_every * args.per_loop_every_mult) == 0
                    and global_step > 0):
                # Reuse the most recent batch for diagnostics; cheap.
                ces = per_loop_ce_diag(model, ids, args.device, autocast_dtype)
                for i, ce in enumerate(ces):
                    writer.add_scalar(f"diag/per_loop_ce_{i}", ce, global_step)

            # --- Histograms ---
            if args.hist_every and global_step % args.hist_every == 0:
                with torch.no_grad():
                    all_s = torch.cat([m.scales.data.flatten()
                                       for m in model.modules()
                                       if isinstance(m, QLinear)]).float()
                    writer.add_histogram("hist/scales/all", all_s, global_step)
                # Sample 1M m values to keep TB happy.
                ms_all: list[torch.Tensor] = []
                opt_for_hist = opt_bop if opt_bop is not None else opt_cmuon
                if opt_for_hist is not None:
                    for group in opt_for_hist.param_groups:
                        for p in group["params"]:
                            st = opt_for_hist.state.get(p)
                            if st and "m" in st:
                                ms_all.append(st["m"].flatten())
                if ms_all:
                    flat = torch.cat(ms_all)
                    if flat.numel() > 1_000_000:
                        idx = torch.randint(0, flat.numel(), (1_000_000,),
                                            device=flat.device)
                        flat = flat[idx]
                    writer.add_histogram("hist/m", flat.float(), global_step)

            # --- Validation ---
            if args.val_every and global_step % args.val_every == 0:
                val_loss = evaluate(model, val_batches, args.device,
                                    autocast_dtype)
                writer.add_scalar("val/loss", val_loss, global_step)
                tqdm.write(f"[val] step {global_step} loss={val_loss:.4f}")
                if args.eval_cycle_sweep:
                    cyc_list = [int(c) for c in args.eval_cycle_sweep.split(",")]
                    sweep = evaluate_at_cycles(model, val_batches, args.device,
                                               autocast_dtype, cyc_list)
                    for nc, l in sweep.items():
                        writer.add_scalar(f"val/cyc_{nc}", l, global_step)
                    sweep_str = " ".join(f"c{nc}={l:.4f}"
                                         for nc, l in sweep.items())
                    tqdm.write(f"[val-sweep] step {global_step} {sweep_str}")

            # --- Checkpoint ---
            samples_at_save = global_step * args.grad_accum * args.batch_size
            if (args.checkpoint_every > 0
                    and global_step % args.checkpoint_every == 0):
                _save_resume(interrupted_path, model, opt_bop, opt_cmuon,
                             opt_lion_trits, opt_lion, global_step,
                             best_snapshot, ctrl, run_name, samples_at_save)
                tqdm.write(f"[ckpt] {interrupted_path} @ step {global_step}")

            if _INTERRUPT["flag"]:
                _save_resume(interrupted_path, model, opt_bop, opt_cmuon,
                             opt_lion_trits, opt_lion, global_step,
                             best_snapshot, ctrl, run_name, samples_at_save)
                writer.flush()
                writer.close()
                pbar.close()
                print(f"[!] saved {interrupted_path}", flush=True)
                sys.exit(0)

    except SystemExit:
        raise
    except BaseException as e:
        try:
            samples_at_save = global_step * args.grad_accum * args.batch_size
            _save_resume(interrupted_path, model, opt_bop, opt_cmuon,
                         opt_lion_trits, opt_lion, global_step,
                         best_snapshot, ctrl, run_name, samples_at_save)
            print(f"[!] emergency save → {interrupted_path} "
                  f"(reason: {type(e).__name__})", flush=True)
        except Exception as save_err:
            print(f"[!!] emergency save failed: {save_err}", flush=True)
        raise
    finally:
        pbar.close()

    # ---- Final ----
    final_path = args.out / "final.safetensors"
    save_safetensors(model, final_path, cfg,
                     extra_meta={"total_steps": global_step})
    print(f"[done] saved {final_path}", flush=True)
    if best_snapshot is not None and ctrl.best_step != global_step:
        model.load_state_dict(best_snapshot, strict=False)
        invalidate_all_q_caches(model)
        best_path = args.out / "final_best.safetensors"
        save_safetensors(model, best_path, cfg,
                         extra_meta={"best_step": ctrl.best_step,
                                     "best_ema": f"{ctrl.best_ema:.6f}"})
        print(f"[done] saved {best_path} (step {ctrl.best_step}, "
              f"EMA {ctrl.best_ema:.4f})", flush=True)
    writer.add_text("end", f"hrm_bop complete at step {global_step}",
                    global_step)
    writer.flush()
    writer.close()
    if interrupted_path.exists():
        interrupted_path.unlink()


def _save_resume(path: Path, model, opt_bop, opt_cmuon, opt_lion_trits,
                 opt_lion, next_step: int, best_snapshot, ctrl,
                 run_name: str, samples_consumed: int) -> None:
    """hrm_bop-flavored resume save: trit-opt (Bop or CMuon or Lion-STE,
    whichever is live) + FP Lion. interrupted.pt is the only on-disk file;
    atomic .tmp rename."""
    payload = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "opt_lion": opt_lion.state_dict(),
        "next_step": int(next_step),
        "samples_consumed": int(samples_consumed),
        "best_snapshot": best_snapshot,
        "ctrl_state": ctrl.state_dict() if ctrl is not None else None,
        "run_name": run_name,
    }
    if opt_bop is not None:
        payload["opt_bop"] = opt_bop.state_dict()
    if opt_cmuon is not None:
        payload["opt_cmuon"] = opt_cmuon.state_dict()
    if opt_lion_trits is not None:
        payload["opt_lion_trits"] = opt_lion_trits.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(tmp))
    tmp.replace(path)


if __name__ == "__main__":
    main()
