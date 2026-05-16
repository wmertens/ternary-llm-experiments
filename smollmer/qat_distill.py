"""qat_distill.py — single-stage QAT distillation with learnable
per-(row, group) codepoint c_{r,g}.

Every QLinear weight is ternarized from step 0 (no promotion schedule,
no commit gate, no rounds):

    forward(w) = sign(w)·c_{r,g}   if |w| > c_{r,g}/2
                 0                  otherwise

The latent w receives identity-via-STE gradient at non-zero-target
slots and 0 at zero-target slots. The codepoint c_{r,g} is an
nn.Parameter trained jointly with w (at --c-lr-mult times the base LR)
and receives a real gradient (sign(w) at non-zero slots).

Deploy fold: snap latents to their ternary targets, then per-(row,
group): w /= c_{r,g}; s *= c_{r,g}. Latent ends in {-1, 0, +1} and
scales absorb the codepoint magnitude.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .build_student import load_student
from .distill import (
    AdamW32, BestEmaTracker, CautiousAdamW, Lion32,
    ShardedDataset, _INTERRUPT, _install_sigint_handler,
    kl_with_rest, lr_at, save_checkpoint, save_resume, snapshot_to_cpu,
)
from .qlinear import QLinear, clamp_qlinear_weights, set_soft_mode
from .teacher_floor import load_or_compute as load_teacher_floor


# ============================================================================
# Setup helpers
# ============================================================================

def attach_learnable_c(model: nn.Module, default_c: float = 2.0 / 3.0) -> int:
    """Add codepoint_c as an nn.Parameter to each QLinear (shape [out_f,
    n_groups], fp32). Trained jointly with the latents and scales.

    Also registers persistent buffers `c_init` and `s_init` (snapshots
    of c and s at first init) so the L2/RMS-drift metric reports motion
    since the very first init across resumes — not since the last script
    start. Initialized to zeros as a sentinel; populated by
    `snapshot_inits_if_unset` after init + resume.
    """
    n = 0
    for m in model.modules():
        if not isinstance(m, QLinear):
            continue
        if hasattr(m, "codepoint_c"):
            continue
        out_f = m.weight.shape[0]
        m.codepoint_c = nn.Parameter(
            torch.full((out_f, m.n_groups), default_c,
                       dtype=torch.float32, device=m.weight.device)
        )
        m.register_buffer(
            "c_init",
            torch.zeros_like(m.codepoint_c.data),
            persistent=True,
        )
        m.register_buffer(
            "s_init",
            torch.zeros_like(m.scales.data),
            persistent=True,
        )
        n += 1
    return n


def snapshot_inits_if_unset(model: nn.Module) -> int:
    """Populate `c_init` and `s_init` with current c and s values *only
    when they're still at their zero sentinel* — i.e., either this is a
    fresh start or we're resuming from a ckpt that predates these
    buffers. Both c (clamped to [c_clamp_min>0, ...]) and s (built from
    max(|w|), always positive) are strictly nonzero, so 'all zeros'
    unambiguously marks 'never snapshotted'."""
    n = 0
    with torch.no_grad():
        for m in model.modules():
            if not isinstance(m, QLinear):
                continue
            if hasattr(m, "c_init") and bool((m.c_init == 0).all()):
                m.c_init.data.copy_(m.codepoint_c.data)
                n += 1
            if hasattr(m, "s_init") and bool((m.s_init == 0).all()):
                m.s_init.data.copy_(m.scales.data)
    return n


@torch.no_grad()
def init_c_from_band_mean(model: nn.Module,
                          target_zero_frac: float | None,
                          fallback_c: float) -> dict[str, float]:
    """Seed codepoint_c with mean(|w| over band) per (row, group). Same
    formula progressive_distill uses — gives the optimizer a sensible
    starting point near the data's center of mass."""
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
        band_sum = (abs_wb * is_band.float()).sum(dim=-1)
        band_count = is_band.float().sum(dim=-1)
        c_rg = band_sum / band_count.clamp_min(1.0)
        c_rg = torch.where(band_count > 0, c_rg,
                           torch.full_like(c_rg, fallback_c))
        m.codepoint_c.data.copy_(c_rg.to(m.codepoint_c.dtype))
        all_c.append(c_rg.flatten())
    if not all_c:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0}
    cat = torch.cat(all_c)
    return {"mean": float(cat.mean()), "min": float(cat.min()),
            "max": float(cat.max()), "p50": float(cat.median())}


def _first_qlinear_latent_sample(model: nn.Module
                                 ) -> tuple[str, torch.Tensor] | None:
    """Histogram source: raw FP latent of the first QLinear. Shows where
    the floating-point weights are sitting — most useful for spotting
    whether they cluster near c/2 boundaries, push against [-1, 1], or
    drift after promotion."""
    for name, m in model.named_modules():
        if not isinstance(m, QLinear):
            continue
        with torch.no_grad():
            w = m.weight.detach()
        return name, w.flatten().to("cpu")
    return None


def _all_c_flat(model: nn.Module) -> torch.Tensor | None:
    """One-sided distribution of every codepoint_c value across all
    QLinears — useful as a histogram alongside scalar c/{mean,min,max}."""
    parts = []
    for m in model.modules():
        if isinstance(m, QLinear) and hasattr(m, "codepoint_c"):
            parts.append(m.codepoint_c.data.detach().flatten().to("cpu"))
    if not parts:
        return None
    return torch.cat(parts)


def _all_cs_flat(model: nn.Module) -> torch.Tensor | None:
    """Distribution of c_{r,g} · s_{r,g} across all QLinears. This is the
    effective per-group magnitude that a ±1 deployed weight will be
    multiplied by after the deploy fold (where s *= c). Watching this
    lets you see whether the model is concentrating its dynamic range in
    a few groups or spreading it broadly."""
    parts = []
    for m in model.modules():
        if (isinstance(m, QLinear)
                and hasattr(m, "codepoint_c")
                and hasattr(m, "scales")):
            cs = (m.codepoint_c.data.detach().float()
                  * m.scales.data.detach().float())
            parts.append(cs.flatten().to("cpu"))
    if not parts:
        return None
    return torch.cat(parts)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None,
                    help="Optional safetensors warm-start.")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-floor", type=float, default=1.0)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--wd", type=float, default=0.001)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--optimizer", default="cautious-adamw",
                    choices=["lion", "adamw", "cautious-adamw"])
    ap.add_argument("--scale-group-size", type=int, default=64)
    ap.add_argument("--scale-lr-mult", type=float, default=None)
    ap.add_argument("--c-lr-mult", type=float, default=0.1,
                    help="LR multiplier for codepoint_c. Default 0.1: c "
                         "is a global-ish parameter (one per row, group, "
                         "not per element), so it should move more slowly "
                         "than the latents to avoid thrashing.")
    ap.add_argument("--c-clamp-min", type=float, default=0.1)
    ap.add_argument("--c-clamp-max", type=float, default=1.0)
    ap.add_argument("--target-zero-frac-init", type=float, default=0.25,
                    help="Used only to seed codepoint_c at init from the "
                         "|w| band mean. Learned c is free to drift after.")
    ap.add_argument("--permute", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--autocast-dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "none"])
    ap.add_argument("--latent-dtype", default="auto",
                    choices=["auto", "float32", "float16", "bfloat16"])
    ap.add_argument("--grad-checkpointing",
                    action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total-steps", type=int, default=20000,
                    help="Total training steps (used by the cosine LR "
                         "schedule's horizon and as the loop's stopping "
                         "condition).")
    ap.add_argument("--ema-warmup", type=int, default=500)
    ap.add_argument("--soft-hist-every", type=int, default=200)
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    ap.add_argument("--tb-dir", type=Path, default=None)
    ap.add_argument("--run-name", type=str, default=None)

    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    _install_sigint_handler()

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

    # ---- Build student ----
    print(f"[build] loading {args.model}, "
          f"group_size={args.scale_group_size}, permute={do_permute}")
    model, _, n_replaced = load_student(
        args.model, dtype=torch.float32, levels=257,
        latent_dtype=latent_dtype, group_size=args.scale_group_size,
        permute=do_permute,
    )
    model.to(args.device)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        print("[build] gradient checkpointing enabled")
    print(f"[build] {n_replaced} QLinear modules "
          f"(latent dtype {latent_dtype})")

    # Soft mode α=0 — but the QLinear.forward will route through the
    # _full_qat_effective_weight path because codepoint_c is a Parameter.
    set_soft_mode(model, alpha=0.0, target_zero_frac=None)
    n_qat = attach_learnable_c(model, default_c=2.0 / 3.0)
    if fresh_start:
        tzf = (args.target_zero_frac_init
               if 0.0 < args.target_zero_frac_init < 1.0 else None)
        cstats = init_c_from_band_mean(model, tzf,
                                       fallback_c=2.0 / 3.0)
        print(f"[qat] codepoint_c init: mean={cstats['mean']:.4f} "
              f"min={cstats['min']:.4f} max={cstats['max']:.4f} "
              f"p50={cstats['p50']:.4f}")
    print(f"[qat] {n_qat} QLinears with learnable codepoint_c; "
          f"fresh_start={fresh_start}")

    # ---- Optimizer: three param groups (latents, scales, codepoint_c) ----
    scale_lr_mult = (args.scale_lr_mult if args.scale_lr_mult is not None
                     else 1.0 / float(args.scale_group_size))
    scale_param_ids = {id(m.scales) for m in model.modules()
                       if isinstance(m, QLinear)}
    c_param_ids = {id(m.codepoint_c) for m in model.modules()
                   if isinstance(m, QLinear)}
    scale_params, c_params, other_params = [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in c_param_ids:
            c_params.append(p)
        elif id(p) in scale_param_ids:
            scale_params.append(p)
        else:
            other_params.append(p)
    param_groups = [
        {"params": other_params, "lr": args.lr, "lr_mult": 1.0,
         "weight_decay": args.wd, "name": "latents"},
        {"params": scale_params, "lr": args.lr * scale_lr_mult,
         "lr_mult": scale_lr_mult, "weight_decay": args.wd, "name": "scales"},
        {"params": c_params, "lr": args.lr * args.c_lr_mult,
         "lr_mult": args.c_lr_mult, "weight_decay": 0.0,
         "name": "codepoint_c"},
    ]
    OptCls = {"lion": Lion32, "adamw": AdamW32,
              "cautious-adamw": CautiousAdamW}[args.optimizer]
    opt = OptCls(param_groups, lr=args.lr, weight_decay=args.wd)
    print(f"[opt] {args.optimizer} lr={args.lr} wd={args.wd} "
          f"scale_lr_mult={scale_lr_mult:g} c_lr_mult={args.c_lr_mult:g}")

    # ---- Resume ----
    interrupted_state = None
    global_step = 0
    samples_consumed = 0
    if args.resume is not None:
        with safe_open(str(args.resume), framework="pt") as f:
            resume_meta = f.metadata() or {}
        sd = load_file(str(args.resume))
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"[resume] warm-start from {args.resume.name} "
              f"(meta={resume_meta}, missing={len(miss)}, "
              f"unexpected={len(unexp)})")
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
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[resume] {interrupted_path} at step {global_step}")

    # Anchor c/{l2,rms}_from_init and s/{l2,rms}_from_init. Only fires
    # when the persistent buffers are still at their zero sentinel —
    # i.e., fresh start, or resuming from a ckpt that predates these
    # buffers. Once a real anchor is in place, it rides in subsequent
    # interrupted.pt saves and persists across resumes.
    n_snap = snapshot_inits_if_unset(model)
    if n_snap > 0:
        print(f"[qat] snapshotted c_init for {n_snap} QLinears "
              "(fresh start or backward-compat resume)")

    # ---- Teacher floor, data, TB ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    floor_data = load_teacher_floor(args.cache_dir, len(tok))
    L_T = float(floor_data["floor"])
    print(f"[qat] L_T = {L_T:.4f}")

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
    autocast_dtype = {"bfloat16": torch.bfloat16,
                      "float16": torch.float16,
                      "none": None}[args.autocast_dtype]
    tb_root = args.tb_dir if args.tb_dir is not None else (args.out / "tb")
    if interrupted_state and interrupted_state.get("run_name"):
        run_name = interrupted_state["run_name"]
    elif args.run_name:
        run_name = args.run_name
    else:
        run_name = datetime.now().strftime("qat_%Y%m%d_%H%M%S")
    run_dir = tb_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir),
                           purge_step=global_step if global_step else None)
    print(f"[tb] {run_dir}")
    writer.add_text("stage", "  \n".join([
        "**qat_distill** (single-stage, learnable c)",
        f"- c init from band mean (target_zero_frac_init={args.target_zero_frac_init})",
        f"- c_lr_mult = {args.c_lr_mult}",
        f"- c clamp = [{args.c_clamp_min}, {args.c_clamp_max}]",
        f"- group_size = {args.scale_group_size}",
        f"- optimizer = {args.optimizer}, lr = {args.lr}, wd = {args.wd}",
        f"- L_T = {L_T:.4f}",
    ]), global_step)

    ctrl = BestEmaTracker(ema_alpha=0.05, ema_warmup=args.ema_warmup)
    if interrupted_state and interrupted_state.get("ctrl_state"):
        ctrl.load_state_dict(interrupted_state["ctrl_state"])
    if (interrupted_state
            and interrupted_state.get("best_snapshot") is not None):
        best_snapshot = interrupted_state["best_snapshot"]
    else:
        best_snapshot = snapshot_to_cpu(model)
    interrupted_state = None  # release CPU memory

    # ---- Train ----
    running = 0.0
    running_n = 0
    pbar = tqdm(desc="qat-distill", dynamic_ncols=True,
                initial=global_step, total=args.total_steps)
    opt.zero_grad(set_to_none=True)
    model.train()

    nominal_total = max(1, args.total_steps)

    try:
        while global_step < args.total_steps:
            cur_lr = lr_at(global_step, nominal_total, args.lr,
                           args.warmup_steps, floor=args.lr_floor)
            for g in opt.param_groups:
                g["lr"] = cur_lr * g.get("lr_mult", 1.0)
                if g.get("name") == "codepoint_c":
                    g["weight_decay"] = 0.0
                else:
                    g["weight_decay"] = args.wd

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
                        f"loss={loss.item()}")
                (loss / args.grad_accum).backward()
                running += loss.item()
                running_n += 1

            grad_norm = None
            if args.max_grad_norm:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm))
            opt.step()
            clamp_qlinear_weights(model)  # keep latents in [-1, 1]
            with torch.no_grad():
                for m in model.modules():
                    if isinstance(m, QLinear):
                        m.codepoint_c.data.clamp_(args.c_clamp_min,
                                                   args.c_clamp_max)
            opt.zero_grad(set_to_none=True)
            global_step += 1

            step_loss = running / max(1, running_n)

            improved = ctrl.update(global_step, step_loss)
            if improved:
                best_snapshot = snapshot_to_cpu(model)

            if global_step % args.log_every == 0:
                pbar.set_postfix({
                    "step": global_step,
                    "loss": f"{step_loss:.4f}",
                    "ema": f"{ctrl.ema:.4f}" if ctrl.ema else "—",
                    "lr": f"{cur_lr:.2e}",
                })
                pbar.update(args.log_every)
                writer.add_scalar("loss/step", step_loss, global_step)
                if ctrl.ema is not None:
                    writer.add_scalar("loss/ema", ctrl.ema, global_step)
                    writer.add_scalar("loss/gap", ctrl.ema - L_T, global_step)
                writer.add_scalar("lr", cur_lr, global_step)
                if grad_norm is not None:
                    writer.add_scalar("grad_norm", grad_norm, global_step)
                # codepoint_c stats — watch the learned c drift
                with torch.no_grad():
                    all_c = torch.cat([m.codepoint_c.data.flatten()
                                       for m in model.modules()
                                       if isinstance(m, QLinear)])
                    writer.add_scalar("c/mean", float(all_c.mean()),
                                      global_step)
                    writer.add_scalar("c/min", float(all_c.min()),
                                      global_step)
                    writer.add_scalar("c/max", float(all_c.max()),
                                      global_step)
                    writer.add_scalar("c/p50", float(all_c.median()),
                                      global_step)
                    # L2/RMS drift from the anchor (set at script start
                    # after init+resume). The aggregate mean/p50 hide
                    # per-cell drift when changes cancel; this captures
                    # the real motion.
                    sq_diff = 0.0
                    n_c = 0
                    for m in model.modules():
                        if not (isinstance(m, QLinear)
                                and hasattr(m, "c_init")):
                            continue
                        diff = (m.codepoint_c.data - m.c_init)
                        sq_diff += float((diff * diff).sum())
                        n_c += diff.numel()
                    if n_c > 0:
                        writer.add_scalar("c/l2_from_init",
                                          sq_diff ** 0.5, global_step)
                        writer.add_scalar("c/rms_from_init",
                                          (sq_diff / n_c) ** 0.5,
                                          global_step)
                    # scales stats — mirrors c/* so we can see whether
                    # scales is absorbing the magnitude work c isn't.
                    all_s = torch.cat([m.scales.data.flatten()
                                       for m in model.modules()
                                       if isinstance(m, QLinear)])
                    writer.add_scalar("s/mean", float(all_s.mean()),
                                      global_step)
                    writer.add_scalar("s/min", float(all_s.min()),
                                      global_step)
                    writer.add_scalar("s/max", float(all_s.max()),
                                      global_step)
                    writer.add_scalar("s/p50", float(all_s.median()),
                                      global_step)
                    sq_diff_s = 0.0
                    n_s = 0
                    for m in model.modules():
                        if not (isinstance(m, QLinear)
                                and hasattr(m, "s_init")):
                            continue
                        diff = (m.scales.data - m.s_init)
                        sq_diff_s += float((diff * diff).sum())
                        n_s += diff.numel()
                    if n_s > 0:
                        writer.add_scalar("s/l2_from_init",
                                          sq_diff_s ** 0.5, global_step)
                        writer.add_scalar("s/rms_from_init",
                                          (sq_diff_s / n_s) ** 0.5,
                                          global_step)
                running = 0.0
                running_n = 0

            if (args.soft_hist_every > 0
                    and global_step % args.soft_hist_every == 0):
                hs = _first_qlinear_latent_sample(model)
                if hs is not None:
                    name, w_flat = hs
                    if torch.isfinite(w_flat).all():
                        writer.add_histogram(f"qat/latent/{name}",
                                             w_flat, global_step, bins=64)
                c_flat = _all_c_flat(model)
                if c_flat is not None and torch.isfinite(c_flat).all():
                    writer.add_histogram("qat/codepoint_c", c_flat,
                                         global_step, bins=64)
                cs_flat = _all_cs_flat(model)
                if cs_flat is not None and torch.isfinite(cs_flat).all():
                    writer.add_histogram("qat/c_times_s", cs_flat,
                                         global_step, bins=64)

            samples_at_save = (global_step
                               * args.grad_accum * args.batch_size)
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
            samples_at_save = (global_step
                               * args.grad_accum * args.batch_size)
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

    # ---- Deploy fold: snap latents to ternary, fold c into scales ----
    # Per element: target = sign(w)·c_{r,g} if |w| > c_{r,g}/2, else 0.
    # Then w /= c_{r,g}, s *= c_{r,g} — latent ends in {-1, 0, +1}.
    print(f"[qat] complete after {global_step} steps")
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
            m.scales.data.mul_(m.codepoint_c.data.to(m.scales.dtype))
            m.invalidate_q_cache()
    out_ckpt = args.out / "stage_qat.safetensors"
    save_checkpoint(model, out_ckpt, args.model, args.scale_group_size,
                    alpha=0.0, target_zero_frac=None)
    print(f"[qat] saved {out_ckpt}")
    writer.add_text(
        "stage_end",
        f"qat_distill complete: {global_step} steps",
        global_step)
    writer.flush()
    writer.close()
    if interrupted_path.exists():
        interrupted_path.unlink()
    print("[done]")


if __name__ == "__main__":
    main()
