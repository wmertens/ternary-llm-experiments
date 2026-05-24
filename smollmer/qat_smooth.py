"""qat_smooth.py — single-stage QAT with smooth ternary forward and
adaptive temperature annealing.

Every QLinear uses a temperature-scaled softmax expectation over {-1,0,+1}
instead of STE. As temperature T→0 the forward converges to hard ternary.

Annealing schedule (no separate compress/anneal phases):
  - T starts at T_init, computed from the weight distribution at init so
    the median latent has 9:1 odds on its nearest codebook value.
  - Each optimizer step where loss < slow_ema: anneal_step += 1.
  - T = T_init × cos(π/2 × anneal_step / anneal_steps).
  - When anneal_step == anneal_steps: T=0, forward falls through to STE.
  - slow_ema tracks a 1/slow_ema_alpha-step horizon (default 200 steps).

Per-layer temperature scaling (optional, from calibrate.py):
  T_layer = T_global × temp_scale_layer
  temp_scale = exp(β·z) where z is z-scored log Hessian trace.
  Sensitive layers stay soft (T > 0) longer.

Optimizer: latents + scales trained; codepoint_c frozen (requires_grad=False).
Scales get a clean gradient through the smooth forward (d/ds of w_q·c·s),
so scale_lr_mult defaults to 1.0 here (vs 1/group_size in qat_distill).
"""
from __future__ import annotations

import argparse
import json
import math
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
    AdamW32, BestEmaTracker, CautiousAdamW, Lion32, PRM32,
    ShardedDataset, _INTERRUPT, _install_sigint_handler,
    kl_with_rest, lr_at, save_checkpoint, save_resume, snapshot_to_cpu,
)
from .qlinear import (
    QLinear, clamp_qlinear_weights, set_soft_mode,
    set_smooth_temp, set_smooth_alpha, compute_T_init,
)
from .qat_distill import (
    attach_learnable_c, init_c_from_band_mean,
    snapshot_inits_if_unset, _first_qlinear_latent_sample,
)
from .teacher_floor import load_or_compute as load_teacher_floor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--resume-pt-weights", type=Path, default=None,
                    help="Load model weights (only) from an interrupted.pt — "
                         "fresh optimizer, fresh anneal state. Use to warm-start "
                         "from a prior run's model without inheriting its "
                         "optimizer momentum.")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-floor", type=float, default=1.0)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--wd", type=float, default=0.001)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--optimizer", default="cautious-adamw",
                    choices=["lion", "adamw", "cautious-adamw", "prm"])
    ap.add_argument("--prm-softness", type=float, default=1.0,
                    help="PRM lam_pop. q=1/2 on the LOO boundary when "
                         "softness=1; larger → more conservative. "
                         "Practical range [0.3, 3]. Ignored for non-PRM opts.")
    ap.add_argument("--scale-group-size", type=int, default=64)
    ap.add_argument("--scale-lr-mult", type=float, default=1.0,
                    help="LR multiplier for scales. Defaults to 1.0 (same as "
                         "latents) since scales have a clean gradient through "
                         "the smooth forward.")
    ap.add_argument("--target-zero-frac-init", type=float, default=0.25,
                    help="Used to seed codepoint_c at init (frozen thereafter).")
    ap.add_argument("--c-clamp-min", type=float, default=0.05)
    ap.add_argument("--c-clamp-max", type=float, default=1.0)
    ap.add_argument("--freeze-scales", action="store_true", default=False,
                    help="Freeze per-group scales (no grad, no optimizer state). "
                         "Use when warm-starting from a checkpoint whose scales "
                         "are already well-calibrated.")
    # Temperature annealing
    ap.add_argument("--anneal-steps", type=int, default=1000,
                    help="Number of 'good' optimizer steps (loss < slow_ema "
                         "and model has plateaued) to complete the T_init→0 "
                         "cosine anneal.")
    ap.add_argument("--slow-ema-alpha", type=float, default=0.005,
                    help="Decay rate for the slow EMA used to gate annealing "
                         "(1/alpha = window in steps, default 200).")
    ap.add_argument("--fast-ema-alpha", type=float, default=0.1,
                    help="Decay rate for the fast EMA (1/alpha = window, "
                         "default 10 steps). Anneal only fires when the "
                         "fast EMA is close to the slow EMA, i.e. the model "
                         "has stopped improving rapidly.")
    ap.add_argument("--anneal-gap-thr", type=float, default=0.005,
                    help="Relative slow-fast EMA gap below which the anneal "
                         "gate fires: (slow-fast)/slow < thr. Default 0.005 "
                         "(0.5%%): anneal pauses while the model is still "
                         "improving by >0.5%% relative to its trend.")
    ap.add_argument("--blend-steps", type=int, default=500,
                    help="Steps over which smooth_alpha anneals linearly 1→0. "
                         "At alpha=1 the forward is FP32-equivalent (teacher "
                         "loss from step 0, no quantization shock). At alpha=0 "
                         "pure smooth ternary. Default 500. Set to 0 to disable.")
    ap.add_argument("--t-floor", type=float, default=0.001,
                    help="Minimum temperature kept after anneal completes. "
                         "Keeps the smooth backward active (avoiding STE dead "
                         "zones) while the forward is already near-hard-ternary. "
                         "Default 0.01: only weights within ~0.05 of the "
                         "decision boundary see non-negligible gradient.")
    ap.add_argument("--t-init-odds", type=float, default=9.0,
                    help="Target odds ratio for T_init computation: the median "
                         "latent starts with this ratio of p(correct)/p(2nd).")
    ap.add_argument("--t-init", type=float, default=None,
                    help="Override the auto-computed T_init. When set, skips "
                         "compute_T_init entirely. Useful for starting softer "
                         "(e.g. --t-init 1.0) to keep boundary weights closer "
                         "to their fp32 values early in training.")
    # Hessian calibration
    ap.add_argument("--calib-file", type=Path, default=None,
                    help="JSON from calibrate.py: per-layer temp_scale values. "
                         "If omitted, all layers use the same temperature.")
    # Standard options
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
    ap.add_argument("--total-steps", type=int, default=40000)
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
    fresh_start = (args.resume is None
                   and args.resume_pt_weights is None
                   and not interrupted_path.exists())
    do_permute = args.permute and fresh_start

    # ---- Load calibration file ----
    temp_scales: dict[str, float] | None = None
    if args.calib_file is not None:
        with open(args.calib_file) as f:
            calib = json.load(f)
        temp_scales = {k: v["temp_scale"] for k, v in calib.items()}
        print(f"[calib] loaded {len(temp_scales)} layer temp_scales "
              f"from {args.calib_file}")
    else:
        print("[calib] no calibration file; uniform temperature across layers")

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
    print(f"[build] {n_replaced} QLinear modules (latent dtype {latent_dtype})")

    set_soft_mode(model, alpha=0.0, target_zero_frac=None)
    attach_learnable_c(model, default_c=2.0 / 3.0)

    if fresh_start:
        tzf = (args.target_zero_frac_init
               if 0.0 < args.target_zero_frac_init < 1.0 else None)
        cstats = init_c_from_band_mean(model, tzf, fallback_c=2.0 / 3.0)
        print(f"[smooth] c init (frozen): mean={cstats['mean']:.4f} "
              f"min={cstats['min']:.4f} max={cstats['max']:.4f}")

    # Freeze codepoint_c entirely — no grad, no optimizer state
    for m in model.modules():
        if isinstance(m, QLinear):
            m.codepoint_c.requires_grad_(False)
            m.codepoint_c.data.clamp_(args.c_clamp_min, args.c_clamp_max)
    print("[smooth] codepoint_c frozen (no grad)")

    if args.freeze_scales:
        for m in model.modules():
            if isinstance(m, QLinear):
                m.scales.requires_grad_(False)
        print("[smooth] scales frozen (no grad)")

    # ---- Compute T_init (placeholder — recomputed after warm-start if needed) ----
    if args.t_init is not None:
        T_init = args.t_init
        print(f"[smooth] T_init = {T_init:.4f} (override, anneal_steps={args.anneal_steps})")
    else:
        T_init = compute_T_init(model, target_odds=args.t_init_odds)
        print(f"[smooth] T_init = {T_init:.4f} "
              f"(target_odds={args.t_init_odds}, anneal_steps={args.anneal_steps})")

    # ---- Optimizer: two param groups (latents, scales) ----
    scale_param_ids = {id(m.scales) for m in model.modules()
                       if isinstance(m, QLinear)}
    scale_params, other_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in scale_param_ids:
            scale_params.append(p)
        else:
            other_params.append(p)
    param_groups = [
        {"params": other_params, "lr": args.lr, "lr_mult": 1.0,
         "weight_decay": args.wd, "name": "latents"},
        {"params": scale_params, "lr": args.lr * args.scale_lr_mult,
         "lr_mult": args.scale_lr_mult, "weight_decay": args.wd,
         "name": "scales"},
    ]
    OptCls = {"lion": Lion32, "adamw": AdamW32,
              "cautious-adamw": CautiousAdamW, "prm": PRM32}[args.optimizer]
    opt_kwargs = dict(lr=args.lr, weight_decay=args.wd)
    if args.optimizer == "prm":
        opt_kwargs["softness"] = args.prm_softness
    opt = OptCls(param_groups, **opt_kwargs)
    print(f"[opt] {args.optimizer} lr={args.lr} wd={args.wd} "
          f"scale_lr_mult={args.scale_lr_mult:g}")

    # ---- Resume ----
    interrupted_state = None
    global_step = 0
    samples_consumed = 0
    anneal_step = 0
    blend_step = 0
    slow_ema: float | None = None
    fast_ema: float | None = None
    T_at_best: float = float("inf")  # T when best_snapshot was taken

    if args.resume_pt_weights is not None:
        pt = torch.load(str(args.resume_pt_weights), map_location="cpu",
                        weights_only=False)
        sd = pt["model"] if "model" in pt else pt
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"[resume] weights-only warm-start from {args.resume_pt_weights.name} "
              f"(missing={len(miss)}, unexpected={len(unexp)})")
        if args.t_init is None:
            T_init = compute_T_init(model, target_odds=args.t_init_odds)
            print(f"[smooth] T_init recomputed from warm-start weights: {T_init:.4f}")
    elif args.resume is not None:
        with safe_open(str(args.resume), framework="pt") as f:
            resume_meta = f.metadata() or {}
        sd = load_file(str(args.resume))
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"[resume] warm-start from {args.resume.name} "
              f"(meta={resume_meta}, missing={len(miss)}, unexpected={len(unexp)})")
        if args.t_init is None:
            T_init = compute_T_init(model, target_odds=args.t_init_odds)
            print(f"[smooth] T_init recomputed from warm-start weights: {T_init:.4f}")
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
        ss = interrupted_state.get("soft_state") or {}
        anneal_step = int(ss.get("anneal_step", 0))
        blend_step = int(ss.get("blend_step", args.blend_steps))  # default: complete
        slow_ema = ss.get("slow_ema", None)
        fast_ema = ss.get("fast_ema", slow_ema)
        T_at_best = float(ss.get("T_at_best", float("inf")))
        # T_init from checkpoint overrides freshly-computed one on resume
        if "T_init" in ss:
            T_init = float(ss["T_init"])
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[resume] {interrupted_path} at step {global_step}, "
              f"anneal_step={anneal_step}, T_init={T_init:.4f}")

    n_snap = snapshot_inits_if_unset(model)
    if n_snap > 0:
        print(f"[smooth] snapshotted s_init for {n_snap} QLinears")

    # ---- Teacher floor, data, TB ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    floor_data = load_teacher_floor(args.cache_dir, len(tok))
    L_T = float(floor_data["floor"])
    print(f"[smooth] L_T = {L_T:.4f}")

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
        run_name = datetime.now().strftime("smooth_%Y%m%d_%H%M%S")
    run_dir = tb_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir),
                           purge_step=global_step if global_step else None)
    print(f"[tb] {run_dir}")
    writer.add_text("stage", "  \n".join([
        "**qat_smooth** (smooth ternary, adaptive anneal)",
        f"- T_init = {T_init:.4f}, anneal_steps = {args.anneal_steps}",
        f"- slow_ema_alpha = {args.slow_ema_alpha} (window ~{1/args.slow_ema_alpha:.0f} steps)",
        f"- calib_file = {args.calib_file}",
        f"- scale_lr_mult = {args.scale_lr_mult:g}",
        f"- c frozen at init (target_zero_frac_init={args.target_zero_frac_init})",
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
        best_snapshot = None  # only snapshot once T reaches floor
    interrupted_state = None

    # ---- Train ----
    running = 0.0
    running_n = 0
    pbar = tqdm(desc="qat-smooth", dynamic_ncols=True,
                initial=global_step, total=args.total_steps)
    opt.zero_grad(set_to_none=True)
    model.train()
    nominal_total = max(1, args.total_steps)

    def _smooth_state() -> dict:
        return {"anneal_step": anneal_step, "blend_step": blend_step,
                "slow_ema": slow_ema, "fast_ema": fast_ema,
                "T_init": T_init, "T_global": T_global,
                "anneal_steps": args.anneal_steps,
                "T_at_best": T_at_best}

    try:
        while global_step < args.total_steps:
            # Current temperature and alpha blend
            frac = min(anneal_step, args.anneal_steps) / args.anneal_steps
            T_global = max(T_init * math.cos(math.pi / 2 * frac),
                           args.t_floor)
            at_floor = (anneal_step >= args.anneal_steps)
            set_smooth_temp(model, T_global, temp_scales, at_floor=at_floor)
            smooth_alpha = (
                1.0 - min(blend_step, args.blend_steps) / args.blend_steps
                if args.blend_steps > 0 else 0.0)
            set_smooth_alpha(model, smooth_alpha)

            cur_lr = lr_at(global_step, nominal_total, args.lr,
                           args.warmup_steps, floor=args.lr_floor)
            for g in opt.param_groups:
                g["lr"] = cur_lr * g.get("lr_mult", 1.0)
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
                        f"non-finite loss at step {global_step}: {loss.item()}")
                (loss / args.grad_accum).backward()
                running += loss.item()
                running_n += 1

            grad_norm = None
            if args.max_grad_norm:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm))
            opt.step()
            clamp_qlinear_weights(model)
            opt.zero_grad(set_to_none=True)
            global_step += 1
            if blend_step < args.blend_steps:
                blend_step += 1

            step_loss = running / max(1, running_n)

            # Update slow + fast EMAs, advance anneal only when plateaued
            if slow_ema is None:
                slow_ema = step_loss
                fast_ema = step_loss
            else:
                slow_ema = ((1.0 - args.slow_ema_alpha) * slow_ema
                            + args.slow_ema_alpha * step_loss)
                fast_ema = ((1.0 - args.fast_ema_alpha) * fast_ema
                            + args.fast_ema_alpha * step_loss)
            plateau_gap = (slow_ema - fast_ema) / (slow_ema + 1e-8)
            if (step_loss < slow_ema
                    and plateau_gap < args.anneal_gap_thr
                    and anneal_step < args.anneal_steps):
                anneal_step += 1

            improved = ctrl.update(global_step, step_loss)
            if improved and at_floor:
                best_snapshot = snapshot_to_cpu(model)
                T_at_best = T_global

            if global_step % args.log_every == 0:
                postfix = {
                    "step": global_step,
                    "loss": f"{step_loss:.4f}",
                    "ema": f"{ctrl.ema:.4f}" if ctrl.ema else "—",
                    "T": f"{T_global:.4f}",
                    "ann": f"{anneal_step}/{args.anneal_steps}",
                    "gap": f"{plateau_gap:.3f}",
                }
                if blend_step < args.blend_steps:
                    postfix["α"] = f"{smooth_alpha:.3f}"
                pbar.set_postfix(postfix)
                pbar.update(args.log_every)
                writer.add_scalar("loss/step", step_loss, global_step)
                if ctrl.ema is not None:
                    writer.add_scalar("loss/ema", ctrl.ema, global_step)
                    writer.add_scalar("loss/gap", ctrl.ema - L_T, global_step)
                writer.add_scalar("lr", cur_lr, global_step)
                writer.add_scalar("smooth/T_global", T_global, global_step)
                writer.add_scalar("smooth/blend_alpha", smooth_alpha, global_step)
                writer.add_scalar("smooth/anneal_step", anneal_step, global_step)
                writer.add_scalar("smooth/anneal_frac",
                                  anneal_step / args.anneal_steps, global_step)
                if slow_ema is not None:
                    writer.add_scalar("smooth/slow_ema", slow_ema, global_step)
                if fast_ema is not None:
                    writer.add_scalar("smooth/fast_ema", fast_ema, global_step)
                    writer.add_scalar("smooth/plateau_gap", plateau_gap,
                                      global_step)
                if grad_norm is not None:
                    writer.add_scalar("grad_norm", grad_norm, global_step)

                with torch.no_grad():
                    all_s = torch.cat([m.scales.data.flatten()
                                       for m in model.modules()
                                       if isinstance(m, QLinear)])
                    writer.add_scalar("s/mean", float(all_s.mean()), global_step)
                    writer.add_scalar("s/min", float(all_s.min()), global_step)
                    writer.add_scalar("s/max", float(all_s.max()), global_step)
                    writer.add_scalar("s/p50", float(all_s.median()), global_step)
                    sq_diff_s = sum(
                        float((m.scales.data - m.s_init).pow(2).sum())
                        for m in model.modules()
                        if isinstance(m, QLinear) and hasattr(m, "s_init"))
                    n_s = sum(m.scales.numel() for m in model.modules()
                              if isinstance(m, QLinear))
                    if n_s > 0:
                        writer.add_scalar("s/l2_from_init",
                                          sq_diff_s ** 0.5, global_step)
                        writer.add_scalar("s/rms_from_init",
                                          (sq_diff_s / n_s) ** 0.5, global_step)
                running = 0.0
                running_n = 0

            if (args.soft_hist_every > 0
                    and global_step % args.soft_hist_every == 0):
                hs = _first_qlinear_latent_sample(model)
                if hs is not None:
                    name, w_flat = hs
                    if torch.isfinite(w_flat).all():
                        writer.add_histogram(f"smooth/latent/{name}",
                                             w_flat, global_step, bins=64)

            samples_at_save = global_step * args.grad_accum * args.batch_size
            if (args.checkpoint_every > 0
                    and global_step % args.checkpoint_every == 0):
                save_resume(interrupted_path, model, opt, global_step,
                            best_snapshot, ctrl.state_dict(), run_name,
                            samples_consumed=samples_at_save,
                            soft_state=_smooth_state())
                tqdm.write(f"[ckpt] {interrupted_path} @ step {global_step}  "
                           f"T={T_global:.4f} ann={anneal_step}/{args.anneal_steps}")

            if _INTERRUPT["flag"]:
                save_resume(interrupted_path, model, opt, global_step,
                            best_snapshot, ctrl.state_dict(), run_name,
                            samples_consumed=samples_at_save,
                            soft_state=_smooth_state())
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
                        soft_state=_smooth_state())
            print(f"[!] emergency save → {interrupted_path} "
                  f"(reason: {type(e).__name__})", flush=True)
        except Exception as save_err:
            print(f"[!!] emergency save failed: {save_err}", flush=True)
        raise
    finally:
        pbar.close()

    # ---- Deploy fold: snap latents to hard ternary, fold c into scales ----
    print(f"[smooth] complete after {global_step} steps  "
          f"(anneal_step={anneal_step}/{args.anneal_steps})")
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

    out_ckpt = args.out / "stage_smooth.safetensors"
    save_checkpoint(model, out_ckpt, args.model, args.scale_group_size,
                    alpha=0.0, target_zero_frac=None)
    print(f"[smooth] saved {out_ckpt}")

    # Also deploy the best-EMA snapshot, but only if it was taken at T≈0
    # (latents optimised for hard ternary). Snapping a snapshot from the
    # smooth phase (T>0) produces a different model than training used there.
    if (best_snapshot is not None
            and ctrl.best_step != global_step
            and T_at_best < 1e-3):
        model.load_state_dict(best_snapshot, strict=False)
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
        best_ckpt = args.out / "stage_smooth_best.safetensors"
        save_checkpoint(model, best_ckpt, args.model, args.scale_group_size,
                        alpha=0.0, target_zero_frac=None)
        print(f"[smooth] saved best-snapshot deploy → {best_ckpt} "
              f"(step {ctrl.best_step}, EMA {ctrl.best_ema:.4f})")
    writer.add_text("stage_end",
                    f"qat_smooth complete: {global_step} steps, "
                    f"anneal_step={anneal_step}/{args.anneal_steps}",
                    global_step)
    writer.flush()
    writer.close()
    if interrupted_path.exists():
        interrupted_path.unlink()
    print("[done]")


if __name__ == "__main__":
    main()
