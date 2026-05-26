"""flip_progressive.py — type-by-type progressive flip ternarization, with
resume and a final joint-flip stage.

Stages:
  1. For each role in --role-order (default q,k,v,o,gate,up,down):
       a. Promote all QLinears of that role from 8-bit passthrough
          (`levels=257`) to ternary (`levels=3`) via LS-init against W_ref.
       b. Run BopTernary+Bet1 (Bop2ndOrder) on just those modules' weights
          until plateau on the loss EMA. Restore the per-role best snapshot.
  2. Joint stage: unfreeze all 210 ternarized modules' weights and run
     BopTernary+Bet1 once more, letting roles re-optimize together.

Resume:
  --resume <interrupted.pt>  — periodic-save snapshot. Restores model,
       opt, ctrl, best_snapshot, stage cursor (role_idx or "joint"),
       and within-stage step. Fresh-built optimizer/ctrl loaded from
       saved state.
  --resume <safetensors>     — fresh start at the next un-ternarized
       role. Determined by inspecting each module's weight values
       (all-in-{-1,0,+1} → role is done). Useful when an interrupted.pt
       is missing or out-of-date.
"""
from __future__ import annotations

import argparse
import math
import signal
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
from .qlinear import QLinear
from .teacher_floor import load_or_compute as load_teacher_floor
from .flip_distill import (
    BopTernary, capture_w_ref, save_w_ref, load_w_ref,
    invalidate_all_q_caches, m_stats, trit_stats,
)


PROJ_ORDER: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


# ---------------------------------------------------------------- helpers


def collect_role_modules(model: nn.Module,
                         role_or_group: str | list[str]
                         ) -> list[tuple[str, QLinear]]:
    """(full_name, module) for every QLinear whose leaf name is in the
    role (or any role in the group). Sorted by name. Groups are how we
    co-train coupled roles in one stage (e.g. ["q_proj", "k_proj"])."""
    roles = (role_or_group,) if isinstance(role_or_group, str) \
        else tuple(role_or_group)
    role_set = set(roles)
    out: list[tuple[str, QLinear]] = []
    for name, m in model.named_modules():
        if isinstance(m, QLinear) and name.rsplit(".", 1)[-1] in role_set:
            out.append((name, m))
    out.sort(key=lambda x: x[0])
    return out


def parse_role_order(spec: str) -> list[list[str]]:
    """Parse a comma-separated role-order spec. Each item may use '+' to
    group co-trained roles. e.g. 'q_proj+k_proj,v_proj+o_proj,gate_proj'
    → [['q_proj','k_proj'], ['v_proj','o_proj'], ['gate_proj']]."""
    out: list[list[str]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        group = [r.strip() for r in item.split("+") if r.strip()]
        for r in group:
            if r not in PROJ_ORDER:
                raise SystemExit(f"unknown role {r!r}; allowed: {PROJ_ORDER}")
        out.append(group)
    return out


def stage_tag(group: list[str]) -> str:
    """Short TB/log tag for a role group."""
    if len(group) == 1:
        return f"role_{group[0]}"
    return "role_" + "+".join(g.replace("_proj", "") for g in group)


def all_qlinears(model: nn.Module) -> list[QLinear]:
    return [m for _, m in model.named_modules() if isinstance(m, QLinear)]


@torch.no_grad()
def promote_role_to_ternary(model: nn.Module,
                            role_or_group: str | list[str],
                            w_refs: dict[str, torch.Tensor],
                            n_iters: int = 5) -> tuple[int, float]:
    """LS-init trits + scales for every module of `role_or_group`;
    switch them to levels=3."""
    n = 0
    nz_sum = 0.0
    for name, m in collect_role_modules(model, role_or_group):
        W_ref = w_refs[name].to(m.weight.device, dtype=torch.float32)
        out_f, in_f = W_ref.shape
        W_blk = W_ref.view(out_f, m.n_groups, m.group_size)
        s = W_blk.abs().amax(-1).clamp_min(1e-8)
        t = torch.zeros_like(W_blk)
        for _ in range(n_iters):
            r = W_blk / s.unsqueeze(-1)
            t = torch.where(r.abs() > 0.5,
                            torch.sign(r),
                            torch.zeros_like(r))
            denom = (t * t).sum(-1)
            numer = (W_blk * t).sum(-1)
            has_nz = denom > 0
            s = torch.where(has_nz, numer / denom.clamp_min(1.0), s)
            s = s.clamp_min(1e-8)
        m.weight.data.copy_(t.view(out_f, in_f).to(m.weight.dtype))
        m.scales.data.copy_(s.to(m.scales.dtype))
        m.levels = 3
        m.invalidate_q_cache()
        nz_sum += float((t != 0).float().mean().item())
        n += 1
    return n, (nz_sum / max(1, n))


@torch.no_grad()
def detect_ternarized_roles(model: nn.Module) -> set[str]:
    """Inspect QLinear weights; a role is "done" iff all its modules have
    weights exactly in {-1, 0, +1}. Side-effect: sets `levels=3` for any
    such module (so the model's forward path is correct after resume)."""
    done: set[str] = set()
    for role in PROJ_ORDER:
        mods = collect_role_modules(model, role)
        if not mods:
            continue
        all_t = True
        for _, m in mods:
            w = m.weight.data
            if not ((w == -1) | (w == 0) | (w == 1)).all().item():
                all_t = False
                break
        if all_t:
            done.add(role)
            for _, m in mods:
                m.levels = 3
                m.invalidate_q_cache()
    return done


def freeze_for_role(model: nn.Module,
                    role_or_group: str | list[str]) -> tuple[int, int]:
    """requires_grad=True only on the named role(s) QLinear weights."""
    active_ids = {id(m.weight)
                  for _, m in collect_role_modules(model, role_or_group)}
    n_t = n_f = 0
    for p in model.parameters():
        if id(p) in active_ids:
            p.requires_grad_(True); n_t += 1
        else:
            p.requires_grad_(False); n_f += 1
    return n_t, n_f


def freeze_for_joint(model: nn.Module) -> tuple[int, int]:
    """requires_grad=True on every QLinear weight (joint stage)."""
    trit_ids = {id(m.weight) for m in all_qlinears(model)}
    n_t = n_f = 0
    for p in model.parameters():
        if id(p) in trit_ids:
            p.requires_grad_(True); n_t += 1
        else:
            p.requires_grad_(False); n_f += 1
    return n_t, n_f


def role_progress_summary(model: nn.Module) -> dict[str, str]:
    summary: dict[str, str] = {}
    for role in PROJ_ORDER:
        mods = collect_role_modules(model, role)
        if not mods:
            continue
        levels_set = sorted({int(m.levels) for _, m in mods})
        summary[role] = "+".join(str(L) for L in levels_set)
    return summary


@torch.no_grad()
def trit_stats_for_modules(modules: list[QLinear]) -> dict[str, float]:
    n = z = pos = neg = 0
    for m in modules:
        t = m.weight.data
        n += t.numel()
        z += int((t == 0).sum().item())
        pos += int((t > 0.5).sum().item())
        neg += int((t < -0.5).sum().item())
    n = max(1, n)
    return {"frac_zero": z/n, "frac_pos": pos/n, "frac_neg": neg/n}


# ---------------------------------------------------------------- resume


def save_prog_resume(path: Path, model, opt, ctrl, best_snapshot,
                     stage_kind: str, stage_idx: int, stage_step: int,
                     role_order: list[str], samples_consumed: int,
                     run_name: str | None) -> None:
    """Atomic write of the per-stage resume snapshot.

    `stage_kind` ∈ {"role", "joint"}.
    `stage_idx` is the role index (0..len(role_order)-1) for "role",
                 0 for "joint".
    `stage_step` is the step within the current stage."""
    payload = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "opt": opt.state_dict() if opt is not None else None,
        "ctrl_state": ctrl.state_dict() if ctrl is not None else None,
        "best_snapshot": best_snapshot,
        "stage_kind": stage_kind,
        "stage_idx": int(stage_idx),
        "stage_step": int(stage_step),
        "role_order": list(role_order),
        "samples_consumed": int(samples_consumed),
        "run_name": run_name,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(tmp))
    tmp.replace(path)


def _resume_state_from_pt(pt_path: Path) -> dict:
    state = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    return state


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None,
                    help="Resume from a saved checkpoint. .pt = full resume "
                         "(model+opt+ctrl+cursor); .safetensors = model only "
                         "(fresh opt, start at the next un-ternarized role).")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    # Bop hyperparams (always with Bet 1 = 2nd-order).
    ap.add_argument("--gamma", type=float, default=1e-3)
    ap.add_argument("--gamma-v", type=float, default=1e-3)
    ap.add_argument("--tau-norm", type=float, default=0.5)
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--reset-on-flip", action="store_true", default=False)
    # Per-role schedule.
    ap.add_argument("--role-order", default=",".join(PROJ_ORDER))
    ap.add_argument("--min-steps-per-role", type=int, default=500)
    ap.add_argument("--max-steps-per-role", type=int, default=5000)
    ap.add_argument("--plateau-patience", type=int, default=500)
    # Joint stage at the end (after all roles).
    ap.add_argument("--joint", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Run a joint-flip stage after all roles complete: "
                         "unfreeze all trits, BopTernary+Bet1 with plateau.")
    ap.add_argument("--joint-min-steps", type=int, default=500)
    ap.add_argument("--joint-max-steps", type=int, default=10000)
    ap.add_argument("--joint-plateau-patience", type=int, default=1000)
    ap.add_argument("--joint-tau-norm", type=float, default=None,
                    help="τ_norm for the joint stage (default = --tau-norm).")
    # Adaptive shrink: when score/max saturates at τ_norm AND no EMA
    # improvement for shrink_patience steps, multiply τ_norm by
    # --shrink-factor. Effectively turns the optimizer's flip rate up
    # when the thermostat locks at the current threshold.
    ap.add_argument("--shrink-on-plateau", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--shrink-patience", type=int, default=200,
                    help="Steps without EMA improvement (and with score/max "
                         "saturated at τ_norm) before shrinking τ_norm.")
    ap.add_argument("--shrink-factor", type=float, default=0.7,
                    help="Multiplicative shrink factor (e.g. 0.7 → τ ← 0.7·τ).")
    ap.add_argument("--shrink-saturation", type=float, default=0.95,
                    help="score/max ≥ this·τ_norm counts as saturated.")
    ap.add_argument("--tau-norm-min", type=float, default=0.05,
                    help="Hard floor for τ_norm under adaptive shrink.")
    # BestEmaTracker.
    ap.add_argument("--ctrl-ema-alpha", type=float, default=0.05)
    ap.add_argument("--ctrl-rel-thr", type=float, default=1e-3)
    # Disk / checkpoint.
    ap.add_argument("--checkpoint-every", type=int, default=500,
                    help="Steps between interrupted.pt saves within a stage.")
    ap.add_argument("--save-per-role", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Save after_NN_<role>.safetensors at end of each "
                         "role (disable to save disk; resume still works "
                         "via interrupted.pt or the final safetensors).")
    # Standard.
    ap.add_argument("--permute", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--scale-group-size", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--autocast-dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "none"])
    ap.add_argument("--latent-dtype", default="float32",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--grad-checkpointing",
                    action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tb-dir", type=Path, default=None)
    ap.add_argument("--run-name", type=str, default=None)

    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    _install_sigint_handler()

    role_groups = parse_role_order(args.role_order)
    role_order_flat = [r for group in role_groups for r in group]

    latent_dtype = {"float32": torch.float32, "float16": torch.float16,
                    "bfloat16": torch.bfloat16}[args.latent_dtype]
    autocast_dtype = {"bfloat16": torch.bfloat16,
                      "float16": torch.float16,
                      "none": None}[args.autocast_dtype]

    interrupted_path = args.out / "interrupted.pt"

    # Resolve resume source: explicit --resume, else any interrupted.pt
    # already in --out, else fresh start.
    resume_src: Path | None = None
    resume_kind: str = "fresh"
    if args.resume is not None:
        resume_src = args.resume
        resume_kind = "pt" if args.resume.suffix == ".pt" else "safetensors"
    elif interrupted_path.exists():
        resume_src = interrupted_path
        resume_kind = "pt"

    # ---- Build student: 8-bit passthrough everywhere (levels=257) ----
    do_permute = args.permute and (resume_kind == "fresh")
    print(f"[build] loading {args.model}, group_size={args.scale_group_size}, "
          f"permute={do_permute}")
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
    print(f"[build] {n_replaced} QLinear modules (init levels=257)")

    # ---- Capture / load W_ref ----
    w_ref_path = args.out / "w_ref.safetensors"
    if w_ref_path.exists():
        w_refs = load_w_ref(w_ref_path)
        print(f"[init] loaded W_ref ← {w_ref_path}")
    else:
        # Fresh build's `weight * scales` is the (permuted) teacher weight.
        w_refs = capture_w_ref(model)
        save_w_ref(w_refs, w_ref_path)
        print(f"[init] captured W_ref, saved → {w_ref_path}")

    # ---- Apply resume (model state + cursor) ----
    resume_payload: dict | None = None
    resume_done: set[str] = set()
    resume_stage_kind = "role"   # "role" | "joint"
    resume_stage_idx = 0          # role index, or 0 for joint
    resume_stage_step = 0
    resume_run_name: str | None = None
    resume_samples = 0

    if resume_kind == "pt":
        resume_payload = _resume_state_from_pt(resume_src)
        miss, unexp = model.load_state_dict(resume_payload["model"],
                                            strict=False)
        invalidate_all_q_caches(model)
        # Set levels=3 on any ternarized modules so forward is correct.
        ternarized = detect_ternarized_roles(model)
        resume_done = ternarized
        resume_stage_kind = resume_payload.get("stage_kind", "role")
        resume_stage_idx = int(resume_payload.get("stage_idx", 0))
        resume_stage_step = int(resume_payload.get("stage_step", 0))
        resume_run_name = resume_payload.get("run_name")
        resume_samples = int(resume_payload.get("samples_consumed", 0))
        print(f"[resume] {resume_src.name} (pt): "
              f"stage={resume_stage_kind}[{resume_stage_idx}] "
              f"stage_step={resume_stage_step} done={sorted(ternarized)}")
    elif resume_kind == "safetensors":
        sd = load_file(str(resume_src))
        miss, unexp = model.load_state_dict(sd, strict=False)
        invalidate_all_q_caches(model)
        ternarized = detect_ternarized_roles(model)
        resume_done = ternarized
        # Cursor: first GROUP not entirely covered by done roles.
        idx = 0
        while idx < len(role_groups) and all(r in ternarized
                                             for r in role_groups[idx]):
            idx += 1
        if idx >= len(role_groups):
            resume_stage_kind = "joint"
            resume_stage_idx = 0
        else:
            resume_stage_kind = "role"
            resume_stage_idx = idx
        resume_stage_step = 0
        print(f"[resume] {resume_src.name} (safetensors): "
              f"done={sorted(ternarized)} → starting "
              f"{resume_stage_kind}[{resume_stage_idx}]")

    # ---- Teacher floor, data, TB ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    floor_data = load_teacher_floor(args.cache_dir, len(tok))
    L_T = float(floor_data["floor"])
    print(f"[progressive] L_T = {L_T:.4f}, role groups = "
          f"{[stage_tag(g) for g in role_groups]}")

    def _worker_init(_id):
        import signal as _sig
        _sig.signal(_sig.SIGINT, _sig.SIG_IGN)

    samples_consumed = resume_samples
    ds = ShardedDataset(args.cache_dir, seed=args.seed,
                        start_skip=samples_consumed)
    dl = DataLoader(ds, batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=args.device.startswith("cuda"),
                    drop_last=True, worker_init_fn=_worker_init)
    it = iter(dl)

    tb_root = args.tb_dir if args.tb_dir is not None else (args.out / "tb")
    run_name = (resume_run_name if resume_run_name
                else (args.run_name or
                      datetime.now().strftime("flipprog_%Y%m%d_%H%M%S")))
    run_dir = tb_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    print(f"[tb] {run_dir}")
    writer.add_text("stage", "  \n".join([
        "**flip_progressive** — type-by-type + joint stage",
        f"- role groups: {[stage_tag(g) for g in role_groups]}",
        f"- adaptive shrink: enabled={args.shrink_on_plateau}, "
        f"patience={args.shrink_patience}, factor={args.shrink_factor}, "
        f"saturation={args.shrink_saturation}, floor={args.tau_norm_min}",
        f"- BopTernary + Bet1, τ_norm={args.tau_norm}, γ={args.gamma}, γ_v={args.gamma_v}",
        f"- reset-on-flip = {args.reset_on_flip}",
        f"- per-role: min/max/patience = "
        f"{args.min_steps_per_role}/{args.max_steps_per_role}/{args.plateau_patience}",
        f"- joint: enabled={args.joint}, min/max/patience = "
        f"{args.joint_min_steps}/{args.joint_max_steps}/{args.joint_plateau_patience}, "
        f"τ_norm = {args.joint_tau_norm if args.joint_tau_norm else args.tau_norm}",
        f"- non-active modules: levels=257 (8-bit passthrough)",
        f"- L_T = {L_T:.4f}",
    ]), 0)

    global_step = 0  # purely for TB x-axis

    # ----------------------------------------------------------------
    # Inner training loop, parameterized over stage.
    # ----------------------------------------------------------------
    def run_stage(stage_kind: str, stage_idx: int,
                  active_trit_params: list[torch.nn.Parameter],
                  tag: str,
                  min_steps: int, max_steps: int, plateau_patience: int,
                  tau_norm: float,
                  load_opt_state: dict | None = None,
                  load_ctrl_state: dict | None = None,
                  load_best_snapshot: dict | None = None,
                  start_step: int = 0) -> tuple[dict | None, BestEmaTracker]:
        """Run one stage (role or joint). Returns (best_snapshot, ctrl)."""
        nonlocal global_step, samples_consumed
        opt = BopTernary(
            active_trit_params, gamma=args.gamma, tau=1e-6,
            use_2nd_moment=True, gamma_v=args.gamma_v,
            tau_norm=tau_norm, eps=args.eps,
            reset_on_flip=args.reset_on_flip)
        if load_opt_state is not None:
            opt.load_state_dict(load_opt_state)
        ctrl = BestEmaTracker(
            ema_alpha=args.ctrl_ema_alpha,
            rel_threshold=args.ctrl_rel_thr,
            ema_warmup=max(1, min_steps // 2))
        if load_ctrl_state is not None:
            ctrl.load_state_dict(load_ctrl_state)
        best_snapshot = load_best_snapshot

        running = 0.0; running_n = 0
        flips_w = elems_w = 0
        step = start_step
        last_shrink_step = start_step
        pbar = tqdm(desc=f"stage[{tag}]", dynamic_ncols=True,
                    total=max_steps, initial=start_step)
        opt.zero_grad(set_to_none=True)
        model.train()

        while step < max_steps:
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
                        f"non-finite loss at stage={tag} step={step}: "
                        f"{loss.item()}")
                (loss / args.grad_accum).backward()
                running += loss.item()
                running_n += 1
                samples_consumed += args.batch_size

            n_flips, n_elems = opt.step()
            flips_w += n_flips; elems_w += n_elems
            invalidate_all_q_caches(model)
            opt.zero_grad(set_to_none=True)
            step += 1; global_step += 1

            step_loss = running / max(1, running_n)
            improved = ctrl.update(step, step_loss)
            if improved:
                best_snapshot = snapshot_to_cpu(model)

            plateau = (step >= min_steps
                       and (step - ctrl.best_step) >= plateau_patience)

            if step % args.log_every == 0:
                rate = flips_w / max(1, elems_w)
                ms = m_stats(opt)
                cur_tau = float(opt.param_groups[0]["tau_norm"])

                # Adaptive shrink: if score/max is saturated at τ_norm,
                # the EMA-best has stalled, and we haven't shrunk recently,
                # multiply τ_norm by --shrink-factor. Releases more flips.
                if (args.shrink_on_plateau and "score_max" in ms):
                    saturated = (ms["score_max"]
                                 >= args.shrink_saturation * cur_tau)
                    no_improve = (step - ctrl.best_step) >= args.shrink_patience
                    no_recent_shrink = (step - last_shrink_step) >= args.shrink_patience
                    can_shrink = cur_tau > args.tau_norm_min
                    if saturated and no_improve and no_recent_shrink and can_shrink:
                        new_tau = max(args.tau_norm_min,
                                      cur_tau * args.shrink_factor)
                        for pg in opt.param_groups:
                            pg["tau_norm"] = new_tau
                        last_shrink_step = step
                        tqdm.write(f"[shrink] {tag} step {step}: τ_norm "
                                   f"{cur_tau:.4f} → {new_tau:.4f}")
                        cur_tau = new_tau

                pbar.set_postfix({
                    "step": step,
                    "loss": f"{step_loss:.4f}",
                    "ema": f"{ctrl.ema:.4f}" if ctrl.ema else "—",
                    "best": f"{ctrl.best_ema:.4f}@{ctrl.best_step}",
                    "flip%": f"{rate*100:.3f}",
                    "τ": f"{cur_tau:.3f}",
                })
                pbar.update(args.log_every)
                writer.add_scalar(f"{tag}/loss/step", step_loss, global_step)
                if ctrl.ema is not None:
                    writer.add_scalar(f"{tag}/loss/ema", ctrl.ema, global_step)
                    writer.add_scalar(f"{tag}/loss/best_ema", ctrl.best_ema,
                                      global_step)
                writer.add_scalar(f"{tag}/flip/rate", rate, global_step)
                writer.add_scalar(f"{tag}/m/rms", ms["m_rms"], global_step)
                if "score_max" in ms:
                    writer.add_scalar(f"{tag}/score/max", ms["score_max"],
                                      global_step)
                writer.add_scalar(f"{tag}/tau_norm", cur_tau, global_step)
                writer.add_scalar(f"loss/global_step", step_loss, global_step)
                if ctrl.ema is not None:
                    writer.add_scalar(f"loss/global_ema", ctrl.ema,
                                      global_step)
                running = 0.0; running_n = 0
                flips_w = elems_w = 0

            # Periodic interrupted.pt save.
            if (args.checkpoint_every > 0
                    and step % args.checkpoint_every == 0):
                save_prog_resume(interrupted_path, model, opt, ctrl,
                                 best_snapshot, stage_kind, stage_idx, step,
                                 role_order_flat, samples_consumed, run_name)
                tqdm.write(f"[ckpt] {interrupted_path} @ {tag} step {step}")

            if plateau or _INTERRUPT["flag"]:
                break

        pbar.close()
        # Always save a snapshot at the end of the stage (covers the case
        # where the stage hit max_steps and may not be at the global best).
        save_prog_resume(interrupted_path, model, opt, ctrl, best_snapshot,
                         stage_kind, stage_idx, step, role_order_flat,
                         samples_consumed, run_name)
        return best_snapshot, ctrl

    # ----------------------------------------------------------------
    # Per-role-group outer loop.
    # ----------------------------------------------------------------
    try:
        for group_idx, group in enumerate(role_groups):
            tag = stage_tag(group)
            # Skip groups whose roles are ALL already ternarized, unless
            # we're mid-resume on this exact group.
            all_done = all(r in resume_done for r in group)
            if all_done and not (resume_stage_kind == "role"
                                 and resume_stage_idx == group_idx):
                print(f"[{tag} {group_idx+1}/{len(role_groups)}] "
                      f"already done (resume), skipping")
                continue
            mid_resume = (
                resume_payload is not None
                and resume_stage_kind == "role"
                and resume_stage_idx == group_idx
                and resume_stage_step > 0
            )
            if not mid_resume:
                n_mods, nz = promote_role_to_ternary(model, group, w_refs)
                n_t, n_f = freeze_for_role(model, group)
                invalidate_all_q_caches(model)
                print(f"\n[{tag} {group_idx+1}/{len(role_groups)}] "
                      f"promoted {n_mods} mods (nz {nz:.3f}), "
                      f"trainable={n_t} frozen={n_f}")
            else:
                n_t, n_f = freeze_for_role(model, group)
                print(f"\n[{tag} {group_idx+1}/{len(role_groups)}] "
                      f"resuming mid-stage at step {resume_stage_step}, "
                      f"trainable={n_t} frozen={n_f}")

            summary = role_progress_summary(model)
            print("  current quantization: " + ", ".join(
                f"{k}={v}" for k, v in summary.items()))

            trit_params = [m.weight
                           for _, m in collect_role_modules(model, group)]

            load_opt = load_ctrl = load_best = None
            start_step = 0
            if mid_resume:
                load_opt = resume_payload.get("opt")
                load_ctrl = resume_payload.get("ctrl_state")
                load_best = resume_payload.get("best_snapshot")
                start_step = resume_stage_step
                resume_payload = None  # consumed

            best_snap, ctrl = run_stage(
                stage_kind="role", stage_idx=group_idx,
                active_trit_params=trit_params,
                tag=tag,
                min_steps=args.min_steps_per_role,
                max_steps=args.max_steps_per_role,
                plateau_patience=args.plateau_patience,
                tau_norm=args.tau_norm,
                load_opt_state=load_opt, load_ctrl_state=load_ctrl,
                load_best_snapshot=load_best, start_step=start_step)

            if best_snap is not None:
                model.load_state_dict(best_snap, strict=False)
                invalidate_all_q_caches(model)
            print(f"[{tag}] best EMA = {ctrl.best_ema:.4f} @ step "
                  f"{ctrl.best_step}; restored best snapshot.")
            writer.add_text(
                f"{tag}/summary",
                f"best_ema={ctrl.best_ema:.4f}@{ctrl.best_step}, "
                f"final_ema={ctrl.ema:.4f}",
                global_step,
            )

            if args.save_per_role:
                stub = "+".join(g.replace("_proj", "") for g in group)
                ckpt_path = args.out / f"after_{group_idx:02d}_{stub}.safetensors"
                save_checkpoint(model, ckpt_path, args.model,
                                args.scale_group_size,
                                alpha=0.0, target_zero_frac=None)
                print(f"  saved {ckpt_path}")

            save_prog_resume(interrupted_path, model, None, None, None,
                             stage_kind="role", stage_idx=group_idx + 1,
                             stage_step=0, role_order=role_order_flat,
                             samples_consumed=samples_consumed,
                             run_name=run_name)

            if _INTERRUPT["flag"]:
                print("[!] SIGINT — stopping after current group.")
                writer.flush(); writer.close()
                sys.exit(0)

        # ----------------------------------------------------------------
        # Joint stage.
        # ----------------------------------------------------------------
        if args.joint and not _INTERRUPT["flag"]:
            mid_joint_resume = (
                resume_payload is not None
                and resume_stage_kind == "joint"
                and resume_stage_step > 0
            )
            n_t, n_f = freeze_for_joint(model)
            invalidate_all_q_caches(model)
            joint_tau = (args.joint_tau_norm
                         if args.joint_tau_norm is not None
                         else args.tau_norm)
            print(f"\n[joint] all 210 modules trainable "
                  f"(trits={n_t} frozen={n_f}), τ_norm={joint_tau}")

            trit_params = [m.weight for m in all_qlinears(model)]

            load_opt = load_ctrl = load_best = None
            start_step = 0
            if mid_joint_resume:
                load_opt = resume_payload.get("opt")
                load_ctrl = resume_payload.get("ctrl_state")
                load_best = resume_payload.get("best_snapshot")
                start_step = resume_stage_step
                resume_payload = None

            best_snap, ctrl = run_stage(
                stage_kind="joint", stage_idx=0,
                active_trit_params=trit_params,
                tag="joint",
                min_steps=args.joint_min_steps,
                max_steps=args.joint_max_steps,
                plateau_patience=args.joint_plateau_patience,
                tau_norm=joint_tau,
                load_opt_state=load_opt, load_ctrl_state=load_ctrl,
                load_best_snapshot=load_best, start_step=start_step)

            if best_snap is not None:
                model.load_state_dict(best_snap, strict=False)
                invalidate_all_q_caches(model)
            print(f"[joint] best EMA = {ctrl.best_ema:.4f} @ step "
                  f"{ctrl.best_step}; restored best snapshot.")
            writer.add_text(
                "joint/summary",
                f"best_ema={ctrl.best_ema:.4f}@{ctrl.best_step}, "
                f"final_ema={ctrl.ema:.4f}",
                global_step,
            )

            if _INTERRUPT["flag"]:
                print("[!] SIGINT — stopping in joint stage.")
                writer.flush(); writer.close()
                sys.exit(0)

    except SystemExit:
        raise
    except BaseException as e:
        try:
            print(f"[!!] {type(e).__name__}: {e} — interrupted.pt was last "
                  "saved at the most recent checkpoint_every cycle.",
                  flush=True)
        finally:
            pass
        raise

    # ---- Final safetensors ----
    out_ckpt = args.out / "stage_flip_progressive.safetensors"
    save_checkpoint(model, out_ckpt, args.model, args.scale_group_size,
                    alpha=0.0, target_zero_frac=None)
    print(f"\n[progressive] saved final → {out_ckpt}")

    # Remove interrupted.pt — run is complete.
    if interrupted_path.exists():
        interrupted_path.unlink()
        print(f"[progressive] removed {interrupted_path}")

    writer.flush(); writer.close()
    print("[done]")


if __name__ == "__main__":
    main()
