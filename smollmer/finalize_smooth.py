"""Finalize a smooth QAT run: snap latents to hard ternary, fold c into scales.

Reads interrupted.pt (or any qat_smooth checkpoint), applies the deploy fold,
writes stage_smooth.safetensors. Does NOT delete interrupted.pt.

Also writes stage_smooth_best.safetensors if the best snapshot was taken at
T≈0 (T_at_best < 1e-3 in soft_state).

Usage:
    python -m smollmer.finalize_smooth --ckpt ckpts.qat-M-smooth/interrupted.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .build_student import load_student
from .distill import save_checkpoint
from .qat_distill import attach_learnable_c
from .qlinear import QLinear, set_soft_mode


def _deploy_fold(model: torch.nn.Module) -> None:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="Path to interrupted.pt (or named checkpoint).")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory. Defaults to the checkpoint's parent.")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--scale-group-size", type=int, default=64)
    args = ap.parse_args()

    out_dir = args.out_dir or args.ckpt.parent

    print(f"[load] {args.ckpt}")
    state = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    next_step = int(state.get("next_step", 0))
    ss = state.get("soft_state") or {}
    anneal_step = int(ss.get("anneal_step", 0))
    anneal_steps = int(ss.get("anneal_steps", 1))
    T_init = float(ss.get("T_init", 0.0))
    T_at_best = float(ss.get("T_at_best", float("inf")))
    ctrl = state.get("ctrl_state") or {}
    best_step = int(ctrl.get("best_step", 0))
    best_ema = float(ctrl.get("best_ema", float("inf")))
    best_snapshot = state.get("best_snapshot")
    print(f"[load]   step={next_step}  anneal={anneal_step}/{anneal_steps}  "
          f"T_init={T_init:.4f}  T_at_best={T_at_best:.4f}")
    print(f"[load]   best_ema={best_ema:.4f} @ step {best_step}  "
          f"T_at_best={'valid (≈0)' if T_at_best < 1e-3 else 'invalid (T>0)'}")

    print(f"[build] {args.model} group_size={args.scale_group_size}")
    model, _, n_replaced = load_student(
        args.model, dtype=torch.float32, levels=257,
        latent_dtype=torch.float32,
        group_size=args.scale_group_size,
        permute=False,
    )
    set_soft_mode(model, alpha=0.0)
    attach_learnable_c(model, default_c=2.0 / 3.0)
    print(f"[build] {n_replaced} QLinear modules")

    miss, unexp = model.load_state_dict(state["model"], strict=False)
    print(f"[load]   missing={len(miss)} unexpected={len(unexp)}")

    print("[fold] snapping latents + folding c into scales ...")
    _deploy_fold(model)

    out_ckpt = out_dir / "stage_smooth.safetensors"
    save_checkpoint(model, out_ckpt, args.model, args.scale_group_size,
                    alpha=0.0, target_zero_frac=None)
    print(f"[save] {out_ckpt}")

    if best_snapshot is not None and T_at_best < 1e-3:
        print(f"[best] deploying best snapshot (step {best_step}, "
              f"EMA {best_ema:.4f}, T_at_best={T_at_best:.4f}) ...")
        model.load_state_dict(best_snapshot, strict=False)
        _deploy_fold(model)
        best_ckpt = out_dir / "stage_smooth_best.safetensors"
        save_checkpoint(model, best_ckpt, args.model, args.scale_group_size,
                        alpha=0.0, target_zero_frac=None)
        print(f"[save] {best_ckpt}")
    else:
        print(f"[best] skipping best snapshot "
              f"(T_at_best={T_at_best:.4f} — was taken during smooth phase)")

    print("[done]")


if __name__ == "__main__":
    main()
