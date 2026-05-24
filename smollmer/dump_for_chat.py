"""Convert a soft-stage interrupted.pt into a safetensors checkpoint
chat.py can load — for mid-training sanity checks without disturbing
a running distill process.

Saves with alpha=0 by default (training-time forward; honest signal
for whether the underlying model still produces coherent text). Pass
--deploy to first apply rescale_well_for_deploy and save with alpha=1
(deploy form; only meaningful once basins are tight, i.e. high α).

Usage:
    smollmer-dump-for-chat --in ckpts.X/interrupted.pt --out /tmp/x.safetensors
    smollmer-chat --ckpt /tmp/x.safetensors --device cpu --dtype float32 \
        --scale-group-size 64 --max-new-tokens 50
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smollmer.build_student import load_student
from smollmer.distill import save_checkpoint
from smollmer.qlinear import rescale_well_for_deploy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, required=True,
                    help="Path to interrupted.pt from a running distill.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Path to write safetensors ckpt.")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--scale-group-size", type=int, default=64)
    ap.add_argument("--target-zero-frac", type=float, default=0.33)
    ap.add_argument("--deploy", action="store_true",
                    help="Apply rescale_well_for_deploy before saving "
                         "(latent /= a, scales *= a per group); save with "
                         "alpha=1. Without this, save with alpha=0 "
                         "(training-time forward; recommended for mid-run "
                         "sanity since the rescale needs tight basins).")
    ap.add_argument("--force-ternary", action="store_true",
                    help="Save with alpha=1 but WITHOUT the deploy rescale "
                         "— forces c(latent) at the ±1/3 boundary on the "
                         "raw latent. Use to inspect what the latents "
                         "would look like under naive ternary rounding "
                         "(typically degenerate when per-group well_a "
                         "differs from 1).")
    ap.add_argument("--smooth-t", type=float, default=None,
                    help="Override smooth temperature for mid-training smooth "
                         "QAT checkpoints. When set, apply smooth forward at "
                         "this T instead of hard-snapping. Useful for "
                         "checkpoints saved before T_global was stored in "
                         "soft_state. Read from log via check_run.sh.")
    args = ap.parse_args()

    print(f"[load] {args.in_path}")
    state = torch.load(str(args.in_path), map_location="cpu",
                       weights_only=False)
    sd = state["model"]
    soft = state.get("soft_state") or {}
    sched = soft.get("schedule", {})
    sched_alpha = float(sched.get("alpha", 0.0))
    next_step = int(state.get("next_step", 0))
    print(f"[load]   step={next_step}, schedule.alpha={sched_alpha:.4f}")

    print(f"[build] arch skeleton (group_size={args.scale_group_size})")
    model, _tok, n_replaced = load_student(
        args.model, dtype=torch.float32, levels=257,
        latent_dtype=torch.float16,
        group_size=args.scale_group_size,
        permute=False,  # ckpt already has permuted weights
    )
    print(f"[build] {n_replaced} QLinear modules; loading state...")
    # If the ckpt is from progressive QAT or qat_distill, attach
    # codepoint_c (+ qat_mask for progressive) buffers FIRST so the
    # state_dict load can populate them. Without this, those keys would
    # be silently dropped as "unexpected".
    has_qat_mask = any(k.endswith(".qat_mask") for k in sd.keys())
    has_codepoint_c = any(k.endswith(".codepoint_c") for k in sd.keys())
    has_progressive = has_qat_mask  # only progressive uses qat_mask
    has_full_qat = has_codepoint_c and not has_qat_mask
    if has_progressive or has_full_qat:
        from smollmer.progressive_distill import attach_progressive_buffers
        attach_progressive_buffers(model)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[load]   missing={len(miss)} unexpected={len(unexp)} "
          f"progressive={has_progressive} full_qat={has_full_qat}")

    if has_progressive or has_full_qat:
        # Smooth QAT (qat_smooth): if T > 0, the effective weight during
        # training is w_q(w/c)*c, not the hard-snapped {-c,0,+c}. Hard-
        # snapping at high T gives garbage output. Apply the smooth forward
        # instead so chat sees the actual training distribution.
        smooth_T = None
        if has_full_qat:
            # Priority: explicit --smooth-t > T_global in checkpoint >
            # recomputed from (T_init, anneal_step, anneal_steps)
            if args.smooth_t is not None:
                t = args.smooth_t
            else:
                ss = state.get("soft_state") or {}
                t = ss.get("T_global")
                if t is None and all(k in ss for k in
                                     ("T_init", "anneal_step", "anneal_steps")):
                    import math as _math
                    frac = (min(int(ss["anneal_step"]), int(ss["anneal_steps"]))
                            / int(ss["anneal_steps"]))
                    t = float(ss["T_init"]) * _math.cos(_math.pi / 2 * frac)
            if t is not None and float(t) > 1e-4:
                smooth_T = float(t)

        with torch.no_grad():
            if smooth_T is not None:
                import torch.nn.functional as _F
                for m in model.modules():
                    if not hasattr(m, "codepoint_c"):
                        continue
                    out_f, in_f = m.weight.shape
                    wb = m.weight.float().view(out_f, m.n_groups, m.group_size)
                    c_b = m.codepoint_c.float().unsqueeze(-1).clamp_min(1e-8)
                    cb = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32,
                                      device=wb.device)
                    probs = _F.softmax(
                        -(wb / c_b).unsqueeze(-1).sub(cb).pow(2) / smooth_T,
                        dim=-1)
                    w_q = probs.mul(cb).sum(-1)
                    eff = w_q.mul(c_b).view(out_f, in_f)
                    m.weight.data.copy_(eff.to(m.weight.dtype))
                    m.invalidate_q_cache()
                n_mod = sum(1 for m in model.modules()
                            if hasattr(m, "codepoint_c"))
                print(f"[smooth] applied smooth forward at T={smooth_T:.4f} "
                      f"across {n_mod} modules (hard snap kicks in at T≈0)")
            else:
                # Hard snap: {-c,0,+c} per slot — correct at T≈0 or for
                # qat_distill (no smooth_T in checkpoint).
                for m in model.modules():
                    if not hasattr(m, "codepoint_c"):
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
                    if has_full_qat:
                        m.weight.data.copy_(target)
                    else:
                        m.weight.data = torch.where(m.qat_mask, target,
                                                    m.weight.data)
                    m.invalidate_q_cache()
                if has_full_qat:
                    n_mod = sum(1 for m in model.modules()
                                if hasattr(m, "codepoint_c"))
                    print(f"[qat] full-QAT: snapped all latents to ternary "
                          f"across {n_mod} modules")
                else:
                    n_mod = sum(1 for m in model.modules()
                                if hasattr(m, "qat_mask")
                                and bool(m.qat_mask.any()))
                    print(f"[qat] progressive: snapped latents to STE-ternary "
                          f"at {n_mod} modules' promoted slots")

    if args.deploy and args.force_ternary:
        raise SystemExit("--deploy and --force-ternary are mutually exclusive")
    if args.deploy:
        n = rescale_well_for_deploy(model)
        print(f"[deploy] rescaled {n} QLinear modules "
              f"(latent /= a, scales *= a)")
        save_alpha = 1.0
    elif args.force_ternary:
        save_alpha = 1.0
    else:
        save_alpha = 0.0
    print(f"[save] {args.out} alpha={save_alpha} "
          f"target_zero_frac={args.target_zero_frac}")
    save_checkpoint(model, args.out, args.model, args.scale_group_size,
                    alpha=save_alpha,
                    target_zero_frac=args.target_zero_frac)
    print("[done]")


if __name__ == "__main__":
    main()
