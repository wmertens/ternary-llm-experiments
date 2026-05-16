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
        # Snap latents to the STE-ternary target so chat sees what the
        # QAT forward was actually producing during training.
        # - Progressive: snap only at qat_mask=True; unpromoted slots
        #   keep their latent (chat shows the actual mid-training forward).
        # - Full QAT (qat_distill): snap everywhere — every slot was
        #   already ternarized in forward.
        with torch.no_grad():
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
                    m.weight.data.copy_(target)  # snap every slot
                else:
                    # Progressive: only at promoted slots.
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
                        if hasattr(m, "qat_mask") and bool(m.qat_mask.any()))
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
