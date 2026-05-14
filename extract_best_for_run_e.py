"""Extract best_snapshot from a Run D interrupted.pt, rescale latents from
c_old (Run D's c) to c_new (Run E's c), strip progressive buffers, and save
as a safetensors warm-start for progressive_distill --resume.

Math: per QLinear,
    w_new = w_old * (c_new / c_old)
    s_new = s_old * (c_old / c_new)
so forward = (w_new * s_new) is unchanged. Committed slots that were at
±c_old in latent are now at ±c_new exactly.
"""
import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--c-old", type=float, default=0.85)
    ap.add_argument("--c-new", type=float, default=2.0 / 3.0)
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()

    print(f"[load] {args.in_path}")
    st = torch.load(str(args.in_path), map_location="cpu", weights_only=False)
    bs = st.get("best_snapshot")
    if bs is None:
        raise SystemExit("no best_snapshot in checkpoint")
    cs = st.get("ctrl_state") or {}
    print(f"[load] best_step={cs.get('best_step')} best_ema={cs.get('best_ema')}")

    mult = args.c_new / args.c_old
    inv = args.c_old / args.c_new
    print(f"[rescale] c_old={args.c_old}  c_new={args.c_new}  "
          f"w *= {mult:.4f}  s *= {inv:.4f}")

    out: dict[str, torch.Tensor] = {}
    n_w = n_s = n_skip = 0
    for k, v in bs.items():
        if "frozen_mask" in k or "frozen_target" in k:
            n_skip += 1
            continue
        if k.endswith(".weight") and v.ndim == 2 and v.dtype.is_floating_point \
                and any(p in k for p in (".q_proj", ".k_proj", ".v_proj",
                                          ".o_proj", ".gate_proj", ".up_proj",
                                          ".down_proj")):
            out[k] = (v.to(torch.float32) * mult).to(v.dtype)
            n_w += 1
        elif k.endswith(".scales"):
            out[k] = (v.to(torch.float32) * inv).to(v.dtype)
            n_s += 1
        else:
            out[k] = v
    print(f"[rescale] adjusted {n_w} weights and {n_s} scales; "
          f"stripped {n_skip} progressive buffers; passthrough {len(out) - n_w - n_s}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(args.out), metadata={
        "model_id": args.model,
        "group_size": str(args.group_size),
        "best_step": str(cs.get("best_step")),
        "best_ema": str(cs.get("best_ema")),
        "c_old": str(args.c_old),
        "c_new": str(args.c_new),
        "mode": "soft",
        "alpha": "0.0",
        "levels": "3",
    })
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
