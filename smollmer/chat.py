"""Interactive chat / inspection script. Pure PyTorch, no CUDA-only deps —
runs on CPU or any PyTorch backend (ROCm, Vulkan via torch-mlir, etc.).

Accepts both stage checkpoints (from `smollmer-distill`) and packed
checkpoints (from `smollmer-finalize`). Auto-detects format. The QLinear
forward at the recorded `levels` reproduces the training-time math.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from .build_student import quantize_in_place
from .pack import dequantize_embed_int8, unpack_ternary_158
from .qlinear import QLinear, set_levels


def _detect_packed(sd_keys) -> bool:
    return any(k.endswith(".T_packed") for k in sd_keys)


@torch.no_grad()
def _load_packed(model: torch.nn.Module, sd: dict[str, torch.Tensor],
                 fmt: str) -> None:
    """Load a packed checkpoint (smollmer-packed-bonsai-v1): 1.58 bpw trits +
    fp16 [out, n_groups] scales per QLinear, plus optional int8 embed/lm_head."""
    if fmt and fmt != "smollmer-packed-bonsai-v1":
        raise ValueError(f"unsupported packed format: {fmt!r} (expected "
                         "'smollmer-packed-bonsai-v1')")
    consumed: set[str] = set()
    for name, m in model.named_modules():
        if not isinstance(m, QLinear):
            continue
        T_key = f"{name}.T_packed"
        s_key = f"{name}.scales"
        if T_key not in sd or s_key not in sd:
            raise KeyError(f"packed checkpoint missing keys for {name}")
        T = unpack_ternary_158(sd[T_key], in_features=m.in_features,
                               dtype=m.weight.dtype)
        m.weight.data.copy_(T.to(m.weight.dtype))
        m.scales.data.copy_(sd[s_key].to(torch.float32))
        consumed.update({T_key, s_key})
        if m.bias is not None and f"{name}.bias" in sd:
            m.bias.data.copy_(sd[f"{name}.bias"].to(m.bias.dtype))
            consumed.add(f"{name}.bias")
    # Dequantize int8-stored embed_tokens / lm_head if present, writing the
    # fp tensor back under the original `*.weight` key so load_state_dict
    # picks it up below.
    for ki in [k for k in sd if k.endswith(".weight_int8")]:
        base = ki[: -len("_int8")]
        ks = f"{base}_scale"
        if ks not in sd:
            raise KeyError(f"int8 weight {ki} missing scale {ks}")
        sd[base] = dequantize_embed_int8(sd[ki], sd[ks])
        consumed.update({ki, ks})
    rest = {k: v for k, v in sd.items() if k not in consumed}
    model.load_state_dict(rest, strict=False)


def _strip_compile_prefix(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """torch.compile wraps the model in OptimizedModule and prefixes every key
    with `_orig_mod.`; strip it so the checkpoint is portable."""
    prefix = "_orig_mod."
    if any(k.startswith(prefix) for k in sd):
        return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}
    return sd


def load_model(model_id: str, ckpt_path: Path, device: str, dtype: torch.dtype,
               group_size: int):
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    if hasattr(model, "config"):
        model.config.use_cache = True  # ok for chat (single-thread inference)
    # Match the latent dtype to the user's chosen inference dtype so the
    # forward path doesn't need an extra cast per linear (and so that
    # load_state_dict from a fp16-trained ckpt lands in matching storage).
    quantize_in_place(model, levels=3, latent_dtype=dtype, group_size=group_size)

    with safe_open(str(ckpt_path), framework="pt") as f:
        meta = f.metadata() or {}
    sd = load_file(str(ckpt_path))
    is_packed = _detect_packed(sd.keys())

    if is_packed:
        _load_packed(model, sd, fmt=meta.get("format", ""))
        set_levels(model, 3)
    else:
        sd = _strip_compile_prefix(sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[load] WARNING: {len(missing)} missing keys (model has them, ckpt does not). "
                  f"First 5: {missing[:5]}")
        if unexpected:
            print(f"[load] WARNING: {len(unexpected)} unexpected keys (ckpt has them, model does not). "
                  f"First 5: {unexpected[:5]}")
        if not missing and not unexpected:
            print(f"[load] all {len(sd)} keys matched")
        set_levels(model, int(meta.get("levels", 3)))

    # Sanity check: at least one QLinear should have non-trivial weight values.
    for name, m in model.named_modules():
        if isinstance(m, QLinear):
            w = m.weight.detach().float()
            print(f"[sanity] {name}: weight |max|={w.abs().max().item():.3f} "
                  f"mean|w|={w.abs().mean().item():.3f} scales|max|={m.scales.abs().max().item():.3f}")
            break

    return model.to(device).eval(), meta, is_packed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M",
                    help="Base model id (used for tokenizer + arch skeleton).")
    ap.add_argument("--device", default="cpu",
                    help="cpu / cuda / hip — keep cpu on the AMD 780M box.")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--scale-group-size", type=int, default=None,
                    help="Per-(row, col-group) scale granularity. Defaults to "
                         "the value recorded in the ckpt metadata.")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--levels", type=int, default=None,
                    help="Override quantization level (overrides ckpt metadata).")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]

    # Peek at metadata first so we can pick the right group_size.
    with safe_open(str(args.ckpt), framework="pt") as f:
        peek_meta = f.metadata() or {}
    group_size = (args.scale_group_size if args.scale_group_size is not None
                  else int(peek_meta.get("group_size", 128)))

    print(f"[load] {args.ckpt} on {args.device}/{args.dtype} "
          f"(group_size={group_size})")
    model, meta, is_packed = load_model(args.model, args.ckpt, args.device, dtype,
                                        group_size)

    if args.levels is not None:
        set_levels(model, args.levels)

    cur_levels = next((m.levels for m in model.modules() if isinstance(m, QLinear)), None)
    fmt = "packed" if is_packed else "stage"
    print(f"[ready] format={fmt} levels={cur_levels} meta={meta}")
    print("Type a prompt and Enter; Ctrl-D / Ctrl-C to exit.\n")

    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt.strip():
            continue
        # Cached training sequences begin with BOS (`<|endoftext|>`); the
        # SmolLM2 tokenizer does NOT prepend it by default (`add_bos_token=False`),
        # so without this the student sees an out-of-distribution prefix and
        # collapses into degenerate loops. Prepend explicitly.
        if tok.bos_token and not prompt.startswith(tok.bos_token):
            prompt = tok.bos_token + prompt
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(args.device)
        streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        with torch.no_grad():
            model.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                min_p=args.min_p,
                streamer=streamer,
                pad_token_id=tok.eos_token_id,
            )
        print()


if __name__ == "__main__":
    main()
