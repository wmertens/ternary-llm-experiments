"""Materialize a packed smollmer checkpoint as a standard HF directory.

Reverses `finalize.to_packed_state_dict`: ternary `T_packed` + per-(row, group)
`scales` become a dense fp16 weight, int8 embed/lm_head are dequantized to fp16,
and everything else is cast to fp16. The output is a vanilla HF model directory
that `optimum-cli export onnx` can consume directly — and that the
transformers.js `quantize.py` script can then re-quantize as q2 / q4f16.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from .pack import dequantize_embed_int8, unpack_ternary_158


@torch.no_grad()
def materialize(model: torch.nn.Module, sd: dict[str, torch.Tensor],
                fmt: str, dtype: torch.dtype) -> None:
    if fmt and fmt != "smollmer-packed-bonsai-v1":
        raise ValueError(f"unsupported packed format: {fmt!r} (expected "
                         "'smollmer-packed-bonsai-v1')")
    consumed: set[str] = set()
    fp_state = dict(model.state_dict())

    # Ternary projections: W[r, c] = T[r, c] * scale[r, c // group_size].
    for name, mod in model.named_modules():
        weight_key = f"{name}.weight"
        T_key = f"{name}.T_packed"
        s_key = f"{name}.scales"
        if T_key not in sd or s_key not in sd:
            continue
        W = fp_state[weight_key]
        T = unpack_ternary_158(sd[T_key], in_features=mod.in_features,
                               dtype=torch.float32)              # [out, in]
        scale = sd[s_key].to(torch.float32)                       # [out, n_groups]
        out_f, in_f = T.shape
        n_groups = scale.shape[1]
        if in_f % n_groups != 0:
            raise ValueError(f"in_features={in_f} not divisible by "
                             f"n_groups={n_groups} for {weight_key}")
        group_size = in_f // n_groups
        T_blocks = T.view(out_f, n_groups, group_size)
        W_dense = (T_blocks * scale.unsqueeze(-1)).view(out_f, in_f).to(dtype)
        if W_dense.shape != W.shape:
            raise ValueError(f"shape mismatch for {weight_key}: "
                             f"unpacked {tuple(W_dense.shape)} vs model {tuple(W.shape)}")
        fp_state[weight_key].copy_(W_dense)
        consumed.update({T_key, s_key})
        bias_key = f"{name}.bias"
        if bias_key in sd:
            fp_state[bias_key].copy_(sd[bias_key].to(dtype))
            consumed.add(bias_key)

    # int8 embed / lm_head -> fp.
    for ki in [k for k in sd if k.endswith(".weight_int8")]:
        base = ki[: -len("_int8")]
        ks = f"{base}_scale"
        if ks not in sd:
            raise KeyError(f"int8 weight {ki} missing scale {ks}")
        W = dequantize_embed_int8(sd[ki], sd[ks], dtype=dtype)
        fp_state[base].copy_(W)
        consumed.update({ki, ks})

    # Everything else (norms, biases not under QLinear, etc.).
    for k, v in sd.items():
        if k in consumed:
            continue
        if k not in fp_state:
            print(f"[warn] checkpoint key not in model, skipping: {k}")
            continue
        fp_state[k].copy_(v.to(dtype) if v.is_floating_point() else v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="Packed safetensors from smollmer-finalize.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory for the unpacked HF checkpoint.")
    ap.add_argument("--model", default=None,
                    help="Base model id; default reads from ckpt metadata.")
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "float32", "bfloat16"])
    args = ap.parse_args()

    dtype = {"float16": torch.float16, "float32": torch.float32,
             "bfloat16": torch.bfloat16}[args.dtype]

    with safe_open(str(args.ckpt), framework="pt") as f:
        meta = f.metadata() or {}
    model_id = args.model or meta.get("model_id")
    if not model_id:
        raise ValueError("--model not given and ckpt metadata has no model_id")
    fmt = meta.get("format", "")
    print(f"[load] base={model_id} fmt={fmt!r} dtype={args.dtype}")

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    sd = load_file(str(args.ckpt))
    materialize(model, sd, fmt=fmt, dtype=dtype)

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.save_pretrained(str(args.out))

    # Sanity: a materialized projection now has up to 3 unique values per
    # (row, group) — i.e. {-s, 0, +s} per group. With multiple groups per
    # row we expect ~3 × n_groups unique values per row.
    for name, mod in model.named_modules():
        if (name + ".T_packed") in sd:
            row = mod.weight.detach()[0].float()
            uniq = torch.unique(row)
            print(f"[sanity] {name}.weight row 0: {len(uniq)} unique values, "
                  f"|max|={row.abs().max().item():.4f}")
            break

    print(f"[done] wrote HF checkpoint to {args.out}")


if __name__ == "__main__":
    main()
