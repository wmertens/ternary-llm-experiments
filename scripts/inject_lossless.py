"""Inject our trained ternary T + int8 embed straight into a q2 ONNX file.

Two passes that reuse standard / com.microsoft ops only — no custom kernels:

1. **Ternary** — for each `MatMulNBits` node corresponding to one of our
   QLinears, overwrite the `B` and `scales` initializers with values derived
   directly from our packed `T_packed` and per-row `scales`. ORT's symmetric
   bits=2 dequant is `(code - 2) * scale`, so encoding `T+2 ∈ {1,2,3}` gives
   `{-1, 0, +1} * scale` — exactly what was trained. Same op, same shapes,
   same runtime — zero re-quantization error.

2. **Int8 embed** — replace the fp16 `Gather` with int8 weight + per-row fp16
   scale + dequant chain (Gather int8, Gather scale, Cast fp16, Unsqueeze,
   Mul). Saves ~28 MB on a SmolLM2-135M-sized vocab. Pure standard-op surgery.

Output is bit-identical in *structure* to a standard q2 ONNX, so transformers.js
loads it with `dtype: "q2"` unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from safetensors import safe_open
from safetensors.torch import load_file

from smollmer.pack import (unpack_sherry, unpack_ternary,
                            unpack_ternary_158)


def _unpacker_for(fmt: str):
    if fmt == "smollmer-packed-sherry-v1":
        return unpack_sherry
    if fmt == "smollmer-packed-158-v1":
        return unpack_ternary_158
    if fmt in ("smollmer-packed-v1", ""):
        return unpack_ternary
    raise ValueError(f"unknown packed format: {fmt!r}")


def _onnx_node_to_qlinear(node_name: str) -> str | None:
    """Map `/model/layers.0/self_attn/q_proj/MatMul[_Q2]` -> `model.layers.0...q_proj`.

    Returns None for nodes that aren't a per-projection MatMul (e.g. the
    lm_head MatMul, or Shape/Constant nodes).
    """
    parts = node_name.strip("/").split("/")
    if not parts:
        return None
    leaf = parts[-1]
    if not leaf.startswith("MatMul"):
        return None
    # `lm_head` lives outside `model/...`; we choose to leave it as ORT-quantized.
    if "model" not in parts[:-1]:
        return None
    return ".".join(parts[:-1])


def _pack_b(T: torch.Tensor, block_size: int, bits: int) -> np.ndarray:
    """[N, K] {-1,0,+1} int8 -> MatMulNBits B [N, K/block_size, block_size*bits/8] uint8."""
    N, K = T.shape
    if K % block_size != 0:
        raise ValueError(f"K={K} not divisible by block_size={block_size}")
    cpb = 8 // bits  # codes per byte
    blob = block_size * bits // 8
    codes = (T.to(torch.int8) + 2).to(torch.uint8)            # ternary -> {1,2,3}
    codes = codes.view(N, K // block_size, blob, cpb)
    out = torch.zeros(N, K // block_size, blob, dtype=torch.uint8)
    for i in range(cpb):
        out |= codes[..., i] << (i * bits)
    return out.contiguous().numpy()


def _expand_scales(s_per_row: torch.Tensor, n_blocks: int) -> np.ndarray:
    """Replicate per-row scale across blocks. Shape [N*n_blocks] fp32 to
    match what ORT's MatMulNBitsQuantizer emits — mixing fp16 scales with
    fp16 activations trips ORT's T1 type-binding validator."""
    return (s_per_row.unsqueeze(1)
            .expand(-1, n_blocks).reshape(-1)
            .to(torch.float32).contiguous().numpy())


def inject_ternary(model: onnx.ModelProto, sd: dict[str, torch.Tensor],
                   fmt: str) -> int:
    unpack = _unpacker_for(fmt)
    inits = {i.name: i for i in model.graph.initializer}
    n_done = 0
    skipped: list[str] = []
    for node in model.graph.node:
        if node.op_type != "MatMulNBits":
            continue
        ql = _onnx_node_to_qlinear(node.name)
        if not ql or f"{ql}.T_packed" not in sd:
            if node.name.endswith("/lm_head/MatMul_Q2") or "lm_head" in node.name:
                skipped.append(node.name)
            continue
        attrs = {a.name: a for a in node.attribute}
        N = attrs["N"].i
        K = attrs["K"].i
        bs = attrs["block_size"].i
        bits = attrs["bits"].i
        if bits != 2:
            raise ValueError(f"{node.name}: expected bits=2, got {bits}")

        u = unpack
        if u is unpack_sherry and K % 32 != 0:
            u = unpack_ternary_158
        T = u(sd[f"{ql}.T_packed"], in_features=K, dtype=torch.int8)
        scale = sd[f"{ql}.scales"]
        n_blocks = K // bs

        b_name = node.input[1]
        s_name = node.input[2]
        if b_name not in inits or s_name not in inits:
            raise KeyError(f"{node.name}: B/scales initializers not found "
                           f"({b_name!r}, {s_name!r})")

        new_B = _pack_b(T, bs, bits)
        new_scales = _expand_scales(scale, n_blocks)
        inits[b_name].CopyFrom(numpy_helper.from_array(new_B, b_name))
        inits[s_name].CopyFrom(numpy_helper.from_array(new_scales, s_name))
        n_done += 1
    print(f"[ternary] replaced {n_done} MatMulNBits nodes")
    if skipped:
        print(f"[ternary] left ORT-quantized (no ternary in pack): {skipped[:3]} ...")
    return n_done


def inject_int8_embed(model: onnx.ModelProto, sd: dict[str, torch.Tensor],
                      embed_init_name: str = "model.embed_tokens.weight") -> bool:
    """Replace the fp16 embed initializer with int8 + per-row scale + a small
    dequant subgraph that re-emits the original tensor name.

    Both consumers (the embedding `Gather` and the tied-weight `Transpose`
    feeding `lm_head/MatMul`) keep working — they just see the output of the
    new `Mul` instead of an initializer.

        int8_W -> Cast(fp16)     -> W_fp16   [V, H]
        scale  -> Unsqueeze(-1)  -> scale_u  [V, 1]
        W_fp16 * scale_u         -> embed_init_name (was the fp16 init)
    """
    int8_key = f"{embed_init_name}_int8"
    scale_key = f"{embed_init_name}_scale"
    if int8_key not in sd or scale_key not in sd:
        print(f"[embed] skip (no {int8_key} in packed file)")
        return False

    g = model.graph
    inits = {i.name: i for i in g.initializer}
    if embed_init_name not in inits:
        raise KeyError(f"{embed_init_name!r} not in graph initializers")

    int8_init = f"{embed_init_name}__int8"
    scale_init = f"{embed_init_name}__scale"
    axes_init = f"{embed_init_name}__neg1_axes"
    cast_out = f"{embed_init_name}__fp16"
    scale_u_out = f"{embed_init_name}__scale_u"

    # Match the original initializer's dtype (typically fp32 even on a fp16
    # export — optimum keeps the embed table at higher precision so the
    # tied lm_head MatMul stays fp32×fp32). Cast/Mul outputs must agree.
    orig_dtype = inits[embed_init_name].data_type
    if orig_dtype == int(TensorProto.FLOAT):
        torch_scale_dtype = torch.float32
        np_scale_dtype = np.float32
        cast_to = int(TensorProto.FLOAT)
    elif orig_dtype == int(TensorProto.FLOAT16):
        torch_scale_dtype = torch.float16
        np_scale_dtype = np.float16
        cast_to = int(TensorProto.FLOAT16)
    else:
        raise ValueError(f"unsupported embed dtype {orig_dtype}")

    g.initializer.remove(inits[embed_init_name])
    g.initializer.append(numpy_helper.from_array(
        sd[int8_key].cpu().numpy(), int8_init))
    g.initializer.append(numpy_helper.from_array(
        sd[scale_key].to(torch_scale_dtype).cpu().numpy().astype(np_scale_dtype),
        scale_init))
    g.initializer.append(numpy_helper.from_array(
        np.array([-1], dtype=np.int64), axes_init))

    new_nodes = [
        helper.make_node(
            "Cast", inputs=[int8_init], outputs=[cast_out],
            name="/embed/dequant_cast", to=cast_to),
        helper.make_node(
            "Unsqueeze", inputs=[scale_init, axes_init], outputs=[scale_u_out],
            name="/embed/dequant_unsq"),
        helper.make_node(
            "Mul", inputs=[cast_out, scale_u_out], outputs=[embed_init_name],
            name="/embed/dequant_mul"),
    ]
    for i, n in enumerate(new_nodes):
        g.node.insert(i, n)
    saved = sd[int8_key].numel() * 1  # int8 vs fp16 = 1 byte saved per element
    print(f"[embed] {embed_init_name}: fp16 -> int8 + per-row scale "
          f"(saves ~{saved / 1e6:.1f} MB on disk)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
                    help="Existing q2 model.onnx (with external data).")
    ap.add_argument("--packed", type=Path, required=True,
                    help="smollmer packed safetensors.")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output ONNX path. _data file written next to it.")
    ap.add_argument("--no-ternary", action="store_true")
    ap.add_argument("--no-embed", action="store_true")
    args = ap.parse_args()

    with safe_open(str(args.packed), framework="pt") as f:
        meta = f.metadata() or {}
    fmt = meta.get("format", "")
    print(f"[load] packed format={fmt!r}")
    sd = load_file(str(args.packed))
    model = onnx.load(str(args.input), load_external_data=True)

    if not args.no_ternary:
        inject_ternary(model, sd, fmt)
    if not args.no_embed:
        inject_int8_embed(model, sd)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model, str(args.output),
        save_as_external_data=True, all_tensors_to_one_file=True,
        location=args.output.name + "_data", size_threshold=1024,
    )
    print(f"[done] {args.output} (+{args.output.name}_data)")


if __name__ == "__main__":
    main()
