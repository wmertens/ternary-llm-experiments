"""Tight packing for ternary weights — Bonsai layout.

* `pack_ternary_158` / `unpack_ternary_158`: 5 trits per uint8 byte,
  base-3 encoding (1.6 bpw, vs theoretical log2(3) = 1.585). Use this
  for the trit values; the per-(row, column-group) scales are stored
  as a separate fp16 tensor of shape [out_features, n_groups].

`out_features` is preserved as the leading dim of the packed tensor.

Plus int8 row-quantization helpers for `embed_tokens` / `lm_head` (kept
high-precision since they don't go through QLinear).
"""
from __future__ import annotations

import torch


# ----- 1.6 bpw: base-3, 5 trits per byte ---------------------------------

@torch.no_grad()
def pack_ternary_158(t: torch.Tensor) -> torch.Tensor:
    """Pack ternary [out, in] (values {-1,0,+1}) into uint8 [out, ceil(in/5)].

    byte = c0 + 3*c1 + 9*c2 + 27*c3 + 81*c4, where c_i = w_i + 1 ∈ {0,1,2}.
    Max byte 242 < 256. Last group is zero-padded if in % 5 != 0.
    """
    if t.dim() != 2:
        raise ValueError(f"expected 2D tensor, got {tuple(t.shape)}")
    out_f, in_f = t.shape
    codes = (t.to(torch.int8) + 1).to(torch.uint8)
    if (codes > 2).any():
        raise ValueError("input contains values outside {-1, 0, +1}")
    pad = (-in_f) % 5
    if pad:
        codes = torch.nn.functional.pad(codes, (0, pad), value=1)  # 1 = trit 0
    codes = codes.contiguous().view(out_f, -1, 5).to(torch.int32)
    packed = (codes[..., 0]
              + 3 * codes[..., 1]
              + 9 * codes[..., 2]
              + 27 * codes[..., 3]
              + 81 * codes[..., 4]).to(torch.uint8)
    return packed.contiguous()


@torch.no_grad()
def unpack_ternary_158(packed: torch.Tensor, in_features: int,
                       dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    if packed.dim() != 2:
        raise ValueError(f"expected 2D packed tensor, got {tuple(packed.shape)}")
    out_f, packed_w = packed.shape
    expected = (in_features + 4) // 5
    if packed_w != expected:
        raise ValueError(f"packed width {packed_w} != ceil(in/5)={expected}")
    p = packed.to(torch.int32)
    c0 = p % 3
    c1 = (p // 3) % 3
    c2 = (p // 9) % 3
    c3 = (p // 27) % 3
    c4 = (p // 81) % 3
    codes = torch.stack([c0, c1, c2, c3, c4], dim=-1).view(out_f, packed_w * 5)
    codes = codes[:, :in_features]
    return (codes.to(torch.int8) - 1).to(dtype)


# ----- int8 row-wise (for embed_tokens / lm_head) ------------------------

@torch.no_grad()
def quantize_embed_int8(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric int8 for [V, H] embedding/lm_head weights.

    Returns (int8 [V, H], fp32 per-row scale [V]). Halves storage vs bf16
    with negligible perplexity hit on token embeddings."""
    if W.dim() != 2:
        raise ValueError(f"expected 2D tensor, got {tuple(W.shape)}")
    W_f = W.detach().to(torch.float32)
    scale = W_f.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    q = (W_f / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()


@torch.no_grad()
def dequantize_embed_int8(q: torch.Tensor, scale: torch.Tensor,
                          dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return (q.to(torch.float32) * scale.unsqueeze(1)).to(dtype)
