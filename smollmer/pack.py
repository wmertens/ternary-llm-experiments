"""2-bit packing for ternary weights.

Encoding: {-1, 0, +1} -> {0, 1, 2}, four weights per uint8 byte
(little-endian within byte: byte = w0 | (w1<<2) | (w2<<4) | (w3<<6)).
Code 3 is unused. Last byte of a row is zero-padded if in_features % 4 != 0.
"""
from __future__ import annotations

import torch


def pack_ternary(t: torch.Tensor) -> torch.Tensor:
    """Pack a ternary tensor of shape [out, in] (values in {-1,0,+1}) to
    uint8 of shape [out, ceil(in/4)]."""
    if t.dim() != 2:
        raise ValueError(f"expected 2D tensor, got {tuple(t.shape)}")
    out_f, in_f = t.shape
    codes = (t.to(torch.int8) + 1).to(torch.uint8)  # {-1,0,1} -> {0,1,2}
    if (codes > 2).any():
        raise ValueError("input contains values outside {-1, 0, +1}")
    pad = (-in_f) % 4
    if pad:
        codes = torch.nn.functional.pad(codes, (0, pad))
    codes = codes.view(out_f, -1, 4)
    packed = (codes[..., 0]
              | (codes[..., 1] << 2)
              | (codes[..., 2] << 4)
              | (codes[..., 3] << 6))
    return packed.contiguous()


def unpack_ternary(packed: torch.Tensor, in_features: int,
                   dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Inverse of pack_ternary. Returns a [out, in_features] tensor in `dtype`
    with values in {-1.0, 0.0, +1.0}."""
    if packed.dim() != 2:
        raise ValueError(f"expected 2D packed tensor, got {tuple(packed.shape)}")
    out_f, packed_w = packed.shape
    expected = (in_features + 3) // 4
    if packed_w != expected:
        raise ValueError(f"packed width {packed_w} != ceil(in_features/4)={expected}")
    p = packed.to(torch.uint8)
    w0 = p & 0b11
    w1 = (p >> 2) & 0b11
    w2 = (p >> 4) & 0b11
    w3 = (p >> 6) & 0b11
    codes = torch.stack([w0, w1, w2, w3], dim=-1).view(out_f, packed_w * 4)
    codes = codes[:, :in_features]
    return (codes.to(torch.int8) - 1).to(dtype)
