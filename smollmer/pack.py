"""Tight packings for ternary weights.

Three formats live here:

* `pack_ternary_158` / `unpack_ternary_158`: 5 trits per uint8 byte,
  base-3 encoding (1.6 bpw, vs theoretical log2(3) = 1.585). Use this
  for plain ternary weights with no extra structure.

* `pack_sherry` / `unpack_sherry`: 5 bits per 4-trit block (1.25 bpw).
  Each block must contain exactly one zero and three ±1 values; the
  code is `zero_pos << 3 | sign_bits`. 8 blocks (40 bits) are LE-packed
  into 5 uint8 bytes, so `in_features` must be a multiple of 32.

* `pack_ternary` / `unpack_ternary`: legacy 2-bpw format kept only so
  `chat.py` can still load the first generation of packed checkpoints.

`out_features` is preserved as the leading dim of the packed tensor
in every format.
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


# ----- 1.25 bpw: sherry blocks (1 zero + 3 signs per 4 trits) -----------

@torch.no_grad()
def pack_sherry(t: torch.Tensor, block: int = 4) -> torch.Tensor:
    """Pack a sherry-constrained ternary [out, in] tensor at 1.25 bpw.

    Each 4-trit block must have exactly one zero. Per-block code (5 bits):
        code = zero_pos << 3 | sign_bits
    where sign_bits enumerates the three nonzero positions in original order
    (MSB-first), 0 = -1, 1 = +1. 8 codes pack LE into 5 uint8 bytes (40 bits),
    so in_features must be a multiple of 32. Output shape: [out, in/32 * 5].
    """
    if t.dim() != 2:
        raise ValueError(f"expected 2D tensor, got {tuple(t.shape)}")
    if block != 4:
        raise NotImplementedError(f"block={block} (only 4 supported)")
    out_f, in_f = t.shape
    if in_f % 32 != 0:
        raise ValueError(f"in_features {in_f} not divisible by 32 (block*8)")

    n_blocks_row = in_f // block
    blocks = t.to(torch.int8).contiguous().view(out_f, n_blocks_row, block)
    zero_mask = (blocks == 0)
    z_per_block = zero_mask.sum(dim=-1)
    if not (z_per_block == 1).all():
        bad = (z_per_block != 1).sum().item()
        raise ValueError(f"{bad} blocks violate sherry constraint "
                         f"(need exactly 1 zero per block of {block})")

    zero_pos = zero_mask.to(torch.int64).argmax(dim=-1)            # [out, n_blocks]
    signs = (blocks > 0).to(torch.int64)                            # 0 (neg) | 1 (pos)

    # Non-zero positions in original order: [0,1,2,3] \ {zero_pos}.
    base = torch.arange(3, device=t.device).view(1, 1, 3).expand(out_f, n_blocks_row, 3)
    z = zero_pos.unsqueeze(-1)
    nz_idx = base + (base >= z).to(torch.int64)
    nz_signs = torch.gather(signs, -1, nz_idx)                      # [out, n_blocks, 3]
    sign_bits = (nz_signs[..., 0] << 2) | (nz_signs[..., 1] << 1) | nz_signs[..., 2]
    codes = (zero_pos << 3) | sign_bits                             # [out, n_blocks], 0..31

    n_groups = n_blocks_row // 8
    codes = codes.view(out_f, n_groups, 8)
    packed40 = (codes[..., 0]
                | (codes[..., 1] << 5)
                | (codes[..., 2] << 10)
                | (codes[..., 3] << 15)
                | (codes[..., 4] << 20)
                | (codes[..., 5] << 25)
                | (codes[..., 6] << 30)
                | (codes[..., 7] << 35))                             # [out, n_groups]
    bytes_out = torch.stack(
        [(packed40 >> (8 * k)) & 0xFF for k in range(5)], dim=-1
    ).view(out_f, n_groups * 5).to(torch.uint8)
    return bytes_out.contiguous()


@torch.no_grad()
def unpack_sherry(packed: torch.Tensor, in_features: int, block: int = 4,
                  dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    if packed.dim() != 2:
        raise ValueError(f"expected 2D packed tensor, got {tuple(packed.shape)}")
    if block != 4:
        raise NotImplementedError(f"block={block} (only 4 supported)")
    if in_features % 32 != 0:
        raise ValueError(f"in_features {in_features} not divisible by 32")
    out_f, packed_w = packed.shape
    n_groups = in_features // 32
    if packed_w != n_groups * 5:
        raise ValueError(f"packed width {packed_w} != {n_groups * 5}")

    p = packed.to(torch.int64).view(out_f, n_groups, 5)
    packed40 = (p[..., 0]
                | (p[..., 1] << 8)
                | (p[..., 2] << 16)
                | (p[..., 3] << 24)
                | (p[..., 4] << 32))
    codes = torch.stack(
        [(packed40 >> (5 * k)) & 0x1F for k in range(8)], dim=-1
    ).view(out_f, n_groups * 8)                                      # [out, n_blocks]

    zero_pos = (codes >> 3) & 0x3
    sign_bits = codes & 0x7
    n_blocks_row = codes.shape[1]

    base = torch.arange(3, device=packed.device).view(1, 1, 3).expand(out_f, n_blocks_row, 3)
    z = zero_pos.unsqueeze(-1)
    nz_idx = base + (base >= z).to(torch.int64)

    s0 = ((sign_bits >> 2) & 1).to(torch.int8) * 2 - 1
    s1 = ((sign_bits >> 1) & 1).to(torch.int8) * 2 - 1
    s2 = (sign_bits & 1).to(torch.int8) * 2 - 1
    nz_vals = torch.stack([s0, s1, s2], dim=-1)                       # [out, n_blocks, 3]

    blocks_out = torch.zeros(out_f, n_blocks_row, block,
                             dtype=torch.int8, device=packed.device)
    blocks_out.scatter_(-1, nz_idx, nz_vals)
    return blocks_out.view(out_f, n_blocks_row * block).to(dtype)


# ----- legacy 2 bpw (read-only path for old checkpoints) ----------------

@torch.no_grad()
def pack_ternary(t: torch.Tensor) -> torch.Tensor:
    """Legacy 2-bpw packer (4 trits per byte). Kept for completeness; new
    checkpoints should use `pack_ternary_158` or `pack_sherry`."""
    if t.dim() != 2:
        raise ValueError(f"expected 2D tensor, got {tuple(t.shape)}")
    out_f, in_f = t.shape
    codes = (t.to(torch.int8) + 1).to(torch.uint8)
    if (codes > 2).any():
        raise ValueError("input contains values outside {-1, 0, +1}")
    pad = (-in_f) % 4
    if pad:
        codes = torch.nn.functional.pad(codes, (0, pad))
    codes = codes.contiguous().view(out_f, -1, 4)
    packed = (codes[..., 0]
              | (codes[..., 1] << 2)
              | (codes[..., 2] << 4)
              | (codes[..., 3] << 6))
    return packed.contiguous()


@torch.no_grad()
def unpack_ternary(packed: torch.Tensor, in_features: int,
                   dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    if packed.dim() != 2:
        raise ValueError(f"expected 2D packed tensor, got {tuple(packed.shape)}")
    out_f, packed_w = packed.shape
    expected = (in_features + 3) // 4
    if packed_w != expected:
        raise ValueError(f"packed width {packed_w} != ceil(in/4)={expected}")
    p = packed.to(torch.uint8)
    w0 = p & 0b11
    w1 = (p >> 2) & 0b11
    w2 = (p >> 4) & 0b11
    w3 = (p >> 6) & 0b11
    codes = torch.stack([w0, w1, w2, w3], dim=-1).view(out_f, packed_w * 4)
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
