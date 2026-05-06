"""Math-preserving permutations to align input columns with per-(row, group)
scale boundaries.

Sorting columns by aggregate magnitude within each permutable dim makes each
scale group's per-row max-abs tighter (similar |w| within a group → fewer
columns waste capacity rounding to 0). For Bonsai-style per-(row, group)
quantization this is pure win at init: forward output is byte-identical
before vs after, but the post-quantize_in_place scales fit the new column
ordering and are usable down to many more weights.

Free dimensions (math-preserving):
  - **MLP intermediate** (down_proj cols + up/gate_proj rows): all internal
    to the MLP sublayer.
  - **Per-KV-head head_dim** (v_proj rows + o_proj cols per Q-head in the
    GQA group): values for KV-head h flow through to its Q-heads' attention
    outputs which o_proj projects back; the permutation is consistent if
    v_proj's output rows and o_proj's input columns rotate together.

Constrained dimensions (not free, skipped):
  - Residual-stream channels (`hidden`): touched by every QKV/down/up/gate
    matrix, every RMSNorm, embeddings, and lm_head simultaneously — one
    global permutation can't simultaneously align everyone's groups.
  - Per-head Q/K dim: RoPE rotates pairs `(k, k + head_dim/2)`, so only
    pair-preserving permutations are valid; gain is small relative to
    down_proj/o_proj anyway (Q/K's in_features = hidden, untouched).

Run BEFORE `quantize_in_place` — operates on plain `nn.Linear`. After
training, the saved checkpoint contains the permuted weights, so finalize
and chat can load the state dict directly without re-permuting.
"""
from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def permute_for_scale_groups(model: nn.Module) -> int:
    """Permute every transformer block's free input dims so columns are sorted
    by descending column score (sum |w| across rows) — coherent magnitude
    inside each future scale group. Operates on plain nn.Linear (call before
    `quantize_in_place`). Returns the count of matrices touched.

    Forward output is byte-identical before vs after; only the column
    ordering inside each free dim changes.
    """
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n_q = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_q)
    if n_q % n_kv != 0:
        raise ValueError(f"GQA group size n_q={n_q} not divisible by n_kv={n_kv}")
    q_per_kv = n_q // n_kv

    n_touched = 0
    for layer in model.model.layers:
        # ---- MLP intermediate dim ----
        # Score = sum_r |down_proj.weight[r, j]|. Sort descending so high-
        # magnitude columns cluster in the early groups.
        down = layer.mlp.down_proj
        up = layer.mlp.up_proj
        gate = layer.mlp.gate_proj
        score = down.weight.detach().abs().sum(dim=0)
        perm = torch.argsort(score, descending=True).to(down.weight.device)
        down.weight.data = down.weight.data[:, perm].contiguous()
        up.weight.data = up.weight.data[perm, :].contiguous()
        gate.weight.data = gate.weight.data[perm, :].contiguous()
        if up.bias is not None:
            up.bias.data = up.bias.data[perm].contiguous()
        if gate.bias is not None:
            gate.bias.data = gate.bias.data[perm].contiguous()
        n_touched += 3

        # ---- Attention V/O per KV-head ----
        # Score = sum over Q-heads in the GQA group of o_proj cols' |w|.
        # All Q-heads in a KV group must rotate by the same perm (they share V).
        v_proj = layer.self_attn.v_proj
        o_proj = layer.self_attn.o_proj
        for h in range(n_kv):
            head_score = torch.zeros(head_dim, device=o_proj.weight.device)
            for q in range(q_per_kv):
                q_idx = h * q_per_kv + q
                cols = slice(q_idx * head_dim, (q_idx + 1) * head_dim)
                head_score = head_score + o_proj.weight.detach()[:, cols].abs().sum(dim=0)
            perm_h = torch.argsort(head_score, descending=True).to(o_proj.weight.device)

            v_rows = slice(h * head_dim, (h + 1) * head_dim)
            block = v_proj.weight.data[v_rows]
            v_proj.weight.data[v_rows] = block[perm_h].contiguous()
            if v_proj.bias is not None:
                b = v_proj.bias.data[v_rows]
                v_proj.bias.data[v_rows] = b[perm_h].contiguous()
            for q in range(q_per_kv):
                q_idx = h * q_per_kv + q
                cols = slice(q_idx * head_dim, (q_idx + 1) * head_dim)
                block = o_proj.weight.data[:, cols]
                o_proj.weight.data[:, cols] = block[:, perm_h].contiguous()
            n_touched += 1 + q_per_kv  # v_proj head + o_proj per-Q-head slices

    return n_touched
