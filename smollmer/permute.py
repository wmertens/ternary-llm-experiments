"""Free-dim permutations that align natural low-magnitude weights with
Sherry block-of-4 boundaries.

The Sherry constraint groups every QLinear's `in_features` into blocks of 4
and forces the smallest-|w| slot in each block to 0. The projection's cost
(how much the model is perturbed by zeroing that slot) is roughly the size
of that smallest weight; we'd rather designate the *globally* smallest
column in each block as the zero, but the permutation must also keep the
forward math identical, which restricts which dimensions we can touch.

What's free vs constrained
--------------------------
- **MLP intermediate dim** (`down_proj` columns ↔ `up_proj`/`gate_proj` rows
  in the same layer): fully internal to the MLP sublayer. Free.
- **Per-KV-head `head_dim`** (`v_proj` rows for KV-head h ↔ `o_proj` columns
  for *every* Q-head in h's GQA group): values for KV-head h flow through
  to its Q-heads' attention outputs, which o_proj then projects back; the
  permutation is consistent if v_proj's output rows and o_proj's input
  columns rotate together. Free per-KV-head.
- Residual-stream channels (`hidden`): touched by every QKV/down/up/gate
  matrix, every RMSNorm, embeddings, and lm_head. One global permutation
  would have to satisfy *all* of those simultaneously, and it can't make
  per-matrix Sherry blocks align — skip.
- Per-head Q/K dim: RoPE rotates pairs `(k, k + head_dim/2)`, so only
  pair-preserving permutations are valid. Skipped here for simplicity;
  the gain on q/k_proj is much smaller than on down_proj/o_proj anyway
  (both Q and K have `in_features = hidden`, which we can't permute).

Math equivalence is at the FP level: forward(model) == forward(permuted_model)
within fp32 rounding when Sherry is off. With Sherry on, the *blocking*
changes (that is the whole point), so quantized outputs differ — by design.
"""
from __future__ import annotations

import torch

from .qlinear import QLinear


@torch.no_grad()
def _build_perm(scores: torch.Tensor, block: int = 4) -> torch.Tensor:
    """Permutation that places the lowest-score n/block elements at block
    position 0 (the "designated zero" slot under Sherry) and distributes
    the rest across positions 1..block-1.

    Greedy heuristic: column j's score = sum of |w| over all rows that
    use column j. Low score => good zero candidate. Pair the lowest n/4
    columns with the top 3n/4 to give every block one weak + three strong.
    """
    n = scores.numel()
    if n % block != 0:
        raise ValueError(f"n={n} not divisible by block={block}")
    n_blocks = n // block
    sorted_idx = torch.argsort(scores)         # ascending
    zeros = sorted_idx[:n_blocks]              # bottom n/block
    fillers = sorted_idx[n_blocks:]            # top (block-1)*n/block
    perm = torch.empty(n, dtype=torch.long, device=scores.device)
    perm[0::block] = zeros
    for k in range(1, block):
        perm[k::block] = fillers[(k - 1) * n_blocks:k * n_blocks]
    return perm


@torch.no_grad()
def _opt_perm_full(opt, param, perm: torch.Tensor, axis: int) -> None:
    """Apply `perm` along `axis` to every tensor in `opt.state[param]` whose
    shape matches `param.data` (Lion: exp_avg; AdamW: exp_avg + exp_avg_sq)."""
    if opt is None:
        return
    state = opt.state.get(param, None)
    if not state:
        return
    for k, v in list(state.items()):
        if isinstance(v, torch.Tensor) and v.shape == param.data.shape:
            state[k] = (v[perm] if axis == 0 else v[:, perm]).contiguous()


@torch.no_grad()
def _opt_perm_slice(opt, param, slc: slice, perm: torch.Tensor, axis: int) -> None:
    if opt is None:
        return
    state = opt.state.get(param, None)
    if not state:
        return
    for _, v in list(state.items()):
        if isinstance(v, torch.Tensor) and v.shape == param.data.shape:
            if axis == 0:
                v[slc] = v[slc][perm].contiguous()
            else:
                v[:, slc] = v[:, slc][:, perm].contiguous()


@torch.no_grad()
def _permute_rows(layer: QLinear, perm: torch.Tensor, opt=None) -> None:
    layer.weight.data = layer.weight.data[perm, :].contiguous()
    layer.scales.data = layer.scales.data[perm].contiguous()
    if layer.bias is not None:
        layer.bias.data = layer.bias.data[perm].contiguous()
    _opt_perm_full(opt, layer.weight, perm, axis=0)
    _opt_perm_full(opt, layer.scales, perm, axis=0)
    if layer.bias is not None:
        _opt_perm_full(opt, layer.bias, perm, axis=0)


@torch.no_grad()
def _permute_cols(layer: QLinear, perm: torch.Tensor, opt=None) -> None:
    layer.weight.data = layer.weight.data[:, perm].contiguous()
    _opt_perm_full(opt, layer.weight, perm, axis=1)


@torch.no_grad()
def _permute_rows_slice(layer: QLinear, row_slice: slice, perm: torch.Tensor,
                        opt=None) -> None:
    """Permute rows of `layer` in [row_slice] by `perm` (perm indexes the
    slice locally). Used for v_proj where we touch only one KV-head's rows."""
    block = layer.weight.data[row_slice]
    layer.weight.data[row_slice] = block[perm].contiguous()
    s = layer.scales.data[row_slice]
    layer.scales.data[row_slice] = s[perm].contiguous()
    if layer.bias is not None:
        b = layer.bias.data[row_slice]
        layer.bias.data[row_slice] = b[perm].contiguous()
    _opt_perm_slice(opt, layer.weight, row_slice, perm, axis=0)
    _opt_perm_slice(opt, layer.scales, row_slice, perm, axis=0)
    if layer.bias is not None:
        _opt_perm_slice(opt, layer.bias, row_slice, perm, axis=0)


@torch.no_grad()
def _permute_cols_slice(layer: QLinear, col_slice: slice, perm: torch.Tensor,
                        opt=None) -> None:
    """Permute columns of `layer` in [col_slice] by `perm` (perm indexes the
    slice locally). Used for o_proj where we touch one Q-head's columns."""
    block = layer.weight.data[:, col_slice]
    layer.weight.data[:, col_slice] = block[:, perm].contiguous()
    _opt_perm_slice(opt, layer.weight, col_slice, perm, axis=1)


@torch.no_grad()
def permute_for_sherry(model: torch.nn.Module, block: int = 4,
                       optimizer=None) -> int:
    """Apply free-dim permutations in place. Returns count of matrices touched.

    Pass `optimizer` to also permute its per-parameter state (Lion's exp_avg,
    AdamW's exp_avg / exp_avg_sq). Required when re-permuting *during* a run
    (e.g. between curriculum stages); the opt state was accumulated under the
    old layout and would otherwise update weights through stale momentum
    indices. Safe to omit at first init (state is empty before the first step).
    """
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n_q = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_q)
    if n_q % n_kv != 0:
        raise ValueError(f"GQA group size n_q={n_q} not divisible by n_kv={n_kv}")
    q_per_kv = n_q // n_kv

    if head_dim % block != 0:
        raise ValueError(f"head_dim={head_dim} not divisible by block={block}; "
                         "per-head permutation would cross sherry block boundaries")

    n_touched = 0
    for layer in model.model.layers:
        # ---- MLP intermediate dim ----
        # Score = sum_r |down_proj.weight[r, j]|. The cost of designating
        # column j as down_proj's per-block zero is dominated by this sum.
        down: QLinear = layer.mlp.down_proj
        up: QLinear = layer.mlp.up_proj
        gate: QLinear = layer.mlp.gate_proj
        score = down.weight.detach().abs().sum(dim=0)
        perm = _build_perm(score, block=block).to(down.weight.device)
        _permute_cols(down, perm, opt=optimizer)
        _permute_rows(up, perm, opt=optimizer)
        _permute_rows(gate, perm, opt=optimizer)
        n_touched += 3

        # ---- Attention V/O per KV-head ----
        v_proj: QLinear = layer.self_attn.v_proj
        o_proj: QLinear = layer.self_attn.o_proj
        for h in range(n_kv):
            # All Q-heads in this KV group see the same (permuted) V values,
            # so their o_proj column slices must rotate by the same perm.
            # Score combines |o_proj cols across all q in group(h)|.
            head_score = torch.zeros(head_dim, device=o_proj.weight.device)
            for q in range(q_per_kv):
                q_idx = h * q_per_kv + q
                cols = slice(q_idx * head_dim, (q_idx + 1) * head_dim)
                head_score = head_score + o_proj.weight.detach()[:, cols].abs().sum(dim=0)
            perm_h = _build_perm(head_score, block=block).to(o_proj.weight.device)

            v_rows = slice(h * head_dim, (h + 1) * head_dim)
            _permute_rows_slice(v_proj, v_rows, perm_h, opt=optimizer)
            for q in range(q_per_kv):
                q_idx = h * q_per_kv + q
                cols = slice(q_idx * head_dim, (q_idx + 1) * head_dim)
                _permute_cols_slice(o_proj, cols, perm_h, opt=optimizer)
            n_touched += 1 + q_per_kv  # v_proj head + o_proj per-Q-head slices
    return n_touched
