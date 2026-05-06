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
def _build_perm_top1(scores: torch.Tensor, block: int = 4) -> torch.Tensor:
    """Permutation for top1 encoding: each block has the i-th strongest
    column at slot 0 (the designated nonzero "anchor"), and the (block-1)
    weakest available columns in slots 1..block-1.

    Pairing: block 0 gets the *globally strongest* column at slot 0
    paired with the (block-1) *globally weakest* columns. Block N-1 gets
    the weakest column from the top quartile (anchor) paired with the
    strongest of the bottom 3 quartiles. Maximum within-block contrast
    in early blocks, gracefully degrading.

    Why this pairing: the strongest top1 column drives the matmul output
    for its block; pairing it with the columns most likely to genuinely
    quantize to 0 (weakest |w|) means the block's "rest is zero"
    constraint costs essentially nothing. Argmax(|w|) per block reliably
    pins to slot 0 — Sherry-style oscillation among slots is structurally
    impossible.
    """
    n = scores.numel()
    if n % block != 0:
        raise ValueError(f"n={n} not divisible by block={block}")
    if block < 2:
        raise ValueError(f"block must be >= 2 to have anchor + zeros; got {block}")
    n_blocks = n // block
    sorted_asc = torch.argsort(scores)                  # ascending |w|
    anchor_pool = sorted_asc[-n_blocks:].flip(0)        # top: STRONGEST first
    zero_pool = sorted_asc[:-n_blocks]                  # bottom (block-1)*n_blocks: weakest first
    if zero_pool.numel() != (block - 1) * n_blocks:
        raise ValueError("internal pool sizing mismatch")
    perm = torch.empty(n, dtype=torch.long, device=scores.device)
    perm[0::block] = anchor_pool                        # slot 0: designated nonzero anchor
    z_per_block = zero_pool.reshape(n_blocks, block - 1)
    for k in range(1, block):
        perm[k::block] = z_per_block[:, k - 1]
    return perm


@torch.no_grad()
def _build_perm(scores: torch.Tensor, block: int = 4) -> torch.Tensor:
    """Permutation that gives every block of `block` columns a guaranteed
    strong|1|, a guaranteed strong-0, and the rest from the middle 50%,
    paired so that block 0 has maximum within-block contrast.

    Three categories after sorting columns by `scores` (= column |w| sum):

      * strong-0   — bottom n/block columns (smallest |w|): natural zeros.
      * strong-|1| — top    n/block columns (largest  |w|): natural ±1s.
      * weak       — middle (block-2)/block: ambiguous.

    Per-block composition (block=4 case):

      slot 0: strong-0[i]                    (smallest |w| in block)
      slot 1: weak.strongest_remaining[2i]   (filler)
      slot 2: weak.strongest_remaining[2i+1] (filler)
      slot 3: strong-|1|.strongest[i]        (largest  |w| in block)

    Pairing: block 0 gets the *globally* smallest column in slot 0 paired
    with the *globally* largest in slot 3. Block N-1 gets the largest of
    the bottom quartile paired with the smallest of the top quartile.
    Maximum within-block contrast in early blocks; weaker but still
    valid contrast in later ones.

    Why this matters: Sherry's argmin (rule 1) always picks the bottom-
    quartile slot as the designated zero, by construction (it has the
    smallest |w| in its block). No oscillation between "which slot is
    the zero," so no flip-rate cliff late in training.
    """
    n = scores.numel()
    if n % block != 0:
        raise ValueError(f"n={n} not divisible by block={block}")
    if block < 3:
        raise ValueError(f"block must be >= 3 to have strong-0, weak, strong-1; got {block}")
    n_blocks = n // block
    sorted_asc = torch.argsort(scores)           # ascending |w|
    z_pool = sorted_asc[:n_blocks]               # bottom: smallest first
    t_pool = sorted_asc[-n_blocks:].flip(0)      # top: STRONGEST first
    m_pool = sorted_asc[n_blocks:-n_blocks].flip(0)  # middle: STRONGEST first
    # Sanity: m_pool must split evenly into (block-2) per block.
    if m_pool.numel() != (block - 2) * n_blocks:
        raise ValueError("internal pool sizing mismatch")
    perm = torch.empty(n, dtype=torch.long, device=scores.device)
    perm[0::block] = z_pool                      # slot 0: designated zero
    perm[block - 1::block] = t_pool              # last slot: designated one
    # Middle slots: block i takes the next (block-2) strongest from m_pool.
    # Reshape m_pool to [n_blocks, block-2] so row i is block i's middle slots.
    m_per_block = m_pool.reshape(n_blocks, block - 2)
    for k in range(1, block - 1):
        perm[k::block] = m_per_block[:, k - 1]
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
def permutation_staleness(model: torch.nn.Module, block: int = 4,
                          mode: str = "sherry") -> float:
    """Fraction of blocks whose argmin/argmax of the perm-score is not in
    slot 0 — i.e. blocks where the permutation's invariant has been broken
    by training drift.

    `mode='sherry'`: slot 0 should have the *smallest* column score
    (designated zero). Staleness counts blocks where some other slot now
    has a smaller score.

    `mode='top1'`: slot 0 should have the *largest* column score
    (designated anchor). Staleness counts blocks where some other slot
    now has a larger score.

    What `permute_for_sherry` actually pins is the *column score* order
    (computed per-matrix as the score `_build_perm` saw at permute time):

      * down_proj:  score[c] = sum_r |down.weight[r, c]|     (per column).
      * o_proj:     per-KV-head, score[c] = sum over Q-heads in the GQA
                    group of |o.weight[:, q*head_dim + c]|.sum(0)
                    (matches the score `permute_for_sherry` computes).

    Per-row staleness (whether each individual row's min lies in slot 0)
    isn't the right metric: Sherry's argmin already runs per-row each step,
    and per-row minimums vary across rows naturally. The score-aligned
    staleness above is the signal that the *structural* alignment has
    drifted.

    Only counts down_proj and o_proj — the matrices whose in_features dim
    is actually aligned to Sherry block boundaries by permute_for_sherry.
    Other QLinears have in_features = hidden, which we don't permute.

    0 = perfectly aligned (just permuted), 1 = fully stale.
    """
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n_q = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_q)
    q_per_kv = n_q // n_kv

    if mode not in ("sherry", "top1"):
        raise ValueError(f"mode must be 'sherry' or 'top1', got {mode!r}")
    n_total = 0
    n_stale = 0
    for layer in model.model.layers:
        # ---- down_proj: column sum across rows ----
        down: QLinear = layer.mlp.down_proj
        if down.in_features % block == 0:
            col_score = down.weight.detach().abs().sum(dim=0)
            blk = col_score.reshape(-1, block)
            slot0_idx = blk.argmin(dim=-1) if mode == "sherry" else blk.argmax(dim=-1)
            n_total += slot0_idx.numel()
            n_stale += int((slot0_idx != 0).sum())

        # ---- o_proj: per-KV-head, sum of |w| across all Q-heads in group ----
        o_proj: QLinear = layer.self_attn.o_proj
        if head_dim % block == 0:
            for h in range(n_kv):
                head_score = torch.zeros(head_dim, device=o_proj.weight.device)
                for q in range(q_per_kv):
                    q_idx = h * q_per_kv + q
                    cols = slice(q_idx * head_dim, (q_idx + 1) * head_dim)
                    head_score = head_score + o_proj.weight.detach()[:, cols].abs().sum(dim=0)
                blk = head_score.reshape(-1, block)
                slot0_idx = blk.argmin(dim=-1) if mode == "sherry" else blk.argmax(dim=-1)
                n_total += slot0_idx.numel()
                n_stale += int((slot0_idx != 0).sum())
    if n_total == 0:
        return 0.0
    return n_stale / n_total


@torch.no_grad()
def _permute_with(model: torch.nn.Module, build_perm,
                  block: int = 4, optimizer=None) -> int:
    """Apply free-dim permutations in place using the given perm builder.
    Mode-agnostic over the choice of `build_perm` (Sherry's `_build_perm`
    or top1's `_build_perm_top1`). Returns count of matrices touched.

    Pass `optimizer` to also permute its per-parameter state (Lion's exp_avg,
    AdamW's exp_avg / exp_avg_sq). Required when re-permuting *during* a run;
    the opt state was accumulated under the old layout and would otherwise
    update weights through stale momentum indices. Safe to omit at first
    init (state is empty before the first step).
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
                         "per-head permutation would cross block boundaries")

    n_touched = 0
    for layer in model.model.layers:
        # ---- MLP intermediate dim ----
        # Score = sum_r |down_proj.weight[r, j]|. The cost of designating
        # column j as down_proj's per-block zero (Sherry) or anchor (top1)
        # is dominated by this sum.
        down: QLinear = layer.mlp.down_proj
        up: QLinear = layer.mlp.up_proj
        gate: QLinear = layer.mlp.gate_proj
        score = down.weight.detach().abs().sum(dim=0)
        perm = build_perm(score, block=block).to(down.weight.device)
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
            perm_h = build_perm(head_score, block=block).to(o_proj.weight.device)

            v_rows = slice(h * head_dim, (h + 1) * head_dim)
            _permute_rows_slice(v_proj, v_rows, perm_h, opt=optimizer)
            for q in range(q_per_kv):
                q_idx = h * q_per_kv + q
                cols = slice(q_idx * head_dim, (q_idx + 1) * head_dim)
                _permute_cols_slice(o_proj, cols, perm_h, opt=optimizer)
            n_touched += 1 + q_per_kv  # v_proj head + o_proj per-Q-head slices
    return n_touched


@torch.no_grad()
def permute_for_sherry(model: torch.nn.Module, block: int = 4,
                       optimizer=None) -> int:
    """Permute for Sherry: place lowest-score columns at slot 0 (designated
    zero), distribute the rest such that block 0 has max contrast. See
    `_build_perm` for the exact strategy."""
    return _permute_with(model, _build_perm, block=block, optimizer=optimizer)


@torch.no_grad()
def permute_for_top1(model: torch.nn.Module, block: int = 4,
                     optimizer=None) -> int:
    """Permute for top1: place highest-score columns at slot 0 (designated
    anchor), fill slots 1..block-1 with the (block-1) weakest columns per
    block. Block 0 has globally strongest paired with globally weakest;
    contrast tapers gracefully toward block N-1."""
    return _permute_with(model, _build_perm_top1, block=block, optimizer=optimizer)
