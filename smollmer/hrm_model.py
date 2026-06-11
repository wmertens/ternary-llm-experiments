"""hrm_model — HRM-Text dual-stack recurrent transformer over QLinear.

Two stacks of HF-Llama-style decoder layers (`H_stack` slow / `L_stack` fast),
iterated H_cycles × (L_cycles + 1) times per token with additive state
injection. q/k/v/o/gate/up/down are all `QLinear` (per-(row, column-group)
scales, levels=3 ternary trits in `weight`). Embeddings, RMSNorms, lm_head
(tied), and the learned `z_L_init` stay FP.

The recurrent core uses the HRM-Text 1-step gradient approximation: only the
final L-iter and final H-iter are differentiable; everything else runs
under `torch.no_grad()`. Cuts BPTT memory through the loop to roughly one
stack-pass worth.

See `smollmer/hrm_bop_spec.md` for the design rationale.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qlinear import QLinear


# ---------------------------------------------------------------- config


@dataclass
class HrmBopConfig:
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_kv_heads: int = 16
    intermediate_size: int = 2752
    H_layers: int = 4
    L_layers: int = 4
    H_cycles: int = 2
    L_cycles: int = 3
    vocab_size: int = 49152
    max_position_embeddings: int = 1024
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    scale_group_size: int = 64
    embedding_scale: float = 1.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


# ---------------------------------------------------------------- norm


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(-1, keepdim=True)
        x_n = x_f * torch.rsqrt(var + self.eps)
        return (self.weight * x_n).to(dtype)


# ---------------------------------------------------------------- RoPE


class RotaryEmbedding(nn.Module):
    """Precomputed RoPE cos/sin tables for `max_position_embeddings` slots.

    Per HF Llama: inv_freq[i] = base^(-2i/D) for i in [0, D/2). Each cached
    table is [max_pos, D] with `cos` and `sin` already cat'd along D
    (cos|cos, sin|sin), so apply_rotary_pos_emb is a single elementwise mul.
    """

    def __init__(self, dim: int, max_pos: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_pos, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)              # [max_pos, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)       # [max_pos, D]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device, dtype):
        return (self.cos_cached[:seq_len].to(device=device, dtype=dtype),
                self.sin_cached[:seq_len].to(device=device, dtype=dtype))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # q/k: [B, H, S, D]; cos/sin: [S, D] → broadcast over B, H.
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_e = (q * cos) + (_rotate_half(q) * sin)
    k_e = (k * cos) + (_rotate_half(k) * sin)
    return q_e, k_e


# ---------------------------------------------------------------- attention


class HrmAttention(nn.Module):
    def __init__(self, cfg: HrmBopConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.kv_groups = self.num_heads // self.num_kv
        self.scale = self.head_dim ** -0.5
        gs = cfg.scale_group_size
        self.q_proj = QLinear(cfg.hidden_size, self.num_heads * self.head_dim,
                              bias=False, levels=3, group_size=gs)
        self.k_proj = QLinear(cfg.hidden_size, self.num_kv * self.head_dim,
                              bias=False, levels=3, group_size=gs)
        self.v_proj = QLinear(cfg.hidden_size, self.num_kv * self.head_dim,
                              bias=False, levels=3, group_size=gs)
        self.o_proj = QLinear(self.num_heads * self.head_dim, cfg.hidden_size,
                              bias=False, levels=3, group_size=gs)

    def forward(self, x: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           scale=self.scale)
        y = y.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(y)


# ---------------------------------------------------------------- MLP


class HrmMLP(nn.Module):
    def __init__(self, cfg: HrmBopConfig) -> None:
        super().__init__()
        gs = cfg.scale_group_size
        self.gate_proj = QLinear(cfg.hidden_size, cfg.intermediate_size,
                                 bias=False, levels=3, group_size=gs)
        self.up_proj = QLinear(cfg.hidden_size, cfg.intermediate_size,
                               bias=False, levels=3, group_size=gs)
        self.down_proj = QLinear(cfg.intermediate_size, cfg.hidden_size,
                                 bias=False, levels=3, group_size=gs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------- block


class HrmDecoderLayer(nn.Module):
    """HF-Llama style: pre-norm → attn → residual → pre-norm → mlp → residual."""

    def __init__(self, cfg: HrmBopConfig) -> None:
        super().__init__()
        self.input_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = HrmAttention(cfg)
        self.post_attn_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = HrmMLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_norm(x), cos, sin)
        x = x + self.mlp(self.post_attn_norm(x))
        return x


class HrmStack(nn.Module):
    """A pile of `n` HrmDecoderLayer applied in sequence."""

    def __init__(self, cfg: HrmBopConfig, n: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([HrmDecoderLayer(cfg) for _ in range(n)])

    def forward(self, x: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, cos, sin)
        return x


# ---------------------------------------------------------------- model


class HrmBopModel(nn.Module):
    """Dual-stack recurrent transformer. Embed → recurrent core → final norm
    → lm_head. CE loss computed externally by the trainer.
    """

    def __init__(self, cfg: HrmBopConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        # Learned init state for z_L. [1, 1, H] broadcasts across (B, S).
        self.z_L_init = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        self.H_stack = HrmStack(cfg, cfg.H_layers)
        self.L_stack = HrmStack(cfg, cfg.L_layers)
        self.final_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        if cfg.tie_word_embeddings:
            self.lm_head = None       # use embed_tokens.weight in forward
        else:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.rotary = RotaryEmbedding(cfg.head_dim,
                                      cfg.max_position_embeddings,
                                      cfg.rope_theta)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(0.0, std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(0.0, std)
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)
        # QLinear weight/scales are populated by `init_trits_and_scales`
        # from the trainer; here we leave them at the nn.Linear defaults
        # (random small floats), which get overwritten before the first
        # forward.

    # ---------- forward helpers ----------

    def _project_to_logits(self, h: torch.Tensor) -> torch.Tensor:
        h = self.final_norm(h)
        if self.lm_head is None:
            return F.linear(h, self.embed_tokens.weight)
        return self.lm_head(h)

    def _core(self, z_H: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
              return_per_loop: bool = False,
              grad_mode: str = "one-step") -> torch.Tensor | list[torch.Tensor]:
        """Recurrent core. Three gradient modes:

          - "one-step" (default, HRM-Text training-time): only the FINAL inner
            L iter and FINAL outer H iter are differentiable; all earlier iters
            run under torch.no_grad(). 2 differentiable stack-apps total
            (1 L + 1 H). Cheapest, but the inner L loop's "convergence" never
            gets gradient signal.

          - "last-per-cycle": for each H cycle, the LAST inner L iter and the
            H iter are both differentiable; the first L_c-1 inner L iters run
            under no_grad. Treats each H cycle as a fixed-point computation
            over L iters whose "exit" is the differentiable handoff. For our
            H_c=2 L_c=3 default: 2 L grads + 2 H grads = 4 differentiable
            stack-apps. Middle ground between one-step and full-bptt.

          - "full-bptt": every iter is in the autograd graph. 6 L grads + 2 H
            grads = 8 differentiable stack-apps on the default config. Highest
            memory, biggest gradient signal — confirmed to massively outperform
            one-step on ternary HRM training (Runs 12/13).

        `return_per_loop=True` always uses full BPTT (it returns a list of
        intermediate z_H values for a diagnostic loop and is only invoked under
        torch.no_grad in practice, so the autograd cost is fine).
        """
        z_L = self.z_L_init.to(z_H.dtype).expand_as(z_H)
        H_c, L_c = self.cfg.H_cycles, self.cfg.L_cycles
        per_loop: list[torch.Tensor] = []
        for h in range(H_c):
            for l in range(L_c):
                if return_per_loop or grad_mode == "full-bptt":
                    diff = True
                elif grad_mode == "last-per-cycle":
                    diff = (l == L_c - 1)
                else:  # "one-step"
                    diff = (h == H_c - 1 and l == L_c - 1)
                if diff:
                    z_L = self.L_stack(z_L + z_H, cos, sin)
                else:
                    with torch.no_grad():
                        z_L = self.L_stack(z_L + z_H, cos, sin)
            if return_per_loop or grad_mode in ("full-bptt", "last-per-cycle"):
                h_diff = True
            else:  # "one-step"
                h_diff = (h == H_c - 1)
            if h_diff:
                z_H = self.H_stack(z_H + z_L, cos, sin)
            else:
                with torch.no_grad():
                    z_H = self.H_stack(z_H + z_L, cos, sin)
            if return_per_loop:
                per_loop.append(z_H)
        if return_per_loop:
            return per_loop
        return z_H

    def forward(self, input_ids: torch.Tensor,
                grad_mode: str = "one-step",
                h_cycles: int | None = None) -> torch.Tensor:
        """Returns logits [B, S, V]. CE/labels handled by the trainer.
        `grad_mode` ∈ {"one-step", "last-per-cycle", "full-bptt"}; see `_core`.
        `h_cycles`: if set, overrides cfg.H_cycles for this single forward —
        used by the trainer to sample variable loop counts per step (fixpoint
        regularization). The model's params don't change; only the number of
        recurrent applications does."""
        B, S = input_ids.shape
        x = self.embed_tokens(input_ids) * self.cfg.embedding_scale
        cos, sin = self.rotary(S, x.device, x.dtype)
        if h_cycles is not None:
            orig_H = self.cfg.H_cycles
            self.cfg.H_cycles = int(h_cycles)
            try:
                z_H = self._core(x, cos, sin, grad_mode=grad_mode)
            finally:
                self.cfg.H_cycles = orig_H
        else:
            z_H = self._core(x, cos, sin, grad_mode=grad_mode)
        return self._project_to_logits(z_H)

    @torch.no_grad()
    def per_loop_logits(self, input_ids: torch.Tensor) -> list[torch.Tensor]:
        """Diagnostic: return one [B, S, V] logits tensor per H-cycle so the
        trainer can log per-loop CE. Run under no_grad() externally — but we
        also force return_per_loop which keeps every iter in the graph; this
        is fine because the caller has no_grad enabled."""
        B, S = input_ids.shape
        x = self.embed_tokens(input_ids) * self.cfg.embedding_scale
        cos, sin = self.rotary(S, x.device, x.dtype)
        z_Hs = self._core(x, cos, sin, return_per_loop=True)
        return [self._project_to_logits(z) for z in z_Hs]
