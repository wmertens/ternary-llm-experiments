"""gpt_model — vanilla decoder-only transformer (GPT-style) using the
ternary QLinear infrastructure from the HRM experiments. No recurrence,
no H/L split, no cycle sweep. Purpose: isolate ternary training
speedups from the recurrence confound.

Reuses HrmDecoderLayer / HrmAttention / HrmMLP / RMSNorm / RotaryEmbedding
from hrm_model so the QLinear init, freezing, optimizer-targeting, and
clamp helpers all work unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hrm_model import (HrmDecoderLayer, RMSNorm, RotaryEmbedding)
from .qlinear import QEmbedding


@dataclass
class GptBopConfig:
    hidden_size: int = 512
    num_attention_heads: int = 8
    num_kv_heads: int = 8
    intermediate_size: int = 1408
    num_layers: int = 6
    vocab_size: int = 49152
    max_position_embeddings: int = 1024
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    scale_group_size: int = 64
    embedding_scale: float = 1.0
    share_kv: bool = False     # Q-K=V (arxiv 2606.04032)
    trit_embeddings: bool = False   # Phase 5a: ternary token embedding
    sandwich_norm: bool = False     # Pre+post RMSNorm around attn/mlp
    # X-former (arxiv 2606.18246v1): per-layer widths along an ⊗ profile,
    # wide residual stream, narrow middle. Each layer reads/writes a slice
    # of the first d_ℓ coords; the rest carry forward. List must be
    # num_layers long. head_dim stays at 64 (cfg.hidden_size//cfg.num_attention_heads)
    # so each d_ℓ must be a multiple of 64 AND of scale_group_size.
    layer_widths: list[int] = field(default_factory=list)
    # SPS — State-Prediction Separation (arxiv 2607.01218). Interleave a
    # learned <predict> token after every input token (T→2T, shared RoPE id).
    # Input-slot K/V are causal+persistent; predict-slot K/V are visible only
    # within `sps_window` and then evicted. Next-token loss is read at the
    # predict slots. Zero extra params; ~2x train FLOPs/activation memory.
    sps: bool = False
    sps_window: int = 64

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def build_sps_mask(T: int, window: int, device=None) -> "torch.Tensor":
    """Boolean attention mask for SPS on the interleaved 2T sequence
    (order x0,p0,x1,p1,...), True = attend. Even indices = input slots,
    odd = predict slots. Rules (arxiv 2607.01218):
      - input query xi (pos 2i): attends all input keys xk, k≤i (causal,
        persistent) + predict keys pk within the window i-w ≤ k < i.
      - predict query pi (pos 2i+1): attends all input keys xk, k≤i
        (it predicts token i+1 from state up to i) + predict keys pk within
        i-w ≤ k ≤ i.
    Returns [2T, 2T] bool. Fed to SDPA as attn_mask (True=keep)."""
    import torch
    idx = torch.arange(2 * T, device=device)
    qi = idx // 2                      # the token-index i of each query slot
    ki = idx // 2                      # token-index of each key slot
    q_is_pred = (idx % 2 == 1)         # query is a <predict> slot
    k_is_pred = (idx % 2 == 1)         # key is a <predict> slot
    Q = qi[:, None]
    K = ki[None, :]
    kp = k_is_pred[None, :]
    qp = q_is_pred[:, None]
    # Input keys (k_is_pred=False): causal & persistent — visible when K<=Q
    # for predict queries, and K<=Q for input queries too (xi sees xk k≤i).
    input_key = (~kp) & (K <= Q)
    # Predict keys: windowed. For input query xi: i-w ≤ k < i  → K<Q and K>Q-w.
    # For predict query pi: i-w ≤ k ≤ i → K<=Q and K>Q-w.
    pred_key_input_q = kp & (~qp) & (K < Q) & (K >= Q - window)
    pred_key_pred_q = kp & qp & (K <= Q) & (K >= Q - window)
    return input_key | pred_key_input_q | pred_key_pred_q


class GptBopModel(nn.Module):
    """Embed → N×HrmDecoderLayer → final_norm → lm_head. CE loss is
    computed externally by the trainer (same convention as HrmBopModel).
    """

    def __init__(self, cfg: GptBopConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.trit_embeddings:
            self.embed_tokens = QEmbedding(
                cfg.vocab_size, cfg.hidden_size,
                group_size=cfg.scale_group_size, levels=3)
        else:
            self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        if cfg.sps:
            assert not cfg.layer_widths, "SPS + X-former not supported yet"
            # Learned <predict> token embedding, interleaved after each input.
            self.predict_embed = nn.Parameter(
                torch.zeros(cfg.hidden_size).normal_(0.0, cfg.initializer_range))
        head_dim = cfg.head_dim
        if cfg.layer_widths:
            widths = list(cfg.layer_widths)
            assert len(widths) == cfg.num_layers, (
                f"layer_widths {widths} must have {cfg.num_layers} entries")
            for d in widths:
                assert d <= cfg.hidden_size and d % head_dim == 0 and (
                    d % cfg.scale_group_size == 0), (
                    f"layer width {d} must be ≤ hidden_size={cfg.hidden_size} "
                    f"and divisible by head_dim={head_dim} and "
                    f"scale_group_size={cfg.scale_group_size}")
        else:
            widths = [cfg.hidden_size] * cfg.num_layers
        self.widths = widths
        layers = []
        for d in widths:
            sub = replace(cfg,
                          hidden_size=d,
                          num_attention_heads=d // head_dim,
                          num_kv_heads=d // head_dim,
                          layer_widths=[])  # no recursion
            layers.append(HrmDecoderLayer(sub))
        self.layers = nn.ModuleList(layers)
        self.final_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        if cfg.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size,
                                     bias=False)
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

    def _lm_head(self, x: torch.Tensor) -> torch.Tensor:
        if self.lm_head is None:
            if hasattr(self.embed_tokens, "quantized_scaled_weight"):
                w = self.embed_tokens.quantized_scaled_weight()
            else:
                w = self.embed_tokens.weight
            return F.linear(x, w)
        return self.lm_head(x)

    def _forward_sps(self, input_ids: torch.Tensor) -> torch.Tensor:
        """SPS (arxiv 2607.01218): interleave a learned <predict> token after
        every input token (T→2T), shared RoPE id per pair, one shared-weight
        stack under the SPS mask; read next-token logits from the predict
        slots so the return is [B, T, vocab] — drop-in for the trainer's loss.
        """
        B, T = input_ids.shape
        H = self.cfg.hidden_size
        tok = self.embed_tokens(input_ids) * self.cfg.embedding_scale  # [B,T,H]
        pred = self.predict_embed.to(tok.dtype).expand(B, T, H)         # [B,T,H]
        # Interleave: x = [x0,p0,x1,p1,...] → [B, 2T, H]
        x = torch.stack((tok, pred), dim=2).reshape(B, 2 * T, H)
        # Doubled RoPE ids: position i shared by (xi, pi).
        cos = self.rotary.cos_cached[:T].repeat_interleave(2, dim=0).to(x.dtype)
        sin = self.rotary.sin_cached[:T].repeat_interleave(2, dim=0).to(x.dtype)
        mask = build_sps_mask(T, self.cfg.sps_window, device=x.device)  # [2T,2T]
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        x = self.final_norm(x)
        x = x[:, 1::2, :]                    # predict slots → [B, T, H]
        return self._lm_head(x)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.cfg.sps:
            return self._forward_sps(input_ids)
        B, S = input_ids.shape
        x = self.embed_tokens(input_ids) * self.cfg.embedding_scale
        cos = self.rotary.cos_cached[:S].to(x.dtype)
        sin = self.rotary.sin_cached[:S].to(x.dtype)
        H = self.cfg.hidden_size
        for layer, d in zip(self.layers, self.widths):
            if d == H:
                x = layer(x, cos, sin)
            else:
                # X-former: feed a [:, :, :d] slice; carry forward [:, :, d:].
                # The layer's internal residual works on the d-wide slice;
                # the wide-residual coords beyond d stay from the previous
                # layer that touched them. RoPE buffers are head_dim-shaped
                # so they don't need slicing.
                head = layer(x[..., :d].contiguous(), cos, sin)
                x = torch.cat([head, x[..., d:]], dim=-1)
        x = self.final_norm(x)
        if self.lm_head is None:
            # Tied. If the embed is QEmbedding the matmul target is the
            # ternary-quantised+scaled table so input and output projections
            # share the same coarse weights. Plain nn.Embedding falls back
            # to the raw FP weight.
            if hasattr(self.embed_tokens, "quantized_scaled_weight"):
                w = self.embed_tokens.quantized_scaled_weight()
            else:
                w = self.embed_tokens.weight
            return F.linear(x, w)
        return self.lm_head(x)
