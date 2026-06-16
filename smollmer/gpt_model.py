"""gpt_model — vanilla decoder-only transformer (GPT-style) using the
ternary QLinear infrastructure from the HRM experiments. No recurrence,
no H/L split, no cycle sweep. Purpose: isolate ternary training
speedups from the recurrence confound.

Reuses HrmDecoderLayer / HrmAttention / HrmMLP / RMSNorm / RotaryEmbedding
from hrm_model so the QLinear init, freezing, optimizer-targeting, and
clamp helpers all work unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hrm_model import (HrmDecoderLayer, RMSNorm, RotaryEmbedding)


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

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


class GptBopModel(nn.Module):
    """Embed → N×HrmDecoderLayer → final_norm → lm_head. CE loss is
    computed externally by the trainer (same convention as HrmBopModel).
    """

    def __init__(self, cfg: GptBopConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [HrmDecoderLayer(cfg) for _ in range(cfg.num_layers)])
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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        x = self.embed_tokens(input_ids) * self.cfg.embedding_scale
        cos = self.rotary.cos_cached[:S].to(x.dtype)
        sin = self.rotary.sin_cached[:S].to(x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.final_norm(x)
        if self.lm_head is None:
            return F.linear(x, self.embed_tokens.weight)
        return self.lm_head(x)
