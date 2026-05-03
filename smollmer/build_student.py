"""Build a quantized student from a HF causal LM by replacing every
attention/MLP projection with a QLinear, initialized so the student's
forward at levels=257 closely matches the teacher.

Init strategy (Bonsai paper-style, not Bonsai-code-style):
  s_i = ||W_i||_2 / sqrt(in_features)   # per-output-row scale
  W'   = W / s.unsqueeze(1)             # rescaled latent weight
The /sqrt(in) factor keeps the typical |W'| around 1 so the curriculum's
clamp to [-1,1] doesn't immediately wipe the row out at low levels.
embed_tokens, lm_head, RMSNorms and biases are left untouched.
"""
from __future__ import annotations

from typing import Iterable

import math
import torch
from torch import nn

from .qlinear import QLinear

PROJ_NAMES: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def _is_target(name: str, targets: Iterable[str]) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    return leaf in set(targets)


@torch.no_grad()
def quantize_in_place(model: nn.Module, levels: int = 257,
                      targets: Iterable[str] = PROJ_NAMES) -> int:
    """Replace target nn.Linear modules with QLinear, copying weights.

    Returns the number of layers replaced.
    """
    targets = tuple(targets)
    replaced = 0
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            if not _is_target(full, targets):
                continue
            if not isinstance(child, nn.Linear):
                continue
            ql = QLinear(child.in_features, child.out_features,
                         bias=child.bias is not None, levels=levels)
            ql.to(device=child.weight.device, dtype=child.weight.dtype)

            w = child.weight.detach().to(torch.float32)
            row_norm = w.norm(dim=1)
            scale = row_norm / math.sqrt(child.in_features)
            scale = scale.clamp_min(1e-8)
            w_scaled = w / scale.unsqueeze(1)

            ql.weight.data.copy_(w_scaled.to(child.weight.dtype))
            ql.scales.data.copy_(scale.to(torch.float32))
            if child.bias is not None:
                ql.bias.data.copy_(child.bias.detach())

            setattr(parent, child_name, ql)
            replaced += 1
    return replaced


def load_student(model_id: str = "HuggingFaceTB/SmolLM2-135M",
                 dtype: torch.dtype = torch.bfloat16,
                 levels: int = 257):
    """Convenience: load a HF causal LM and quantize its projections."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    n = quantize_in_place(model, levels=levels)
    return model, tok, n
