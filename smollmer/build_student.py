"""Build a quantized student from a HF causal LM by replacing every
attention/MLP projection with a QLinear, initialized so the student's
forward at levels=257 closely matches the teacher.

Init strategy (Bonsai-style per-(row × column-group) scaling):
  G          = group_size                  (e.g. 128)
  s_{r,g}    = max(|W[r, g*G : (g+1)*G]|)  # per-(row, group) max abs
  W'_{r,c}   = W[r, c] / s_{r, c // G}     # latent in [-1, 1] per (row, group)

At L=257 this gives `quantize(W') * S ≈ W` (quantizer error tiny because W'
is in the box). At L=3 only the largest few elements per (row, group)
survive as ±1, others round to 0 — the curriculum walks the student toward
this. Compared to a plain per-row scale, per-(row, group) scaling lets
mid-magnitude weights round to ±1 (62% nonzero density on Bonsai-1.7B vs
~15% with per-row), which is the key fidelity win.

embed_tokens, lm_head, RMSNorms and biases are left untouched.
"""
from __future__ import annotations

from typing import Iterable

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
                      targets: Iterable[str] = PROJ_NAMES,
                      latent_dtype: torch.dtype = torch.float16,
                      group_size: int = 128) -> int:
    """Replace target nn.Linear modules with QLinear, copying weights.

    `latent_dtype` controls the storage dtype of the QLinear latent weight
    (bounded to [-1, 1] per (row, group), so fp16 is a strict win over bf16
    near zero).

    `group_size` controls the per-(row, column-group) scale granularity:
    each group of `group_size` consecutive input columns shares one scale.
    Bonsai uses 128. Must divide every target's in_features.

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
                         bias=child.bias is not None, levels=levels,
                         group_size=group_size)
            ql.to(device=child.weight.device, dtype=child.weight.dtype)
            ql.weight.data = ql.weight.data.to(latent_dtype)

            w = child.weight.detach().to(torch.float32)
            out_f, in_f = w.shape
            n_groups = in_f // group_size
            w_blocks = w.view(out_f, n_groups, group_size)
            scales = w_blocks.abs().amax(dim=-1).clamp_min(1e-8)  # [out, n_groups]
            w_scaled = (w_blocks / scales.unsqueeze(-1)).view(out_f, in_f)

            ql.weight.data.copy_(w_scaled.to(latent_dtype))
            ql.scales.data.copy_(scales.to(torch.float32))
            if child.bias is not None:
                ql.bias.data.copy_(child.bias.detach())

            setattr(parent, child_name, ql)
            replaced += 1
    return replaced


def load_student(model_id: str = "HuggingFaceTB/SmolLM2-135M",
                 dtype: torch.dtype = torch.bfloat16,
                 levels: int = 257,
                 latent_dtype: torch.dtype = torch.float16,
                 group_size: int = 128):
    """Convenience: load a HF causal LM and quantize its projections."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    n = quantize_in_place(model, levels=levels, latent_dtype=latent_dtype,
                          group_size=group_size)
    return model, tok, n
