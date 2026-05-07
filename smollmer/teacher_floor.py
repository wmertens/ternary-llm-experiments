"""Compute the cross-entropy floor of the cached teacher distribution.

The distillation loss `kl_with_rest` is computed against a LUMPED K+1
distribution (top-K bins + 1 "rest" bin); it has no per-token resolution
beyond top-K. When the student matches the teacher's lumped distribution
exactly, the cross-entropy floors at:

  H_lumped = -Σ_topk p_topk · log(p_topk)  -  rest_mass · log(rest_mass)

This is what we compute here and what `--soft-floor theoretical` should
hand to the soft-stage diagnostics — anything else over-states the floor
because the loss never tries to predict each rest-token individually.

(We previously approximated H_rest as the max-entropy uniform spread over
V-K vocab tokens, giving a strict upper bound on H(p_t) the FULL teacher
entropy. That floor was correct for H(p_t) but WRONG for `kl_with_rest`,
which lumps. The two differ by `rest_mass · log(V-K)` per position — for
SmolLM2-135M that's ~0.4 nats, exactly the gap users observed between the
estimate and the actual achievable loss.)

The denominator matches the training loss: sum over ALL positions
(including the seed_len-1 prefix and final position that have zero target
mass), divided by total positions. Those positions contribute 0 to both
the loss and the floor, so the averages stay comparable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file


def _shard_paths(cache_dir: Path) -> list[Path]:
    paths = sorted(Path(cache_dir).glob("shard_*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"no shard_*.safetensors under {cache_dir}")
    return paths


def compute_teacher_floor(cache_dir: Path, vocab_size: int,
                          eps: float = 1e-12) -> dict[str, float | int]:
    """Return a dict with the lumped cross-entropy floor and sanity counters.

    `floor = mean( -Σ_topk p_topk·log(p_topk) - rest_mass·log(rest_mass) )`
    matches what the training loss `kl_with_rest` floors at when the
    student exactly reproduces the teacher's lumped K+1 distribution.

    Keys:
      `floor`           : mean lumped cross-entropy floor per position
      `H_topk_mean`     : mean of the top-K contribution alone
      `H_rest_mean`     : mean of the rest-mass-self-entropy contribution
      `total_positions` : total (S,T) positions counted (matches loss denom)
      `valid_positions` : positions with non-zero teacher target
      `vocab_size`      : echoed input (kept for cache-validation only)
      `top_k`           : K detected from the first shard
    """
    paths = _shard_paths(cache_dir)
    total_H = 0.0
    total_H_topk = 0.0
    total_H_rest = 0.0
    total_pos = 0
    valid_pos = 0
    K_detected: int | None = None
    for p in paths:
        sh = load_file(str(p))
        topk_prob = sh["topk_prob"].float()      # [S, T, K]
        rest_mass = sh["rest_mass"].float()      # [S, T]
        K = topk_prob.shape[-1]
        if K_detected is None:
            K_detected = K
        elif K != K_detected:
            raise ValueError(f"shard {p.name} has K={K}, "
                             f"expected K={K_detected} from earlier shards")
        log_p_topk = (topk_prob + eps).log()
        H_topk = -(topk_prob * log_p_topk).sum(dim=-1)            # [S, T]
        # Lumped: rest is one bin with mass rest_mass; H = -m·log(m).
        # 0·log(0+eps) ≈ 0 numerically (eps prevents -inf, the 0 zeros it).
        H_rest = -(rest_mass * (rest_mass + eps).log())           # [S, T]
        H = H_topk + H_rest
        total_H += H.sum().item()
        total_H_topk += H_topk.sum().item()
        total_H_rest += H_rest.sum().item()
        total_pos += H.numel()
        valid_pos += int(((topk_prob.sum(dim=-1) > 0) | (rest_mass > 0)).sum())
    return {
        "schema_version": 2,  # 1 = old uniform-rest upper bound; 2 = lumped
        "floor": total_H / max(1, total_pos),
        "H_topk_mean": total_H_topk / max(1, total_pos),
        "H_rest_mean": total_H_rest / max(1, total_pos),
        "total_positions": total_pos,
        "valid_positions": valid_pos,
        "vocab_size": int(vocab_size),
        "top_k": int(K_detected or 0),
    }


def teacher_floor_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "teacher_floor.json"


CURRENT_SCHEMA = 2


def load_or_compute(cache_dir: Path, vocab_size: int,
                    refresh: bool = False) -> dict[str, float | int]:
    """Cached version: read from teacher_floor.json if present and both
    the schema_version and vocab_size match; otherwise recompute."""
    fp = teacher_floor_path(cache_dir)
    if fp.exists() and not refresh:
        try:
            data = json.loads(fp.read_text())
            sv = int(data.get("schema_version", 1))
            vs = int(data.get("vocab_size", -1))
            if sv == CURRENT_SCHEMA and vs == int(vocab_size):
                return data
            print(f"[teacher_floor] {fp.name} schema_version={sv} "
                  f"vocab_size={vs}; recomputing for schema "
                  f"{CURRENT_SCHEMA}, vocab {vocab_size} (the old schema "
                  f"used a max-entropy uniform-rest upper bound that "
                  f"over-stated the floor by ≈ rest_mass·log(V-K))")
        except Exception as e:
            print(f"[teacher_floor] failed to read {fp}: {e}; recomputing")
    data = compute_teacher_floor(cache_dir, vocab_size)
    fp.write_text(json.dumps(data, indent=2))
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--vocab-size", type=int, default=None,
                    help="If unset, loaded from --model's tokenizer.")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M",
                    help="Used only when --vocab-size is unset.")
    ap.add_argument("--refresh", action="store_true",
                    help="Recompute even if teacher_floor.json exists.")
    args = ap.parse_args()
    if args.vocab_size is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        args.vocab_size = len(tok)
        print(f"[teacher_floor] vocab_size={args.vocab_size} from {args.model}")
    data = load_or_compute(args.cache_dir, args.vocab_size, refresh=args.refresh)
    print(f"[teacher_floor] floor (lumped cross-entropy) = {data['floor']:.4f}")
    print(f"[teacher_floor]   H_topk_mean = {data['H_topk_mean']:.4f}")
    print(f"[teacher_floor]   H_rest_mean = {data['H_rest_mean']:.4f}")
    print(f"[teacher_floor]   {data['valid_positions']:,} / "
          f"{data['total_positions']:,} positions have a teacher target")
    print(f"[teacher_floor] saved to {teacher_floor_path(args.cache_dir)}")


if __name__ == "__main__":
    main()
