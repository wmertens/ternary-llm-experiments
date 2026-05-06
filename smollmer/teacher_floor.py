"""Compute the cross-entropy floor of the cached teacher distribution.

Distillation loss (per kl_with_rest) is the cross-entropy

  H(p_t, p_s) = -Σ_topk p_t log p_s  -  rest_mass * log p_s_rest

with the teacher constant H(p_t) dropped. So when the student matches the
teacher exactly, the reported step loss equals H(p_t). That's the floor we
need to drive the soft-ternary α schedule.

We can't compute H(p_t) exactly from the cache (only top-K + rest_mass
total are stored), but we can bound it tightly: the rest-mass entropy is
maximized when spread uniformly over the V-K remaining vocab tokens, giving

  H_rest_upper = -rest_mass * log(rest_mass / (V - K))

So the returned floor is an UPPER bound on H(p_t). The practical floor —
the converged stage-0 (L≈FP) EMA — is what the schedule actually steers
on; the theoretical estimate here is logged alongside as a sanity check.

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
    """Return a dict with the floor estimate and a few sanity counters.

    Keys:
      `floor`           : mean H(p_t) per position (cross-entropy floor)
      `H_topk_mean`     : mean of the top-K entropy contribution alone
      `H_rest_mean`     : mean of the rest-mass entropy contribution alone
      `total_positions` : total (S,T) positions counted (matches loss denom)
      `valid_positions` : positions with non-zero teacher target
      `vocab_size`      : echoed input
      `top_k`           : K detected from the first shard
    """
    paths = _shard_paths(cache_dir)
    total_H = 0.0
    total_H_topk = 0.0
    total_H_rest = 0.0
    total_pos = 0
    valid_pos = 0
    K_detected: int | None = None
    rest_denom_floor = max(vocab_size - 1, 1)  # safe even if K not yet known
    for p in paths:
        sh = load_file(str(p))
        topk_prob = sh["topk_prob"].float()      # [S, T, K]
        rest_mass = sh["rest_mass"].float()      # [S, T]
        K = topk_prob.shape[-1]
        if K_detected is None:
            K_detected = K
            rest_denom_floor = max(vocab_size - K, 1)
        elif K != K_detected:
            raise ValueError(f"shard {p.name} has K={K}, "
                             f"expected K={K_detected} from earlier shards")
        log_p_topk = (topk_prob + eps).log()
        H_topk = -(topk_prob * log_p_topk).sum(dim=-1)            # [S, T]
        # rest_mass * log(rest_mass / (V-K)); 0 * log(0) defined as 0.
        ratio = rest_mass / rest_denom_floor
        H_rest = -(rest_mass * (ratio + eps).log())               # [S, T]
        H = H_topk + H_rest
        total_H += H.sum().item()
        total_H_topk += H_topk.sum().item()
        total_H_rest += H_rest.sum().item()
        total_pos += H.numel()
        valid_pos += int(((topk_prob.sum(dim=-1) > 0) | (rest_mass > 0)).sum())
    return {
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


def load_or_compute(cache_dir: Path, vocab_size: int,
                    refresh: bool = False) -> dict[str, float | int]:
    """Cached version: read from teacher_floor.json if present and the
    vocab_size matches; otherwise recompute and persist."""
    fp = teacher_floor_path(cache_dir)
    if fp.exists() and not refresh:
        try:
            data = json.loads(fp.read_text())
            if int(data.get("vocab_size", -1)) == int(vocab_size):
                return data
            print(f"[teacher_floor] {fp.name} has vocab_size="
                  f"{data.get('vocab_size')}; recomputing for {vocab_size}")
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
    print(f"[teacher_floor] floor (upper bound on H(p_t)) = {data['floor']:.4f}")
    print(f"[teacher_floor]   H_topk_mean = {data['H_topk_mean']:.4f}")
    print(f"[teacher_floor]   H_rest_mean = {data['H_rest_mean']:.4f}")
    print(f"[teacher_floor]   {data['valid_positions']:,} / "
          f"{data['total_positions']:,} positions have a teacher target")
    print(f"[teacher_floor] saved to {teacher_floor_path(args.cache_dir)}")


if __name__ == "__main__":
    main()
