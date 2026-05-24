"""Offline Fisher-information calibration for per-layer temperature scaling.

Runs the teacher model forward+backward on N batches from the cached corpus,
accumulating squared gradients as a proxy for the per-layer Hessian trace
(Fisher information). Layers with high trace are sensitive to weight changes
and should anneal more slowly in qat_smooth.py.

Output JSON: {module_name: {"trace": float, "z": float, "temp_scale": float}}
  trace      : mean squared-gradient sum over calibration batches
  z          : z-score of log(trace) across all QLinear layers
  temp_scale : exp(beta * z)  — multiply global T by this per layer

Usage:
  python -m smollmer.calibrate \
      --cache-dir smollmer/cache_corpus \
      --out smollmer/calib.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from .build_student import load_student
from .distill import ShardedDataset
from .qlinear import QLinear


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fisher calibration for per-layer temperature scaling.")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-batches", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="Teacher dtype. bfloat16 (default) uses ~2× less VRAM "
                         "than float32; gradients are still valid for trace ranking.")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="Sensitivity scale strength: temp_scale = exp(beta*z). "
                         "beta=0 → uniform temperature. beta=0.5 (default) → "
                         "~2.7× range across layers.")
    ap.add_argument("--scale-group-size", type=int, default=64)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # Load teacher
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print(f"[load] {args.model} on {args.device} ({args.dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype).to(args.device)
    model.eval()

    # Data: stream from cached corpus, use raw tokens for CE loss
    ds = ShardedDataset(args.cache_dir, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=1,
                    drop_last=True)
    it = iter(dl)

    # Accumulate squared gradients per named Linear layer
    traces: dict[str, float] = {}
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            traces[name] = 0.0

    print(f"[calib] {len(traces)} Linear layers, {args.n_batches} batches")
    pbar = tqdm(total=args.n_batches, desc="calib")

    for _ in range(args.n_batches):
        batch = next(it)
        tokens = batch["tokens"].to(args.device, non_blocking=True).long()
        ids = tokens[:, :-1]
        targets = tokens[:, 1:]

        model.zero_grad()
        logits = model(ids).logits          # [B, T-1, V]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
        loss.backward()

        with torch.no_grad():
            for name, m in model.named_modules():
                if isinstance(m, torch.nn.Linear) and m.weight.grad is not None:
                    traces[name] += float(m.weight.grad.pow(2).sum())

        pbar.update(1)
    pbar.close()

    # Average over batches
    for k in traces:
        traces[k] /= args.n_batches

    # Z-score the log traces
    log_traces = {k: math.log(v + 1e-30) for k, v in traces.items()}
    vals = list(log_traces.values())
    mean_log = sum(vals) / len(vals)
    std_log = (sum((v - mean_log) ** 2 for v in vals) / len(vals)) ** 0.5

    result: dict[str, dict] = {}
    for name in traces:
        z = (log_traces[name] - mean_log) / (std_log + 1e-8)
        ts = math.exp(args.beta * z)
        result[name] = {
            "trace": traces[name],
            "z": z,
            "temp_scale": ts,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    # Summary
    ts_vals = [v["temp_scale"] for v in result.values()]
    print(f"[done] {len(result)} layers → {args.out}")
    print(f"       temp_scale range: {min(ts_vals):.3f} – {max(ts_vals):.3f}  "
          f"(beta={args.beta})")


if __name__ == "__main__":
    main()
