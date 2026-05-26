"""Offline sensitivity calibration for per-layer temperature scaling.

Runs the teacher model on N batches from the cached corpus and estimates a
per-Linear-layer sensitivity score. Layers with high score are sensitive to
weight changes and should anneal more slowly in qat_smooth.py.

Two methods:
  --method grad     (cheap): trace ≈ mean over batches of Σ (∂L/∂w)²
                    This is the Fisher diagonal trace surrogate. Conflates
                    "moving" with "sensitive"; cheapest path (1 backward
                    per batch).
  --method hutchpp  (Hessian): per-layer tr(H) via Hutch++ (Meyer et al.
                    2021). Sketch S → Y = HS → Q from QR(Y) → tr_low =
                    Σ Q·HQ; then Ω query → Ω_⊥ = (I-QQ^T)Ω → tr_res =
                    (1/m) Σ Ω·HΩ_⊥. tr(H) = tr_low + tr_res. Variance
                    reduction by deflating the top-r eigenspace. Slower
                    but unbiased near a stationary point where ‖g‖²
                    collapses.

Output JSON: {module_name: {"trace": float, "z": float, "temp_scale": float}}
  trace      : sensitivity estimate (Fisher trace or Hessian trace)
  z          : z-score of log(trace) across all Linear layers
  temp_scale : exp(beta * z)  — multiply global T by this per layer

Usage:
  python -m smollmer.calibrate \
      --cache-dir smollmer/cache_corpus \
      --out smollmer/calib.json --method hutchpp
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

from .distill import ShardedDataset


def _ce_loss(model: torch.nn.Module, ids: torch.Tensor,
             targets: torch.Tensor) -> torch.Tensor:
    logits = model(ids).logits
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )


def calibrate_grad(model: torch.nn.Module, iter_dl, n_batches: int,
                   device: str) -> dict[str, float]:
    """Fisher diagonal trace ≈ mean ‖∇w L‖²."""
    traces: dict[str, float] = {
        n: 0.0 for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear)
    }
    pbar = tqdm(total=n_batches, desc="grad")
    for _ in range(n_batches):
        batch = next(iter_dl)
        tokens = batch["tokens"].to(device, non_blocking=True).long()
        ids, targets = tokens[:, :-1], tokens[:, 1:]
        model.zero_grad()
        loss = _ce_loss(model, ids, targets)
        loss.backward()
        with torch.no_grad():
            for name, m in model.named_modules():
                if isinstance(m, torch.nn.Linear) and m.weight.grad is not None:
                    traces[name] += float(m.weight.grad.pow(2).sum())
        pbar.update(1)
    pbar.close()
    for k in traces:
        traces[k] /= n_batches
    return traces


def _rademacher(shape, device, dtype=torch.float32) -> torch.Tensor:
    return (torch.randint(0, 2, shape, device=device, dtype=dtype) * 2 - 1)


def _hvp_for_param(grad_flat_with_graph: torch.Tensor,
                   param: torch.Tensor, vec_flat: torch.Tensor,
                   retain: bool) -> torch.Tensor:
    """Single-vector HVP given a pre-computed grad (with create_graph=True).

    grad_flat_with_graph: ∂L/∂param, flattened, autograd graph still alive
    vec_flat: probe vector, same numel as param, on same device
    Returns: H @ vec_flat, flattened.
    """
    gv = (grad_flat_with_graph * vec_flat).sum()
    hvp = torch.autograd.grad(gv, param, retain_graph=retain)[0]
    return hvp.view(-1)


def calibrate_hutchpp(model: torch.nn.Module, iter_dl, n_batches: int,
                      num_sketch: int, num_query: int,
                      device: str) -> dict[str, float]:
    """Per-Linear-layer Hessian trace via Hutch++.

    Per batch we share the forward + first backward across layers and
    issue per-(layer, probe) second-backwards. State lives in CPU until
    needed.
    """
    layer_modules: dict[str, torch.nn.Linear] = {
        n: m for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and m.weight.requires_grad
    }
    names = list(layer_modules.keys())
    params = [layer_modules[n].weight for n in names]
    numels = [p.numel() for p in params]
    print(f"[hutch++] {len(names)} Linear layers, "
          f"r={num_sketch}, m={num_query}, batches={n_batches}")

    # Probe vectors (CPU) per layer
    S = {n: _rademacher((numels[i], num_sketch), "cpu")
         for i, n in enumerate(names)}
    Omega = {n: _rademacher((numels[i], num_query), "cpu")
             for i, n in enumerate(names)}

    # Phase 1 + 2 accumulators: Y = H @ S, G = H @ Q, all CPU
    Y_acc = {n: torch.zeros_like(S[n]) for n in names}

    def _shared_first_backward(batch):
        tokens = batch["tokens"].to(device, non_blocking=True).long()
        ids, targets = tokens[:, :-1], tokens[:, 1:]
        loss = _ce_loss(model, ids, targets)
        grads = torch.autograd.grad(loss, params, create_graph=True,
                                    retain_graph=True)
        return [g.view(-1) for g in grads], loss

    # --- Phase 1: Y = H @ S, averaged over batches
    pbar = tqdm(total=n_batches * len(names), desc="hutch++ p1")
    for _ in range(n_batches):
        batch = next(iter_dl)
        grads_flat, loss = _shared_first_backward(batch)
        for i, name in enumerate(names):
            S_i_dev = S[name].to(device, non_blocking=True)
            param, g_flat = params[i], grads_flat[i]
            out = torch.empty_like(S_i_dev)
            for j in range(num_sketch):
                out[:, j] = _hvp_for_param(g_flat, param, S_i_dev[:, j],
                                           retain=True)
            Y_acc[name] += out.cpu()
            del S_i_dev, out
            pbar.update(1)
        del grads_flat, loss
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    pbar.close()

    # QR per layer (CPU) → Q
    Q = {}
    for name in names:
        Y_avg = Y_acc[name] / n_batches
        q, _ = torch.linalg.qr(Y_avg, mode="reduced")
        Q[name] = q
    del Y_acc, S

    # --- Phase 2: G = H @ Q, averaged. tr_low = Σ Q * G_avg.
    G_acc = {n: torch.zeros_like(Q[n]) for n in names}
    pbar = tqdm(total=n_batches * len(names), desc="hutch++ p2")
    for _ in range(n_batches):
        batch = next(iter_dl)
        grads_flat, loss = _shared_first_backward(batch)
        for i, name in enumerate(names):
            Q_i_dev = Q[name].to(device, non_blocking=True)
            param, g_flat = params[i], grads_flat[i]
            out = torch.empty_like(Q_i_dev)
            for j in range(Q_i_dev.shape[1]):
                out[:, j] = _hvp_for_param(g_flat, param, Q_i_dev[:, j],
                                           retain=True)
            G_acc[name] += out.cpu()
            del Q_i_dev, out
            pbar.update(1)
        del grads_flat, loss
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    pbar.close()

    trace_low = {n: float((Q[n] * (G_acc[n] / n_batches)).sum())
                 for n in names}
    del G_acc

    # --- Phase 3: Ω_⊥ = (I-QQ^T)Ω, Z = H @ Ω_⊥, tr_res = (1/m) Σ Ω · (I-QQ^T) Z
    Omega_perp = {n: Omega[n] - Q[n] @ (Q[n].T @ Omega[n]) for n in names}
    Z_acc = {n: torch.zeros_like(Omega[n]) for n in names}
    pbar = tqdm(total=n_batches * len(names), desc="hutch++ p3")
    for _ in range(n_batches):
        batch = next(iter_dl)
        grads_flat, loss = _shared_first_backward(batch)
        for i, name in enumerate(names):
            Op_dev = Omega_perp[name].to(device, non_blocking=True)
            param, g_flat = params[i], grads_flat[i]
            out = torch.empty_like(Op_dev)
            for j in range(num_query):
                out[:, j] = _hvp_for_param(g_flat, param, Op_dev[:, j],
                                           retain=True)
            Z_acc[name] += out.cpu()
            del Op_dev, out
            pbar.update(1)
        del grads_flat, loss
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    pbar.close()

    trace_res = {}
    for name in names:
        Z_avg = Z_acc[name] / n_batches
        Z_perp = Z_avg - Q[name] @ (Q[name].T @ Z_avg)
        trace_res[name] = float((Omega[name] * Z_perp).sum() / num_query)

    return {n: trace_low[n] + trace_res[n] for n in names}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-layer sensitivity calibration for qat_smooth.")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--method", default="hutchpp",
                    choices=["grad", "hutchpp"],
                    help="grad: Fisher diagonal ‖∇w L‖² (cheap). "
                         "hutchpp: per-layer Hessian trace via Hutch++.")
    ap.add_argument("--n-batches", type=int, default=None,
                    help="Number of batches per phase. Defaults: grad=50, "
                         "hutchpp=3.")
    ap.add_argument("--num-sketch", type=int, default=8,
                    help="Hutch++ sketch rank r (deflated eigenspace). "
                         "Larger → lower variance, more compute.")
    ap.add_argument("--num-query", type=int, default=12,
                    help="Hutch++ residual probe count m. Larger → lower "
                         "variance on the residual trace.")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="Teacher dtype. bfloat16 (default) uses ~2× less VRAM "
                         "than float32; HVPs are valid for trace ranking.")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="Sensitivity scale strength: temp_scale = exp(beta*z). "
                         "beta=0 → uniform temperature. beta=0.5 (default) → "
                         "~2.7× range across layers.")
    ap.add_argument("--scale-group-size", type=int, default=64)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.n_batches is None:
        args.n_batches = 50 if args.method == "grad" else 3

    torch.manual_seed(args.seed)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print(f"[load] {args.model} on {args.device} ({args.dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype).to(args.device)
    # Hutch++ needs double-backward through every linear's weight, so we
    # don't .eval() in a way that disables grad. Eval mode is fine; only
    # dropout/BN are toggled.
    model.eval()

    ds = ShardedDataset(args.cache_dir, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=1,
                    drop_last=True)
    it = iter(dl)

    if args.method == "grad":
        traces = calibrate_grad(model, it, args.n_batches, args.device)
    else:
        traces = calibrate_hutchpp(model, it, args.n_batches,
                                   args.num_sketch, args.num_query,
                                   args.device)

    # Z-score the log |trace| — Hutch++ traces are signed but should be ≥0
    # in expectation; we abs() to stay safe with the log.
    log_traces = {k: math.log(abs(v) + 1e-30) for k, v in traces.items()}
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

    ts_vals = [v["temp_scale"] for v in result.values()]
    print(f"[done] method={args.method}, {len(result)} layers → {args.out}")
    print(f"       temp_scale range: {min(ts_vals):.3f} – {max(ts_vals):.3f}  "
          f"(beta={args.beta})")


if __name__ == "__main__":
    main()
