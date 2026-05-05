"""Stage-2 finalize: freeze ternary T, train per-row scales + RMSNorm weights.

After training, packs the ternary weights to 2 bits and writes a deployment
checkpoint suitable for `smollmer-chat`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from .build_student import load_student
from .distill import ShardedDataset, kl_with_rest, lr_at
from .pack import pack_sherry, pack_ternary_158, quantize_embed_int8

EMBED_INT8_KEYS: tuple[str, ...] = ("model.embed_tokens.weight", "lm_head.weight")
from .qlinear import (QLinear, clamp_qlinear_weights, quantize_levels,
                      quantize_sherry, set_levels, set_sherry)


def freeze_for_finalize(model: torch.nn.Module, freeze_embed: bool,
                        freeze_lm_head: bool) -> tuple[int, int]:
    """Freeze QLinear latent weights; keep scales + norms + biases trainable.

    Returns (#trainable_params, #frozen_params).
    """
    for name, p in model.named_parameters():
        leaf = name.rsplit(".", 1)[-1]
        if leaf == "weight":
            owner = model
            for part in name.split(".")[:-1]:
                owner = getattr(owner, part)
            if isinstance(owner, QLinear):
                p.requires_grad_(False)
                continue
            if freeze_embed and "embed_tokens" in name:
                p.requires_grad_(False)
                continue
            if freeze_lm_head and "lm_head" in name:
                p.requires_grad_(False)
                continue
        p.requires_grad_(True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return n_train, n_frozen


@torch.no_grad()
def to_packed_state_dict(model: torch.nn.Module,
                         dtype: torch.dtype = torch.bfloat16,
                         quant_embed: bool = True) -> dict[str, torch.Tensor]:
    """Build a state_dict with QLinear weights replaced by tightly-packed
    ternary tensors. Other params are cast to `dtype` for storage.

    Per-module `sherry` flag picks both the projection and the packer:
      sherry=True  -> quantize_sherry + pack_sherry (1.25 bpw)
      sherry=False -> quantize_levels + pack_ternary_158 (1.6 bpw)

    Sherry packing requires `in_features % 32 == 0`; we fall back to the
    1.6 bpw packer for any oddly-shaped sherry layer rather than failing."""
    qlinear_skip: set[str] = set()
    out: dict[str, torch.Tensor] = {}
    for name, m in model.named_modules():
        if not isinstance(m, QLinear):
            continue
        qlinear_skip.update({f"{name}.weight", f"{name}.scales"})
        if m.bias is not None:
            qlinear_skip.add(f"{name}.bias")
        quant = quantize_sherry if m.sherry else quantize_levels
        T = quant(m.weight.detach(), 3).to(torch.int8).cpu()
        if m.sherry and m.in_features % 32 == 0:
            out[f"{name}.T_packed"] = pack_sherry(T)
        else:
            out[f"{name}.T_packed"] = pack_ternary_158(T)
        out[f"{name}.scales"] = m.scales.detach().cpu().to(torch.float32).contiguous()
        if m.bias is not None:
            out[f"{name}.bias"] = m.bias.detach().cpu().to(dtype).contiguous()
    for k, v in model.state_dict().items():
        if k in qlinear_skip:
            continue
        v = v.detach().cpu()
        if v.is_floating_point():
            v = v.to(dtype)
        out[k] = v.contiguous()
    if quant_embed:
        for k in EMBED_INT8_KEYS:
            if k in out:
                q, s = quantize_embed_int8(out[k])
                del out[k]
                out[f"{k}_int8"] = q
                out[f"{k}_scale"] = s
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--resume", type=Path, required=True,
                    help="Final L=3 stage checkpoint from distill.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Directory to write `final_packed.safetensors`.")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--autocast-dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "none"])
    ap.add_argument("--store-dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--freeze-embed", action="store_true")
    ap.add_argument("--freeze-lm-head", action="store_true")
    ap.add_argument("--no-quant-embed", action="store_true",
                    help="Skip int8 quant of embed_tokens / lm_head; store at --store-dtype.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"[build] loading {args.model}")
    # See distill.py for the latent_dtype rationale; same fp16/fp32 fallback
    # rule applies (autocast handles the cast in-kernel).
    latent_dtype = torch.float32 if args.autocast_dtype == "none" else torch.float16
    model, _tok, n_replaced = load_student(args.model, dtype=torch.float32, levels=3,
                                           latent_dtype=latent_dtype)
    print(f"[build] {n_replaced} QLinear modules (latent dtype: {latent_dtype})")
    model = model.to(args.device)

    with safe_open(str(args.resume), framework="pt") as f:
        meta = f.metadata() or {}
    sd = load_file(str(args.resume))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[resume] {args.resume.name} (meta={meta})")
    if missing:
        print(f"[resume] missing: {len(missing)}")
    if unexpected:
        print(f"[resume] unexpected: {len(unexpected)}")

    set_levels(model, 3)
    sherry = meta.get("sherry") == "1"
    if sherry:
        n_sherry = set_sherry(model, True)
        print(f"[finalize] sherry constraint active on {n_sherry} layers (from ckpt metadata)")
    n_train, n_frozen = freeze_for_finalize(model, args.freeze_embed, args.freeze_lm_head)
    print(f"[freeze] trainable={n_train:,} frozen={n_frozen:,}")

    from lion_pytorch import Lion
    opt = Lion([p for p in model.parameters() if p.requires_grad],
               lr=args.lr, weight_decay=args.wd)

    ds = ShardedDataset(args.cache_dir, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=(args.device.startswith("cuda")),
                    drop_last=True)
    it = iter(dl)

    autocast_dtype = {"bfloat16": torch.bfloat16,
                      "float16": torch.float16,
                      "none": None}[args.autocast_dtype]
    store_dtype = {"bfloat16": torch.bfloat16,
                   "float16": torch.float16,
                   "float32": torch.float32}[args.store_dtype]

    model.train()
    opt.zero_grad(set_to_none=True)
    running, running_n = 0.0, 0
    pbar = tqdm(range(args.steps), desc="finalize", dynamic_ncols=True)
    for step in pbar:
        cur_lr = lr_at(step, args.steps, args.lr, args.warmup_steps)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        for _ in range(args.grad_accum):
            batch = next(it)
            tokens = batch["tokens"].to(args.device, non_blocking=True)
            topk_idx = batch["topk_idx"].to(args.device, non_blocking=True)
            topk_prob = batch["topk_prob"].to(args.device, non_blocking=True)
            rest_mass = batch["rest_mass"].to(args.device, non_blocking=True)
            ctx = (torch.amp.autocast(args.device.split(":")[0], dtype=autocast_dtype)
                   if autocast_dtype is not None
                   else torch.amp.autocast(args.device.split(":")[0], enabled=False))
            with ctx:
                out = model(tokens)
                loss = kl_with_rest(out.logits, topk_idx, topk_prob, rest_mass)
            (loss / args.grad_accum).backward()
            running += loss.item()
            running_n += 1
        if args.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.max_grad_norm)
        opt.step()
        clamp_qlinear_weights(model)
        opt.zero_grad(set_to_none=True)
        if (step + 1) % args.log_every == 0:
            pbar.set_postfix(loss=f"{running / max(1, running_n):.4f}",
                             lr=f"{cur_lr:.2e}")
            running, running_n = 0.0, 0

    packed_sd = to_packed_state_dict(model, dtype=store_dtype,
                                     quant_embed=not args.no_quant_embed)
    out_path = args.out / "final_packed.safetensors"
    fmt = "smollmer-packed-sherry-v1" if sherry else "smollmer-packed-158-v1"
    save_file(packed_sd, str(out_path), metadata={
        "format": fmt,
        "model_id": args.model,
        "store_dtype": args.store_dtype,
        "sherry": "1" if sherry else "0",
        "embed_int8": "0" if args.no_quant_embed else "1",
    })
    print(f"[done] wrote {out_path} ({sum(v.numel() * v.element_size() for v in packed_sd.values()) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
