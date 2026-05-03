"""Distillation curriculum loop.

For each (levels, n_steps) stage in the curriculum:
  1. Set every QLinear.levels to `levels`.
  2. Train n_steps with KL-distillation against cached teacher top-K + rest_mass.
  3. Save a checkpoint with metadata.

Loss (per position):
  KL(p_teacher || p_student)
    = sum_i p_t[i] * log(p_t[i] / p_s[i])           over top-K i
    + p_rest_t * log(p_rest_t / p_rest_s)
where p_rest_s = 1 - sum(p_s[topk_idx]).  Drop teacher constants.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from .build_student import load_student
from .qlinear import QLinear, set_levels


DEFAULT_CURRICULUM: list[tuple[int, int]] = [
    (257, 300), (129, 300), (65, 400), (33, 600),
    (17, 1000), (9, 1500), (5, 2500), (3, 5000),
]


class ShardedDataset(IterableDataset):
    def __init__(self, shard_dir: Path, seed: int = 0) -> None:
        self.paths = sorted(Path(shard_dir).glob("shard_*.safetensors"))
        if not self.paths:
            raise FileNotFoundError(f"no shard_*.safetensors under {shard_dir}")
        self.seed = seed

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        wid = worker.id if worker else 0
        nworkers = worker.num_workers if worker else 1
        rng = random.Random(self.seed + wid * 7919)
        my_paths = self.paths[wid::nworkers] or self.paths
        while True:
            order = list(my_paths)
            rng.shuffle(order)
            for p in order:
                shard = load_file(str(p))
                S = shard["tokens"].shape[0]
                idx_order = list(range(S))
                rng.shuffle(idx_order)
                for i in idx_order:
                    yield {
                        "tokens": shard["tokens"][i].long(),
                        "topk_idx": shard["topk_idx"][i].long(),
                        "topk_prob": shard["topk_prob"][i].float(),
                        "rest_mass": shard["rest_mass"][i].float(),
                    }


def kl_with_rest(student_logits: torch.Tensor,
                 topk_idx: torch.Tensor,
                 topk_prob: torch.Tensor,
                 rest_mass: torch.Tensor,
                 eps: float = 1e-7) -> torch.Tensor:
    # Avoid materializing the full [B,T,V] log_softmax tensor:
    # log_p[i] = logits[i] - logsumexp(logits).  We only need the K
    # gathered positions plus the lse scalar per (B,T).
    lse = torch.logsumexp(student_logits, dim=-1, keepdim=True)        # [B,T,1]
    selected = torch.gather(student_logits, -1, topk_idx)              # [B,T,K]
    log_p_topk = (selected - lse).float()                              # [B,T,K]
    p_topk = log_p_topk.exp()
    p_rest = (1.0 - p_topk.sum(dim=-1)).clamp_min(eps)                 # [B,T]
    log_p_rest = p_rest.log()
    loss_topk = -(topk_prob * log_p_topk).sum(dim=-1)                  # [B,T]
    loss_rest = -(rest_mass * log_p_rest)                              # [B,T]
    return (loss_topk + loss_rest).mean()


def parse_curriculum(spec: str) -> list[tuple[int, int]]:
    if not spec:
        return DEFAULT_CURRICULUM
    out: list[tuple[int, int]] = []
    for chunk in spec.split(","):
        L, n = chunk.split(":")
        out.append((int(L), int(n)))
    return out


def lr_at(step: int, total: int, base_lr: float, warmup: int) -> float:
    if warmup and step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    # Cosine to 10% of base_lr over the rest of the stage.
    import math
    progress = (step - warmup) / max(1, total - warmup)
    progress = max(0.0, min(1.0, progress))
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def save_checkpoint(model, path: Path, levels: int, stage: int, model_id: str) -> None:
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(sd, str(path), metadata={
        "levels": str(levels),
        "stage": str(stage),
        "model_id": model_id,
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None,
                    help="Stage checkpoint (.safetensors) to load before training.")
    ap.add_argument("--start-stage", type=int, default=0,
                    help="Skip curriculum stages before this index.")
    ap.add_argument("--curriculum", type=str, default="",
                    help="Override default curriculum, e.g. `33:200,17:200,9:300,5:500,3:1000`")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup-steps", type=int, default=30)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--autocast-dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "none"])
    ap.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=True,
                    help="Trade some compute for activation memory (recommended on <=8GB).")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"[build] loading {args.model} and quantizing projections")
    model, _tok, n_replaced = load_student(args.model, dtype=torch.float32, levels=257)
    print(f"[build] {n_replaced} QLinear modules")
    model = model.to(args.device)
    if hasattr(model, "config"):
        model.config.use_cache = False  # never needed for training fwd/bwd
    if args.grad_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("[build] gradient checkpointing enabled")

    if args.resume is not None:
        with safe_open(str(args.resume), framework="pt") as f:
            meta = f.metadata() or {}
        sd = load_file(str(args.resume))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[resume] {args.resume.name} (meta={meta})")
        if missing:
            print(f"[resume] missing keys: {len(missing)} (showing 5): {missing[:5]}")
        if unexpected:
            print(f"[resume] unexpected keys: {len(unexpected)} (showing 5): {unexpected[:5]}")

    if args.compile:
        model = torch.compile(model)

    from lion_pytorch import Lion
    opt = Lion(model.parameters(), lr=args.lr, weight_decay=args.wd)

    curriculum = parse_curriculum(args.curriculum)
    print(f"[plan] curriculum: {curriculum}")
    if args.start_stage:
        print(f"[plan] starting at stage {args.start_stage}")

    ds = ShardedDataset(args.cache_dir, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=(args.device.startswith("cuda")),
                    drop_last=True)
    it = iter(dl)

    autocast_dtype = {"bfloat16": torch.bfloat16,
                      "float16": torch.float16,
                      "none": None}[args.autocast_dtype]

    for stage_idx, (levels, n_steps) in enumerate(curriculum):
        if stage_idx < args.start_stage:
            continue
        n_set = set_levels(model, levels)
        print(f"\n[stage {stage_idx}] levels={levels} on {n_set} layers, {n_steps} steps")
        model.train()
        opt.zero_grad(set_to_none=True)
        running = 0.0
        running_n = 0
        pbar = tqdm(range(n_steps), desc=f"L={levels}", dynamic_ncols=True)
        for step in pbar:
            cur_lr = lr_at(step, n_steps, args.lr, args.warmup_steps)
            for g in opt.param_groups:
                g["lr"] = cur_lr
            for _ in range(args.grad_accum):
                batch = next(it)
                tokens = batch["tokens"].to(args.device, non_blocking=True)
                topk_idx = batch["topk_idx"].to(args.device, non_blocking=True)
                topk_prob = batch["topk_prob"].to(args.device, non_blocking=True)
                rest_mass = batch["rest_mass"].to(args.device, non_blocking=True)
                ctx = (torch.amp.autocast(args.device.split(":")[0], dtype=autocast_dtype)
                       if autocast_dtype is not None else torch.amp.autocast(args.device.split(":")[0], enabled=False))
                with ctx:
                    out = model(tokens)
                    loss = kl_with_rest(out.logits, topk_idx, topk_prob, rest_mass)
                (loss / args.grad_accum).backward()
                running += loss.item()
                running_n += 1
            if args.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if (step + 1) % args.log_every == 0:
                pbar.set_postfix(loss=f"{running / max(1, running_n):.4f}",
                                 lr=f"{cur_lr:.2e}")
                running = 0.0
                running_n = 0
        ckpt_path = args.out / f"stage_{stage_idx:02d}_L{levels}.safetensors"
        save_checkpoint(model, ckpt_path, levels, stage_idx, args.model)
        print(f"[stage {stage_idx}] saved {ckpt_path}")

    print("\n[done] curriculum complete.")


if __name__ == "__main__":
    main()
