"""gsq.py — Gumbel-Softmax Quantization (GSQ) PTQ for SmolLM2-135M.

Block-wise progressive PTQ: each transformer block's QLinear layers are
optimized via Gumbel-Softmax relaxation to minimize block output MSE
against the FP16 teacher. Blocks are processed sequentially; each block
sees activations from already-quantized prior blocks (progressive).

Reference: arXiv:2604.18556 (Alistarh lab, 2026)

Key design choices from the paper:
  - Lion optimizer: sign-based momentum is stable when GS gradient
    magnitude vanishes near saturation at low temperature.
  - τ: 2→0.05 linear, κ (noise scale): 100→500 linear.
  - α=3 warm-start from nearest-ternary of FP16 weights.
  - Block-level MSE loss (not per-layer, not cross-entropy).
  - codepoint_c (per-group scale) trained alongside logits.
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader

from .build_student import load_student
from .distill import Lion32, ShardedDataset
from .qat_distill import attach_learnable_c, init_c_from_band_mean
from .qlinear import (
    QLinear, init_gsq_logits, set_gsq_temp, snap_gsq_to_weight,
)


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def _causal_mask(
    seq_len: int, batch_size: int, device: str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Upper-triangular additive causal mask: 0 for attend, -inf for mask."""
    m = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),
        diagonal=1,
    )
    return m.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)


@torch.no_grad()
def collect_calibration_tokens(
    cache_dir: Path,
    n_seqs: int,
    seed: int,
) -> torch.Tensor:
    """Pull n_seqs token sequences from ShardedDataset."""
    ds = ShardedDataset(cache_dir, seed=seed)
    dl = DataLoader(ds, batch_size=min(n_seqs, 32), drop_last=False)
    seqs = []
    for batch in dl:
        seqs.append(batch["tokens"])
        if sum(t.shape[0] for t in seqs) >= n_seqs:
            break
    tokens = torch.cat(seqs, dim=0)[:n_seqs]
    print(f"[calib] {tokens.shape[0]} sequences × {tokens.shape[1]} tokens")
    return tokens


@torch.no_grad()
def collect_all_block_io(
    teacher: nn.Module,
    calib_tokens: torch.Tensor,
    run_batch: int,
    device: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Single teacher forward pass collecting (x_calib, y_ref) for every block.

    x_calibs[k] = teacher hidden state entering layer k
    y_refs[k]   = teacher hidden state leaving layer k

    All stored as CPU float16 [n_seqs, seq_len, hidden]. Each block's
    calibration is independent — no student activations, no error compounding.
    """
    n = calib_tokens.shape[0]
    seq_len = calib_tokens.shape[1]
    n_blocks = len(teacher.model.layers)
    teacher.eval()

    # Precompute RoPE once
    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    dummy_h = torch.zeros(1, seq_len, teacher.config.hidden_size,
                          device=device, dtype=torch.float16)
    pos_emb_fixed = teacher.model.rotary_emb(dummy_h, position_ids=pos_ids)

    # accumulators[k] collects batch chunks for layer k's input
    inputs_acc:  list[list[torch.Tensor]] = [[] for _ in range(n_blocks)]
    outputs_acc: list[list[torch.Tensor]] = [[] for _ in range(n_blocks)]

    for i in range(0, n, run_batch):
        tokens = calib_tokens[i:i + run_batch].to(device)
        bs = tokens.shape[0]
        h = teacher.model.embed_tokens(tokens).half()
        mask = _causal_mask(seq_len, bs, device, dtype=h.dtype)
        pos_emb = (pos_emb_fixed[0].expand(bs, -1, -1),
                   pos_emb_fixed[1].expand(bs, -1, -1))
        for k in range(n_blocks):
            inputs_acc[k].append(h.cpu())
            out = teacher.model.layers[k](
                h,
                attention_mask=mask,
                position_ids=pos_ids,
                position_embeddings=pos_emb,
                past_key_values=None,
                use_cache=False,
            )
            h = out[0] if isinstance(out, tuple) else out
            outputs_acc[k].append(h.cpu())

    x_calibs = [torch.cat(chunks, dim=0) for chunks in inputs_acc]
    y_refs   = [torch.cat(chunks, dim=0) for chunks in outputs_acc]
    return x_calibs, y_refs


# ---------------------------------------------------------------------------
# Per-block optimization
# ---------------------------------------------------------------------------

def optimize_block(
    student_layer: nn.Module,
    rotary_emb: nn.Module,      # student.model.rotary_emb
    x_calib: torch.Tensor,      # [N, seq, hidden] CPU float16
    y_ref: torch.Tensor,        # [N, seq, hidden] CPU float16
    max_steps: int,
    min_steps: int,
    plateau_patience: int,
    plateau_ema_alpha: float,
    tau_init: float,
    tau_end: float,
    kappa_init: float,
    kappa_end: float,
    lr_logits: float,
    lr_scales: float,
    weight_decay: float,
    batch_size: int,
    autocast_dtype: torch.dtype,
    device: str,
    writer: SummaryWriter | None = None,
    global_step_offset: int = 0,
) -> tuple[float, float, int]:
    """Optimize one block's GSQ logits + codepoint_c via block MSE.

    Steps until plateau (loss stops improving) or max_steps, whichever comes
    first. Plateau detection starts after min_steps.

    Returns (final_loss, final_loss / ref_var, steps_taken).
    """
    n = x_calib.shape[0]
    seq_len = x_calib.shape[1]
    ref_var = float(y_ref.float().var().clamp_min(1e-8))

    # Precompute fixed position context (same for all sequences)
    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    with torch.no_grad():
        dummy_h = torch.zeros(1, seq_len, x_calib.shape[2],
                              device=device, dtype=autocast_dtype)
        pos_emb_fixed = rotary_emb(dummy_h, position_ids=pos_ids)
        # pos_emb_fixed = (cos, sin) each [1, seq, head_dim]

    # Separate param groups: logits and codepoint_c
    logit_params, scale_params = [], []
    for m in student_layer.modules():
        if not isinstance(m, QLinear):
            continue
        if getattr(m, "mask_logits", None) is not None:
            logit_params += [m.mask_logits, m.sign_logits]
        if hasattr(m, "codepoint_c") and m.codepoint_c.requires_grad:
            scale_params.append(m.codepoint_c)

    opt = Lion32(
        [
            {"params": logit_params, "lr": lr_logits, "weight_decay": weight_decay},
            {"params": scale_params, "lr": lr_scales, "weight_decay": weight_decay},
        ],
        lr=lr_logits,
        weight_decay=weight_decay,
    )

    student_layer.train()
    last_loss = float("nan")
    plateau_ema: float | None = None
    plateau_count = 0
    steps_taken = 0

    for step in range(max_steps):
        frac = step / max(max_steps - 1, 1)
        tau = tau_init + (tau_end - tau_init) * frac
        kappa = kappa_init + (kappa_end - kappa_init) * frac
        set_gsq_temp(student_layer, tau, kappa)

        idx = torch.randperm(n, device="cpu")[:batch_size]
        x_b = x_calib[idx].to(device)
        y_b = y_ref[idx].to(device)
        bs = x_b.shape[0]
        mask = _causal_mask(seq_len, bs, device, dtype=autocast_dtype)
        pos_emb = (pos_emb_fixed[0].expand(bs, -1, -1),
                   pos_emb_fixed[1].expand(bs, -1, -1))

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.split(":")[0], dtype=autocast_dtype):
            out = student_layer(
                x_b.to(autocast_dtype),
                attention_mask=mask,
                position_ids=pos_ids,
                position_embeddings=pos_emb,
                past_key_values=None,
                use_cache=False,
            )
        y_pred = out[0] if isinstance(out, tuple) else out
        loss = F.mse_loss(y_pred.float(), y_b.float())
        loss.backward()
        opt.step()
        last_loss = loss.item()
        steps_taken = step + 1

        if writer is not None:
            gs = global_step_offset + step
            writer.add_scalar("loss/step", last_loss, gs)
            writer.add_scalar("loss/step_norm", last_loss / ref_var, gs)
            writer.add_scalar("gsq/tau", tau, gs)
            writer.add_scalar("gsq/kappa", kappa, gs)

        # Plateau detection: EMA tracks slow trend; stop if not improving
        if plateau_ema is None:
            plateau_ema = last_loss
        else:
            plateau_ema = (plateau_ema_alpha * last_loss
                           + (1.0 - plateau_ema_alpha) * plateau_ema)
        if step >= min_steps:
            if last_loss >= plateau_ema:
                plateau_count += 1
                if plateau_count >= plateau_patience:
                    break
            else:
                plateau_count = 0

    student_layer.eval()
    return last_loss, last_loss / ref_var, steps_taken


# ---------------------------------------------------------------------------
# Deploy fold (mirrors finalize_smooth._deploy_fold)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _deploy_fold(model: nn.Module) -> None:
    """Snap ternary weights and fold codepoint_c into scales."""
    for m in model.modules():
        if not isinstance(m, QLinear):
            continue
        out_f, in_f = m.weight.shape
        c_elem = (m.codepoint_c.unsqueeze(-1)
                  .expand(out_f, m.n_groups, m.group_size)
                  .reshape(out_f, in_f)
                  .to(m.weight.dtype))
        thresh = c_elem * 0.5
        target = torch.where(m.weight.abs() > thresh,
                             torch.sign(m.weight) * c_elem,
                             torch.zeros_like(m.weight))
        m.weight.data.copy_(target)
        m.weight.data.div_(c_elem)
        m.scales.data.mul_(m.codepoint_c.data.to(m.scales.dtype))
        m.invalidate_q_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="GSQ PTQ: block-wise Gumbel-Softmax ternary optimization")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-calib-seqs", type=int, default=128,
                    help="Number of calibration sequences.")
    ap.add_argument("--run-batch", type=int, default=16,
                    help="Batch size for the teacher calibration forward pass.")
    ap.add_argument("--opt-batch-size", type=int, default=16,
                    help="Batch size for per-block optimization.")
    ap.add_argument("--steps-per-block", type=int, default=1000,
                    help="Maximum optimizer steps per block.")
    ap.add_argument("--min-steps-per-block", type=int, default=50,
                    help="Minimum steps before plateau early-stopping kicks in.")
    ap.add_argument("--plateau-patience", type=int, default=20,
                    help="Consecutive non-improving steps before stopping a block.")
    ap.add_argument("--plateau-ema-alpha", type=float, default=0.1,
                    help="EMA alpha for plateau detection (window ≈ 1/alpha steps).")
    ap.add_argument("--tau-init", type=float, default=2.0,
                    help="Initial GS temperature (paper: 2.0).")
    ap.add_argument("--tau-end", type=float, default=0.05,
                    help="Final GS temperature (paper: 0.05).")
    ap.add_argument("--kappa-init", type=float, default=100.0,
                    help="Initial GS noise scale (paper: 100).")
    ap.add_argument("--kappa-end", type=float, default=500.0,
                    help="Final GS noise scale (paper: 500).")
    ap.add_argument("--alpha-init", type=float, default=3.0,
                    help="Warm-start logit strength from nearest-ternary "
                         "(paper: α=3 for ternary).")
    ap.add_argument("--logit-std", type=float, default=0.01,
                    help="Std of initial logit noise (paper: 0.01).")
    ap.add_argument("--lr-logits", type=float, default=1e-4,
                    help="Lion LR for mask+sign logits (paper: 1e-4).")
    ap.add_argument("--lr-scales", type=float, default=5e-5,
                    help="Lion LR for codepoint_c scales (paper: 5e-5).")
    ap.add_argument("--weight-decay", type=float, default=1.0,
                    help="Weight decay — very high per paper (paper: 1.0).")
    ap.add_argument("--scale-group-size", type=int, default=64)
    ap.add_argument("--target-zero-frac", type=float, default=0.25,
                    help="Target sparsity for initial codepoint_c.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--autocast-dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "none"])
    ap.add_argument("--tb-dir", type=Path, default=None,
                    help="TensorBoard log root. If omitted, no TB logging.")
    ap.add_argument("--run-name", type=str, default=None,
                    help="TB run name. Defaults to a timestamp.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    autocast_dtype = {"bfloat16": torch.bfloat16,
                      "float16": torch.float16,
                      "none": torch.float32}[args.autocast_dtype]

    # ---- Load teacher (FP16, frozen) ----
    print(f"[build] loading teacher {args.model}")
    from transformers import AutoModelForCausalLM
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16)
    teacher.to(args.device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    n_blocks = len(teacher.model.layers)
    print(f"[build] teacher: {n_blocks} blocks")

    # ---- Build student (FP32 QLinear) ----
    print(f"[build] building student with QLinear group_size={args.scale_group_size}")
    student, _, n_replaced = load_student(
        args.model, dtype=torch.float32, levels=257,
        latent_dtype=torch.float32,
        group_size=args.scale_group_size,
        permute=False,
    )
    student.to(args.device)
    attach_learnable_c(student, default_c=2.0 / 3.0)
    tzf = args.target_zero_frac if 0 < args.target_zero_frac < 1 else None
    init_c_from_band_mean(student, tzf, fallback_c=2.0 / 3.0)
    for m in student.modules():
        if isinstance(m, QLinear):
            m.codepoint_c.requires_grad_(True)
            m.scales.requires_grad_(False)  # frozen; absorbed at the end
    print(f"[build] {n_replaced} QLinear modules")

    # ---- Checkpoint resume ----
    interrupted_path = args.out / "interrupted.pt"
    start_block = 0
    resumed_run_name: str | None = None
    if interrupted_path.exists():
        ckpt = torch.load(str(interrupted_path), map_location="cpu",
                          weights_only=True)
        student.load_state_dict(ckpt["model"], strict=True)
        start_block = int(ckpt.get("next_block", 0))
        resumed_run_name = ckpt.get("run_name")
        # Freeze all already-snapped blocks
        for j in range(start_block):
            for p in student.model.layers[j].parameters():
                p.requires_grad_(False)
        print(f"[resume] {interrupted_path.name}: resuming from block {start_block}")
    student.to(args.device)

    # ---- TensorBoard ----
    writer: SummaryWriter | None = None
    if args.tb_dir is not None:
        # Re-use saved run name on resume so TB scalars are contiguous
        run_name = args.run_name or resumed_run_name or datetime.now().strftime("gsq_%Y%m%d_%H%M%S")
        run_dir = args.tb_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(run_dir))
        print(f"[tb] {run_dir}")
    else:
        run_name = None

    # ---- Calibration data ----
    calib_tokens = collect_calibration_tokens(
        args.cache_dir, args.n_calib_seqs, args.seed)

    # ---- Collect all (x_calib, y_ref) from teacher in one pass ----
    print("[calib] running teacher forward to collect all block I/O …")
    x_calibs, y_refs = collect_all_block_io(
        teacher, calib_tokens, args.run_batch, args.device)
    # Teacher no longer needed on GPU
    teacher.cpu()
    torch.cuda.empty_cache()

    # ---- Block-wise optimization ----
    pbar = tqdm(range(n_blocks), desc="gsq-blocks")
    for block_idx in pbar:
        pbar.set_description(f"gsq block {block_idx}/{n_blocks}")

        if block_idx < start_block:
            pbar.set_postfix({"status": "resumed"})
            continue

        x_calib, y_ref = x_calibs[block_idx], y_refs[block_idx]

        # Attach GSQ logits to this block's QLinear layers only
        n_init = init_gsq_logits(
            student.model.layers[block_idx],
            alpha=args.alpha_init,
            std=args.logit_std,
        )
        # codepoint_c already attached globally; ensure this block's are trainable
        for m in student.model.layers[block_idx].modules():
            if isinstance(m, QLinear) and hasattr(m, "codepoint_c"):
                m.codepoint_c.requires_grad_(True)

        # Optimize
        last_loss, last_loss_norm, steps_taken = optimize_block(
            student.model.layers[block_idx],
            student.model.rotary_emb,
            x_calib, y_ref,
            max_steps=args.steps_per_block,
            min_steps=args.min_steps_per_block,
            plateau_patience=args.plateau_patience,
            plateau_ema_alpha=args.plateau_ema_alpha,
            tau_init=args.tau_init,
            tau_end=args.tau_end,
            kappa_init=args.kappa_init,
            kappa_end=args.kappa_end,
            lr_logits=args.lr_logits,
            lr_scales=args.lr_scales,
            weight_decay=args.weight_decay,
            batch_size=args.opt_batch_size,
            autocast_dtype=autocast_dtype,
            device=args.device,
            writer=writer,
            global_step_offset=block_idx * args.steps_per_block,
        )

        # Hard snap + freeze this block
        n_snapped = snap_gsq_to_weight(student.model.layers[block_idx])
        for p in student.model.layers[block_idx].parameters():
            p.requires_grad_(False)

        # Save checkpoint after each snapped block
        ckpt = {
            "next_block": block_idx + 1,
            "model": student.cpu().state_dict(),
            "run_name": run_name,
        }
        torch.save(ckpt, str(interrupted_path))
        student.to(args.device)

        # Block-level TB summary
        if writer is not None:
            writer.add_scalar("loss/block_final", last_loss, block_idx)
            writer.add_scalar("loss/block_final_norm", last_loss_norm, block_idx)
            writer.add_scalar("gsq/steps_taken", steps_taken, block_idx)
            total_w, zero_w = 0, 0
            for m in student.model.layers[block_idx].modules():
                if isinstance(m, QLinear):
                    total_w += m.weight.numel()
                    zero_w += (m.weight == 0).sum().item()
            if total_w > 0:
                writer.add_scalar("gsq/zero_frac", zero_w / total_w, block_idx)

        pbar.set_postfix({"loss": f"{last_loss:.4f}", "norm": f"{last_loss_norm:.4f}",
                          "steps": steps_taken, "snapped": n_snapped})
        print(f"[block {block_idx:2d}] loss={last_loss:.4f}  norm={last_loss_norm:.4f}  "
              f"steps={steps_taken}  snapped={n_snapped} QLinears")

    # ---- Deploy fold + save ----
    print("[gsq] folding codepoint_c into scales …")
    student.cpu()
    _deploy_fold(student)

    out_path = args.out / "stage_gsq.safetensors"
    # SmolLM2 ties lm_head.weight to embed_tokens.weight; safetensors refuses
    # shared storage — clone the second occurrence keyed by data_ptr.
    sd: dict[str, torch.Tensor] = {}
    seen: dict[int, str] = {}
    for k, v in student.state_dict().items():
        t = v.detach().cpu().contiguous()
        ptr = t.data_ptr()
        if ptr in seen:
            t = t.clone()
        else:
            seen[ptr] = k
        sd[k] = t
    save_file(sd, str(out_path),
              metadata={"model_id": args.model,
                        "method": "gsq",
                        "steps_per_block": str(args.steps_per_block),
                        "n_calib_seqs": str(args.n_calib_seqs),
                        "group_size": str(args.scale_group_size)})
    print(f"[gsq] saved → {out_path}")


if __name__ == "__main__":
    main()
