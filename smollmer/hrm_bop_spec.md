# hrm_bop — design spec

A ~150M ternary recurrent LM. HRM-Text's dual-timescale loop, smollmer's
per-(row, group) ternary scales, Bop-style flip optimization on the trits
from random init. Pure causal, pretrained from scratch on a mixed web/edu
corpus on a 6 GB laptop GPU.

Status: **design only**, no code yet. This document is the implementation
contract. Empirical findings get appended below once the first run lands.

---

## Why this exists

Three threads collide here.

1. `bitlooplm-small` (Wout, 2025) — 245M ternary recurrent transformer, single
   shared layer stack × 4 loops, BitNet STE with per-tensor scale, Lion. Hit
   held-out CE 3.22 / ppl 25 on cosmopedia-v2 from scratch. Effective depth
   converged to 3, exit-gate was real.
2. `smollmer/flip_distill.py` — Bop2ndOrder on per-(row, group)-scale ternary
   QLinears, latent-free. Stable as a refinement step (`feedback_flip_findings`:
   "Stage 0 + Bet 1 + Bet 5 didn't beat P; ternary 135M floor may be ~1.9").
   Not yet validated as a from-scratch pretraining method.
3. HRM-Text (Sapient, arxiv:2506.21734, HF: sapientinc/HRM-Text-1B) — dual
   stack (H slow / L fast) with `H_cycles × L_cycles` iterations and a
   **1-step gradient approximation** that backpropagates only through the
   final inner L iter + final outer H iter. Cuts loop BPTT memory to ~one
   stack-pass worth.

This project is the cross product: HRM's dual-stack recurrence with smollmer's
QLinear, optimized with Bop on the trits and Lion on everything else, no teacher.

The honest framing: **almost every dimension here is a research bet**. Bop has
never been used to pretrain an LLM, never been run on recurrent weight
sharing, never been combined with learned per-group scales from random init.
This document is partly the design and partly a list of the bets we're
taking simultaneously. If the run dies in 2k steps we'll know which one bit
us only after isolating them.

---

## Architecture

Pure causal, no PrefixLM. HRM-Text's PrefixLM is plausibly worth the
complexity later but earns no place in v1 — it complicates streaming
pretraining for no measured benefit at this scale.

| component | value |
|---|---|
| `hidden_size` | 1024 |
| `num_attention_heads` | 16 |
| `head_dim` | 64 |
| `num_kv_heads` | 16 (MHA, no GQA in v1) |
| `intermediate_size` | 2752 (≈2.7× hidden, SwiGLU) |
| `H_layers` (slow stack) | 4 |
| `L_layers` (fast stack) | 4 |
| `H_cycles` | 2 |
| `L_cycles` | 3 |
| `vocab_size` | 49152 (SmolLM2 tokenizer) |
| `max_position_embeddings` | 1024 |
| `rope_theta` | 10000 |
| `tie_word_embeddings` | True |
| activation | SwiGLU |
| norm | learned RMSNorm, eps=1e-6 |
| pos enc | RoPE on q,k |
| attention | causal SDPA, `is_causal=True` |

Per-block layout follows smollmer's reuse target (HF LlamaDecoderLayer-style),
not HRM-Text's sandwich norm. Rationale: smollmer's `QLinear` swap and
`build_student.py` plumbing expect HF-style `pre_norm → attn → residual →
pre_norm → mlp → residual` blocks. Sandwich norm (4 norms per layer, used by
HRM-Text and Ouro for "loop stability") is a separate research bet (#R7
below) — not adopting in v1.

### The recurrent core

```python
def core(x_embed):
    z_H = x_embed * embedding_scale          # scale TBD; 1.0 default
    z_L = z_L_init.expand_as(z_H)            # learned param [1,1,H]
    H_c, L_c = cfg.H_cycles, cfg.L_cycles
    total = H_c * L_c
    step = 0
    for h in range(H_c):
        for l in range(L_c):
            step += 1
            if step < total:                 # all but last inner iter
                with torch.no_grad():
                    z_L = L_stack(z_L + z_H)
            else:
                z_L = L_stack(z_L + z_H)
        if h < H_c - 1:                      # all but last outer iter
            with torch.no_grad():
                z_H = H_stack(z_H + z_L)
        else:
            z_H = H_stack(z_H + z_L)
    return z_H
```

Forward cost: 2·(3·4) = 24 L-layer applications + 2·4 = 8 H-layer applications
= **32 effective transformer-layer applications per token**. Identical
order-of-magnitude to SmolLM2-135M's 30 layers, but with shared weights.

Backward cost: 4 L-layer + 4 H-layer = **8 layer-grads per training step**,
same as a plain 8-layer feedforward transformer of the same hidden size.

### Parameter count (rough)

Per QLinear block at hidden=1024, intermediate=2752:
- q+k+v+o: 4 · 1024² = 4.19 M (trits) + 64 row × 16 groups × 4 = ~16 K (scales)
- gate+up: 2 · 1024 · 2752 = 5.63 M (trits) + scales ~22 K
- down: 2752 · 1024 = 2.82 M (trits) + scales ~21 K
- 4 norms × 1024 = 4 K (FP)
- Total per block: ~12.6 M

Embedding (tied with lm_head): 49152 · 1024 = **50.3 M FP**.

Total: 50.3 M (embed) + 8 · 12.6 M (blocks) = **~151 M**.

Trits (Bop-managed): ~101 M. FP (Lion-managed): ~50.4 M.

---

## Quantization scheme

Reuse `smollmer.qlinear.QLinear`. Same forward as `flip_distill.py`:
`levels=3`, per-(row, group=64) scales, mutable scales as `nn.Parameter`,
trits stored directly in `weight` (latent-free, exactly `{-1, 0, +1}`).

### Trit init (Bet B-init)

Uniform discrete: `P(0) = 0.5, P(+1) = 0.25, P(-1) = 0.25`. Bonsai converged
to ~38% zero and bitlooplm to ~33%; starting at 50% gives Bop the most
free-flips early (a zero can move either direction). All trits are
independently sampled.

### Scale init

Standard fan-in init, accounting for the trit variance:

```
var(z_l) = var(sum_k s_g · t_k · x_k)
        = s_g² · in_features · var(t) · var(x)    (assuming uncorrelated)
        = s_g² · in_features · 0.5 · 1.0
```

For `var(z_l) = 1`: `s_g = sqrt(2 / in_features)` per group, broadcast across
rows. Concretely:
- attention proj (in=1024): `s_g = sqrt(2/1024) ≈ 0.0442`
- gate/up (in=1024): same
- down (in=2752): `s_g ≈ 0.0270`

Scales are `nn.Parameter` from step 0 — Lion takes over the moment training
starts.

### What does *not* get quantized

- `embed_tokens` (= `lm_head` via tying): FP
- All RMSNorms: FP
- `z_L_init` (learned [1, 1, H] tensor): FP
- RoPE buffers: not trainable

---

## Optimizers

| param class | optimizer | state (fp32) |
|---|---|---|
| trits (QLinear.weight) | `BopTernary` Bet 1 | `m` + `v` ≈ 2 × 4 B × 101 M = **808 MB** |
| scales (QLinear.scales) | `Lion32` | ≈ 4 B × ~1.5 M = ~6 MB |
| embed (tied) + norms + z_L_init | `Lion32` | ≈ 4 B × 50.4 M = ~202 MB |

### Bop hyperparameters

Starting point (will need tuning during the validation gate):
- `--use-2nd-moment` (Bet 1 mandatory; raw `|m|` is too noisy at this scale)
- `γ = 1e-3` (EMA window ~1000 steps)
- `γ_v = 1e-3` (same window for second moment)
- `τ_norm = 0.5` (smollmer-P refinement value; almost certainly wrong here)
- `eps = 1e-12`
- `reset_on_flip = False`, `refractory = 0`

**Open**: the τ_norm distribution from random ternary init is unknown. The
`flip_findings` memo notes that the refinement regime has different |m|/sqrt(v)
statistics than scratch; we will be tuning blind for the first ~1k steps.
Watch `bop/score_rms` and `bop/score_max` in TB. If `score_rms ≫ τ_norm` the
threshold needs to rise; if `score_max ≪ τ_norm` it needs to fall.

### Lion hyperparameters

bitlooplm sweep: sweet spot 3e-4 to 1e-3, diverges at 3e-3, underfits at 1e-4.
Start: peak `lr = 5e-4`, betas (0.95, 0.98), weight decay 0.0 (Lion's update
norm is already bounded). Cosine schedule: warmup 200 steps → peak → cosine
to `min_lr = 5e-5` over `--total-steps`. Gradient clip 2.0 on the FP params
only (Bop is gradient-clip-agnostic at the trit level; the clip changes the
g_t magnitude distribution and would invalidate τ_norm — gated by
`--clip-bop`, default off).

### Optimizer state breakdown at 6 GB

- Trits storage (fp16 latents, even though they're {-1,0,+1}): 200 MB
- Bop `m` + `v` (fp32): 808 MB
- FP params (embed + norms + z_L_init): 200 MB
- Lion momentum (fp32) on FP + scales: 208 MB
- Activations + grads for the 1-step backward slice: ~400 MB at batch=2 seq=1024
- Cuda kernels + cache: ~500 MB headroom

Total budget: ≈ 2.3 GB. **Comfortably under 6 GB**, leaves margin for
batch-up to 4–8 if it trains stably.

### Trit storage dtype

Default `latent_dtype = float16`. Spec says fp32 for correctness; we have
ample evidence from flip_distill that fp16 is fine when the value set is
literally `{-1, 0, +1}` (no rounding regime). Saves 200 MB.

---

## Data

Streaming mix, packed to seq_len=1024, no per-dataset epochs. Weights are
sampling weights, not pre-mixed ratios.

| dataset | HF id | config | weight | format |
|---|---|---|---|---|
| FineWeb-Edu | HuggingFaceFW/fineweb-edu | `sample-10BT` | 0.60 | text |
| Cosmopedia v2 | HuggingFaceTB/smollm-corpus | `cosmopedia-v2` | 0.25 | text |
| Python-Edu | HuggingFaceTB/smollm-corpus | `python-edu` | 0.10 | text |
| OpenMathInstruct-2 | nvidia/OpenMathInstruct-2 | (default) | 0.05 | problem+solution |

- python-edu's text availability needs to be verified at implementation
  time (`feedback_python_edu_dropped` notes the cache_teacher path stopped
  using it because HF made it metadata-only; this needs re-verifying for
  the raw pretraining path).
- If python-edu is unavailable as text, **fall back to 0.70 FW-Edu /
  0.25 Cosmopedia / 0.05 OpenMath** (reallocate the 0.10 to FW-Edu).
- OpenMathInstruct-2 format: `f"Problem: {problem}\n\nSolution: {generated_solution}"`
  (matches bitlooplm `_format_openmath`).
- Packing: tokenize each example, append `eos_token_id`, pack into
  fixed `seq_len=1024` non-overlapping chunks. Drop the tail of each
  example. No document attention masking (treat the packed sequence
  as a single causal stream — same trade-off as bitlooplm and smollmer
  cache, simplicity over correctness).

Sampling: at each batch position, draw a dataset by weight, then pull the
next packed chunk from that dataset's iterator. Per-dataset iterators are
streaming (no full materialization). Shard each dataset by
`worker_id / num_workers` so DataLoader workers don't duplicate data.

### Held-out validation

A fixed held-out shard from `fineweb-edu` (the dominant component) — say
the *last* 128 packed sequences from a different `sample-XX` split, frozen
on disk after a one-time fetch. Loss computed every `--val-every` steps
(default 500), reported as `val/loss` in TB. No val-loss-based scheduling;
purely for monitoring.

---

## Loss

CE on the final `z_H` only, after one final RMSNorm + lm_head:

```python
logits = lm_head(final_norm(z_H))               # [B, S, V]
loss = F.cross_entropy(
    logits[:, :-1].reshape(-1, V),
    labels[:, 1:].reshape(-1),
)
```

No deep supervision per loop, no exit gate, no entropy bonus. Justification:
the 1-step gradient means non-final loops have `requires_grad=False`, so any
per-loop loss term gets gradient only via the final iter — duplicate signal
with diminishing marginal information. HRM-Text's setup confirms: they
supervise only the final output.

Per-loop CE is still computed as a **diagnostic** under `torch.no_grad()`
every `--log-every` steps, logged as `diag/per_loop_ce_{0..5}` to verify
that later loops produce lower CE than earlier ones (the bitlooplm
"L0 abandoned, L2 ≈ L3" finding).

---

## Interruptibility

Model after `flip_distill.py`. Components:

1. **SIGINT handler** — `_install_sigint_handler()` reused from
   `smollmer/distill.py`. First Ctrl-C sets `_INTERRUPT["flag"]`; the train
   loop checks at the top of each step and saves before exiting. Second
   Ctrl-C raises KeyboardInterrupt and the `except BaseException` block
   emergency-saves.

2. **`interrupted.pt`** — `torch.save` of:
   ```
   {
     "model": state_dict,
     "bop": opt_bop.state_dict(),
     "lion": opt_lion.state_dict(),
     "lion_sched": sched.state_dict(),
     "next_step": global_step,
     "samples_consumed": global_step * grad_accum * batch_size,
     "run_name": run_name,
     "ctrl_state": best_ema_tracker.state_dict(),
     "best_snapshot": best_snapshot_or_none,
   }
   ```
   Written every `--checkpoint-every` (default 1000) steps and on
   SIGINT / exception.

3. **Auto-resume** — at startup, if `args.out / interrupted.pt` exists
   AND `--resume` is None, load it. Otherwise fresh init.

4. **`--resume <safetensors>`** — warm-start *weights only*. Fresh Bop
   `m`/`v` (zero EMA), fresh Lion momentum, step 0. Useful for restarting
   with different hyperparameters from a converged-ish point.

5. **Atomic save** — write to `interrupted.pt.tmp`, fsync, rename. Standard
   `save_resume` pattern, already in `smollmer/distill.py`.

6. **DataLoader resume** — `ShardedDataset`-style `start_skip` parameter
   advances the streaming iterator by `samples_consumed` examples on
   resume. Same pattern as `flip_distill.py`. Per-worker shard offsets are
   preserved across resume because the shard list is seeded.

7. **Final outputs** —
   - `args.out / final.safetensors` — model state at `--total-steps`
   - `args.out / final_best.safetensors` — best EMA-loss snapshot from
     the tracker
   - `interrupted.pt` deleted on clean completion

---

## TensorBoard

`writer = SummaryWriter(log_dir=args.out / "tb" / run_name)`. Default
`run_name = datetime.now().strftime("hrmbop_%Y%m%d_%H%M%S")`,
overridable via `--run-name`.

Logged every `--log-every` (default 25) steps unless noted:

| key | what |
|---|---|
| `loss/step` | mean CE over the log window |
| `loss/ema` | exponential mean (BestEmaTracker, α=0.05) |
| `loss/best` | best EMA seen so far |
| `bop/flip_rate` | flips this window / trit-elems |
| `bop/flip_count` | total flips this window |
| `bop/m_rms`, `bop/m_max` | distribution of EMA `m` |
| `bop/score_rms`, `bop/score_max` | distribution of `|m|/sqrt(v)` |
| `bop/score_p99`, `bop/score_p99_9` | tail of the flip-decision score |
| `trits/frac_zero`, `trits/frac_pos`, `trits/frac_neg` | overall |
| `trits/H_frac_zero`, `trits/L_frac_zero` | per-stack zero fraction |
| `scales/mean`, `scales/min`, `scales/max`, `scales/p50` | overall |
| `scales/H_mean`, `scales/L_mean` | per-stack |
| `lion/lr` | current LR (cosine sched) |
| `lion/grad_norm` | pre-clip L2 norm of FP grads |
| `throughput/tokens_per_sec` | (tokens this window) / (wall-clock this window) |
| `throughput/steps_per_sec` | |
| `val/loss` | every `--val-every` (default 500) steps |
| `diag/per_loop_ce_{0..5}` | every `--log-every` × 4 steps (no-grad eval) |
| `diag/zL_norm` | RMS of `z_L_init` (for parameter health) |

Histograms (`add_histogram`, every `--hist-every` default 1000 steps):
- `hist/scales/all`
- `hist/scales/H` and `hist/scales/L`
- `hist/m` (sample 1M elements, full Bop `m` is 101M too large)
- `hist/score`

At step 0, `writer.add_text("config", "...")` dumps the full argparse Namespace
+ git SHA + py/torch versions. Same at run end with `stage_end`.

---

## File layout

| file | role |
|---|---|
| `smollmer/hrm_bop.py` | top-level trainer (single-file script, argparse main, modeled on `flip_distill.py`) |
| `smollmer/hrm_model.py` | `HrmBopConfig`, `HrmBopModel`, `HrmDecoderLayer`, the recurrent core |
| `smollmer/hrm_data.py` | weighted streaming mix DataLoader |
| `smollmer/hrm_bop_spec.md` | this document |

Reuse from existing smollmer:
- `qlinear.QLinear`, `qlinear.set_levels`
- `flip_distill.BopTernary` (Bet 1 path)
- `distill.Lion32` (or move it to a shared `optimizers.py` if it lives elsewhere — verify at impl time)
- `distill.BestEmaTracker`, `distill._install_sigint_handler`, `distill._INTERRUPT`,
  `distill.save_resume`, `distill.snapshot_to_cpu`
- `pack.py` and `finalize.py` at deploy time (Stage 1 of the spec, see below)

New entry point in `pyproject.toml`:
```
[project.scripts]
smollmer-hrm-bop = "smollmer.hrm_bop:main"
```

---

## Validation gate

Before declaring the spec validated and tuning hyperparameters, confirm in
order:

1. **No NaN / Inf** for 2k steps from random init.
2. **Loss decreases** from initialization loss (≈ `log(vocab) = 10.8`) to
   below 8.0 within 2k steps. (bitlooplm hit ~5.5 by step 2k.)
3. **Flip rate is high early and decays** — expect 1–10% in the first
   ~500 steps, decaying toward <0.1% steady-state.
4. **`bop/score_rms` and `bop/score_max`** show a stable gap — `score_max`
   well above `τ_norm`, `score_rms` below. If they collapse together,
   τ_norm is wrong.
5. **Per-loop CE diagnostic** shows monotone descent with loop index by
   step 2k (later loops produce lower CE). If the model collapses to
   "all loops produce identical logits", the 1-step gradient is failing
   to teach the recurrence.
6. **Scales don't blow up** — `scales/max` stable within a factor of 10
   of init. Lion's sign-update has bounded step size, so this is mostly
   a sanity check on the Bop/Lion coupling.
7. **Validation loss tracks training loss** within 0.3 nats by step 5k
   (cosmopedia-v2 + fineweb-edu mix is fairly homogeneous; gap should
   stay small).

If any of these fails, isolate which research bet failed (see below)
before changing hyperparameters.

---

## Research bets (what we're betting on simultaneously)

This is the candor section. Each item is independently untested at this
combination; the run's outcome is the joint probability that none of them
bite.

### R1. Bop from random init for ternary LLM
Bop-CIFAR converged from random init on CNNs. flip_distill never tried
random-init pretraining — only refinement. `feedback_flip_findings`:
"BopTernary criterion is too conservative to compensate quickly enough."
This bet says that "too conservative" was specific to the refinement
regime (where smooth-QAT had already eaten the easy descent), and that
from random init Bop has plenty of high-`|m|` trits to flip every step.

**Risk:** flip rate collapses to ~0 within hundreds of steps because
nothing crosses τ_norm. Watch `bop/score_max` and `bop/flip_rate`.

### R2. Bop on shared (recurrent) weights with 1-step gradient
The trit `w_{ij}` in the L-stack participates in 3 (×2 = 6) forward
applications but receives gradient from only the final inner L iter.
The Bop EMA accumulates this "fixed-point" gradient over many steps.
Open question: does the fixed-point gradient point in the same direction
as the "true" full-BPTT gradient? Untested.

**Risk:** the flip decisions are misaligned with what would actually
reduce loss. Catastrophic case: loss diverges as iterations make the
fixed-point gradient unreliable. Watch `loss/step` for divergence past
step 2k. Mitigation: a quick ablation run with full-BPTT (no `no_grad`
wrappers) on a smaller batch — see if it reaches lower loss faster.

### R3. Bop with concurrently learned scales
`flip_research.md` Bet 2 warned: learnable scales aggravate oscillation
(OFQ, arXiv:2302.02210). That warning was given for the refinement case
where the model is near-optimum and small scale wiggles cause stable
trits to flip. From-scratch differs: there is no near-optimum yet, both
trits and scales are moving fast and roughly synchronously. Plausible
this works; possible it does not.

**Risk:** oscillation between scale-flip-scale-flip. Diagnostic: if
`bop/flip_rate` plateaus at high values (>1%) past step 5k while `loss/ema`
stops improving, scales are racing trits.

### R4. HRM dual-stack with shared weights inside each stack
HRM-Text shares neither: the H_stack and L_stack are independent
parameter sets. We're keeping that. The shared-weight aspect of our
model is *within* each stack iteration (the L stack is the same 4 layers
applied 3 times per cycle). This isn't HRM's contribution — it's
recurrence in the Ouro/LoopLM sense, layered on top of HRM's two
timescales. The combination is new.

**Risk:** the H and L stacks don't differentiate at our scale — both
converge to similar weights, defeating the dual-timescale point. Watch:
`scales/H_mean` vs `scales/L_mean` distributions, `trits/H_frac_zero`
vs `trits/L_frac_zero`. If they're identical by step 10k, the
dual-stack adds parameters but no compute.

### R5. Pure CE pretrain, no teacher
bitlooplm showed this works with STE + per-tensor scale (ppl 25 from
cosmopedia-v2). We're swapping STE for flip-opt and per-tensor for
per-group scales. Independent of R1–R4.

**Risk:** none new beyond R1–R4 actually. The bet is mostly that the
prior bitlooplm result generalizes when we substitute the quantization
scheme. If R1 succeeds, R5 follows.

### R6. Trit init zero-fraction at 50%
bitlooplm settled near 33%; Bonsai at 38%. Starting at 50% maximizes
early-flip freedom but may take longer to find a useful representation
than starting at the steady-state distribution.

**Risk:** low. Worst case: convergence is a few hundred steps slower than
optimal init. Ablation cost: trivial (re-run with `--init-zero-frac 0.33`).

### R7. No sandwich norm
HRM-Text and Ouro both use sandwich norm (4 norms per layer: pre-attn,
post-attn, pre-ffn, post-ffn) for "loop stability". We're using HF Llama
style (2 norms, pre-attn + pre-ffn). bitlooplm used sandwich and it
worked; whether the loop is stable without it at recurrence depth 6 is
untested in this codebase.

**Risk:** activation magnitude grows across loop iterations, hidden
state diverges. Diagnostic: `diag/zL_norm` per loop iteration (would need
to instrument). If it grows unbounded across iterations, swap to sandwich
norm. Cost of swap: small (extra 2 norms × 4 layers × 2 stacks × 1024 =
~16K FP params).

---

## Deferred / out of scope for v1

- **PrefixLM mask** — earn its place once we have a baseline working causal model.
- **Adaptive early-exit inference** — bitlooplm's "if exit gate fires, skip remaining loops". Inference engineering, not a training change. Defer until v2.
- **Sandwich norm** — see R7; only adopt if `zL_norm` diverges.
- **Per-loop deep supervision** — only worth revisiting if R2 fails (1-step gradient is genuinely insufficient).
- **Pack for deploy (`pack.py`)** — once the model converges, fold into Bonsai format with `pack.py` and `finalize.py`. Should work as-is since QLinear is the same.
- **ONNX export** — same as smollmer.
- **GQA** — 16 KV heads now (full MHA). Switch to GQA only if attention becomes the memory bottleneck (it won't at hidden=1024).
- **Activation 8-bit quant** — bitlooplm used `activation_quant(x, 8)` inside its `BitLinear.forward`. smollmer's QLinear doesn't quantize activations. We follow smollmer (FP activations) — the per-(row, group) scale handles representation, no need to also crush activations to int8 unless we're targeting a hardware kernel that requires it. Defer.

---

## Empirical findings

*(append after the first runs land — date them. Mirror `flip_research.md`'s
"What was tried / What we learned / Implication" structure.)*

### Run log

#### 2026-06-01 / 2026-06-04 — hrm-A through hrm-H

**Settings shared across all runs unless noted:** spec defaults (hidden=1024,
H=4/L=4, 2×3 cycles, vocab=49152, seq=1024). Bop Bet 1, γ=γ_v=1e-3. Lion32
on FP params, peak lr=5e-4 cosine to 5e-5 over `--total-steps`. fp16
trits. Same `--init-seed 0` everywhere, so init differences are zero —
post-init divergence is entirely due to hyperparams.

| run | config | result | notes |
|---|---|---|---|
| hrm-A | bs=2 ga=8, τ_norm=0.5 | loss/ema 7.14 @ step 6175 (still descending slowly); flip rate ≤ 1e-7 throughout | Failed: scales drifted to ±2.9 (Lion's sign update has no positivity bias); `g_t = s·dL/dW` had random sign per group → Bop incoherent. Killed. |
| hrm-B | identical to A + positive clamp on scales | loss/ema 6.97 @ step 4400; flip rate 0.000 in postfix throughout | Scales stayed sane (clamp worked); but τ_norm=0.5 is way above the achievable score_max (~0.2). Loss reduction came entirely from Lion on FP. |
| hrm-C/D | bs=4/bs=3 | OOM | bs=2 ga=16 (effective batch 32) is the largest that fits 6 GB. |
| hrm-E | bs=2 ga=16, τ_norm=0.2 | loss/ema 7.12 @ step 1750; flip rate 1e-7 | Same "score_max stuck at τ_norm" plateau, just shifted from 0.5 to 0.2. Tried hot-retune τ_norm=0.15 mid-run at step 1752 — catastrophic: 5M trits flipped in one log window (8.59% in 25 steps), loss spiked from 7.11 to 29.4, v EMA contaminated with Inf → score=NaN for those positions permanently. Killed. |
| hrm-F | fresh from scratch, τ_norm=0.15 | loss/ema 7.00 @ step 1500; flip rate 1e-6 | Real flips from step 0, but still plateau'd. 0.3% of trits permanently flipped. Killed at 1575. |
| **hrm-G** | hrm-F config + `--random-scales`, `--freeze-scales`, `--freeze-non-embed-fp` (only `embed_tokens` is trainable FP) | **loss/ema 6.08, val/loss 6.28** @ step ~7000; per-loop CE gap 0.33 nats; 2.5% cumulative flips | The winning config. Beat the previous plateau by 0.85 nats. |
| **hrm-H** | identical to hrm-G + `--freeze-trits` (ablation: Bop is a no-op) | **loss/ema 6.93, val/loss 7.20** @ step 7850 — plateau'd | Embed-alone-Lion floor. Same as hrm-A/B/F's plateau; the trit-fixed baseline doesn't break the wall. Per-loop CE gap collapsed to 0.02. |

### What we learned

1. **Bop contributes ~0.92 nats of CE reduction**, measured by the
   hrm-G − hrm-H val/loss gap (7.20 − 6.28). That is a clean isolation:
   same architecture, same random ternary init, same random frozen
   scales, same frozen norms — the only difference is whether trits
   are allowed to flip. So the 0.92 nats is the *floor* on Bop's
   contribution at this size; with norms/scales also trainable
   (hrm-A/B/F) the model can't extract the same value from flips
   because Lion-on-FP absorbs the gradient first.

2. **The dual-stack recurrence is parasitic on trit motion.** With trits
   frozen at random init, the per-loop CE gap is 0.02 nats — the H and
   L stacks produce essentially identical logits, recurrence is
   useless. With Bop flipping trits, the gap reaches 0.33 nats by
   step ~7000 and keeps growing. The architecture and the optimizer
   co-depend: there's no "we get HRM-style recurrence for free from
   the structure" — recurrence has to be carved into the trits.

3. **Per-(row, group) scales going negative is a real failure mode of
   Lion + Bop.** Lion's sign update has no positivity bias. Scales
   drift through zero. With `s < 0`, `g_t = s · ∂L/∂W` has sign opposite
   to the loss-reducing direction, and Bop's flip decisions become
   random. Fix: `clamp_min_(1e-6)` on `scales.data` after each Lion
   step. (Equivalent math: absorb the sign into the trits in the
   affected group; we just clamp.) This was added in hrm-B.

4. **Hot retuning τ_norm down is dangerous.** If `m` has been
   accumulating signal for thousands of steps but never crossing the
   active threshold, lowering the threshold all at once fires every
   accumulated-but-saturated trit in one step. Observed in hrm-E:
   5.3 M trits flipped in one 25-step log window, loss spiked 4×,
   `v` EMA got Inf-contaminated by the huge gradients of the spike
   step, score became NaN for those positions forever. Mitigation
   (not yet implemented): when lowering τ_norm mid-run, also zero the
   Bop `m` and `v` state so the EMAs start fresh against the new
   threshold. Cold-start at the lower threshold has no such problem.

5. **τ_norm sits right at the score_max ceiling — there is no useful
   "headroom"**. Whatever τ_norm we choose, `score_max` asymptotes to
   roughly that value and `flip_rate` decays toward a steady-state
   set by the tail of the score distribution. The per-step SNR per
   trit at this scale is too small for τ_norm=0.5 to fire on
   anything; τ_norm=0.15 is the smallest we've tried that gives
   sustained activity without going noise-dominated. The
   `flip_rate · steps` integral is what matters for total loss
   movement, not the threshold setpoint per se.

6. **Effective batch matters more than per-step compute.** bs=2 ga=16
   reaches a meaningfully lower loss than bs=2 ga=8 (the original
   hrm-A config). The noise reduction from larger effective batch
   lets `score_max` cross τ_norm more often. This is consistent with
   the S/√(S²+σ²/N) model: doubling N at fixed S gives a meaningful
   score bump only when S²/σ² is small, which is exactly our regime.

### Implication for next steps

- The R2 risk in the spec ("Bop on shared recurrent weights with 1-step
  gradient") **did not materialize** as a blocker. Bop does train the
  trits in this regime, and the recurrence does become useful once
  trits move. We don't need to add full BPTT through the loops.
- The R3 risk ("learnable scales aggravate oscillation") **did
  materialize**. The combination of trainable per-(row, group) scales
  + Bop + Lion produced the runs that plateau'd at 6.9. Freezing
  scales unlocks Bop. This is the inverse of the spec's prediction
  for the from-scratch case — turns out the scales-flips coupling is
  bad here too.
- Next bets to try, in rough order of expected payoff:
  1. **Resume hrm-G to completion** (40k steps). At step 7000 we were
     at val 6.28 with 21h on the clock; the remaining schedule should
     keep descending given Bop is still firing and per-loop gap is
     still growing.
  2. **Unfreeze RMSNorms partway through** (e.g. at step 20k or once
     ema plateau is hit). The R3 finding suggests scale-coupling is
     the problem, but norms might be OK to train once the trit
     pattern has stabilized.
  3. **Layer-wise scale fitting at deploy**, à la `pack.py` — compute
     the best per-(row, group) scale post-hoc given the final trits.
     Cheap and may close some of the gap to a hypothetical
     scale-trained version.
  4. **Wider effective batch** (ga=32 → 64 tokens/sample × 1024 seq
     = 64k tokens/step). Slower per step but probably more flips per
     unit compute.
