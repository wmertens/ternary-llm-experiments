# flip_distill — research plan

Stage 0 (the `flip_distill.py` baseline) is the only component with empirical
grounding — and even that is grounding-by-analogy (Bop is binary, CNN/ViT
scale). Every bet below is a research bet against an LLM-scale, ternary,
loop-friendly distillation target. Add one at a time; each must beat the
running best on a fixed held-out eval to be kept. Bets 3–5 all target
post-flip stability; do a head-to-head, do not stack blindly.

## Stage 0 — what's already in `flip_distill.py`

- Latent-free: `self.weight` *is* the trit, exactly in `{-1, 0, +1}`.
- Per-(row, group) scale `s_g` initialized by iterative LS against the
  teacher's pre-quant projection weights (`W_ref`), recomputed every
  `--scale-recompute-every` steps via the same closed-form
  `s_g = <W_ref, t> / <t, t>`. **Not learned.**
- Per-trit fp32 EMA `m` of the trit-space gradient `g_t = s_g · dL/dW`
  (already what `weight.grad` carries thanks to QLinear's `levels=3` STE).
- Bop flip rule with **two hyperparams** (γ, τ) and a rail clamp on `m`
  (the only ternary-specific extension; no Bop precedent but unambiguously
  correct). No reset, no refractory, no throttle, no second moment.
- Per-step instrumentation: flip rate, `m` rms / max, per-class trit
  fractions, scale stats.

## Validation gate — run before any bet

Before adding any bet, confirm Stage 0 trains. Concretely:

1. Loss decreases for the first ~2k steps from the LS-init starting point.
2. Flip rate is high early (1–10%) and decays toward steady state.
3. `m/rms` settles at a magnitude below `--tau` but `m/max` stays above
   (so a small tail of trits crosses the threshold each step — that's
   how Bop works).
4. Scale recompute every N steps doesn't visibly perturb loss.
5. Spot-check 5–10 trits: do any oscillate or sweep rail-to-rail more
   than a handful of times in a row? Expect occasional sweeps under a
   strongly signed gradient; persistent oscillation is the failure mode
   Bets 3–5 address.

Record the Stage-0 number on the eval. That is the bar.

## Hyperparameter notes for Stage 0

Bop CIFAR settings (`γ ~ 1e-3`, `τ ~ 1e-6`) are a starting region, not a
recipe at LLM scale. The signal magnitude of `g_t = s_g · dL/dW` depends on:

- `s_g`: per-group scale from LS fit to the teacher; typically O(1e-2) to
  O(1e-1) for SmolLM2.
- `dL/dW`: bf16-autocast KL gradients, typically O(1e-4) to O(1e-5).
- So `g_t` lands around O(1e-6) to O(1e-7).

Watch `m/rms` and `m/max` in TB after a few hundred steps. Set τ so that
roughly 0.01–1% of trits flip per step early. If too few flip, lower τ; if
the layer goes to a uniform pattern, raise τ.

---

## Bet 1 — second-moment normalization (Bop2ndOrder)

**Why first:** strongest external evidence. Bop2ndOrder (arXiv:2104.05124)
reports accuracy gains over Bop on CIFAR/ImageNet. Cost is one extra
buffer per trit; risk is low.

**Change in `BopTernary.step()`:**

- Add state buffer `v` (fp32, same shape as `m`).
- `v.mul_(1 - γ_v).addcmul_(g_t, g_t, value=γ_v)` — separate decay
  `γ_v` (start with γ_v = γ).
- Flip criterion: `|m| / (sqrt(v) + eps) > τ_norm`. With normalization,
  τ becomes scale-invariant — a single τ_norm should work across layers.

**CLI:** `--gamma-v`, `--eps`, `--tau-norm` (replaces `--tau` when this
bet is on). Document the new τ range — expect τ_norm in [0.1, 1.0] since
the ratio is unitless.

**Ablation:** does it beat Stage 0 on the eval? If yes, keep; subsequent
bets layer on top of the normalized criterion.

## Bet 2 — learned group scales

**Why:** at LLM scale the scale-vs-trit coupling is plausibly large enough
that LS-to-teacher leaves accuracy on the table. The bet is honest gradient
descent on a log-parameterized scale, with optimizer state preserved across
training (no resets).

**Change:**

- Add parameter `ρ_g` per group; replace `m.scales` use with `exp(ρ_g)`
  inside the forward (or back ρ_g → scales each step before forward).
- Honest gradient: `dL/dρ_g = sum_{i in g} dL/dW_i · s_g · t_i` —
  i.e. `(scales · weight.grad_along_row_group_axis).sum(group_dim)` —
  PyTorch autograd computes this for free if `scales = exp(ρ)` is in the
  forward graph.
- Optimizer for ρ: AdamW or Lion at a small LR (likely 1e-3 to 1e-4),
  with weight decay 0 in log-space.
- **Disable** the periodic LS recompute when this bet is on (otherwise
  the learned ρ gets clobbered).

**Risk:** OFQ (arXiv:2302.02210) shows learnable scales aggravate
oscillation. Log-space decoupling mitigates but does not eliminate. If
this bet wins, watch the flip rate after enabling — a jump suggests
exactly the OFQ failure mode.

**Ablation:** strictly vs the Stage-0 (or Stage 0 + Bet 1) baseline. If
this bet wins, the scale optimizer state survives subsequent bets.

## Bet 3 — stochastic flipping

**Why:** the 2025 probabilistic flip paper (He et al.) diagnosed instability
near the threshold. A Bernoulli flip with `p ∝ |m|` near τ replaces the hard
cutoff and may remove the need for Bets 4 and 5 entirely.

**Change:**

- Replace `(|m| > τ) & valid` with `(Bernoulli(p_flip(|m|)) > 0) & valid`.
- Candidate `p_flip(x) = sigmoid((x - τ) / w)` where `w` is a small
  bandwidth around τ. Or piecewise: `p = clamp((|m| - τ) / w, 0, 1)`.
- At `|m| >> τ + w` flips are deterministic (matches Bop); at `|m| ≈ τ`
  flips become stochastic.

**CLI:** `--flip-bandwidth w` (default 0.5·τ? tune).

**Ablation:** head-to-head with Bets 4 and 5 on a per-trit
oscillation-rate metric, not just final eval loss. If this fixes both
oscillation and the eval gap to Stage 0, no need to pursue 4 or 5.

## Bet 4 — adaptive per-weight threshold (SGDAT)

**Why:** Bop's lineage's preferred answer to over-flipping. The same
flip-count tensor also enables OvSW-style detection of silent trits
(never-flippers) for a possible nudge.

**Change:**

- Add per-trit counter `c_flip` (uint8 saturating, or fp16).
- After flip: `c_flip += flip`.
- Maintain a per-trit τ as `τ_i = τ_base · (1 + η · c_flip_i)` or similar
  monotone schedule.
- Periodically (every M steps) snapshot `c_flip` to derive a "silent" mask
  (`c_flip == 0`) for OvSW-style nudges — e.g. a one-shot relaxed flip
  candidate that uses a lower τ.

**CLI:** `--adapt-eta`, `--silent-nudge-every`, `--silent-threshold`.

**Ablation:** vs Bet 3 and the Stage-0 (+1) baseline.

## Bet 5 — reset-on-flip / refractory

**Why:** least external support (Bop lineage chose 3 and 4 over this).
Only pursue if both 3 and 4 underperform.

**Change:**

- After a flip: `m[flip] = 0`.
- Optionally: maintain a per-trit `lockout` counter, decrement each step;
  block flips where `lockout > 0`.

**CLI:** `--refractory T`.

**Ablation:** vs Bets 3, 4. Likely loses to one or both, but a positive
ablation here would be informative — it would mean the rail-to-rail sweep
is doing real damage at this scale.

## Bet 6 — curvature gate

**Why:** mostly subsumed by Bet 1's normalization. Mainly relevant if Bet 2
(learned scales) is on, because the predicted delta gains a scale-squared
conservatism term. Low priority.

**Change:**

- Maintain `c = EMA(g_t^2)` per trit (similar to Bet 1's `v`).
- Admit flip only if `m·Δt + 0.5·c·s_g^2·Δt^2 < 0`, where `Δt = direction`.

**CLI:** `--curvature-gate`, `--gamma-c`.

**Ablation:** only if Bet 2 wins.

## Bet 7 — per-row concurrency throttle

**Why:** no flip-optimizer prior art; the only motivation is OFQ's
finding of query/key weight coupling causing correlated oscillation.
Looped models like Ouro densify weight-tying, so coupling may bite
sooner. Only enable if instrumentation shows correlated oscillation
within a row.

**Change:**

- After computing the `flip` mask: for each row, keep at most `k` flips
  with largest `|m|`. `k=1 → argmax`.
- Implementation: `mask non-candidates to -inf; torch.topk(|m|·flip, k,
  dim=row_dim); rebuild flip from indices`.
- **Per-row only** — per-layer would need an all-reduce in distributed
  setups; per-row stays shard-local.

**CLI:** `--row-flip-k` (0 = off).

**Ablation:** condition on seeing coupled oscillation in the row-wise
flip-rate distribution. Otherwise skip.

---

## Cross-cutting future work

- **Memory:** quantize `m` to int16 fixed-point only once Stage 0 +
  surviving bets are stable. fp8/int8 needs stochastic rounding to avoid
  within-interval swamping. Pack trits to 2-bit only at deploy, not in the
  optimizer's hot path.
- **Trainable non-trit params:** Stage 0 freezes everything but trits. A
  small fp AdamW over embeddings/norms/biases at low LR is the obvious
  next addition; treat as Bet 0.5 — earn its place too.
- **Looped models (Ouro):** physical weight applied N times per forward;
  the gradient already sums over loop iterations — no special handling.
  Weight-tying densifies coupling, so if Bet 7 ever kicks in, expect to
  need it sooner here than on a feedforward net of similar width.
- **W_ref drift:** the LS scale recompute targets the teacher's
  pre-quant weights, but the KL loss cares about the teacher's *logits*.
  These are different optima. If Bet 2 (learned scales) wins decisively,
  W_ref's only role is the LS init; the persistent buffer can then be
  dropped from later runs.
- **Non-stationary objectives (RL / self-play):** spec calls this out as
  the regime where the coordinate-descent flavor of flip optimization
  is expected to weaken. Out of scope for distillation.

---

# Empirical findings (2026-05-25 / 2026-05-26)

## Setting

- Model: SmolLM2-135M, ternary student via per-(row,group) scales
  (group_size=64); 210 QLinears, ~107M trits in QLinear weights.
- Teacher cache: regenerated 2026-05-25 with stratified cadence-60
  (51F/9C, python-edu dropped); L_T = 1.7444 on n=128 mean.
- Reference: smooth-QAT run P, deploy-folded ternary, **n=128 mean KL
  2.108 ± 0.171**.

## What was tried

1. **flip_distill.py** (all trits trainable from start):
   - flip-B from scratch, Stage 0, τ=1e-7 → plateaued at ~6 EMA
   - flip-D resume P, Stage 0, τ=1e-4 → loss 2→6.5 in 200 steps
   - flip-E resume P, Bet 1, τ_norm=0.5 → stable at ~2.0 EMA, no real
     improvement (multi-batch eval: 1.950 vs P's 1.927)
   - flip-F resume E-best, Bet 1+5 reset → over-suppression, no gain

2. **flip_progressive.py** (one role/group ternarized at a time):
   - A: q,k,v,o,gate,up,down sequential → 3.78 EMA pre-joint, 3.28 post
   - B: q+k,v+o,gate+up+down + adaptive shrink → 6.19 pre-joint (MLP
     group was disastrous; adaptive shrink actively harmful)
   - C: q,k+v,o,gate+up,down + roomier caps, no shrink → 3.43 at down
     (best per-role trajectory; killed before joint)

## What we learned

1. **Direct snap-to-ternary causes loss damage that flip-opt can't
   efficiently recover.** Each module promotion adds 0.1–0.5 nats; the
   damage compounds across 7 roles to ~+2 nats from L_T even with
   per-stage flip optimization to plateau. None of the schedules reach
   smooth-QAT P's 2.11 (multi-batch eval).

2. **Co-training coupled roles is roughly neutral.** Q+K grouped ≈ K
   after Q. V+O grouped ≈ O after V. The expected "let them adapt
   together" benefit was not observed. Q alone first does help slightly
   (1.85 vs 1.90 for q-only).

3. **MLP all-at-once is disastrous.** 90 modules promoted simultaneously
   (B) jumped to EMA 11+. Flip-opt couldn't recover; the joint stage
   from a 6+ start barely budged. 60 modules at once (C's gate+up) is
   recoverable but still worse per Δ than sequential gate then up.

4. **Adaptive τ_norm shrink (Bet 5 idea applied to threshold) is
   actively harmful.** When |m|/sqrt(v) saturates at τ, that's not the
   threshold being too tight — it's the optimizer telling you the
   confident flips at that threshold are exhausted. Lowering τ releases
   noise-driven flips that immediately destroy the optimum (loss 2→14
   in 10 steps in both B's q+k and v+o stages).

5. **Bet 1 (Bop2ndOrder) at τ_norm=0.5 is the only flip configuration
   that keeps the model stable**. Stage-0 raw |m|>τ is too noisy at any
   τ (Bop CIFAR's τ ≈ 1e-6 destroys the model in 10 steps; τ ≈ 1e-4
   degrades over ~100 steps). Bet 5 (reset-on-flip) over-suppresses.

## Implication for the broader research direction

Smooth QAT remains the right tool for the bulk of ternarization.
Flip-based optimization, in any form tested here, is at best a refining
step on top of an already-near-optimal ternary checkpoint, and even then
it has not produced a measurable improvement over the smooth-QAT
champion on multi-batch eval.

The progressive scheduling pursued here (Q alone, then coupled pairs,
then MLP) did not change this conclusion. The fundamental issue is that
direct ternarization injects a large precision-loss perturbation per
module, and the BopTernary criterion (even with Bet 1's normalization)
is too conservative to compensate quickly enough — it would need many
thousands of full forward+backward passes per module, which is
effectively a full training run per module.

## What would change the picture

- **Soft-then-flip hybrid**: at the start of each module's stage, run a
  short smooth-temperature anneal on just that module (re-using
  qat_smooth's machinery, restricted to one module), then lock at hard
  ternary and flip-optimize. Bridges the strengths of both methods.
  Not built.
- **Hessian-weighted promotion order** from calibrate.py: promote
  low-sensitivity modules first so the high-sensitivity ones have the
  most "absorbable" remaining capacity. Could reduce compounding damage.
- **A different optimizer entirely** for the per-module refinement —
  e.g. small-LR AdamW on a smooth-anneal latent inside the per-module
  loop, rather than flip-based.

What is *not* worth more time without a fundamental change: continuing
to tune τ_norm / γ / γ_v / patience for vanilla flip_progressive. The
plateau is structural to the snap-then-flip approach.
