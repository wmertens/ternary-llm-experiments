# Autoresearch worklog: ternary-hrm-fast-recipe

Started: 2026-06-04
Branch: `autoresearch/ternary-hrm-fast-recipe-2026-06-04`
Step budget per experiment: **2500 steps** (~4.5h on RTX 4050)

## Goal

Fastest from-scratch recipe (lowest val/loss at 2500 steps) for a ~150M
ternary HRM-text-style recurrent transformer. See `autoresearch.md` for
the search space and metric definitions.

## Baseline strategy

The hrm-G-bop config (random lognormal frozen scales, frozen
norms+z_L_init, only embed FP trainable, Bop Bet1 on trits, Lion32 @
lr=5e-4 cosine, bs=2 ga=16) was the last sustained run on `main` and
reached val/loss 6.28 by step 7000. We replicate it for 2500 steps as
the autoresearch baseline. Every subsequent experiment perturbs one
knob.

## Run-by-run log

### Run 1: baseline-hrm-G — val_loss=6.9074 (KEEP, baseline)
- Timestamp: 2026-06-04 11:04 → 16:24 (5h20m)
- What ran: hrm-G recipe at 2500 steps. `--random-scales --freeze-scales
  --freeze-non-embed-fp`, bs=2 ga=16, τ_norm=0.15, γ=γ_v=1e-3, lr=5e-4.
- Result: val_loss=**6.9074**, loss_ema=6.8269, flip_rate=3.86e-6,
  frac_zero δ=-1.1% (1.1% of trits left zero), per-loop gap=0.039 nats.
- Insight: val at 2500 (6.91) is much higher than expected (predicted
  ~6.6 from prior hrm-G trajectory). Reason: this run uses
  `--ema-warmup 200` (vs prior hrm-G's 500), so the EMA tracker is
  different — but val is direct, not EMA. The real diff: prior hrm-G
  was at val 6.83 at *step 2000* in a 40k-budget cosine schedule, not
  step 2500 in a 2500-budget schedule. The shorter budget means the
  cosine decay is much steeper — at step 2500/2500 we're already deep
  in the cosine tail (lr ≈ 0.55e-4) whereas the 40k-budget at step
  2500 was still near peak (lr ≈ 4.84e-4). The val descent slowed
  dramatically in the last 500 steps as Lion's LR collapsed.
- **Implication for the budget**: 2500 steps with cosine-to-floor IS
  the test budget, not "what hrm-G looked like at step 2500 on the
  longer run." This is fine — we're optimizing the FIXED-BUDGET val
  loss. But it means improvements are measured against 6.9074, not
  the 6.83 we'd have predicted.
- Tooling note: autoresearch.sh's grep '^[val]' missed tqdm's CR-
  embedded val lines on the first try; fixed in commit 64dfda5
  (pre-Run-1 commit, separate from the result).
- Next: Run 2 — add **cautious mask** (`m·g_t > 0` AND existing flip
  rule) to BopTernary. Lowest-cost win in the queue: one-line code
  change, expected to let τ_norm drop without random-walking, which
  should boost flip activity without hurting per-flip quality.

### Run 2: cautious-bop — val_loss=6.8893 (KEEP, -0.018 vs baseline)
- Timestamp: 2026-06-04 16:24 → 22:16 (5h52m)
- What changed: added `cautious=True` flag to `BopTernary`. In the
  flip rule, after computing `flip = (score > τ_norm) & valid`, AND
  with `(m·g_t > 0)` so only coords where the EMA still agrees with
  the current gradient direction actually flip.
- Result: val 6.8893 vs 6.9074 (-0.018), loss_ema 6.8183 vs 6.8269
  (-0.009), flip_rate 4.07e-6 vs 3.86e-6 (+5%), per_loop_gap
  +0.065 vs +0.039 (+67%), wall +10% (more masking work per step).
- Insight: small absolute win but a clean signal that filtering
  oscillating trits via the cautious mask is helpful in our
  regime — without it, the no-2nd-order-aware flips do happen
  occasionally on noise. Per-loop gap improving more than val_loss
  suggests the recurrence is benefiting disproportionately
  (cautious-filtered flips concentrate on coords that help the
  later H-cycle's representation).
- Next: Run 3 — **C-Muon STE** (user steer). Replace BopTernary
  with Cautious Muon (Jordan 2024 + Liang 2024) on STE'd ternary
  latents. The latent in [-1, 1] gets continuous Muon updates;
  forward STE-quantizes to {-1, 0, +1}; code flips happen as
  latents cross ±1/3 boundaries. New optimizer module
  `smollmer/cmuon.py`, new CLI flags `--ste-trits` and `--c-muon`.

### 2026-06-05 — user steers captured into autoresearch.ideas.md
1. "Warmup with full BPTT against small English corpus, then switch to
   1-step gradient" curriculum (parking lot — non-trivial framework change)
2. Tiny non-looped Bop vs Muon screening (HIGH priority, run after Run 3
   completes but before Run 4 — gives optimizer-choice signal without
   recurrence confound)

### Run 3: c-muon-ste — STARTING 2026-06-04 22:30
- Config: hrm-G structural baseline (random lognormal frozen scales,
  frozen non-embed FP, Lion-on-embed) with trit optimizer swapped
  from BopTernary to CMuon. Cautious mask ON. muon-lr=0.02, muon-
  beta=0.95, ns_steps=5. Latents start at discrete random ternary
  (else quantize-of-Normal(0, 0.02) gives all-zero forward output and
  no gradient signal). After each opt step latents are clamped to
  [-1, 1] via `clamp_qlinear_weights`.
- Expected: Muon's orthogonalized update should drive faster code
  flips than Bop (which is gated by score>τ). On the other hand, the
  cautious mask plus the per-coord update magnitude ~lr/sqrt(N) ≈
  6e-4 means flipping a code (cross ±1/3) takes ~500+ consistent
  steps. If too slow, we may need higher muon-lr.
- Result (2026-06-05): **val_loss 6.5320 (-0.357 vs Run 2)**. Big
  win. flip_rate jumped to 2.63e-4 (65× Run 2), 2× more cumulative
  trit motion. The per-loop gap actually *shrank* from 0.065 to
  0.032, suggesting the trit pattern itself became more agile and
  recurrence is less load-bearing. C-Muon STE is the new baseline.

### Screening Round 1 (Segment 1) — 2026-06-05

Tiny non-looped models (h=384, H=L=2, cycles=1×1, ~26M params, 1500
steps) to compare optimizer choice without the recurrence confound.

| run | optimizer | val_loss | flip_rate | Δfrac_zero | wall |
|---|---|---|---|---|---|
| 4 (s1) | Bop+cautious | 5.4674 | 1.71e-5 | -0.046 | 32min |
| 5 (s2) | CMuon-STE | **5.1850** | 3.94e-4 | -0.004 | 32min |
| 6 (s3) | Lion-STE | 5.5221 | 2.64e-4 | -0.006 | 28min |

- CMuon-STE wins by 0.28 nats vs Bop, 0.34 vs Lion-STE.
- Same direction as the Run-3 finding (where CMuon beat Bop by 0.36
  on the 153M model). Optimizer-choice signal isn't an artifact of
  recurrence dynamics — it's a property of CMuon at this STE setup.
- Interesting: s1 (Bop) has 13× *more* net trit motion away from
  zero (Δfrac_zero -0.046 vs s2's -0.004) but much lower flip rate.
  CMuon-STE's continuous latents oscillate around ±1/3 boundaries so
  trit codes flip back and forth often; the cumulative one-way drift
  is smaller. The bidirectional churn is apparently more useful for
  loss than monotonic drift.
- Lion-STE's underperformance is puzzling given bitlooplm's success.
  Possible explanations: (a) 1500 steps is too short for STE's slower
  per-step convergence to fully express, (b) frozen lognormal scales
  hurt Lion specifically (bitlooplm used per-tensor recomputed
  scales), (c) Lion's sign update is too coarse without the per-tensor
  scale's softening effect. Could revisit in a longer-budget run.

### Screening Round 2 — starting 2026-06-05

Stacking precision tricks on the round-1 winner (CMuon-STE).

- s4: CMuon-STE + int8 per-token-absmax activations (BitNet style).
  Adds STE wrapper around QLinear input. Tests whether BitNet's
  inference-time quantization is also a free training-time trick.
- s5 (planned): CMuon-STE + fp16 opt state (with stochastic rounding).
  Tests whether Muon's m can be stored at lower precision.

Tooling note: switched from `Bash run_in_background=True` (kept
returning "undefined is not an object") to `nohup ... & disown` plus
a `Monitor` watcher that fires when `METRIC val_loss=` appears in the
log. Gives a single completion signal per run without polling.

#### Round 2 results

- **s4 (int8 activations)**: val 5.1848 vs s2 5.1850 — **free** on
  the tiny model (within noise, +0.0002 nats). int8 is the BitNet-
  style per-token-absmax STE applied to QLinear input. New
  `m.int8_activations = True` attribute on QLinear, gated by
  `--int8-activations` CLI flag.

- **s5 (fp16 opt state on CMuon's m)**: val 5.1962 vs s2 5.1850 —
  **small but real penalty (+0.011 nats)**. Stored m as fp16 via
  `state_dtype` param to CMuon; NS5 still fp32 by upcast. The EMA
  rounding error per step is small but does accumulate. Discard from
  pure-quality view; revisit if memory pressure on main run forces it.

### Run 9: main HRM 153M + CMuon-STE + int8 act — STARTED 2026-06-05 ~16:00

Back to full HRM 153M config (hidden=1024, H=L=4, cycles=2×3, bs=2 ga=16,
2500 steps) with the screening winners stacked: CMuon-STE + int8
per-token-absmax activations.

- Expected: val_loss ≈ 6.53 (same as Run 3) if int8 act stays free at
  main-run scale. Better than 6.53 if STE regularization happens to
  help on the bigger model.
- Wall time **observed at step 300**: 11.7 s/step (vs Run 3's 8.1) =
  +45% wall-clock overhead. The screening's "int8 act is free" claim
  was on a model with 1/4 the QLinear forwards per step (cycles 1×1
  vs 2×3). The relative overhead is bigger on main.
- **Recipe-fastest implication**: if val ≈ Run 3 → discard int8 for
  main (wall-time cost without quality gain). If val < Run 3 (real
  improvement) → keep despite wall cost. Decision pending Run 9
  completion (~7h remaining).

---

### Runs 10–32 — consolidated (fast-A surrogate → LR tuning → fixpoint)

Detailed per-run numbers live in `autoresearch.jsonl`; the narrative arc:

- **Runs 10–13 (full BPTT).** fast-A (38M) validated as a 5× faster
  surrogate for the 153M model (Run 10 val 6.55 ≈ Run 3 6.53). Switching
  from the 1-step gradient to **full BPTT throughout** was the big
  unlock: Run 12 fast-A 4.96, Run 13 main 153M 4.78. The 1-step
  approximation — the thing the HRM-Text paper sells — was the bottleneck.

- **Runs 14–25 (CMuon LR).** Cautious mask confirmed helpful (Run 14
  non-cautious +0.11). LR swept up: 0.05 → 0.10 → 0.20; peak at 0.10
  constant, then **0.20 cosine→0.02** edged ahead (Runs 22/23), warmup
  neutral (Run 24). Recipe landed the all-time champion **Run 25: main
  153M, val 4.1562**.

- **Runs 26–29 (does recurrence pay?).** Stripping the outer H cycle
  (1×3, Run 26) and then all recurrence (1×1, Run 27) cost nothing at
  2500 steps; extended to 5000 steps, 1×1 (Run 29, 4.178) ties 2×3
  (Run 28, 4.171) at **53% less wall time**. Fixed-cycle recurrence
  doesn't earn its compute here — the ~0.39 per-loop CE gap is
  under-trained loop layers, not refinement.

- **Runs 30–32 (the pivot — variable-loop fixpoint).** User insight:
  randomize H_cycles per step so the model can't depend on a fixed loop
  count. Run 31 ([1,4]) collapsed the per-loop CE gap **100×**
  (0.39 → 0.004) with val unchanged (4.1863) — the model converges to a
  **true fixed point**, doing genuine iterative refinement. Run 32 ([1,8])
  confirms the property is robust to a 2× wider range (gap 0.016, still
  ~24× below fixed) but costs +56% wall (mean loop count ~4.5 vs 2.5) for
  no quality gain. **[1,4] is the sweet spot.**

- **Run 33 (test-time cycle sweep).** Re-ran fast-A [1,4] and at val time
  swept H_cycles ∈ {1,2,3,4,6,8,12,16,24,32}. val flat at 4.1873 ± 0.0002
  from c2 to c24 — a **stable wide-basin fixed point**, holding 6× past
  the training max. c1 = 4.1989 (one iter under-converged). c32 = 10.8125
  = ln(V): the iterate over-smooths to a degenerate constant fixed point
  (~96 stack-applies cumulative). So ternary recurrence is *stable but
  not improving* at test-time, with a cliff past ~24 loops.

- **Run 34 (FP-weights control — diagnostic, off-leaderboard).** Mirrored
  Run 33 EXACTLY but with `--fp-weights` (CMuon directly on raw FP
  weights, no STE/quantize/scales). Result: val 6.3229 vs ternary 4.1873
  (+2.14 nats), and the cycle sweep has IDENTICAL shape (flat c2..c12,
  collapse at c16, NaN by c48). NOT a clean precision verdict: lr=0.20
  was tuned for ternary latents in [-1,1] and is likely too hot for
  LeCun-scale FP weights (step-500 val 6.86 already underwater). Next:
  sweep FP CMuon lr (0.05 / 0.02 / 0.01) to find FP's own optimum before
  comparing.

---

## Key Insights

- Full BPTT >> 1-step gradient on this HRM/ternary setup (the paper's
  central efficiency claim doesn't hold for us).
- Trit optimizer: CMuon-STE with cautious mask, lr=0.20 cosine→0.02.
- Plain val/loss does not reward recurrence at ≤5000 steps; 1×1 is the
  fastest recipe for pure loss.
- **The reason to keep recurrence is the fixed-point property, and you
  only get it by training with a variable per-step loop count.** Fixed
  cycles leave the loop layers under-trained.

## Next Ideas (queue)

Roughly ordered by expected gain. Replenish as old ideas land.

1. **Cautious mask in BopTernary** — flip only when `sign(m) == -sign(g_t)`.
   Cheapest win in the queue: a single elementwise AND in the flip rule.
   Should let τ_norm drop below 0.15 without random-walking.
2. **Per-tensor BitNet-style scale** recomputed `1/mean(|w|)` each forward,
   not learned. Strips the 1.5M per-group scales down to 56 numbers and
   removes the scale-gradient confound entirely. Tests whether per-group
   granularity matters vs the *frozenness* in hrm-G.
3. **Per-group fan-in fixed** (no lognormal noise). Strict ablation
   of hrm-G — does the noise help or hurt?
4. **STE+Lion on trits** as a direct comparison. We know it's ~1000×
   faster per flip but flips may be less informed. Where's the break-even
   on val/loss at 2500 steps?
5. **τ_norm sweep** with cautious mask: 0.05 / 0.10 / 0.15.
6. **γ / γ_v sweep**: 3e-4 / 1e-3 / 3e-3.
7. **INT8 per-token activations** (BitNet style) to halve act memory
   and try bs=4.
8. **Lion LR sweep on FP**: 1e-4 / 3e-4 / 5e-4 / 1e-3.
9. **Trit init zero-frac**: 33% vs 50%.
10. **Curvature gate** (Bet 6 in `flip_research.md`).
11. **Sandwich norm** in HrmDecoderLayer (4 norms instead of 2).
12. **Cycle ratio**: 1×3 vs 2×3 vs 3×2.
