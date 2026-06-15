# Autoresearch Ideas Backlog

Ideas captured during the loop. Used as inspiration when the immediate
queue is exhausted, or to remember good thoughts that aren't right for
the next experiment but worth keeping.

## 2026-06-05 — User steers (captured mid Run 3)

### A. "Warmup then switch" curriculum for the 1-step gradient
The HRM 1-step-gradient approximation trains only the FINAL L iter and
FINAL H iter; earlier loops are no_grad. This is cheap but maybe pays
a cost early: the loop layers never get differentiable signal *as
intermediate states*, only as fixed-points-reached. Hypothesis: maybe
HRM has to be trained first with **all** loop layers in the autograd
graph (full BPTT through 6 L applications + 2 H applications) against a
**small English corpus** until they "sort of grok language", then switch
to the faster 1-step-gradient regime for the long tail of training.

Suggested test:
1. Bring up a "warmup" mode: re-implement `_core` with no `torch.no_grad()`
   wrappers (full BPTT). Memory cost ~6× normal backward.
2. Run for N steps (~500-1500) on cosmopedia-v2 (smaller, cleaner dataset
   than fineweb-edu) with everything trainable, see how fast loss drops.
3. Switch to 1-step gradient (the current spec) and continue.
4. Compare final val/loss vs same compute spent entirely in 1-step.

Risks:
- Memory: full BPTT through 8 layer-stack applications at bs=2 seq=1024
  may OOM on the 4050. Might need bs=1 for warmup.
- Switch shock: when we drop from full BPTT to 1-step, the gradient
  magnitudes change ~6×; LR or momentum may need re-tuning at the
  transition point.

Status: parking lot. Needs careful design before launching as an autoresearch experiment because the "warmup phase" is a non-standard config that doesn't fit the current fixed-budget framework.

### B. Tiny non-looped optimizer + precision-tricks screening
Before investing many hours in C-Muon-STE-on-HRM-recurrent, do a fast
screening on a non-recurrent baseline to compare optimizers on their
own merits:
- Tiny non-looped ternary transformer (e.g. hidden=512, 6 layers, no
  recurrence, ~20-30M params)
- Same data feed
- 500-1000 steps each (~30 min)
- Three variants: BopTernary (Bet 1 + cautious) / CMuon-STE / Lion-STE
  on trits

This would tell us cleanly whether the optimizer choice matters
without the recurrence confound. If Bop and CMuon are within noise on
a non-recurrent setup, the recurrence-specific dynamics in the HRM
runs are dominating, and we should focus there. If CMuon dominates on
the simple setup, recurrence isn't the issue and we should keep pushing
on optimizer for HRM too.

Suggested implementation:
- New trainer file `smollmer/screen_trit_opt.py` (small, non-HRM model)
  or reuse hrm_bop.py with `--H-cycles 1 --L-cycles 1 --H-layers 0`
  (degenerate "loop" = single forward pass through L_stack). Easier:
  add `--no-loop` flag that runs `z_H = H_stack(embed); z_H = L_stack(z_H)`
  once, no recurrence.
- Output to `experiments/screen_<opt>` / `tb/screen_<opt>`.

Priority: HIGH. This is cheap (~1.5h total for 3 variants) and gives
strong signal for the rest of the autoresearch. Could run interleaved
with main experiments by reducing main step budget to 1500 temporarily.

**Two-round structure** (per user steer 2026-06-05):

Round 1 — optimizer head-to-head (3 runs, ~1.5h):
  - s1: BopTernary (Bet 1 + cautious) on STE'd ternary latents
  - s2: CMuon on STE'd ternary latents
  - s3: Lion32 on STE'd ternary latents (bitlooplm-style)
  All with bf16 activations, fp32 optimizer state, non-looped 1-pass
  through L_stack.

Round 2 — tricks on the round-1 winner (2-3 runs, ~1-1.5h):
  - s4: winner + int8 per-token-absmax activations (BitNet style)
  - s5: winner + fp16 optimizer state with stochastic rounding
  - (s6 optional: both, only if both individually neutral or positive)

Implementation notes:
- Non-loop = add `--no-loop` to hrm_bop.py that bypasses the recurrent
  core and just runs `embed → H_stack → L_stack → final_norm → lm_head`
  once. Tiny model: hidden=512, H=L=3 layers, no cycles.
- int8 activations: per-token absmax quant inside QLinear.forward
  (or a wrapping module). 1-line STE: `qx = (x*s).round().clamp(Qn,Qp)/s`
  with `s = Qp / abs(x).amax(-1, keepdim=True).clamp_min(1e-5)`. Add
  `--int8-activations` flag.
- fp16 opt state: subclass BopTernary / CMuon to cast m, v to fp16 with
  stochastic rounding on the EMA update. Add `--low-precision-opt-state`
  flag. Stochastic rounding is REQUIRED: without it, fp16 underflow at
  low grads silently zeroes the EMA late in training.
- Muon's NS5 must stay fp32 (matrix iteration; fp16 would lose
  orthogonality precision); only the stored momentum buffer can be fp16.

Status: queued. To launch after Run 3 completes, before Run 4.

## 2026-06-11 — User steer (captured during Run 33, fixpoint extrapolation)

### F. FP-weights control: is the recurrence brittle *because* it's ternary?
User question: the runs so far (fixed-cycle recurrence doesn't pay; you
only get a true fixed point by training with a variable per-step loop
count) are all on **ternary** weights. Maybe recurrence is brittle
specifically under ternary quantisation, and an FP-weight HRM would (a)
benefit from fixed-cycle recurrence and/or (b) reach the fixpoint more
easily / extrapolate further. Repeat the variable-vs-fixed cycle and the
test-time cycle-sweep experiments with FP weights to isolate the cause.

Feasibility: QLinear soft mode already has α=0 = identity (FP passthrough)
— so an FP HRM is a bounded change, NOT a rewrite. Caveats:
- **Keep CMuon, just drop the STE.** The STE only exists to bridge ternary
  quantisation (forward quantises to {-1,0,1}, backward passes the gradient
  straight through to a latent FP weight, CMuon updates the latent). FP
  weights have no quantiser, so apply CMuon directly to the actual weight
  matrices used in the forward — same cautious-Muon optimiser, same lr=0.20
  cosine. This holds the optimiser FIXED across the ternary-vs-FP
  comparison so **weight precision is the only variable** (do NOT switch to
  Lion/CAdamW — that confounds precision with optimiser). Needs an
  --fp-weights path in hrm_bop that feeds the FP weights to CMuon and skips
  quantise-in-forward.
- This is a **diagnostic / control line**, off the main fastest-*ternary*-
  recipe metric. Keep results in a separate segment; don't let FP val
  numbers contaminate the ternary leaderboard.

Sequencing: gated on Run 33. If the ternary [1,4] fixpoint extrapolates
cleanly past its training range, recurrence is NOT too brittle for
ternary and the FP control is lower priority. If ternary extrapolation is
poor/unstable, the FP control becomes the key next experiment.

### G. FP CMuon-LR sweep (follow-up to Run 34, gated on its result)
Run 34 early read (step 500): FP val 6.86 vs ternary 5.57 at matched
lr=0.20 cosine — FP trains slower, and its fixpoint over-smooths earlier
(collapse at cyc16 vs ternary cyc24; NaN by cyc48 from unclamped weight
growth). The slow start suggests lr=0.20 (tuned for ternary latents in
[-1,1]) is too hot for LeCun-scale FP weights. If Run 34 finishes below
ternary, DON'T conclude "FP is worse" yet — first sweep CMuon lr on FP
(0.05, 0.02, 0.01) to find FP's own optimum, then compare best-FP vs
best-ternary. Only then is the precision conclusion clean.
Also consider: add inter-iteration RMSNorm on z_H, or re-enable a
(wider) weight clamp for FP, to tame the cyc>16 over-smoothing/NaN — the
extrapolation-stability gap may be an artefact of unbounded FP residual
growth, not of precision per se.

## 2026-06-15 — User steer (captured during Run 41)

### H. Two-phase curriculum: fixpoint → reasoning specialisation
Hypothesis: a stable wide-basin fixpoint (now proven achievable with var
[1,4] training, e.g. r040 per-loop-gap=0.01) is the RIGHT initialisation
for "loops as iterative refinement" on hard problems. Phase A trains the
recurrence to be a clean refiner on generic data (per_loop_gap → 0, flat
test-time sweep on FineWeb). Phase B continues training on reasoning data
and asks: does per_loop_gap RE-OPEN in a useful direction — c4 < c2 < c1
on a reasoning val set, meaning the model has learned to USE the deep
loops for harder tokens?

Key design choices:
- **Variable [1,4] vs fixed-high in Phase B.** Variable preserves the
  fixpoint guarantee but gives no incentive to use deeper loops if shallow
  ones suffice. Fixed-high (h_cyc=4 always) forces all examples through 4
  cycles → rewards meaningful use of each iteration on hard examples, but
  risks reverting to the brittle fixed-cycle regime (per-loop-gap 0.39
  with no test-time scaling). Preferred: start with variable [1,4]; the
  cleaner test is "do loops self-organise into refinement under reasoning
  pressure" without forcing it.
- **Data mix.** Current is 70/25/5 FineWeb/Cosmopedia/OpenMath. Phase B
  should weight OpenMath heavily — 80/20 OpenMath/Cosmopedia (keeps some
  language) or 100% OpenMath (max reasoning signal). Pure-OpenMath risks
  catastrophic language drift; mix is safer.
- **Optimiser LR.** Phase A used lr=0.20 cosine→0.02. Phase B should
  fine-tune with lower peak (lr=0.05 cosine→0.005) to preserve the
  fixpoint while specialising — full lr=0.20 might shake the fixpoint
  loose.
- **Measurement.** Two things matter: (a) per_loop_gap signed — positive
  on reasoning val = model recruits depth for reasoning; (b) test-time
  cycle sweep on a held-out reasoning eval (math word problems, e.g.
  GSM-style) — if val_c4 < val_c2 < val_c1, test-time compute helps
  reasoning, a Big Result we can't get from FineWeb-only training.

Implementation cost: add a --data-mix CLI flag to hrm_data.py (currently
the 70/25/5 ratio is hard-coded). Resume mechanism for Phase A → Phase B
already exists: --resume-pt-weights from r040's final.safetensors keeps
weights and resets optimiser, which is exactly what we want for a
distribution shift.

Sequencing: queue after r041 completes (main-153M var [1,2] scaling
test). r042 = phase B fine-tune from r040 on OpenMath-heavy mix.

**Decision after user steer 2026-06-15**: cascade rather than commit.
- r042 = variable [1,4] (NOT fixed-high) on OpenMath-heavy data, fine-
  tune LR (lr=0.05 cosine→0.005), resume from r040's final.safetensors.
  Cheapest test of whether reasoning data alone can re-open per_loop_gap
  *positively* on a math-eval split.
- If r042 stays flat (c4 ≈ c2 ≈ c1 on reasoning val) → variable cycles
  alone don't generate the depth pressure. Then escalate to a per-loop
  exit gate (the bitnet+looplm pattern: small head decides "stop here?"
  per loop). That's a real code change (head + gate loss + scheduled
  expectation), ~1-2 days.
- If r042's reasoning still incoherent at fast-A scale → scale issue,
  not recipe. Reasoning use of loops likely needs 250M+, as in the
  user's prior bitnet+looplm work where 250M-with-exit-gate was still
  "not very coherent". Note that finding; don't keep scaling tweaks at
  fast-A.

Why variable not fixed in Phase B: our fixed-cycle ternary runs (r028,
r031 comparison) showed per_loop_gap=0.39 in fixed, but c1=c2=c4 at
test-time (the gap was scheduled refinement, not test-time-useful
compute). Even at extra-step budgets, r027 (1x1) matched r028 (2x3) on
val. Fixed cycles consume compute without test-time payoff. Variable
preserves the fixpoint and gives the model freedom to allocate depth
adaptively if data pressure is present.

Risk to manage: if reasoning eval shows DEEPER loops MORE WRONG (overfit
or over-smoothing under reasoning pressure), the fixpoint hypothesis
breaks down — escalate to exit gate, not fixed-high.

### Note: "The Topological Trouble With Transformers" (arxiv 2604.17121v3, 2026-06-15)
Position paper read mid r041. Core argument: depth-recurrence alone is
fundamentally insufficient for state tracking; each input step pushes
representations upward through layers, eventually exhausting depth. The
paper advocates step-recurrence (block-recurrent / fully recurrent
models that carry state ACROSS input positions) instead.

Direct mapping to our findings:
- Our HRM-text is purely depth-recurrent (no inter-token carrier state).
- Explains why r027 (1×1, no recurrence) matched r028 (2×3 fixed) on
  FineWeb val: depth-recurrence doesn't address what next-token CE on
  generic text rewards. The recurrence is "free refinement" the loss
  doesn't ask for.
- The paper cites that variable-depth models can succeed "with only
  fine-tuning, sometimes with no training at all", attributed to
  residual alignment across layers — exactly what our fixpoint training
  enforces by construction (c2..c24 flat). r042 (phase B fine-tune from
  r040 on OpenMath, lr=0.05) is therefore a clean test of this claim.

Predictions for r042 from the paper:
- **Optimistic**: residual alignment + reasoning pressure → per_loop_gap
  re-opens positively on OpenMath val sweep (c4 < c2 < c1 on math
  tokens). Corroborates the "alignment-enables-test-time-compute"
  reading.
- **Pessimistic**: paper predicts depth-recurrence can't substitute for
  state tracking; r042 may still show flat OpenMath sweep even with the
  reasoning curriculum.

Escalation path if r042 lands flat: do NOT add an exit gate (paper
suggests it won't fix the fundamental issue for per-sequence tasks).
Instead, prototype a **block-recurrent variant** — process seq_len in K
chunks with a carrier state (RNN-style) across chunks. Real code rewrite
(~1-2 weeks, new model class). Defer until r042 verdict is in.
