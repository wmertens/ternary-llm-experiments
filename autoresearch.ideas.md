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
- With FP weights the trit optimiser (CMuon-STE) no longer applies; train
  the weights with Lion/CAdamW instead. So the FP control is a genuinely
  different recipe, not a one-flag toggle — needs an --fp-weights path in
  hrm_bop that swaps the optimiser wiring.
- This is a **diagnostic / control line**, off the main fastest-*ternary*-
  recipe metric. Keep results in a separate segment; don't let FP val
  numbers contaminate the ternary leaderboard.

Sequencing: gated on Run 33. If the ternary [1,4] fixpoint extrapolates
cleanly past its training range, recurrence is NOT too brittle for
ternary and the FP control is lower priority. If ternary extrapolation is
poor/unstable, the FP control becomes the key next experiment.
