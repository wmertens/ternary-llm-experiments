# Ternary text diffusion — future research

Parked direction, gated on: **ternary from-scratch training working first**.
Sequence: ternary from-scratch → ternary text diffusion → progressive
refinement. Do not start until the ternary recipe is solid.

## 2026-06-17 — User steer (captured during r042/r043)

### Why diffusion at all
Same bet as ternary: an inference-economics win, not a loss-floor win.
Diffusion's one *structural* (not just efficiency) advantage in the lit is
planning tasks — Dream-7B 81 vs Qwen-7B 21 on Sudoku, replicated on LLaDA.
Plausibly because global revision beats left-to-right commitment. If ternary
ever shows that edge too, the combination is the interesting story.

### Conversion is the cheap path (don't train diffusion from scratch)
AR cross-entropy = absorbing-state discrete diffusion as a special case
(left-to-right masking, T=N steps), so AR weights transfer.
- DiffuLLaMA/DiffuGPT (arxiv 2410.17891): anneal attention causal→
  bidirectional, keep AR "shift" alignment, drop time embeddings. ~30B tokens
  (GPT2-S), ~65B (LLaMA2-7B) — roughly 0.5–1× original pretrain budget.
- Dream-7B (2508.15487): init from Qwen2.5-7B, 0.6T tokens, matches LLaDA-8B
  (2.3T from scratch). Conversion is clearly the efficient route.
- LLaDA-8B (2502.09992): from-scratch counterexample; beats LLaMA3 on math+zh.
Implication for us: a ternary diffusion LM probably starts as a *converted*
ternary AR checkpoint, not a fresh run.

### I. Progressive-refinement block diffusion via `<refine>` references
User's core idea. Build on Block Diffusion (BD3-LM, arxiv 2503.09573, ICLR
oral) — diffuse within a block, stream blocks L→R with KV cache, arbitrary
length.

Mechanism:
1. Assume model already emits null/pad tokens for unused canvas space.
2. Train the model to emit a `<refine>` token inside text spans that are too
   complex to fit/resolve in the current block. Treat each `<refine>` as a
   *reference/placeholder*.
3. Next block: append the just-completed block to context, then generate the
   expansion for each `<refine>` reference (the placeholder's content).
4. Present to the user with references recursively filled in.
5. A `<refine>` at the *end* of a block degenerates to plain continuation —
   i.e. exactly BD3-LM's streaming. So end-refine and inline-refine are the
   same primitive; continuation is the boundary case.

This is the capacity-triggered recursive expander that the lit review found
**does not yet exist as learned end-to-end behavior**. Closest partials:
- LR-DLLM (2602.07546, claimed): length-as-explicit-variable at inference —
  capacity-aware but length only, not hierarchical, and post-hoc not learned.
- PLANNER (2306.02531, Apple): latent plan → AR decode; fixed-width plan, not
  recursive, not capacity-triggered.
- Diffusion-in-Diffusion (2601.13599, claimed): draft → remask low-confidence
  → global re-diffuse. Coarse-to-fine within diffusion; −22% gen-ppl at 26%
  budget. The refinement primitive, without the reference/outline semantics.
- RecurrentGPT (2305.13304), recursive book summarization (2109.10862):
  scaffold versions — hierarchy externalized in a harness, not learned.

### Guardrails (negative results already on the board)
- **Outline-then-expand failed human eval once already** (hierarchical-outline
  gen, arxiv 1810.08802, 2018): improved perplexity, NOT subjective quality.
  Whatever we build must be judged on output quality, not just ppl/CE.
- **Depth-recurrence ≠ state tracking** (Topological Trouble, 2604.17121; our
  own r027≈r028 result). `<refine>` references carry state ACROSS blocks =
  step-recurrence (the right axis), NOT HRM-style depth-recurrence (the wrong
  one). The reference-as-carrier-state framing is what makes this defensible
  where HRM looping wasn't. cf. COCONUT (2412.06769) feeding last hidden state
  back as next-input embedding — the step-recurrent precedent.

### Open training-signal question (the hard part)
How do you supervise "emit `<refine>` when the span is too complex / won't
fit"? Needs a mixed-granularity corpus where the same content appears both
as a flat passage and as outline+expansion, with refine boundaries labeled.
No published recipe. Meta-CoT's linearized-search-trace supervision
(2501.04682) is the nearest paradigm for "learn to recognize when the direct
approach fails." Likely the first real subproblem to solve before any model
work.

### Minimal first experiment (when unparked)
Convert a small ternary AR checkpoint to block diffusion (DiffuLLaMA recipe),
add `<refine>` as the only new token, supervise on a synthetic mixed-
granularity set (programmatically generate flat ↔ outline+expansion pairs —
e.g. nested lists, structured docs where ground-truth hierarchy is free).
Success metric: does refine-expansion beat flat generation on *quality* (not
ppl) at matched compute, AND does the per-block sweep show state actually
carried across the reference boundary (the step-recurrence test r042/r043
is probing on the AR side).
