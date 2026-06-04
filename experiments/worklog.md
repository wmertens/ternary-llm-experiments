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

(Updated by the loop after each experiment.)

---

## Key Insights

(Updated when a run reveals something architecturally important.)

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
