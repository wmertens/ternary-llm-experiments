# Autoresearch Ideas Backlog — Ternary NanoGPT line

Sibling research to the HRM autoresearch arc. HRM is parked at r044
(autoresearch.jsonl segment 0). This line uses vanilla decoder-only GPT
to isolate ternary training speedups from any recurrence confound.

Outputs live in:
- `experiments_gpt/g<NNN>-<tag>/` — checkpoints, train.log
- `tb_gpt/g<NNN>-<tag>/` — TensorBoard scalars
- `autoresearch_gpt.jsonl` — structured state

## Protocol
- **Metric**: val_loss on FineWeb-Edu sample-100BT held-out (16 batches × bs=2 × seq=1024).
- **Fixed compute budget**: 5000 steps unless explicitly noted otherwise.
- **Baseline (g001)**: 6-layer fast-A-sized ternary GPT, CMuon-STE + cautious + lr=0.20 cosine, per-(row, group) frozen lognormal scales, freeze non-embed FP. Same data mix (70/25/5 FW/Cosmo/OpenMath).
- **Inheritance from HRM line**: champion recipe (Run 25 → Run 40) transferred wholesale. Everything else is "what speedup can we get on top of this?".

## Sequencing (priority order, per user steer 2026-06-16)

### Phase 1 — Compute dials (do these first)
Knobs that change the effective work per step or the optimisation regime, with no architectural changes. Each ~5h on fast-A.

1. **g001** — Baseline (above).
2. **CMuon momentum dtype**: fp32 → bf16 → fp16+SR. We saw a +0.011 penalty for fp16-m in HRM Run 8 (tiny screening); revisit with CMuon-STE on the GPT baseline.
3. **Optimizer state offloading** (Lion + CMuon state on CPU, copy on-step). Trades wall-time for VRAM headroom; if the headroom isn't needed, just measure the cost.
4. **CMuon LR sweep** on the GPT baseline. HRM hit 0.20 optimal at fast-A; check whether GPT has the same optimum or wants something different (no recurrence = different gradient norm regime).
5. **LR floor / cosine shape** sweep: cosine→0.02 / →0.01 / →0.05; linear decay / inverse-sqrt comparison.
6. **Warmup** sweep: 0 / 100 / 400 steps.
7. **Bigger batch via grad accum** (bs=2 ga=16 → 32 / 64). Tests whether effective batch matters or step count matters.

### Phase 2 — Quantisation knobs
After Phase 1 stabilises the compute regime.

8. **Per-tensor BitNet-style scale** vs our per-(row, group) frozen. Strips 1.5M scale params to ~6 numbers (one per QLinear). Mirrors BitNet original design.
9. **INT8 per-token activations** (revisit Run 7 finding cleanly on GPT).
10. **Per-group scale granularity sweep**: 32, 64, 128, 256.
11. **Random scale variance**: σ sweep on the lognormal init.
12. **Trit init zero-frac**: 33% / 50% / 67%.
13. **Cautious mask off** as a control (Run 14 confirmed it helps HRM; reconfirm on GPT).

### Phase 3 — Architectural levers
After quantisation settles.

14. **Sandwich norm** (RMSNorm before AND after attention/MLP).
15. **Tie / untie lm_head**.
16. **Pre-LN vs post-LN**.
17. **GLU variants** in the MLP.

### Phase 4 — Speedrun-style
Switch metric to wall-time-to-target-val if Phase 1-3 results justify it.

## Reference points from HRM line
- HRM fast-A champion: Run 25/40 — val 4.16 (5000 steps) / 4.00 (10000 steps) at this recipe.
- Likely GPT baseline floor: val ≈ 4.5-5.0 at 5000 steps (no recurrence means less effective depth; HRM ran the L_stack 6× and H_stack 2×, far more compute per token than a flat 6-layer GPT). Actual baseline TBD.

## Open questions the loop can answer
- Is CMuon's optimum LR architecture-dependent or recipe-universal?
- Does per-tensor BitNet scale tie/beat per-(row, group) frozen on a non-recurrent model?
- Do optimizer-state precision savings hold up on fast-A scale GPT (HRM screening showed +0.011 nats for fp16-m at tiny scale)?
- Where does INT8-act sit on the wall/quality tradeoff for the actual fast-A GPT baseline?
