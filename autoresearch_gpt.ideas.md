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
7. **Bigger batch via grad accum** (bs=2 ga=16 → 32 / 64). Tests whether effective batch matters or step count matters. **User steers 2026-06-17**: g003 only used 2.5GB / 6.1GB → 3.3GB headroom NOW, but later phases (tritised embeddings, larger models, INT8 act buffers) will eat that. Use the headroom only for the wall-time experiment, NOT for stacked effective-batch growth:
   - 7a. **bs=4 ga=8** — effective batch stays 32, pure wall-time win (more sequences per forward, same per-step gradient). Run this FIRST after the LR sweep settles.
   - 7b. **bs=2 ga=32** — effective batch doubles to 64 with NO VRAM increase. Tests the effective-batch effect without burning headroom we'll need later. Quality effect, may need LR re-tune.
   - (Dropped 7c bs=8 — would assume VRAM headroom we may not have in later phases.)

### Phase 2 — Quantisation knobs
After Phase 1 stabilises the compute regime.

8. **Per-tensor BitNet-style scale** vs our per-(row, group) frozen. Strips 1.5M scale params to ~6 numbers (one per QLinear). Mirrors BitNet original design.
9. **INT8 per-token activations** (revisit Run 7 finding cleanly on GPT).
10. **Per-group scale granularity sweep**: 32, 64, 128, 256.
11. **Random scale variance**: σ sweep on the lognormal init.
12. **Trit init zero-frac**: 33% / 50% / 67%.
13. **Cautious mask off** as a control (Run 14 confirmed it helps HRM; reconfirm on GPT).

### Phase 3 — Architectural levers
After quantisation settles. **First entry promoted to g002 per user steer
2026-06-17.**

13a. **Q-K=V** — share weights of the K and V projections, keep Q separate
     (arxiv 2606.04032v2). Paper: +2.48% PPL at 1.2B, **50% KV cache
     reduction**, stacks with GQA/MQA and quantisation. Mechanism: K and V
     have natural cosine similarity 0.73 — the projection sharing absorbs
     a redundancy. For our fast-A geometry (num_kv_heads=8, head_dim=64):
     **1.57M fewer trits** (-8% of trit total) and ~8% less optimiser
     state across the 6 layers. RoPE is applied to K after the shared
     projection; V uses the pre-RoPE output of the same projection, so
     W_K = W_V but post-application K ≠ V. ~30 lines of code: add
     `share_kv: bool` to GptBopConfig + conditional in HrmAttention.
     **g002**: same baseline recipe as g001 with --share-kv. If PPL cost
     < ~+5%, this becomes the new baseline and propagates forward.
14. **Sandwich norm** (RMSNorm before AND after attention/MLP).
15. **Tie / untie lm_head**.
16. **Pre-LN vs post-LN**.
17. **GLU variants** in the MLP.

### Phase 4 — Speedrun-style
Switch metric to wall-time-to-target-val if Phase 1-3 results justify it.

### Phase 5 — "Only trits + block scales where needed" (user goal 2026-06-17)
End-state of the research line: **every weight tensor is ternary**, with FP
scale blocks present only where trit-alone resolution is provably
insufficient. Today's g001 baseline is 44.7M params, of which 25.2M
(56%) are FP — almost entirely the 49152 × 512 token embedding. The
Phase 5 experiments tritise each FP source in turn.

Run in parallel with Phases 1-3 (does not block them), but each Phase 5
experiment is paired with a matched non-Phase-5 control to isolate the
quality cost.

P5a. **Embedding → ternary, per-row FP scale.**
     49152 rows × 512 cols → 49152 trits per row + 1 FP scale per row =
     25.2M trits + 49k scales. Massive memory win. Risk: tokens whose
     row is mostly-zero trits become indistinguishable; per-row scale
     may need to be a (small) vector. Compare val on FW val: any > +0.05
     nats penalty triggers a per-(row, group) scale follow-up.

P5b. **Tied lm_head**: automatic if P5a works (current cfg ties weights).
     If untied, repeat the experiment for the output projection.

P5c. **RMSNorm weights → trits + per-tensor scale.** ~3.5K params, near
     free. Mostly a "does it break training" check, not a memory win.

P5d. **Scale precision sweep.** Currently fp32 scales. Try fp16, bf16,
     and int4-log-quantised with a per-tensor master scale. Tests how
     much FP precision the scale block really needs.

P5e. **RoPE table dtype.** Currently fp32 cos/sin tables. Cast to int8
     fixed-point (since |cos/sin| ≤ 1). Pure inference-time saving.

P5f. **Per-tensor BitNet scale revisited under Phase 5.**
     Phase 2 #8 with the additional pressure of tritised embeddings:
     does per-tensor scale on QLinears still hold up when embeddings are
     also coarse?

P5g. **Block sparsity** (later, after Phases 1-5a settle). Some blocks
     of the trit matrix might be all-zero across rows; collapse those
     to a "skip" flag. Memory + compute win at inference. Not a Phase 5
     blocker but the natural follow-up once trit density is the binding
     constraint.

P5h. **Re-evaluate share-kv at trit-emb regime.** g017 partial (step
     1500 truncated): trit-emb WITHOUT share-kv beat trit-emb WITH
     share-kv by -0.143 nats. Suggests coarse ternary embeddings need
     more K/V degrees of freedom than FP embeddings. Full 5000-step run
     + 10k extended needed before locking. Tradeoff: dropping share-kv
     costs 1.6M extra trits (-8pct savings reversed) for potentially
     -0.1 to -0.2 nat val improvement.

## Reference points from HRM line
- HRM fast-A champion: Run 25/40 — val 4.16 (5000 steps) / 4.00 (10000 steps) at this recipe.
- Likely GPT baseline floor: val ≈ 4.5-5.0 at 5000 steps (no recurrence means less effective depth; HRM ran the L_stack 6× and H_stack 2×, far more compute per token than a flat 6-layer GPT). Actual baseline TBD.

### Phase 6 — Architectural (queue once Phase 5 recipe is locked)

**User scope reminder (2026-06-22)**: end goal is fastest wall-clock to high-quality ternary on currently limited hardware, with planned scale-up. Don't dismiss architectural wins for being "scale-marginal at 43M" — if they compound at the eventual target scale, validating cheap at 43M is the right move.

P6a. **Variable-width "⊗-former" layers (arxiv 2606.18246v1).** Per-layer width follows an ⊗-shape (wide ends, ~30pct width at ~75pct depth). Residual stream stays at max d; each block reads/writes a slice with **carry-forward** copy of untouched coords (beats zero-pad/learned-projection). Reported -0.6 to -1.3pct loss, -4 to -9pct PPL, +1 pt NLU acc, param-matched (also -2-4.6pct FLOPs, -10.5pct KV). Validated 200M-3B; no quantization data. Composes cleanly with our stack (ternary QLinear/STE, RoPE, RMSNorm, SwiGLU, GQA with divisibility constraint on d_ℓ × n_kv). Per-layer scale buffers must be sized to d_ℓ. Implementation ~120-180 LOC + carry-forward state, ~6h. **Worth running at 43M even though below validated range** — likely small absolute gain here but the architectural change should compound at our scale-up target.

P6b. **Fixed-Point Reasoners (arxiv 2606.18206v1) — single-loop recurrence + adaptive halting.** Movahedi et al. show looped depth-recurrence DOES recruit deeper computation for reasoning at **7M params**, with:
- **Single weight-tied block, looped k times** (NOT HRM's H+L dual-stack). Paper explicitly outperforms HRM at this size — **park the HRM dual-stack design**.
- Per-layer learnable α₁,β₁ residual scalars
- Per-iteration learnable α₂,β₂ residual scalars
- **Adaptive halting at inference** via damped fixed-point solver (criterion ||z_i − f_θ(z_i;x)||∞ / ||f_θ(z_i;x)||∞ < 0.1)
- Training: fixed BPTT window of k iters, NOT variable-cycles supervision

Numbers vs TRM at 7M: Sudoku-Extreme 94.2 vs 74.7 (+19.5), A₅ state-tracking 98.1 vs ~65, ARC-1 47.5 vs 44.6, length-generalizes to 4× training. FP32, no quantization, but architecture transfers cleanly to ternary.

**Revises the parked HRM-line conclusion**: r043 phase B failure wasn't "depth-recurrence structurally insufficient" — it was the HRM dual-stack AND missing residual scalars AND adaptive halting AND tested on OpenMath (open-form) instead of symbolic-reasoning evals where FPRM's wins are. The "Topological Trouble" verdict was overgeneralized.

**User design questions (2026-06-22), worth experimental ablation:**
- **Pre/post non-looped layers**: should the first and/or last transformer layers be unlooped (different weights, applied once), with only the middle k layers being weight-tied and looped? Symmetric: input-conditioning prefix and output-decoding suffix outside the recurrence, computation in the middle. Easy ablation: --unlooped-prefix N, --unlooped-suffix N.
- **X-former + looped block = big BPTT win**: variable-width with narrow middle (~30pct) means the looped block has ~3x smaller activations than fixed-width. Full BPTT through k=8 iters of an ⊗-shape costs ~3x less activation memory than fixed-width. Combined with adaptive halting at inference, this could let us train deeper *effective* compute on the same 6GB VRAM. This is the multiplier — P6a × P6b together, not either alone.

Recommended re-entry sequence:
1. **Re-implement single-loop recurrence** (single weight-tied transformer block, loop k times, α/β scalars). Park HrmBopModel; build new LoopedGptModel reusing HrmDecoderLayer as the inner block. Test on Sudoku-Extreme.
2. If signal at 38M → add ⊗-shape width schedule to looped block (cheap activation memory, deeper k for same VRAM).
3. If signal at 38M → ablate unlooped prefix/suffix.

Implementation ~250-400 LOC (LoopedGptModel + α/β params + damped solver + Sudoku data + eval) ~2-3 days. Queue after current ternary recipe sweep closes.

**Queue priority**: Phase 6, after current sweep closes. Best paired with P6a (X-former) for the BPTT memory win.

## Queued-after-current-sweep (user steers 2026-06-23)

### Per-row scales at bigger model size (revisit g034)
g034 at 43M fast-A: per-row scales (81K) cost +0.131 nats vs per-(row,
gs=128) 350K — 4.3x storage win, +3% PPL fail at this scale. User
hypothesis: at bigger model, per-row gap may shrink because row
capacity becomes less restrictive relative to representation depth.
Worth testing on first scale-up (e.g. 8 layers, hidden=768, ~80M).
If gap shrinks to <+0.05 nats, adopt per-row as new baseline since
scale storage matters more in absolute terms at scale.

### Per-layer NS step count (Muon-spectra paper, arxiv 2606.04058v2)
g035 currently tests a global ns=3. The paper shows NS iter count
should be per-layer because spectral norm decay follows a power law
in model size with layer-type-dependent exponents (final output proj
needs more iters, mid-late layers fewer). Implementation:
- Per-QLinear `ns_steps` override, or
- Auto-schedule in CMuon: ns = base + extra_for_top_K_layers
For us: embed_tokens (49152×512) is much bigger than the rest (512×512
to 512×1408). Heuristic: assign higher ns to embed and output (if
untied), lower to mid-layers. Total NS FLOPs could go DOWN while
quality goes UP. ~30 LOC in cmuon.py + a CLI flag.

Both queued for after current Phase 5d/compute-dial sweep closes.

## Reference-only (not queued at current scale)

- **Muon momentum spectra at scale (arxiv 2606.04058v2)** — empirical
  power laws for the per-layer singular-value spectra of Muon's momentum
  buffer (77M-2.8B). Recipe variant: rank-p truncated Newton-Schulz
  (p=0.5 ≈ full Muon, p=0.25 -10 to 20% perf, p=0.1 -50%). Could cut NS
  iteration cost. **User correction 2026-06-23**: per-layer NS schedule
  is the actionable insight. Moved to Queued-after-current-sweep above.

## Open questions the loop can answer
- Is CMuon's optimum LR architecture-dependent or recipe-universal?
- Does per-tensor BitNet scale tie/beat per-(row, group) frozen on a non-recurrent model?
- Do optimizer-state precision savings hold up on fast-A scale GPT (HRM screening showed +0.011 nats for fp16-m at tiny scale)?
- Where does INT8-act sit on the wall/quality tradeoff for the actual fast-A GPT baseline?
