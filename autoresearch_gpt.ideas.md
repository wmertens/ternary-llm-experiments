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

**Decision pinned 2026-06-28**: build P6b *immediately after g051 (8L 30k) completes*. That run locks the flat-recipe ceiling as the baseline FPRM has to beat on language-modeling val loss, then we pivot to the build (~2-3 days). At that point, kill the loss-sweep cadence — FPRM evaluates on Sudoku-Extreme + A₅ state-tracking, not fineweb perplexity.

**Steer 2026-06-28**: BEFORE FPRM, run the latent-precision sweep below — un-played knob, cheap to try (recipe-only changes, no new code beyond a CLI flag), and the answer informs whether FPRM's BPTT memory budget gets even cheaper.

## P5b — Latent precision sweep (queued before P6b)

**Motivation**: trit latents are 63.7M fp32 params (~255MB). If bf16 latents give equivalent loss curves we save ~125MB and free VRAM for bigger effective batch / deeper FPRM unroll. CMuon optimizer state is already bf16; latents themselves are still fp32 by default. Unknown whether the small gradient updates the STE needs survive bf16's 7-bit mantissa.

Queue order (each ~5k probe, compare to g045 8L 5k baseline 4.2390; promote winners to 10k):
1. **bf16 latents only** (scales fp32, opt state stays bf16). Wall: same as baseline. Quality: ?
2. **bf16 latents + bf16 scales** (full bf16 trainable params). Maximally lazy.
3. **If (1) muddled**: per-block-type dtype sweep — embed-trit / attn-trit / MLP-trit / scales / final-norm individually as fp32 vs bf16 vs fp16. Identifies which block tolerates low precision and which doesn't.
4. **8-bit latents** (int8 with fp scale, custom): only if (1)/(2) shows quality survives precision drop — then we know there's slack to push further.

Implementation: ~30-60 LOC for `--latent-dtype {fp32,bf16,fp16}` flag in `gpt_bop` + dtype cast in QLinear `__init__`. Per-block-type would be an extra ~50 LOC of per-module dispatch.

Closes when: (a) bf16 latents within +0.02 nats of fp32 → adopt bf16 as new baseline before FPRM build; or (b) any precision drop costs >0.05 nats → fp32 stays, FPRM builds on fp32 latents.

## P5c — Optimizer state quantization (queued after P5b)

ml-research survey 2026-06-30 returned three actionable items, ranked:

**P5c.1 — Lion scales bf16 momentum (FREE LUNCH, zero code)**. Use existing `--scales-dtype bfloat16`. Lion's `sign(beta·m + (1-beta)·g)` rule depends only on the EMA *sign*, not magnitude precision — so bf16 momentum is theoretically the easiest sub-fp32 step among all known optimizers. No paper needed; first-principles argument is unambiguous. Run 8L 5k as a free check vs g045 fp32-scales baseline; if within +0.02 nats, adopt.

**P5c.2 — MuonQ int4 state (arxiv 2605.11396, May 2025)**. Drop-in for CMuon: replaces stored bf16 momentum with int4 via three tricks — (a) per-step gradient normalization so quantization error is direction-uniform, (b) power-iteration to separate dominant singular components and quantize *magnitudes* only, (c) mu-law companding for dense central region. Paper claims ~0 quality cost on GPT/LLaMA pretraining. NS5 runs in fp32 anyway in our CMuon (cmuon.py:141), so int4 only affects the stored buffer. Cuts CMuon state 4x further (bf16 0.5MB/100M→int4 0.13MB). Code: github.com/YupengSu/MuonQ. ~200-400 LOC integration.

**P5c.3 — Theory anchor (arxiv 2510.21314)**. Validates direction: Adam needs tight mantissa-bound second-moment quantization due to β₂→1, but Muon's NS5 acts as a quantization buffer — published convergence rates hold with logarithmically-shrinking mantissa. Cite this in the MuonQ PR.

**Queue order after P5b**: 5c.1 first (zero-code probe), then 5c.2 if 5c.1 confirms Lion's robustness (~2-day commit), then FPRM (P6b).

**Fork audit 2026-07-01** (precision-knob triangulation): three gaps to close before trusting the bf16 verdict. Precision-lever inventory: `--autocast-dtype` (bf16, HIGH conf ✓), `--latent-dtype` (bf16, MODERATE-HIGH after user 07-01 confirm re: g052 resume), `--scales-dtype` (fp32 by g053 rejection of bf16, but fp16 untested — see below), `--cmuon-state-dtype` (bf16, HIGH conf, NS5 always upcasts to fp32 per cmuon.py:141), Lion32 momentum (fp32 hard-coded, NO flag exists).

**User steer 2026-07-01 (priority)**: run **fp16-scales probe FIRST** — shortest of the three, and if it wins it directly helps wall-time via halved scale memory (potentially faster forward on fp16-fast hardware). Reordered gap queue:
1. **fp16 scales** (g05x, ~3h, no code change — `--scales-dtype float16` already exists)
2. **fp32-latents control** (~2.5h)
3. **Lion32 `--lion-state-dtype` flag + bf16 momentum probe** (~30 LOC in distill.py:303/349/405 + 3h)
4. Then MuonQ (P5c.2), then FPRM (P6b).

Original gap descriptions:
- **P5b-control**: fp32-latents 8L 5k has never been run. Default was fp16; we jumped to bf16 without triangulating against fp32. The g052 -0.098 win vs g045 may be a mid-run resume artifact (g052 got SIGINT-resumed at step 1500 with lr_snapshot re-anchor). g054 midpoint gap of only -0.017 supports that concern. Run g055 = 8L 5k fp32-latents (~2.5h) as the missing control before scaling bf16 further.
- **P5c.1 correction**: --scales-dtype (which g053 tested) changes scale STORAGE, not Lion32 momentum. The survey's actual free-lunch is Lion32's internal momentum buffer dtype. Add --lion-state-dtype flag (~30 LOC in distill.py:303/349/405 where Lion32 currently hard-codes fp32 momentum), then re-probe. This is the actual P5c.1 that got misfired.
- **P5b-scales-fp16**: g053 tested bf16 scales (7-bit mantissa) which lost +0.22 nats. **fp16 scales never tested** (10-bit mantissa, 5-bit exponent). Per-(row, group) scales cluster near 0.01-0.1 — modest dynamic range demand, mantissa resolution matters more. fp16 scales should sit close to fp32 while halving scale memory. Probe: 8L 5k --latent-dtype bfloat16 --scales-dtype float16, ~3h. Bar within +0.02 of g052 (4.1409) → adopt fp16 scales as the memory lever, not bf16.

**Out of scope on this hardware**: FP8-LM (H100-only matmul), bnb 8-bit Adam (we don't use Adam), GaLore (conflicts with NS5), APOLLO/Shampoo (Adam-family alternatives).

**KNOWN GAP — SF-Lion resume untested**: ScheduleFreeWrapper stores averaging state (the `z` buffer in self.state, plus `k`/`weight_sum`/`lr_max` in param_groups, plus train/eval-mode which determines whether model params are the y-iterate or x-average). `_save_resume` calls `opt_lion.state_dict()` = `base.state_dict()`, which captures state+param_groups but NOT `train_mode`, and the saved MODEL params depend on which mode was active at save time. Until verified round-trip-correct, SF-Lion runs must complete WITHOUT SIGINT/resume (restart fresh on interruption). g058 was restarted fresh after a system crash for exactly this reason. Fix if SF-Lion proves worthwhile: save/restore train_mode + ensure eval() (x-iterate) before checkpoint so saved params are deployment-ready and resumable. ~20 LOC.

## P5d — LR schedule: WSD (Warmup-Stable-Decay, arxiv 2404.06395)

**Motivation from g051 failure**: 30k cosine (38.5h wall) landed 3.7030 vs g048 20k cosine (8.9h wall) 3.7272 — 4.3x wall for -0.024 nats. Postmortem revealed the tail-end floor-LR fine-tune (steps 28k-30k) is where most of the win materialized; the mid-run high-LR phase (5k-25k) largely spun wheels. This is exactly the failure mode WSD is designed to fix.

**Recipe**: warmup ~200 → constant peak LR until val slope < ε over 2 consecutive checkpoints → 15% cosine (or linear) anneal to floor.

**Why it wins over cosine**: (a) no pre-committed step budget — val-plateau-detect decides when to anneal; (b) if plateau comes earlier than expected, save the compute; if later, keep running at productive LR instead of being stuck in the cosine tail; (c) MiniCPM/Deepseek/Hu 2024 all report WSD matches or beats cosine at equal compute with the flexibility to extend.

**Implementation**: ~50-80 LOC in gpt_bop.py — add `--schedule {cosine,wsd}` flag, `--wsd-anneal-frac 0.15`, `--wsd-plateau-tol 0.005`, `--wsd-plateau-window 2`. lr_snapshot works unchanged (WSD is stateful in a different way: which phase, val-history buffer).

**Queue priority**: HIGH — orthogonal to the precision-lever sweep and directly attacks our biggest weakness (guessing total-steps up front). Run once after fp16-scales / fp32-latents / Lion-momentum probes close. First WSD run: same recipe as champion (8L bf16-latents), let val-plateau decide when to anneal, target beating g054 at less wall.

**RESULT 2026-07-06**: WSD wins at 5k (g056 4.0867 beat cosine 4.14) but LOSES at 20k (g060 3.8063 vs g048 cosine 3.7272). Fixed 15% anneal-frac makes the stable phase too long at 20k — loss plateaued ~4.16 for steps 11000-17000 (wasted compute at peak LR). Two fixes, both DEFERRED (cosine stays champion schedule; pursue only if WSD becomes worth revisiting):
- **anneal-frac scaled to horizon**: at 20k, anneal-frac ~0.45 would start the anneal at the plateau onset (~step 11000). One-flag test.
- **plateau-triggered anneal** (the proper WSD): track val slope, start anneal when it flattens over K checkpoints (--wsd-plateau-tol, --wsd-plateau-window). ~40 LOC. This is what MiniCPM/DeepSeek actually do; the fixed-fraction version is a simplification that only works when the fraction happens to match the plateau point.

**Alternatives worth considering (fork survey 2026-07-01)**:
- **Schedule-Free** (Defazio et al., arxiv 2405.15682). No schedule at all — uses a weighted iterate average with a specific momentum formula that provably matches cosine-tuned baseline. Ships in `schedulefree` package. Ideal fit for Lion on scales (drop-in replacement). Weaker composability with CMuon (Muon's orthogonalization changes momentum semantics). Try: Schedule-Free-Lion for scales while keeping CMuon+cosine or CMuon+WSD for trit latents.
- **AdaLRS** (arxiv 2506.13274, June 2025). Multiplicative velocity-monitor overlay on cosine/WSD with rollback-on-failure. Fork deep-read verdict 2026-07-01: **skip for our regime**. Two of three published failure modes match us — "well-tuned baseline + slight loss spikes" and "aggressive α,β on short training → oscillations". Paper's k window is 1000-2500; our 5k-30k runs would need k≤200 (untested). Rollback snapshots (~313MB each) don't fit our 6GB VRAM. No released code — 2-3 day implementation risk on us. Revisit only if we scale to 60k+ runs.
- **Prodigy** (arxiv 2306.06101) — no LR to tune, but doesn't compose with orthogonalized momentum (skip for CMuon).

## P5e — Plateau-triggered LR warm-restart with rollback (user steer 2026-07-06) — NEXT SCHEDULE EXPERIMENT

**Idea (user)**: fixed WSD flat/anneal is too simple. We have EXACT full-state resume (model + all optimizer state in interrupted.pt) and preserve-mode schedule handling. Exploit it: detect a long-term val plateau → snapshot → re-anchor the LR to a *slightly spicier* (higher) point and restart a cosine from there → observe. If val rises over a decent window, REVERT to the snapshot (exact rollback) and either anneal or lower the restart peak. If val improves, keep the escape and continue.

**Why it beats WSD/cosine**: cosine and fixed-WSD are open-loop — they commit a schedule up front and can't react to a plateau. g060's failure was exactly this: 6000 steps stuck at ~4.16 because peak LR had extracted all it could and the fixed schedule wouldn't anneal yet. A warm restart re-injects LR to knock the model out of the basin; the val-gated rollback makes it *safe* (worst case = revert to the plateau snapshot, cost = the observation window). This is SGDR warm-restarts (Loshchilov & Hutter 2017, arxiv 1608.03983) BUT plateau-triggered and accept/reject-gated on validation — the gating is what SGDR lacks and what makes it safe on a tuned recipe.

**Mechanism**:
1. **Plateau detect**: track val loss over a sliding window of W checkpoints; if best-in-window improvement < ε (e.g. slope flatter than 0.005 nats over 4 vals), declare plateau.
2. **Snapshot**: save best-so-far full state as `restart_anchor.pt` (we already write interrupted.pt with full opt state — reuse that machinery; keep a separate best-anchor copy).
3. **Rewarm**: re-anchor LR to `spice × current_LR` (start modest, e.g. spice=2-3×, NOT back to full peak — "slightly spicier"), begin a fresh short cosine over the observation window.
4. **Observe** N steps (e.g. 1000-2000).
5. **Accept/reject**: if best val in the observation window < snapshot val by margin → ACCEPT (plateau escaped), continue from the improved state, update anchor. Else → REVERT to `restart_anchor.pt` (exact model+opt rollback), and either (a) retry with lower spice, or (b) if retries exhausted, trigger the final anneal to floor.
6. **Caps**: max K restart attempts; each accepted restart resets the counter; each rejected one decrements spice.

**Implementation** (~120-180 LOC): plateau detector on the val history (already have val loop), a best-anchor save/load (reuse _save_resume + a copy), LR re-anchor (reuse the existing re-anchor path — this is what re-anchor mode was BUILT for), rollback = load the anchor mid-run. Add flags: `--restart-on-plateau`, `--restart-spice 2.5`, `--restart-observe-steps 1500`, `--restart-plateau-tol 0.005`, `--restart-plateau-window 4`, `--restart-max-attempts 3`, `--restart-accept-margin 0.01`. Rollback fits our 6GB (one extra 312MB anchor on disk, not VRAM — unlike AdaLRS which needed in-VRAM snapshots).

**Queue priority**: HIGHEST schedule experiment — supersedes the two deferred WSD fixes above. Run after g061 (X-former champion) resolves. First test: 8L 20k with restart-on-plateau, compare vs g048 cosine 3.7272 — the plateau that killed g060 is the exact case this should convert into a win. Note: this composes with cosine base schedule (not WSD); the base is cosine, the restarts are the adaptation layer.

**REFINEMENT 2026-07-08 (user): rewarm-UP vs drop depends on WHERE the plateau sits in LR-space.**
- Cosine-tail plateau = stuck at LOW LR (decayed too far) → escape by REWARMING up (P5e as built, spice>1). Correct pairing.
- WSD stable-phase plateau (g060: ~4.16 for 6000 steps) = stuck at HIGH/peak LR (never dropped) → escape by DROPPING LR early (anneal), NOT rewarming. That's plateau-triggered-anneal, a different action from P5e-rewarm. So do NOT pair P5e-rewarm with WSD; pair it with cosine.

**g062 status + snapshot-seeded sweep plan (user 2026-07-08)**: g062 (8L 30k cosine + P5e, tol=0.008) fired ZERO rewarms through step 24000 — cosine descends too smoothly (~0.01/500) to ever hit the 0.008/2000-window trigger; dead-even with g051 plain (step 20000: 3.9291 vs 3.9294), heading to ~3.70. Plan: let g062 RUN TO COMPLETION (confirm clean bf16-latents 30k stats + get the end-of-run interrupted.pt). Then g062b/c/… = RESUME from that snapshot (model sits at floor LR = a genuine low-LR plateau, the ideal rewarm seed) with adjusted P5e params (looser tol so it fires immediately, sweep spice/observe), running only a few k extra steps each. Cheap iteration — tests whether a rewarm from the converged floor escapes below 3.70, without a fresh 30k per config. Resume is cheap now (fresh-stream fix). Requires bumping --total-steps past 30000 on resume (re-anchor path handles the schedule; P5e override drives the rewarm LR).

## PRE-CHAMPION-RUN BLOCKER — fix lr_snapshot to preserve original cosine

**Cause of g054's null result**: lr_snapshot is memoryless. On resume it treats snapshot LR as a new "peak" and re-cosines from there over remaining steps to floor. Multiple SIGINTs concatenate multiple shorter cosines with lower-and-lower peaks; composite schedule spends less wall at productive mid-tail LR, more at near-floor. Correct only for "extend past original schedule end" mode; wrong for "clean mid-run continuation" mode. g054 got hit twice, plateaued at 3.77 instead of the projected 3.69.

**Fix (~15 LOC)**: store `(peak_lr, original_total_steps, floor_frac, step_offset)` in interrupted.pt. Add `--resume-schedule {preserve,re-anchor}` flag; default `preserve` for pure mid-run resumes, keep `re-anchor` for the extend-past-end case. On resume with `preserve`, compute `lr_at(current_step)` from stored originals — same math a fresh run would use.

**Why blocker for champion runs**: X-former + long schedules are the highest-signal experiments. Any run that gets SIGINT'd gets silently sabotaged by the current re-anchor. Champion attempts must survive resumes without schedule contamination.

**Must land before g056 (X-former 8L 20k) and any future multi-hour run.**

## P6a-revisit — X-former with champion recipe (fork audit 2026-07-01)

**Gap**: X-former was closed prematurely at 6L fp16 baseline (g042/g043/g044). Never re-tested against the current 8L + bf16-latents + 20k champion recipe.

**What g042 showed at 6L fp16 5k**: 4.1487 val, 39M params (-24% vs 6L flat 51.3M), wall 14009s. Tied on val with g052 (8L flat bf16 5k = 4.1409) at -24% params.

**Compound stack never tried**: X-former ⊗ profile at 8L (e.g. [512,384,256,256,256,256,384,512]) + bf16 latents + 20k schedule. Each of the three levers landed a real win alone; product-of-wins likely lands champion. Projection: if g042 5k → g054-equivalent 20k tracking holds with -24% params and -0.10 bf16 improvement, lands ~3.55-3.65 at ~7h wall (beats g048 champion 3.7272 at less wall).

**Queue priority**: HIGHEST post-g054. Ahead of P5b-scales-fp16, fp32-latents control, Lion-momentum flag — because X-former alone is a bigger swing (-24% params + wall) than any single precision knob. Confirm g054's flat-arch bf16 20k first (~5h remaining) as the comparison baseline, then g055 = X-former 8L bf16 20k as champion attempt.

**g061 2026-07-07 (too narrow, NOT closed)**: X-former ⊗[512,384,256,256,256,256,384,512] 8L 20k cosine, bf16-latents+fp16-scales, 41.7M params. val 3.9033 vs g048 flat 3.7272 = +0.176. Capacity-limited: tracked +0.06 behind flat to step 15k, then PLATEAUED at ~3.90 while flat descended to 3.7272. The 256-wide middle (4/8 layers at HALF width) saturates at convergence. +24% throughput / -19% params can't offset +0.176 nats.

**User steer 2026-07-07: don't close X-former — make it LESS narrow at small scale.** The failure was over-narrowing, not the ⊗ idea. Narrowing degree should scale with model size: a 51M model can't spare 4 layers at half-width, but a gentler taper preserves capacity while still saving some params/wall. Next X-former variant to try (after P5e): a shallow ⊗ that stays ≥384 in the middle, e.g. [512,448,384,384,384,384,448,512] (widths must be multiples of head_dim=64 AND scale_group=128 → valid: 512,384; 448 is mult of 64 but NOT 128, so use [512,384,384,384,384,384,384,512] = min-width 384, ~46M params, -10% vs flat). Bar: within +0.05 of g048 at less wall. The g042 5k "tie" and g044 gentle-6L both under-tested this at convergence; a 384-floor 8L 20k is the real gentle test. Only truly close X-former if even a 384-floor taper loses at convergence.

## lr_snapshot: preserve-vs-re-anchor fix (~15 LOC, blocker for 20k+ champion attempts)

**Problem** (fork 2026-07-02): current lr_snapshot is memoryless — on resume it treats the saved LR as a fresh cosine peak over `total_steps - current_step` remaining. For a single SIGINT this loses the mid-tail LR-time; for two SIGINTs (g054) you get three concatenated shortened cosines, each with its own near-peak start + near-floor end. Composite spends more wall at extremes than a clean cosine.

**Fix**: store `{peak_lr, original_total_steps, floor_frac, step_offset, warmup_steps}` in interrupted.pt. On resume, compute `lr_at(current_step)` from those originals instead of treating snapshot as new peak. Add `--resume-schedule {preserve,re-anchor}` flag; `preserve` = default for mid-run resumes, `re-anchor` = current behavior for extending past original schedule end (where new peak is correct).

**Why blocker**: g054's 3.7742 result is confounded by this — can't isolate "bf16 latents don't compound at 20k" from "double-SIGINT distorted the schedule." Any future 20k+ run with SIGINT contact will have the same artifact. Cheap fix (~15 LOC) that unblocks clean champion-tier data.

**Queue priority**: BEFORE any 20k+ champion attempt (X-former, WSD, or otherwise). ~30min implementation.

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

## P6c — State-Prediction Separation (SPS) dual-stream (arxiv 2607.01218, user steer 2026-07-13)

**Paper**: "The State-Prediction Separation Hypothesis" — Monea, Godey, Brantley, Artzi (Cornell/lil-lab). Code: github.com/lil-lab/sps. Standard Transformers conflate two roles in one forward pass — maintaining state for FUTURE predictions vs predicting the NEXT token. SPS splits them into a dual-stream decoder. Claims: consistent data+compute-efficiency gains across scales, +2-3pp downstream, better val loss.

**Why promising for us**: architectural (orthogonal to quantization) → composes with the ternary QLinear/STE/CMuon stack, same as X-former. "Compute efficiency across scales" fits the fastest-wall-clock + scale-up goal. User flagged very promising 2026-07-13.

**Mechanism (ml-research 2026-07-13, repo github.com/lil-lab/sps)**: NOT two parameter sets — one shared-weight Transformer on a sequence-DOUBLED input. Interleave a learned `<predict>` token after every input token: (x1,p1,x2,p2,...) so T→2T, both xi and pi share the SAME RoPE position id i. Every block (attn+SwiGLU+RMSNorm, identical to standard) runs on all 2T positions with the SAME weights. Only change vs standard causal Transformer = the attention mask: INPUT-slot K/V globally visible + persistent (normal causal, stays in KV cache); PREDICT-slot K/V visible only in a sliding window w (default 64, and w=0 barely hurts per Fig4) then evicted. Next-token loss computed ONLY at predict positions (lm_head(x[:,1::2])).

**Overhead**: ZERO extra params. ~2x TRAIN FLOPs + ~2x train activation memory (2T positions through same weights, both streams carry gradient). Inference is train-time-cost-only: 1.01x memory, 0.94x throughput. On our 6GB card the 2x train activation memory means halving bs (4→2) or seq (1024→512) — feasible for 8L/512.

**Results at OUR arch**: their XS config = 8L, hidden 512, 8 heads, head_dim 64, FFN 1536 = essentially our target. 20B tok FineWeb-Edu, AdamW. SPS vs STANDARD at 53M: val NLL 3.065 vs 3.107 (-0.042), OOD corpus NLL -0.092, zero-shot +2.5pp. Backed at <200M (also S=131M -0.048/+2.3pp), gap WIDENS with scale. Ablations rule out capacity/compute-alone: DELAYED_STATE (same 2x compute, no separation) gets only half the gain; 2X_MEMORY (more compute) is worse than SPS. Per Fig2 SPS reaches STANDARD loss on ~half the tokens → net GPU-hours favorable despite 2x/step.

**Ternary composability**: NO blockers. SPS touches only mask + sequence layout; every linear (c_attn/c_proj/gate/up/down) unchanged in shape+role → ternary QLinear/STE drops in exactly. RoPE/RMSNorm unaffected (kept FP as usual). One UNTESTED flag: SPS Fig5a shows systematically asymmetric per-stream gradient magnitudes; untested with Muon orthogonalization (normalizes update direction not magnitude — likely fine, not a known incompatibility).

**Implementation (~1-2 days, NO Triton at our scale)**: reuse existing RMSNorm/SwiGLU/RoPE/QLinear unchanged. Add: token-interleave + doubled RoPE ids (~15 LOC), a torch flex_attention mask_mod = causal + predict-windowed/input-persistent (~20 LOC; adapt repo `_sliding_window_reference.py::window_score_mod_factory`), read-logits-from-odd-slots loss (~10 LOC). Validate the mask against repo reference on a toy 2L model first (their test_dense_only.py pattern), then a short 1-2B-tok FineWeb-Edu A/B at 50M.

**Queue**: HIGH-value Phase-6 architectural bet (composes with ternary, backed at our exact 53M/8L arch, cheaply falsifiable). Run after the fp16-vs-bf16-latents precision decision (g066) closes.

**STATUS 2026-07-14: IMPLEMENTED + smoke-tested.** smollmer/gpt_model.py: build_sps_mask (validated causality/persistence/ephemerality) + GptBopModel._forward_sps (interleave predict tokens, doubled RoPE ids, SPS mask via SDPA attn_mask, read logits from odd slots → returns [B,T,vocab] drop-in). hrm_model.py: optional attn_mask threaded through HrmAttention/HrmDecoderLayer (None=causal, HRM untouched). gpt_bop.py: --sps + --sps-window flags. Smoke-tested on CPU with --ste-trits --c-muon: loss descends, no crash. NOTE: bare-model causality test is invalid (QLinear is input-independent until the trainer's STE setup runs) — validate via the trainer, not a bare GptBopModel. NEXT RUN g068 = SPS A/B at 8L/512 fp16: --sps --batch-size 2 (the ~2x train activation memory needs bs 4→2 on 6GB) vs a bs=2 flat baseline at matched tokens/step. Bar: SPS beats flat by the paper's ~-0.04 val NLL at our arch.


**SPS A/B RESULT 2026-07-16 (g068 vs g069)**: at 53M/8L/10k/bs=2, SPS 4.0902 vs matched flat 4.0872 = PARITY (+0.003 noise) at 1.9x wall. The paper's -0.04 val NLL advantage did NOT appear at our small scale + 10k (~0.3B tokens vs their 20B). SPS is SCALE-GATED (paper's own gap grows XS-0.042→XL-0.068 and needs long training). Implementation is validated + committed and ready. RETRY only at much larger model or token budget. Champion recipe stays flat.

## INT8 activations — interpretive frame (fork 2026-07-16) + deep-anneal caveat

Why int8 activations work at all (no learned "coupled dimensions" needed): each QLinear output sums ~d features; per-element int8 error (width Δ=absmax/127) is ~independent zero-mean, so it accumulates like Δ√d while signal accumulates √d → OUTPUT SNR >> any single input's. The reduction averages away int8 coarseness for free (this is why ternary-weight × int8-act BitNet works). Training adapts mostly MAGNITUDE conditioning (outlier suppression, à la SmoothQuant), not pairwise coupling.

**CAVEAT for judging g070 (int8-act A/B):** int8 has FIXED spacing → much coarser than fp16 for SMALL values (fp16 has fine near-zero resolution). Predicted failure mode = a LATE-training/low-LR fine-convergence penalty, MIRRORING the bf16-latent deep-anneal gap (g066: fine to step 25k, then +0.073 in the deepest anneal). So a 10k int8-act A/B (g070) may look ~parity mid-run but UNDER-report the true cost — the penalty could surface only in a deep 30k anneal. If g070 tracks g069 to ~step 8500 then diverges in the 8500-10000 anneal tail, that's the fixed-spacing-hurts-small-values effect → run a 30k int8-act confirmation before adopting. If g070 is flat-parity throughout including the tail, int8-act is likely genuinely cheap.

## FP8/E4M3 activations — the finer-grained fix (fork 2026-07-16) → g071

**Why FP8 E4M3 should beat int8 for activations on THIS hardware:** the int8 caveat above is a *fixed-spacing* problem — a uniform grid gives absolute step Δ=absmax/127, so within a token whose absmax is set by an outlier, the *small* values get a coarse relative representation. E4M3 (4-bit exp, 3-bit mantissa) has a **floating** exponent → constant ~2^-3 *relative* step across magnitudes. Small activations keep their relative precision. This directly attacks the predicted int8 anneal penalty. Standalone check: STE identity grad ✓, ~1.8% mean quant err, and the relative error is magnitude-independent (int8's grows for small values).

**Hardware angle (why fp8 > int8 as the lever to build):** RTX 4050 = Ada sm_89 → **native FP8 tensor cores**. int8×ternary needs a custom matmul kernel to realize any speed win; fp8 has a real speedup path through cuBLAS/TE-style fp8 GEMM without bespoke kernel work. So even at *parity* quality, fp8 is the more valuable format to validate.

**IMPLEMENTED 2026-07-16** (commit 3436cd4): `--fp8-activations` per-token absmax E4M3 fake-quant STE in qlinear.py (`elif` after int8; scales into [-448,448], casts to float8_e4m3fn and back, STE). `--fp8-activations`/`--int8-activations` mutually exclusive (assert at init).

**QUEUED g071 = fp8-act bs=2 10k**, matched A/B vs g069 (4.0872), same recipe as g070 but `--fp8-activations` in place of `--int8-activations`. Runs after g070 (single GPU). Gives a clean 3-way **flat (g069) / int8 (g070) / fp8 (g071)** quality gate → pick the activation format worth a real kernel. Same anneal-tail caveat applies (watch 8500-10000); if fp8 tracks the flat tail where int8 diverges, that confirms the near-zero-resolution advantage and fp8 becomes the adopted lever.
