# Autoresearch: ternary-hrm-fast-recipe

## Objective

Find the fastest recipe for training a ~150M ternary HRM-text-style recurrent
transformer **from scratch** on a 6 GB RTX 4050. Optimize for **lowest
held-out val/loss at a short fixed step budget** so we get many iterations
per day.

Baseline = the `hrm-G-bop` recipe (last sustained run before autoresearch):
HRM dual-stack (hidden=1024, H=L=4 layers, 2×3 cycles), per-(row, group=64)
**random lognormal frozen** scales, frozen RMSNorms + z_L_init, **only
`embed_tokens` is FP-trainable**, Bop Bet 1 on trits (γ=γ_v=1e-3,
τ_norm=0.15), Lion32 on FP @ lr=5e-4 cosine to 0.1·lr, bs=2 ga=16 (effective
batch 32), seq=1024.

## Metrics

- **Primary**: `val_loss` — fineweb-edu `sample-100BT` held-out, 16 batches,
  CE per token. Lower is better. Measured at the final `[val]` step in the
  run's log.
- **Secondary** (always tracked once introduced):
  - `loss_ema` — last `ema=` value from the tqdm postfix (training EMA at end)
  - `flip_rate` — last `bop/flip_rate` value from TB (Bop activity at end)
  - `frac_zero_delta` — `trits/frac_zero` final minus initial, signed
    (negative = trits left zero, magnitude = how much Bop moved)
  - `wall_seconds` — duration of the run (compute spent per result)
  - `per_loop_gap` — `diag/per_loop_ce_0 - diag/per_loop_ce_1` (last sample);
    positive = recurrence helping
- Tie-break: if primary metric is within noise (~±0.03 at this scale), prefer
  the simpler recipe (fewer lines of code / fewer knobs).

## How to Run

`./autoresearch.sh` — runs ONE training experiment with whatever config is
currently set in the script, blocks until completion (or crash), parses TB
and log, prints `METRIC name=number` lines.

The script:
- Uses `experiments/<run-name>/` as the trainer output dir (interrupted.pt,
  logs go there; `hrm_bop.py` deletes interrupted.pt on clean completion).
- Forces TB output to `./tb/<run-name>/` so all experiments accumulate under
  a single directory the user can compare in TensorBoard.
- Run name format: `r<NNN>-<short-tag>` where NNN is zero-padded run number
  (matches the JSONL `run` field). The harness edits the script per
  experiment to set the run-name and config knobs.

## Step Budget

**2500 steps per experiment**, ~4.5 hours on the 4050 at bs=2 ga=16. Step
2500 is well past the hrm-A/B/F "no Bop, all FP trainable" plateau (~6.9
EMA) and gives clear separation between baselines and improvements based on
hrm-G's trajectory (val 6.83 at step 2000, 6.39 at step 5000).

Shorter budgets considered: 1500 steps gets noisy comparisons; 1000 steps
the "warmup phase still descending" effect dominates. Use 1500 only as a
screening round if confident, with promotion to 2500 for finalists.

## Files in Scope

- `smollmer/hrm_bop.py` — trainer entry point, init, optimizer assignment, TB.
  Most knobs land here.
- `smollmer/hrm_model.py` — HrmBopModel, the recurrent core with 1-step grad.
  Architectural variants (cycles, hidden, sandwich norm, etc.) here.
- `smollmer/qlinear.py` — QLinear forward path, per-(row, group) scales,
  `quantized_weight()` at `levels=3`. Scale parameterization changes here.
- `smollmer/flip_distill.py` — BopTernary optimizer (Bet 1 path).
  Optimizer variants (cautious mask, curvature gate, third moment) here.
- `smollmer/hrm_data.py` — DataLoader.
- `run_hrm.sh` — the standalone launcher (separate from autoresearch.sh).
- `autoresearch.sh` — the experiment runner. Edited each iteration.

## Off Limits

- Anything outside `smollmer/hrm_*` (and `qlinear.py`, `flip_distill.py`
  when needed for optimizer changes).
- `smollmer/distill.py`, `smollmer/qat_smooth.py`, `smollmer/flip_progressive.py`,
  `smollmer/calibrate.py`, all `smollmer/cache_*.py`, `smollmer/finalize*.py`,
  `smollmer/chat*.py`, `smollmer/build_student.py`, `smollmer/teacher_floor.py`,
  `smollmer/permute.py`, `smollmer/pack.py`, `smollmer/dump_for_chat.py`,
  `smollmer/export_onnx.py`, `smollmer/gsq.py`, `smollmer/progressive_distill.py`,
  `smollmer/qat_distill.py` — different research lines, leave alone.
- `Hestia/`, `smollmer/ckpts.*` — other experiments / archives, don't touch.

If a change requires touching `flip_distill.py` (e.g. adding a cautious mask
to BopTernary), edit the class in-place — the other smollmer trainers that
import it should keep working since they pass `use_2nd_moment=False` and
won't hit the new code paths.

## Constraints

- Runs MUST be interruptible (SIGINT saves interrupted.pt) and resumable
  (auto-resume on next launch if interrupted.pt exists). hrm_bop.py already
  does both.
- interrupted.pt MUST NOT survive a clean completion. hrm_bop.py deletes
  it at end-of-run; don't break that.
- All TB data MUST land under `./tb/<run-name>/` so the user can compare
  every experiment in one TensorBoard.
- VRAM ceiling is 5.64 GB usable on the 4050. Anything we add (extra opt
  state, bigger model, more activations) has to fit. The 808 MB Bop m+v
  baseline is the biggest single consumer.
- No new pip deps unless absolutely necessary.
- Don't introduce backwards-compatibility shims — this is a research
  branch, not production.

## Knobs to Explore (search space)

Roughly ordered by expected information gain.

### Trit optimizer (Bop variants and alternatives)
- **Cautious mask** in BopTernary (Liang et al. 2024): flip only when
  `sign(m) == -sign(g_t)`. Filters oscillating coords, should let τ_norm drop.
- **Curvature gate** (Bet 6 in `flip_research.md`): flip only when expected
  ΔL = `m·Δt + 0.5·v·Δt²` is negative.
- **τ_norm sweep**: 0.05 / 0.10 / 0.15 / 0.20. Current 0.15 is OK; lower
  with cautious mask, higher to test stability.
- **γ / γ_v sweep**: 3e-4 / 1e-3 / 3e-3. Shorter EMA = faster reaction,
  more noise.
- **BitNet-STE alternative**: replace Bop with STE-on-latent + Lion. Same
  trit codebook at forward, latent in `[-1, 1]` learns via STE, Lion drives
  flips by walking the latent across `±1/3` boundary. Known fast from
  bitlooplm. Loses Bop's "informed flip" property; gains speed.
- **Adaptive τ** (Bet 4 SGDAT): per-trit `τ_i = τ_base · (1 + η · c_flip)`.

### Scale parameterization
- **Per-tensor BitNet style**: single `s = 1/mean(|w|)` per matrix,
  recomputed per forward, not learned.
- **Per-row, frozen at fan-in**: per-out-row scale, no per-group, fixed.
- **No scale at all**: bare ternary matmul.
- **hrm-G config (per-group lognormal frozen)**: baseline.
- **Per-group fan-in fixed (no noise)**: removes the lognormal noise from
  hrm-G to test whether the randomness mattered.
- **Late-stage unfreeze**: train trits with frozen scales for N steps,
  then unfreeze scales for the last fraction.

### Activations
- **INT8 per-token absmax** (BitNet style): potentially 2× act memory,
  may unlock bs=4.

### Freezing / unfreezing
- Vary which FP params are trainable: `embed only` (current), `embed+norms`,
  `embed+z_L_init`, `all FP`.

### Architecture
- **Sandwich norm** (4 norms per layer, HRM-Text style): tests R7 in spec.
- **Cycle ratio**: 1×3, 2×2, 2×3, 3×2. Current 2×3.
- **Hidden / layer count**: 4+4 vs 6+2 (asymmetric, more L compute).

### Optimization side-effects
- **Lion LR sweep**: 1e-4 / 3e-4 / 5e-4 / 1e-3 on FP.
- **Trit init zero-frac**: 33% / 50% / 67%.
- **Effective batch**: 16 / 32 / 64.
- **AdamW32 or CautiousAdamW on FP** instead of Lion.

## What's Been Tried (update after each ~5 runs)

### Prior to autoresearch (see `smollmer/hrm_bop_spec.md` for full detail)

- hrm-A/B/F variants with all FP trainable + Bop on trits: plateau at
  loss/ema ~6.9. Lion-on-FP absorbs most loss; Bop barely flips because
  per-(row, group) trainable scales destabilize Bop's gradient sign.
- hrm-G config (random frozen scales, frozen norms+z_L_init, only embed
  trainable): val/loss 6.28 at step 7000; 0.92 nats better than
  embed-alone ablation (hrm-H). Recurrence (per-loop CE gap) only
  develops when Bop flips → architecture+optimizer co-depend.
- Hot τ_norm decrease mid-run = catastrophe (saturated m fires all at
  once, loss spikes, `v` Inf-contaminated). Cold-start at lower τ is fine.

### Autoresearch session

(Filled in by the loop. Update after every ~5 runs.)

---

## Resume Notes

If you're a fresh agent resuming this loop:

1. Read `experiments/worklog.md` for the narrative.
2. Read `autoresearch.jsonl` for the structured history; find current
   segment via the last `{"type":"config",...}` line, then current run via
   the last result line's `run` field + 1.
3. Read `autoresearch-dashboard.md` for visual summary.
4. Check whether a run is currently in flight: `ps -ef | grep
   "smollmer.hrm_bop" | grep -v grep`. If yes, decide whether to wait,
   SIGINT and resume after measurement, or kill and move on.
5. Continue the loop.

The trainer auto-resumes from any `experiments/<run-name>/interrupted.pt`
if present. To force a fresh run, delete that file before launching.
