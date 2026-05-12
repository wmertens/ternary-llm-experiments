# Ternary distillation: soft-stage findings

Notes from the investigation into why `bins/frac_zero` drifted from
target 0.33 to ~0.50 during soft-stage training, the design of the
fix (per-group calibrated triple-well attractor), and what the live
run with the new recipe is showing.

## Problem

Run with `--soft-attractor l2 --soft-zero-frac 0.33`: `frac_zero`
hit 0.33 at step 10 (as designed by `nearest_ternary_quantile`'s
per-group cutoff), then drifted monotonically up to 0.50 over ~17k
steps. By α≈0.88 the loss had stalled at gap 0.17, weights had
"frozen" (stopped changing), and only scales kept learning.

## Root cause: three colluding effects

### 1. The quantile classifier is `<=` inclusive

```python
cutoff = abs_wb.quantile(target_zero_frac, dim=-1)
is_zero = abs_wb <= cutoff
```

Gives *exactly* 33% zeros only when the |w| values are distinct.
Once enough weights cluster at exactly 0 (or any common value),
the cutoff *is* that common value and `<=` admits all of them.
With more than 33% of weights at exactly 0, the cutoff collapses
to 0 and `frac_zero = (weights == 0)`, which can climb arbitrarily.

### 2. wd feeds the absorbing-zero state

Decoupled weight decay (`w ← w·(1 − lr·wd)`) shrinks every weight
toward 0 each step, regardless of basin classification. In fp16
latents, weights below the smallest representable subnormal
underflow to exactly 0 and can't escape:
- L2 attractor gradient at w=0 with c=0 is `2·λ·w/N = 0`.
- Lion's sign-only update with grad ≈ 0 is dominated by noise
  but biased toward 0 by wd.
- Once a weight hits exact 0, both forces vanish — it stays.

### 3. L2/wd cancellation freezes non-zero-bin weights

For a weight in the ±1 bin (c(w) = sign(w)):
- L2 gradient: `2λ(w − sign(w))/N` — pulls toward sign(w).
- wd gradient: `wd·w` — pulls toward 0.
- These have opposite signs in the bin. Under Lion's sign-only
  update, `sign(L2 + wd + KL)` cancels out as an integrated
  signal when L2 and wd magnitudes are comparable.
- KL gradient on latents is `(1−α)·∂L/∂w`, which vanishes at
  high α — so once α is large there's nothing breaking the tie.

**Net effect:** weights crossing into the 0-bin get absorbed
permanently; weights in the ±1 bin can't move; the system
asymmetrically accretes mass at 0. The 50% asymptote isn't
universal — it's roughly where the median |w| lives in the
max-abs-normalized weight distribution. Different inits would
land at different asymptotes; the *direction* (drift up from
target) is universal.

## Solution: per-group calibrated triple-well

### Triple-well attractor

`U(w) = w²(w²−1)²` has minima at {−1, 0, +1}, saddles at ±1/√3,
saddle height 4/27. Smooth everywhere. Replaces both the
piecewise nearest-ternary classifier and the L2 penalty with a
single C^∞ regularizer added to the loss:
`L_total = L_kl + α·U(W)`. No boundary flicker, no
classifier-flip-induced momentum chatter.

But the canonical well's basin width is fixed: with saddle at
±1/√3 ≈ 0.577 and Gaussian-ish max-abs-normalized init, ~50−70%
of weights fall in the 0-basin from the start. Far worse than
33%.

### Per-group `a` calibration

Generalize to `U_a(w) = U(w/a)` with minima at `{-a, 0, +a}`
and saddle at `±a/√3`. Coordinate rescale of the canonical
form, saddle height preserved. Setting

```
a = √3 · |w_init|.quantile(target_zero_frac)   per (row, group)
```

makes the saddle land at exactly the target |w| quantile. So
~target_zero_frac of init weights per group sit inside the
0-basin naturally — same intent as the L2 quantile cutoff,
just frozen at init and embedded in the attractor's smooth
geometry rather than recomputed every step.

Stored as a `[out, n_groups]` buffer on `QLinear` (`well_a`).
Survives state_dict load on resume; only re-initialized on
fresh start so resume keeps the calibration that was used.

### Math-preserving deploy rescale

At end of soft training: per (row, group),
`latent /= a` and `scales *= a`. Forward output
`(latent/a) · (scales·a) = latent · scales` is unchanged. Latents
that drifted past ±a get clipped to ±1 (small information loss
only for weights the well never fully captured). After this,
the codebook is back to the standard {−1, 0, +1} that
`finalize.py`, `chat.py`, and `pack.py` expect.

**Key caveat:** the rescale is math-preserving *only* when basins
are tight enough that `|latent| ≤ a` per group. Mid-training (low
α) the well's pull is weak relative to KL noise, so many latents
sit at |w| > a (still in the correct basin per the saddle-defined
classifier, but past the rescale's clamp boundary). After the
rescale these get clipped to ±1 instead of preserving their
basin-relative magnitude, and the deploy-time classifier
(c(w) using ±1/3) misclassifies as ±1 weights that the well
intended for the 0-basin. So **deploy-form chat will be
degenerate until α is high enough that basins tighten**, even
when the training-time (α=0) forward produces coherent text.

This was confirmed via mid-training chat sanity at step 22k
(α=0.165): training-time forward gave coherent SmolLM2-style
output ("Once upon a time…" → coherent story; "The capital of
France is…" → "Paris"); deploy-rescaled forward gave "to to and
and…" gibberish. Mean |latent| was 0.42 pre-rescale, 0.957
post-rescale — most weights got clamped because λ·α was still
small enough for KL noise to keep latents away from the well
minima.

Practical rule: only do deploy-form sanity checks once
`grad_norm_penalty/grad_norm_kl` has been ≫1 for many bumps and
α is well past 0.5 — basins need to be tight before the rescale
is fair.

## Recipe (live run, started 2026-05-07 23:34)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True smollmer-distill \
  --no-grad-checkpointing \
  --cache-dir cache/ \
  --out ckpts.well-zerofrac \
  --tb-dir tb \
  --scale-group-size 64 \
  --soft-attractor well \
  --soft-zero-frac 0.33 \
  --wd 0 \
  --lr 2e-4 \
  --warmup-steps 1000 \
  --soft-bump 0.0005 \
  --soft-patience 50 \
  --soft-tolerance 0.02
```

Key choices vs the failed prior run:
| Flag | Prior | New | Reason |
|---|---|---|---|
| `--soft-attractor` | `l2` | `well` | C^∞, no boundary flicker; KL keeps full strength |
| `--soft-zero-frac` | (default 0.33) | `0.33` | now also drives per-group `well_a` calibration |
| `--wd` | 0.05 | `0` | confirmed culprit for drift + L2/wd freeze |
| `--lr` | `7e-5` | `2e-4` | Lion sweet spot for ternary (3e-4..1e-3 per prior sweep) |
| `--warmup-steps` | 300 | `1000` | momentum tracks KL before α contracts |
| `--soft-bump` | 0.001 | `0.0005` | half-speed, KL has room to react per bump |
| `--soft-patience` | 20 | `50` | 2.5× more steady steps required per bump |
| `--soft-tolerance` | 0.05 | `0.02` | stricter "stable" check, fewer false bumps |

## Live-run observations (through ~9h, step ~18k, α≈0.135)

- **`frac_zero` drift is dramatically slower** than L2: 0.329 → 0.347
  over 18k steps vs prior 0.33 → 0.50 in 17k. Drift is structural
  to any stochastic gradient + absorbing-zero combo, but the well
  + wd=0 makes it gradual rather than runaway.
- **Loss EMA: 1.59 → 1.81 (warmup peak) → 1.66** (and still slowly
  descending). New min at every check. Below `L_T + 0.04` consistently.
- **Gap from teacher floor: 0.04 mean** with batch noise dominating.
  Prior run never got below 0.10 at any α.
- **Lion sign update is fully well-driven on latents** at α≈0.13:
  `grad_norm_penalty/grad_norm_kl` ≈ 30× and growing. KL on latents
  is suppressed; latents are essentially locked in basins.
- **Scales carry the KL signal** — they're outside the contraction
  and outside the well, so they receive full-strength KL gradient.
  This is the actual learning channel during the polish phase.
- **`weights/flip_rate` ≈ 0.3%** — negligible bin reclassification
  per step. Basins are stable.

### Histogram structure (q_proj, current run)

Sharp tall spikes at ±0.5 (the per-group `a` for q_proj specifically
landed near 0.5 because q_proj's 33rd-percentile-of-|w| ≈ 0.289).
Broad lower hump around 0. **Crucially, no spike at exactly 0** —
that's the wd=0 difference: weights in the 0-basin wobble freely
under KL noise rather than getting trapped at exact zero. The
breadth of the 0-cluster is what enables (slow) bin migration via
saddle crossings.

## Insights

### "Polish phase" arrives early under strong attractors

Once `λ·α` × per-element well force exceeds KL signal magnitude
on latents, latents are locked and only scales learn. With Lion's
sign-only update and λ_max=1e-2, this transition happens around
α=0.05 in the new run (vs prior run not reaching it cleanly even
at α=0.88). Recognizing this regime early lets you reason about
what *can* still improve: scale fine-tuning of a fixed discrete
configuration.

### Lion is well-suited to this dynamics

Sign updates normalize attractor gradient magnitude — even when
penalty grad is 30× KL grad, the per-parameter step is ±lr.
KL retains influence on parameters where it's the only signal
(scales, embeddings) and is overruled only on latents where the
attractor wants to enforce the discrete codebook. AdamW would
need careful gradient scaling to avoid the well overflowing the
update direction.

### `frac_zero` is a target, not a stable equilibrium

Both the L2 quantile cutoff (per-step) and the well per-group `a`
(frozen at init) hit the target at step 0, but neither holds it
exactly over training. Mechanism is structural:
- Boundary weights drift across the saddle/cutoff under stochastic
  gradients.
- Once on the 0-side, the basin pulls them in.
- With or without wd, the symmetry breaks.

The well + wd=0 combo slows the drift from "minutes" to "hours"
and makes it bounded-ish (depends on how many weights are
boundary-flippable). For a hard-pinned `frac_zero`, you'd need
either an argpartition-based exact-quantile classifier (expensive)
or periodic re-calibration.

### Bonsai's 38% and BitNet's 85% are both fine

`frac_zero` is a hyperparameter, not a universal optimum. The
"right" value depends on the model's natural |w| distribution
(controllable via init), the distillation target's structure,
and the deployment storage budget. Drift past the initial target
isn't necessarily a failure mode — it's the system finding its
own equilibrium. What matters is loss/gap, not the bin counts.

## Open questions and future tuning

### Likely improvements for a follow-up run

0. **`--soft-alpha-init 0.03`** (eliminate the warmup overshoot).
   At α=0 the well/L2 penalty is off, KL is small (model ≈ teacher),
   and Lion's sign-of-momentum update is dominated by gradient noise.
   As LR ramps to 2e-4, those noisy updates accumulate into a random
   walk away from the teacher (1.59 → 1.81 EMA climb in this run).
   Starting at α=0.03 gives λ=3e-4 from step 1 — coherent gradient
   signal toward basins, which Lion can lock onto consistently. With
   per-group `well_a` calibrated to the natural |w|-quantile, the
   "obvious" basin assignment matches what KL would prefer anyway,
   so capturing weights early is essentially free. Likely also lets
   you shorten `--warmup-steps` (the warmup was largely there to
   mask the overshoot).

1. **`--soft-l2-coef 5e-3`** (half current). With grad ratio 30×
   and growing, halving λ_max gives KL real per-latent authority
   throughout training. Probably faster convergence and less
   `frac_zero` drift. Likely the single highest-impact knob to try.
2. **`--soft-bump 0.001 --soft-patience 30`**. Compresses schedule
   to ~25h total. KL still has 30 steady steps per bump.
3. **Try `--soft-zero-frac 0.38`** (Bonsai) or **`0.5`** to see
   whether wider 0-basin is more drift-stable — wider basin means
   more weights are deep in it (vs near saddle), so fewer boundary
   crossings.
4. **Skip well backward at very low α**. Currently the backward
   runs every step regardless of `cur_lambda`. Gating on
   `cur_lambda > 1e-5` would save ~5−10% of step time during the
   early ramp. Easy patch.

### Open questions

- Does the gap stabilize as α saturates, or keep slowly closing
  as scales fine-tune indefinitely? (Live run will tell us.)
- Is the 0-cluster broadness from wd=0 *necessary* for the model
  to find a working configuration, or just a side effect? Could
  test with tiny wd (e.g. 1e-4) to see whether basin sharpness
  helps or hurts.
- Does the per-group calibration help at α=1 (post-rescale,
  finalize stage) or is it purely a soft-stage convenience?
  Likely the latter — finalize uses fixed ±1/3 boundary anyway,
  and the well_a rescale puts weights at ±1 where the boundary
  classification is unambiguous.
- Is the 30× attractor/KL ratio too strong? An A/B with
  λ_max ∈ {1e-2, 5e-3, 1e-3} would establish the working range
  and show how it trades against `frac_zero` drift speed.

## Cycle experiments and histogram-shape analysis (2026-05-09 → 2026-05-10)

After the well + per-group calibration locked the basin assignments
(structurally `frac_zero = 0.3333` to 4 decimals across thousands of
steps), the polish phase plateaued: EMA bouncing in [1.59, 1.66],
no new lows for hours, scales doing the only learning. Two follow-up
intuitions drove the experiments:

1. Periodically loosen the well so latents near the saddle can
   re-cross under KL guidance, then re-tighten so they re-lock in
   their (possibly new) basin. Simulated-annealing-flavored.
2. Make the zero-basin actually peak at zero, not just *contain*
   33% of the mass spread across `[-saddle, +saddle]`.

### α-cycle: from taper-down to quadratic growth

Original implementation (2026-05-09): sinusoidal cycle on `effective_α`
with linear taper to 0 in the top 20% of the α range. Period 200,
amplitude 0.05. Rationale: keep basins tight at the deploy end.

Outcome: `effective_α` swung ±0.05 around the schedule. Loss EMA
found a project record (1.5905 vs prior 1.6152) but `flip_rate` and
`flip_rate_fixed` stayed at zero across thousands of samples — bulk
distribution metrics (`saturation_frac`, `near_boundary_frac`,
`frac_zero`) bit-identical across 200+ samples for hours.

Revised (2026-05-10): replaced the taper with quadratic growth:

```
cycle_amp = soft_cycle_amp · (1 + soft_cycle_grow · (α/α_max)²)
effective_α = clamp(schedule.α + cycle_amp · sin(2π·step/period), 0, 1)
```

`soft_cycle_grow=4` widens swings from ±0.05 to ±0.15 at α=0.7,
±0.25 at α=0.95. Combined with shorter period (80) and 2× lr
(4e-4), this *did* produce non-zero flip_rate_fixed for the first
time — but at trivial scale: max ~1.7e-7 (≈10 weights of ~50M per
log window), with rate decaying over time. Bulk distribution
metrics still bit-identical. Verdict: the extra aggression unstuck
edge cases at the saddle; it didn't trigger global re-optimization.

### wd cycle

Added `--wd-cycle-amp / --wd-cycle-period`:
`effective_wd = wd · (1 + amp · sin(2π·step/period))`, clamped ≥0.
With `--wd 1e-3 --wd-cycle-amp 1.5 --wd-cycle-period 1200`, wd swings
0→2.5e-3 in slow lobes. Did not visibly change basin sharpness or
flip rate — the per-step magnitude is still tiny against KL.

### Latent histogram inspection

Sampled QLinear latent histograms after ~50k steps (q_proj layers,
mlp.up_proj layers; same shape across all):

```
weight bin    count
±1.00         ~3-10k
±0.86         ~4-12k
±0.71         ~8-23k
±0.57    ~15-110k     ← peak (the +a basin minimum)
±0.43    ~20-130k     ← peak (the −a basin minimum, asymmetric per layer)
±0.29     ~24-85k
±0.14     ~35-110k
 0.00     ~40-115k    ← hump, NOT a spike (only marginally taller
                        than ±0.14 neighbors)
```

Trimodal, but the 0-basin is a broad shallow hump from `-saddle` to
`+saddle` instead of a sharp peak at exactly 0. Per-layer stats:
frac `|w|<0.05` only 8-9%, frac `|w|<0.1` only 16-17%. If the zero
basin were tightly packed at 0 you'd expect frac `|w|<0.05` near
30%.

(This refines the earlier "no spike at exactly 0 — wd=0 difference"
note: the absence of a 0-spike is intrinsic to the well+KL
equilibrium, not just a wd-related artifact.)

### Basin equilibrium width: λ/g_KL ratio sets the spike sharpness

For a latent inside the 0-basin, the well's restoring force toward 0
is approximately linear: `F_well(w) ≈ λ · 2w/a²`. The KL gradient on
the latent is `g_KL` (per-param, varying). At equilibrium:

```
w_eq ≈ g_KL · a² / (2λ)
```

Plug in `λ = soft_l2_coef · α` ≈ 4.75e-3 at α=0.95 (current
recipe), `a` ≈ 0.52 (typical), `g_KL` ≈ 5e-3 (rough per-param
average): `w_eq` ≈ 0.14. **This matches the histogram's hump
half-width.**

The basin width is set by `λ/g_KL` ratio, **not by training time or
cycle aggression**. Continuing to train at fixed soft_l2_coef won't
narrow the basin further; only ~6% improvement is expected as α
saturates. To get a real spike at 0, `soft_l2_coef` must be raised
(constant change to mean λ), reducing equilibrium width by ~`√k`
per `k×` increase in coef.

### Lion sign-momentum × λ-cycling: why penalty cycling is weak

Cycling λ would matter strongly for SGD/Adam where the per-step
update magnitude scales with gradient. With Lion, the per-step
update is `lr · sign(momentum)` — fixed magnitude regardless of λ.
λ only affects Lion's update if it's large enough to flip the
*sign* of the momentum direction.

For latents at basin equilibrium, well-sign and KL-sign are aligned
(both point toward `w_eq`). Cycling λ scales both forces but doesn't
flip their sum's sign. So Lion takes the same sign-step regardless
of cycle phase, and the cycle's intended annealing effect is
muted to whatever boundary cases are right at the saddle.

This explains why `flip_rate_fixed` was bit-zero in the original
cycle, and why doubling lr + halving period (the "flip-push"
experiment) only produced trivial flips at boundary edges.

### Hypothesis check: variable penalty ≠ variable latent perturbation

The intent behind cycling is "let weights move enough during dips
to find a better basin assignment, then lock them in during peaks."
This is simulated annealing — it requires actual *position* noise
(or magnitude-aware update rules) to work. What we implemented is
variable *penalty strength*, which under Lion's sign update only
changes behavior at the saddle.

A more direct implementation of the original hypothesis: add
Gaussian noise to latents proportional to `(1 − effective_α)`
during troughs. This would be true Langevin-style annealing and
would actually let latents migrate. Not yet tried — flagged as a
future experiment if histogram tightening alone doesn't suffice.

### Recipe iterations (this session)

```
2026-05-10 13:04  pre-cycle-grow snapshot     (α=0.66, EMA min 1.585)
2026-05-10 15:30  cycle-grow4-p200 snapshot   (4 hrs of taper→grow conversion;
                                               EMA min 1.608, no flips)
2026-05-10 16:29  flip-push-lr4e4 snapshot    (1 hr of lr 4e-4, period 80,
                                               grow 6; first nonzero flips at
                                               1.7e-7 scale)
2026-05-10 16:30  spike-push run (in progress) lr 4e-4, period 80, grow 10,
                                               soft-l2-coef 1.5e-2 (3×)
```

All snapshots preserved as `interrupted.<tag>.pt` for replay /
parameter sweeps once the recipe is dialed.

### Updated open questions

- Does `soft_l2_coef = 1.5e-2` produce the intended ~3× basin
  tightening, and at what loss cost? (In progress.)
- Is the histogram-spike goal achievable without sacrificing
  loss EMA, or is the basin-width / loss tradeoff fundamental?
- Would direct latent perturbation (Gaussian noise during cycle
  troughs) produce real flip rates where penalty cycling failed?
- Is a clean from-scratch run with the dialed-in recipe (rather
  than further mid-train iterations) the right next step? Per
  user: yes once params are settled.

## Code map

- `qlinear.py:triple_well_potential(w, a)` — `U_a(w) = U(w/a)`
- `qlinear.py:triple_well_loss(model)` — sums per-group U_a using
  each module's `well_a` buffer
- `qlinear.py:init_well_a(model, target_zero_frac)` — fills
  `well_a = √3·|w|.quantile(target)` per (row, group)
- `qlinear.py:rescale_well_for_deploy(model)` — math-preserving
  end-of-training rescale to {−1, 0, +1} codebook
- `distill.py` calls `init_well_a` after `set_soft_mode` on fresh
  start, `triple_well_loss(model)` per step in well mode, and
  `rescale_well_for_deploy(model)` before the final
  `stage_soft.safetensors` save.
- `distill.py` cycles (per-step):
  - `effective_wd = wd · (1 + wd_cycle_amp · sin(2π·step/period))`,
    clamped ≥0, applied to optimizer param groups.
  - `effective_α = clamp(schedule.α + cycle_amp · sin(...), 0, 1)`
    with `cycle_amp = soft_cycle_amp · (1 + soft_cycle_grow · (α/α_max)²)`.
  - `cur_lambda = soft_l2_coef · effective_α` modulates the well
    penalty backward.
- `dump_for_chat.py` — new utility to dump `interrupted.pt` to
  safetensors for mid-training chat sanity (default α=0 training-time
  forward; `--deploy` for post-rescale α=1).
- `qlinear.py:module_ternary_fixed(m)` — frozen-threshold
  classifier (`well_a/√3`) used by `weights/flip_rate_fixed`,
  separate from the moving-quantile `weights/flip_rate`.
