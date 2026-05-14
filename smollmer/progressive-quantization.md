# Progressive ternary clamping

Alternative recipe to `distill.py`'s soft α-anneal: commit one weight per
`(row, column-group)` at a time, lowest-disturbance first, with KL
distillation training between commits to absorb each perturbation.

Implementation: `progressive_distill.py`.

## Idea

The Bonsai forward is `(x @ T) * s` where `T ∈ {-1, 0, +1}` per element and
`s` is one scale per `(row, group)` of `group_size` columns. Group =
`group_size` weights sharing one scale.

`distill.py`'s soft anneal commits *all* weights simultaneously by ramping
`α: 0 → 1` so the residual-contraction forward `T_α(w) = c(w) + (1-α)(w -
c(w))` converges to hard ternary as α saturates.

Progressive clamping commits *one weight per group per round*:

1. **Initialize** with the math-preserving `s = max(|w|)` Bonsai scale and an
   extra `1/c` rescale: `w *= c, s /= c`. The largest-magnitude weight per
   group now sits at exactly ±c (the leeway codepoint), and forward output
   is identical to the FP teacher at step 0.
2. **Warmup**: train the soft model (α=0, forward = identity = `w * s`) for
   `--warmup-rounds-steps` so optimizer momentum populates.
3. **Commit round**: for each group, pick the unfrozen weight minimizing
   `|q_error| + λ_m · |momentum| / median(|momentum|)`, where `q_error =
   min(|w - (-c)|, |w|, |w - c|)`. Snap that weight to its target ∈ `{-c,
   0, +c}`. Zero `exp_avg` and `exp_avg_sq` at the committed slot. Mark it
   frozen.
4. **Train** until either (a) loss EMA is stable AND `fast_ema - L_T <
   commit-gap-threshold`, or (b) `commit-max-round-steps` is hit. The gap
   condition prevents us from locking weights in while the model is still
   recovering from a previous bad perturbation.
5. **Repeat** for `group_size` rounds. Every weight is committed by the
   end.
6. **Deploy rescale**: `w /= c, s *= c`. Latents are now exactly `{-1, 0,
   +1}`, matching Bonsai's deployment codebook. `finalize.py` / `chat.py`
   work unchanged.

## Why leeway (c < 1)?

Bonsai's deployment codebook is `{-1, 0, +1}`, but training with codepoints
at exactly ±1 means committed weights sit at the boundary of `[-1, 1]`. The
optimizer has no soft "stay inside the box" pressure on the *unfrozen*
neighbors of a committed weight, so they can drift past ±1 and the
quantizer's `clamp(-1, 1)` produces a discontinuous saturating zone.

With `c < 1`:
- Codepoints `{-c, 0, +c}` sit interior to `[-1, 1]`.
- Unfrozen weights have a "safe zone" `(c, 1)` and `(-1, -c)` to move
  through.
- A soft barrier `relu(|w|-1)²` keeps them mostly in `[-1, 1]` without a
  hard discontinuity.
- The end-of-run `w /= c, s *= c` rescale folds c into the scale
  (math-preserving), so the deployed codebook is `{-1, 0, +1}` regardless
  of which c we trained with.

`--c` defaults to 0.85 (~18% headroom).

## Mask, not optimizer-state encoding

We considered encoding "frozen" status entirely in optimizer state (e.g.
`exp_avg_sq = +inf` → update is zero). Rejected because:

- AdamW weight decay does `w -= lr * wd * w` unconditionally, drifting
  frozen weights toward 0.
- Sentinel-value tricks (`NaN`, `+inf`) are fragile and break on optimizer
  swap.
- 1 bit per weight is trivial overhead.

Two persistent buffers per `QLinear`:
- `frozen_mask: bool[out, in]` — True for committed slots.
- `frozen_target: int8[out, in]` — sign of the committed value (`-1/0/+1`).
  Actual latent value is `frozen_target * c`.

Pre-step: `zero_frozen_grad` masks the gradient at frozen slots.
Post-step: `enforce_frozen` overwrites the latent at frozen slots back to
target. This is bulletproof against weight decay, barrier pull, autocast
roundtrips, optimizer-state corruption, etc.

## Selection criterion

```
# Boundary mode A (default): per-(row, group) |w|-quantile cutoff
cutoff = quantile(|w|, target_zero_frac)         per group
target(w) = 0                if |w| ≤ cutoff
            sign(w) * c      otherwise

# Boundary mode B (fallback when target-zero-frac is disabled): fixed midpoint
target(w) = sign(w) * c       if |w| ≥ c/2
            0                  otherwise

q_error(w) = |w - target(w)|

score(w) = q_error(w) + λ_m * |exp_avg(w)| / median(|exp_avg|)

commit argmin score(w)   per (row, group)   over unfrozen w
```

Mode A is the default with `--target-zero-frac=0.38` (Bonsai-like). Mode B
(`--target-zero-frac ≤ 0 or ≥ 1`) is the fixed midpoint — kept as a
fallback for ablation but DO NOT use it for real runs: see "Findings"
below for why it produces ~91% zeros on Gaussian-ish init.

The momentum term penalizes committing weights the optimizer is actively
trying to move (high `|m|`). `--momentum-weight` (default 0) controls
λ_m. The median-normalization makes the penalty dimensionless and
comparable across layers.

At commit time we also zero `exp_avg` and `exp_avg_sq` for the chosen
slot, so no stale momentum leaks into the locked-but-still-in-the-tensor
position.

## Convergence gate per round (`CommitGate`)

Track fast and slow loss EMA (1/0.05 ≈ 20-step and 200-step windows by
default). "Stable" = fast EMA hasn't outrun slow by more than
`--commit-tolerance` for `--commit-patience` consecutive steps.

Commit fires when:
- `stable AND (fast_ema - L_T < --commit-gap-threshold)`, OR
- `step_in_round >= --commit-max-round-steps` (safety cap).

When stable but gap is wider than threshold, we keep training — the user's
explicit preference. Locking weights in while the model is still
recovering from the previous commit yields a worse final basin.

## Optimizer-state hygiene at commit

1. Zero `exp_avg[r, c]` (and `exp_avg_sq` for AdamW family) at chosen slot.
2. Optionally `--post-commit-momentum-damp < 1` multiplies *all*
   `exp_avg` by that factor (cools momentum globally to absorb the
   perturbation). 1.0 = off.
3. `enforce_frozen` reapplies the target value, so even if the optimizer's
   next step touches the slot (it shouldn't, since we also zeroed the
   grad), we still overwrite back.

## TensorBoard scalars

User-requested round tracking:
- `progressive/round` — integer round count.
- `progressive/committed_frac` — global fraction of weights committed.
- `progressive/max_per_group` — worst-case `(row, group)` commit count
  across the model. Equals `group_size` when fully done.

Per-commit diagnostics (logged at the moment of each commit):
- `progressive/last_commit_q_err` — mean `|q_error|` of the chosen
  weights this round. Should grow round-over-round (early commits are
  near codepoints; late commits are the hard middle cases).
- `progressive/last_commit_target_frac_{neg,zero,pos}` — sign split of
  this round's commits.

Per-step (every `--log-every`):
- `loss/step`, `loss/ema`, `loss/gap` (= step_loss − L_T)
- `progressive/loss_ema_{fast,slow}` from the gate
- `progressive/step_in_round`, `progressive/steady`, `progressive/barrier`
- All distill.py metrics (`bins/`, `weights/flip_rate*`, `scales/`,
  `soft/latent/*`, `embed/drift_l2*`)

Caveat: `soft/bins/frac_{neg,zero,pos}` uses the ±1/3 c(w) classifier
inherited from `qlinear.py`, NOT our `±c/2` codebook boundary. Treat as
a coarse "which way is this weight leaning" signal, not codebook
occupancy. The actual codebook occupancy is reflected in
`progressive/committed_frac` and the per-commit fractions.

## Math-preserving identities

Two identities keep the forward output unchanged across reparameterizations:

| Operation | Effect |
|-----------|--------|
| `w *= c; s /= c` (init) | `(w/s) * s = (w·c)/(s/c) * (s/c) · ... = same` |
| `w /= c; s *= c` (deploy) | inverse of above |

Verified by inspection: forward is `(x @ T.T) * s_broadcast`, and any
`(w, s) → (αw, s/α)` preserves the product `w * s`.

## Things to think about (open questions)

These are tunable / arguable choices, flagged for ablation:

### 1. Commit boundary `c/2` — **RESOLVED, see Findings**

The c/2 midpoint over-zeros heavily on Gaussian-ish weight distributions
(Run A produced 91% zeros at round 21). Replaced with a per-(row, group)
`|w|`-quantile cutoff (`--target-zero-frac` flag, default 0.38 matching
Bonsai's ~62% non-zero deployed ratio). The c/2 path is kept as a
fallback when the flag is disabled, but it is no longer the default.

### 2. Permute at start

`--permute` defaults to True (matches `distill.py`), clustering
high-magnitude columns together. With permute=True, "tight" groups (all
high-magnitude or all low-magnitude) may want all-one-target commits,
producing lopsided per-group target distributions. permute=False keeps
heterogeneous groups → more variety in commit targets per group.

Worth an ablation once we have a baseline.

### 3. Lopsided target distribution

If after a few rounds a group's tally is e.g. `(+1: 20, 0: 5, -1: 0)`, do
we want to force the next pick toward 0 or -1?

Pros: balanced groups make `0 < ‖T·s‖ < ‖s‖·group_size` regardless of
input direction, which may help expressivity.

Cons: we'd be deviating from the gradient-driven natural choice, locking
in a worse position.

Not implemented in the first cut. Easy to add: a per-group penalty in
`select_and_commit_one_per_group` scaled by deviation from target
distribution (e.g. `(actual - target)²` over the three bins). Add
`--target-distribution 0.33,0.33,0.33` and a coefficient.

### 4. EMA-overshoot at start of each round

Each commit is a small perturbation, but cold momentum after
`exp_avg`-zeroing can produce a brief loss spike before the model
re-stabilizes. Mitigations available:

- `--post-commit-momentum-damp` (already implemented).
- `--warmup-rounds-steps` (initial only).
- Could add per-round LR warmup (not implemented; one knob too many for
  the first cut).
- Could add `--opt-warmup-passes` (lr=0 forward+backward passes) like
  `distill.py` — easy port if needed.

### 5. Cosine LR horizon

Default `--lr-floor=1.0` = flat LR, because total step count is
data-dependent (depends on how quickly each commit round converges).
For a known horizon, set `--lr-floor=0.1` and rely on the nominal
horizon = `group_size × commit_max_round_steps` (deliberately
pessimistic; you can override by short-circuiting after fewer rounds).

### 6. `enforce_frozen` cost

Uses fancy indexing `m.weight.data[fm] = target[fm]` per QLinear per
step. Negligible at low commit count; could become measurable when
~half the weights are frozen and we're paying ~30M bools of selection
work per step. Profile and optimize if needed (e.g. `where(fm, target,
w)` in a single op, or a sparse-index list).

### 7. CautiousAdamW interaction with frozen slots

CAdamW's "cautious mask" is `(m * grad > 0)`. At frozen slots we zero
the grad, so the mask is 0 there — fine. But the mask is then
normalized by `mask.mean()`, which now includes a bunch of forced-0s.
Effective LR shifts as the fraction of frozen slots grows. May want to
exclude frozen slots from the mean: a cleaner fix is to mask out frozen
slots from the param iteration entirely (e.g. multiply update by
`~frozen_mask`), but that requires touching the optimizer. Watch the
TB `grad_norm` and `progressive/loss_ema_fast` for late-stage
trajectory shifts that correlate with `committed_frac`.

### 8. Group-finished detection

We loop `while round_idx < group_size` assuming every group commits once
per round. True for the first round. False for later rounds if any
group is fully frozen early (e.g. small groups, or unusual
distributions). `max_committed_per_group` is the right exit gate;
swap the `while` condition to `max_committed_per_group(model) <
group_size`. Not critical for early experiments but should be fixed
before a real run.

### 9. Resume robustness

The frozen buffers ride along in `state_dict`, so `load_state_dict`
restores them. But `select_and_commit_one_per_group` reads optimizer
state, and `opt.load_state_dict` happens before we'd ever call it
again, so this should work. Verify on first resume.

## Findings from tuning runs (2026-05-12)

Goal of the tuning: *minimize starting EMA overshoot above L_T*, the
teacher KL floor. Math-preserving init means the student is AT L_T at
step 0, so any drift is pure noise injection that the model later has
to undo.

### Run A — `lr=3e-4 wu=100 wd=0.001 c/2 boundary` (no `--target-zero-frac`)
- **Overshoot**: EMA drifted from L_T=1.6348 up to ~1.71-1.72 by step
  ~100 (LR-warmup end) and parked there. ~0.07-0.09 nats above floor.
- **Distribution**: 91% zero commits per round by round 21 — confirmed
  the c/2 boundary over-zeros Gaussian-ish weights.
- **Killed** at step ~1500.

### Run B — `lr=3e-4 wu=400 --target-zero-frac=0.38` (quantile boundary)
- **Distribution fix worked**: round 1 commits were 37/27/37 (neg/zero/pos),
  vs Run A's 4/91/4. By round 8 the per-round skew swung toward zero
  (the natural argmin trajectory: max-abs to ±c first, smallest-|w| to 0
  next). The end-state should still hit the quantile target.
- **Overshoot during warmup**: EMA stayed AT or slightly BELOW L_T for
  steps 0-400. Long LR ramp covering the no-commit phase kept drift
  near zero. **Big win.**
- **Overshoot post-warmup**: EMA jumped to ~1.72 immediately after
  commits started, then trended DOWN to ~1.67 over the next ~1000
  steps. Commits perturb, model recovers, EMA closes on L_T.
- **Killed** by disk-full at step ~1700 mid-checkpoint (unrelated to
  recipe).

### Run C — `lr=1e-3 wu=600 gap=0.05 min-steps-per-round=50`
- **Regression on overshoot**: even during the warmup window, EMA drifted
  UP to ~1.74 (0.10 above L_T). With zero commits in flight, the only
  cause was gradient updates injecting noise. lr=1e-3 is too aggressive
  for the math-preserving regime.
- **Commits stalled**: with `--commit-gap-threshold=0.05` but actual
  gap stuck at ~0.10, commits only fired on the `max_round_steps`
  safety cap.
- **Killed** at step ~900 (first commit just fired).

### Run D — `lr=3e-4 wu=400 gap=0.10 patience=50 min-steps-per-round=50` (✓ WINNER)
- Same LR/wu as Run B; **added** the new safety floor
  (`min-steps-per-round=50`, `commit-patience=50`) and slightly
  longer `commit-max-round-steps=300`.
- **Overshoot**: ~0 during warmup; post-first-commit peak ~+0.07; then
  trended **down through L_T** by step ~2000.
- At step 2060 / round 16: **EMA = 1.6323, below L_T = 1.6348**. First
  recipe where the gap closed to ≤ 0.
- Cadence ~130 steps/round (slower than B's ~70 but cleaner per commit).
- Letting it run to completion of all 64 rounds.

### Lessons
1. **LR is the dominant overshoot lever.** Lower LR = less drift on the
   math-preserving init. The optimizer's role here is "absorb commit
   perturbations," not "find a better optimum" — KL is already
   minimized at step 0. lr=3e-4 was good; lr=1e-3 was clearly too high.
2. **Long LR-warmup covering the no-commit phase is essential.** With
   wu=warmup-rounds-steps, the LR is small while the model is at
   step-0 optimum, so noise is minimized. After the first commit, full
   LR is needed to recover quickly. Setting `--warmup-steps =
   --warmup-rounds-steps` gives clean overshoot ≈ 0 during warmup.
3. **`--target-zero-frac` is required**, not optional. Without it (Mode
   B), the distribution collapses to ~91% zeros. Always set it
   (default 0.38 is fine).
4. **`--commit-gap-threshold` must be reachable.** If the steady-state
   overshoot is X nats, then `gap-threshold << X` stalls all commits to
   the max-round-steps cap. Set gap_threshold no tighter than what the
   recipe can plausibly achieve.
5. **Per-round target-distribution skew is a transient.** Argmin commits
   max-abs first (→ ±c), then smallest-|w| (→ 0) until small-|w|s
   exhaust. The per-round target_frac_zero swings 27% → 95% → and will
   swing back as small-|w|s run out. End-state matches the quantile
   target. Don't panic if rounds 5-15 look 95% zero.
6. **Disk: each `interrupted.pt` is ~2.4GB** (135M model weights, fp32
   optimizer state for `m + v`, best_snapshot clone, frozen buffers).
   Atomic rename needs 2.4GB old + 2.4GB new tmp = 4.8GB free per
   ckpt. Plan ckpt-every accordingly when disk is tight.

### Best recipe (Run D)
```
--optimizer cautious-adamw
--lr 3e-4
--warmup-steps 400 --warmup-rounds-steps 400   # ramp covers no-commit phase
--target-zero-frac 0.38
--c 0.85
--commit-gap-threshold 0.10   # reachable; 0.05 stalls (Run C)
--commit-patience 50 --min-steps-per-round 50
--commit-max-round-steps 300
--commit-tolerance 0.05
--scale-group-size 64   # SmolLM2-135M
--wd 0.001
```

Achieved EMA = 1.6323 (below L_T = 1.6348) at step 2060 / round 16 of
64. Full schedule needs ~12k more steps (~5-6 hours wall time at
~2.3s/step on RTX 4050 6GB).

## Hyperparameter starting points

| Flag | Default | Notes |
|------|---------|-------|
| `--c` | 0.85 | 18% headroom to ±1. Start here; 0.9 has less safety but is closer to deployment. |
| `--barrier-coef` | 1e-4 | Tiny — only outliers feel it. |
| `--momentum-weight` | 0.0 | Pure q-err greedy. Bump to 0.1-1.0 once basic loop works. |
| `--warmup-rounds-steps` | 200 | Pre-first-commit warmup; lets exp_avg populate. |
| `--commit-patience` | 50 | ~10x the fast-EMA window. |
| `--commit-tolerance` | 0.02 | nats; matches distill.py's α-schedule. |
| `--commit-gap-threshold` | 0.05 | Don't commit if fast_ema − L_T > 0.05. |
| `--commit-max-round-steps` | 2000 | Safety cap per round. |
| `--post-commit-momentum-damp` | 1.0 | Off by default; try 0.5 if overshoot. |
| `--scale-group-size` | 64 | SmolLM2-135M; 128 for Qwen3. |
| `--optimizer` | cautious-adamw | Bonsai recipe; sweep `--lr 3e-4..1e-2`. |
