# Autoresearch Dashboard: ternary-hrm-fast-recipe

## Segment 0 — main HRM 153M runs, 2500-step budget

**Runs:** 3 | **Kept:** 3 | **Discarded:** 0 | **Crashed:** 0
**Baseline:** val_loss: 6.9074 nats (#1)
**Best:** val_loss: 6.5320 nats (#3, -5.4%)

| # | commit | val_loss | loss_ema | flip_rate | Δfrac_zero | per_loop_gap | wall_s | status | description |
|---|--------|----------|----------|-----------|------------|--------------|--------|--------|-------------|
| 1 | 64dfda5 | 6.9074 | 6.8269 | 3.86e-6 | -0.011 | 0.039 | 19189 | keep | baseline hrm-G replay |
| 2 | 1c018e7 | 6.8893 (-0.26%) | 6.8183 | 4.07e-6 | -0.011 | 0.065 | 21109 | keep | + cautious mask on Bop |
| 3 | 4e87cd6 | **6.5320 (-5.4%)** | 6.4223 | 2.63e-4 | -0.019 | 0.032 | 20333 | keep | replace Bop with CMuon-STE |

## Segment 1 — tiny-non-loop screening (26M model, 1500 steps)

Goal: optimizer head-to-head + precision tricks, no recurrence confound.

**Runs:** 3 (+1 in flight) | **Round 1 winner:** CMuon-STE

| # | commit | val_loss | loss_ema | flip_rate | Δfrac_zero | wall_s | status | description |
|---|--------|----------|----------|-----------|------------|--------|--------|-------------|
| 4 | 4e87cd6 | 5.4674 | 5.2768 | 1.71e-5 | -0.046 | 1927 | keep | s1: Bop+cautious (round 1) |
| 5 | 4e87cd6 | **5.1850 (-5.2%)** | 4.9728 | 3.94e-4 | -0.004 | 1946 | keep | s2: CMuon-STE (round 1) |
| 6 | 4e87cd6 | 5.5221 (+1.0%) | 5.3654 | 2.64e-4 | -0.006 | 1679 | discard | s3: Lion-STE (round 1) |
| 7 | — | — | — | — | — | — | running | s4: CMuon-STE + int8 act (round 2) |

**Round 1 take:** CMuon-STE beats both alternatives on the tiny model.
The Bop result is consistent with Run 3 (where CMuon beat Bop by 0.36
on the big model; here by 0.28). Lion-STE underperforms despite
bitlooplm's reported success — likely the budget (1500 steps too
short for STE's slower per-step convergence) or the no-trainable-scale
restriction.

**Round 2** stacks int8 activations / fp16 opt state on CMuon-STE to
test which precision tricks are free wins.
