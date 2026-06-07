# Autoresearch Dashboard: ternary-hrm-fast-recipe

## Segment 0 — main HRM runs

**Runs:** 6 kept + 2 discarded + 1 crashed/running
**Baseline:** val_loss: 6.9074 nats (#1)
**Best:** val_loss: **4.9572 nats (#12, -28.2%)** — fast-A + full BPTT

| # | commit | val_loss | loss_ema | flip_rate | wall_s | status | description |
|---|--------|----------|----------|-----------|--------|--------|-------------|
| 1 | 64dfda5 | 6.9074 | 6.8269 | 3.86e-6 | 19189 | keep | baseline hrm-G replay (153M) |
| 2 | 1c018e7 | 6.8893 | 6.8183 | 4.07e-6 | 21109 | keep | + cautious mask on Bop |
| 3 | 4e87cd6 | 6.5320 | 6.4223 | 2.63e-4 | 20333 | keep | replace Bop with CMuon-STE |
| 9 | c553358 | n/a | 7.1094 | — | 5790 | discard | int8 act on main, +50% wall (interrupted) |
| 10 | (s4) | 6.5510 | 6.4606 | 2.68e-4 | 4270 | keep | fast-A 38M baseline (val ≈ Run 3) |
| 11 | (s5) | 6.5714 | 6.4869 | 2.77e-4 | 3606 | discard | fast-A + full-BPTT 500 → 1-step. val_step500=6.21! |
| 12 | (s6) | **4.9572** | 4.6984 | 3.52e-4 | 6508 | keep | **fast-A + full BPTT throughout** ★ |
| 13 | running | tbd | tbd | tbd | tbd | running | main 153M + full BPTT, bs=1 ga=32, ~10h ETA |

## Segment 1 — tiny non-loop screening (26M model, 1500 steps)

**Best:** val_loss: 5.1848 nats (#7, CMuon-STE + int8 act ≡ noise vs s2)

| # | commit | val_loss | wall_s | status | description |
|---|--------|----------|--------|--------|-------------|
| 4 | 4e87cd6 | 5.4674 | 1927 | keep | s1: Bop+cautious |
| 5 | 4e87cd6 | 5.1850 | 1946 | keep | s2: CMuon-STE (winner round 1) |
| 6 | 4e87cd6 | 5.5221 | 1679 | discard | s3: Lion-STE |
| 7 | c553358 | 5.1848 | 1930 | keep | s4: CMuon-STE + int8 act (free) |
| 8 | c553358 | 5.1962 | 1869 | discard | s5: CMuon-STE + fp16 m (+0.011) |

## Headline findings

1. **CMuon-STE >> BopTernary** on HRM ternary training. Confirmed at 38M
   (screening) and 153M (Run 3). Run 12 with full BPTT gets val 4.96.
2. **1-step gradient is the bottleneck** on HRM in our setup, not the
   feature the HRM-Text paper claims. Full BPTT through all 8 stack
   applications is required for the recurrence to actually train the
   loop layers — switching to 1-step mid-training visibly degrades the
   model (Run 11: val went 6.21 → 6.57 over the 1-step phase).
3. **fast-A (38M HRM) is a good iteration surrogate**: 38M reaches val
   6.55 at 1.2h vs 153M's val 6.53 at 5.6h (Δ 0.02 nats, 5× speedup).
   Same conclusions about optimizer/gradient approximation.
4. int8 activations are free at small scale (noise-level Δ) but
   add ~45% wall time on main runs — discard for fastest-recipe metric.
5. fp16 opt state has a small (+0.011 nats) penalty on CMuon — discard
   unless memory pressure forces it.
