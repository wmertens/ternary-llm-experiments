# Autoresearch Dashboard: ternary-hrm-fast-recipe

## Segment 0 — main HRM runs

**Runs:** 18 kept + 7 discarded
**Baseline:** val_loss: 6.9074 nats (#1)
**Best:** val_loss: **4.1562 nats (#25, -39.8%)** — main 153M + full BPTT + CMuon lr=0.20 cosine

| # | commit | val_loss | loss_ema | flip_rate | wall_s | status | description |
|---|--------|----------|----------|-----------|--------|--------|-------------|
| 1 | 64dfda5 | 6.9074 | 6.8269 | 3.86e-6 | 19189 | keep | baseline hrm-G replay (153M) |
| 2 | 1c018e7 | 6.8893 | 6.8183 | 4.07e-6 | 21109 | keep | + cautious mask on Bop |
| 3 | 4e87cd6 | 6.5320 | 6.4223 | 2.63e-4 | 20333 | keep | replace Bop with CMuon-STE |
| 10 | (s4) | 6.5510 | 6.4606 | 2.68e-4 | 4270 | keep | fast-A 38M baseline (val ≈ Run 3) |
| 12 | (s6) | 4.9572 | 4.6984 | 3.52e-4 | 6508 | keep | fast-A + full BPTT throughout |
| 13 | ea2d2cb | 4.7779 | 4.5447 | 1.25e-4 | 33408 | keep | main 153M + full BPTT throughout |
| 14 | ea2d2cb | 5.0628 | 4.8162 | 2.72e-4 | 6537 | discard | non-cautious CMuon (+0.11, cautious helps) |
| 16 | ea2d2cb | 4.5923 | 4.3500 | 1.03e-3 | 6736 | keep | fast-A CMuon lr=0.05 (higher LR wins) |
| 18 | ea2d2cb | 4.5001 | 4.2680 | 1.82e-3 | 6544 | keep | fast-A CMuon lr=0.10 |
| 19 | ea2d2cb | 4.5808 | 4.3409 | 2.98e-3 | 6557 | discard | fast-A CMuon lr=0.20 (peak past 0.10 const) |
| 20 | ea2d2cb | 4.2905 | 4.0523 | 1.18e-3 | 33878 | keep | main 153M lr=0.10 + cautious |
| 22 | ea2d2cb | 4.3607 | 4.1128 | 6.36e-4 | 6738 | keep | fast-A lr=0.20 cosine→0.02 (best fast-A 2500) |
| 23 | ea2d2cb | 4.3566 | 4.1175 | 9.56e-4 | 6542 | keep | fast-A lr=0.40 cosine (plateau 0.20–0.40) |
| 25 | ea2d2cb | **4.1562** | 3.9286 | 2.35e-4 | 29598 | keep | **main 153M lr=0.20 cosine→0.02** ★ |
| 26 | ea2d2cb | 4.3479 | 4.1015 | 6.44e-4 | 5517 | keep | fast-A cycles 1×3 (H cycle redundant) |
| 27 | ea2d2cb | 4.3524 | 4.1252 | 6.71e-4 | 4464 | keep | fast-A cycles 1×1 (no recurrence, fastest) |
| 28 | ea2d2cb | 4.1706 | 3.7973 | 7.26e-4 | 17380 | keep | fast-A 2×3, 5000 steps (= main 153M!) |
| 29 | ea2d2cb | 4.1783 | 3.8226 | 7.51e-4 | 8131 | keep | fast-A 1×1, 5000 steps (53% faster, tied) |
| 31 | ea2d2cb | 4.1863 | 3.8215 | 7.09e-4 | 19522 | keep | fast-A var H_cycles [1,4] — FIXPOINT (gap 0.004) |
| 32 | cc165f1 | 4.1910 | 3.8169 | 7.14e-4 | 30463 | keep | fast-A var H_cycles [1,8] — fixpoint robust (gap 0.016) |

(Discarded along the way: #9 int8-act main, #11 full-BPTT→1-step, #15 lr=0.01,
#17 last-per-cycle, #24 cosine+warmup, #30 interrupted pivot.)

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
6. **CMuon lr=0.20 cosine→0.02 is the winning trit-optimizer setting.**
   LR sweep peaked at 0.10 constant (Run 18) and 0.20–0.40 with cosine
   (Runs 22/23 tied); warmup neutral (Run 24). Drove the 153M champion
   to val 4.1562 (Run 25).
7. **Recurrence doesn't earn its compute at this horizon.** fast-A 1×1
   (no loop) ties 2×3 at both 2500 and 5000 steps (Runs 27/29) while
   running 34–53% faster. The per-loop CE gap in fixed-cycle runs (~0.39)
   reflects under-trained loop layers, not useful iterative refinement.
8. **Variable per-step H_cycles forces a TRUE fixed point.** Sampling
   H_cycles uniformly each step collapses the per-loop CE gap ~100×
   (0.39 → 0.004 at [1,4], Run 31) with val unchanged — the model
   becomes robust to loop count, i.e. it actually converges. Robust to a
   2× wider range (Run 32 [1,8]: gap 0.016, still ~24× below fixed) but
   [1,4] is the sweet spot: cleaner fixpoint at 56% less wall time, since
   wall scales with the mean loop count. This is the property the HRM
   theory predicts but fixed-cycle training never delivered.
