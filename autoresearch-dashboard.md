# Autoresearch Dashboard: ternary-hrm-fast-recipe

**Runs:** 2 | **Kept:** 2 | **Discarded:** 0 | **Crashed:** 0
**Baseline:** val_loss: 6.9074 nats (#1)
**Best:** val_loss: 6.8893 nats (#2, -0.26%)

| # | commit | val_loss | loss_ema | flip_rate | Δfrac_zero | per_loop_gap | wall_s | status | description |
|---|--------|----------|----------|-----------|------------|--------------|--------|--------|-------------|
| 1 | 64dfda5 | 6.9074 | 6.8269 | 3.86e-6 | -0.011 | 0.039 | 19189 | keep | baseline hrm-G replay |
| 2 | 1c018e7 | 6.8893 (-0.26%) | 6.8183 | 4.07e-6 | -0.011 | 0.065 | 21109 | keep | + cautious mask on Bop |
