# Ternary LLM experiments

This repo contains a bunch of experiments done with the help of Claude, working towards a ternary-valued LLM trained from scratch on a single 6 GB RTX 4050.

There are three lines of research, roughly in chronological order.

## 1. Converting SmolLM2-135M to ternary

The first step was distilling SmolLM2-135M into the Bonsai weight format (ternary weights with per-(row, column-group) scales), using a curriculum of decreasing quantization levels (257 → 129 → … → 3). See [smollm-distill.md](smollm-distill.md) and [smollmer/FINDINGS.md](smollmer/FINDINGS.md).

- The best attempt uses a Hestia-style sequenced QAT ([smollmer/qat_smooth.py](smollmer/qat_smooth.py)). It gives the best results so far, with a clear quality gap to the FP model still to close.
- Weight decay plus a nearest-ternary L2 attractor makes zeros "absorbing": the zero fraction drifts from 33% to 50%. A smooth per-group triple-well attractor with no weight decay fixes most of that drift.
- There's a browser demo of the result in [docs/](docs/).

## 2. Ternary HRM from scratch

Next up was training a recurrent HRM-style transformer from scratch, fully ternary. See [autoresearch.md](autoresearch.md), [autoresearch-dashboard.md](autoresearch-dashboard.md) and [experiments/worklog.md](experiments/worklog.md).

- **Cautious Muon with STE on latent weights (CMuon-STE) beats Bop** (flip-based ternary optimizer) by a wide margin. The cautious mask matters.
- **Full BPTT is needed**; the 1-step gradient approximation from the HRM paper keeps the loop layers from training.
- **Recurrence didn't earn its compute** at this scale: a 1×1 (no loop) model ties 2×3 cycles while running ~50% faster.
- **Randomizing the number of cycles per step makes the model converge to a true fixed point**, stable from 2 to 24 loops, but extra loops don't lower the loss.
- An FP-weights control with a tuned LR lands ~0.3 nats lower; FP wants a ~20× lower CMuon LR than ternary and shows the same fixed-point behavior.

This line is parked in favor of a plain GPT, to measure ternary training without the recurrence confound.

## 3. Ternary GPT from scratch

The current line: a plain decoder-only GPT with **98.5% of parameters ternary**, including the embeddings. See [autoresearch_gpt.ideas.md](autoresearch_gpt.ideas.md) and [autoresearch_gpt.jsonl](autoresearch_gpt.jsonl).

Current recipe (8 layers, 512 wide, ~51M params): ternary embeddings, CMuon-STE (lr 0.15, cautious, 4 Newton-Schulz steps) on the trits, Lion on the remaining FP params, trainable per-(row, 128-column) scales, 90% zero trits at init, effective batch 64, cosine schedule, fp16 latents and scales, `torch.compile`.

Best validation loss on FineWeb-Edu: **3.70** (30k steps). With the current snapshot of the training data the best is 3.75; data from earlier runs gave ~0.05 lower loss and can no longer be reproduced.

What mattered:

- **Ternary embeddings work** once the scales are trainable. At equal compute they end up within ~0.05 nats of FP embeddings.
- **Fine-grained scales are essential.** Per-tensor (BitNet-style) scales fail catastrophically, per-row costs ~0.13 nats; group size 128 is as good as 64.
- **Start mostly zero.** Initial zero fraction 0.90 beats 0.33–0.80; the model then fills in to ~65% zeros.
- **Depth helps.** 8 layers beat 6, and 12 layers are better again per step (but slower per wall-clock hour).
- Larger effective batch, 4 Newton-Schulz steps instead of 5, and not sharing K/V projections all helped.

Things that are free (no quality cost):

- int8 per-token activations (fp8 and mu-law int8 were slightly worse).
- fp16 scales and fp16 latents. bf16 latents look better early but lose in the final anneal.
- `torch.compile`: a ~1.3× speedup.

Things that didn't help at this scale:

- Variable-width (⊗-former) layers, State-Prediction Separation, sandwich norm, per-layer Newton-Schulz schedules.
- WSD and plateau-triggered LR restarts versus plain cosine for long runs.

## Next

Scaling up the GPT recipe, a real int8×ternary kernel for inference, and later ternary text diffusion ([autoresearch-text-diffusion.ideas.md](autoresearch-text-diffusion.ideas.md)).
