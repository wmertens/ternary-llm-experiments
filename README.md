# smollmer

Ternarize HF causal LMs into the **Bonsai weight format** (per-(row × column-group) scales + ternary T) by **distilling** from the FP base model with a **curriculum of decreasing odd quantization levels** (so 0 is always representable):

```
257 → 129 → 65 → 33 → 17 → 9 → 5 → 3
```

Bonsai inference math, per `Linear`:

```
y[r, c] = sum_k T[r, k] * S[r, k // G] * x[c, k] + bias[r]
   T ∈ {-1, 0, +1}^{out × in},   S ∈ ℝ^{out × n_groups},   G = group_size
```

Each (row, column-group) of `G=128` consecutive input columns has its own scale. This finer scaling is the key fidelity lever: trained Bonsai-1.7B has ~62% nonzero density vs ~15% with plain per-row scales — the model can use most of its weights instead of forcing them to 0.

Quantized layers per transformer block: `q/k/v/o_proj`, `gate/up/down_proj`. `embed_tokens` and `lm_head` stay full precision (and may be int8-quantized at finalize for storage).

## Install

```bash
# Training (CUDA box, e.g. RTX 4050)
uv venv && source .venv/bin/activate
uv pip install -e .                           # CPU-only
# or for CUDA 12.x:
# uv pip install torch --index-url https://download.pytorch.org/whl/cu124
# uv pip install -e .

# Inference (AMD 780M box, CPU is the safe default)
uv pip install -e .
```

## Workflow

### 1. Cache teacher targets (run once)

Generates self-text from the FP teacher with min-P=0.05 sampling, then forwards through the teacher to extract top-K=64 + `rest_mass` per token.

```bash
smollmer-cache --out cache/ --total-tokens 10_000_000 \
               --seq-len 1024 --gen-batch 4 --logit-batch 1 --top-k 64
```

Disk: ~33 MB per shard of 128 sequences × 1024 tokens. 100M tokens ≈ 25 GB.

### 2. Distillation curriculum

```bash
smollmer-distill --cache-dir cache/ --out ckpts/ \
                 --batch-size 4 --grad-accum 4 \
                 --lr 2e-4 --autocast-dtype bfloat16 \
                 --scale-group-size 128
```

`--scale-group-size` must divide every projection's `in_features`. Defaults to **128** (Bonsai/Qwen3). For SmolLM2-135M (hidden=576, intermediate=1536) use 32 or 64.

Default curriculum (longer at low L where the codebook actually settles):
`(257,5000) (129,2000) (65,2000) (33,3000) (17,4000) (9,5000) (5,8000) (3,15000)`.

Override with `--curriculum 33:200,17:200,9:300,5:500,3:1000`.

Resume mid-curriculum: `--resume ckpts/stage_03_L33.safetensors --start-stage 4`.

### 3. Stage 2: freeze T, polish scales + RMSNorms

```bash
smollmer-finalize --cache-dir cache/ \
                  --resume ckpts/stage_07_L3.safetensors \
                  --out ckpts/ --steps 1000 --lr 5e-5
```

Writes `ckpts/final_packed.safetensors` (`format=smollmer-packed-bonsai-v1`):
- ternary trits at **1.58 bpw** (base-3, 5 trits per uint8 byte)
- per-(row, group) scales as fp16 `[out_features, n_groups]`
- norms / biases at `--store-dtype` (default bf16)
- embed_tokens / lm_head as int8 with per-row fp16 scale (disable with `--no-quant-embed`)

Total: ~1.7 bpw on the projections (1.58 trit + ~0.13 scale). Matches Bonsai's deployment overhead.

### 4. Inspect any stage

```bash
# On the AMD 780M box (CPU):
smollmer-chat --ckpt ckpts/stage_05_L9.safetensors --device cpu --dtype float32

# Final packed model:
smollmer-chat --ckpt ckpts/final_packed.safetensors --device cpu

# Force a different L on a high-precision checkpoint to see degradation:
smollmer-chat --ckpt ckpts/stage_00_L257.safetensors --levels 5
```

## Memory budget on RTX 4050 (6 GB)

With Lion32 + bf16 autocast, fp16 latents, fp32 optimizer state:

| component             | size (135M, default settings) |
|----------------------|------|
| latent weights (fp16) | 270 MB |
| per-(row, group) scales (fp32) | ~6 MB |
| grads (fp16)          | 270 MB |
| Lion32 state (fp32)   | 540 MB |
| q-cache (fp16, persistent across grad_accum) | 270 MB |
| frozen teacher        | (none — cached) |
| **subtotal**          | **~1.4 GB** |
| activations + KV      | rest |

If you OOM, add `--grad-checkpointing` (default on) and/or shrink `--batch-size`.

## Files

| file | purpose |
|------|---------|
| `smollmer/qlinear.py` | `QLinear` with per-(row, group) scales, mutable `levels`, STE, q-cache |
| `smollmer/pack.py` | 1.58 bpw base-3 ternary packing + int8 embed helpers |
| `smollmer/build_student.py` | swap projections in any HF causal LM, init per-(row, group) scales |
| `smollmer/cache_teacher.py` | self-text generation + top-K logit cache |
| `smollmer/distill.py` | curriculum loop, Lion32 / AdamW32 / CautiousAdamW (all with fp32 state), KL+rest-bucket loss, plateau controller (loss EMA + flip_rate gate) |
| `smollmer/finalize.py` | stage-2 freeze T + train scales/norms, write packed ckpt |
| `smollmer/chat.py` | interactive generation, auto-detects ckpt format |
| `smollmer/export_onnx.py` | materialize packed ckpt as a dense HF directory for ONNX export |

## Notes / divergences from upstream Bonsai

- Bonsai trains from scratch on ~3.8B tokens. We distill from the FP base — much cheaper at small scale, and the per-(row, group) scaling means we recover Bonsai's deployment fidelity without retraining from zero.
- Curriculum over odd L is novel to this repo (Bonsai is L=3 from step 0). The closest analog in the literature is Quant-Noise (FAIR, 2004.07320). Empirically the curriculum lets the model find a good ternary configuration smoothly rather than slamming into L=3 from random init.
- KL loss includes an explicit "rest mass" bucket so the student is penalized for putting probability outside the teacher's top-K (without it, ternary outliers can drift unchecked).
- Lion's sign-momentum has no implicit norm control on the latent weights, so we project them back into `[-1, 1]` per element after every opt step (`clamp_qlinear_weights`). The q-cache is invalidated each clamp, so the next forward sees the current latent.
- We use Lion32 / AdamW32 / CautiousAdamW (fp32 momentum, fp16 latents). fp16 latents save ~270 MB on a 135M model and have ~8× finer ULP than bf16 in the latents' actual `[-1, 1]` range, with no overflow risk by construction.
