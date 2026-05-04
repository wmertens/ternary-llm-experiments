# smollmer

Ternarize SmolLM2-135M into the **Bonsai weight format** (per-output-row scale + ternary T) by **distilling** from the FP base model with a **curriculum of decreasing odd quantization levels** (so 0 is always representable):

```
257 → 129 → 65 → 33 → 17 → 9 → 5 → 3
```

Bonsai inference math, per `Linear`:

```
y = (x @ T.T) * s + bias       T ∈ {-1,0,+1}^{out×in}, s ∈ ℝ^out
```

Quantized layers per transformer block: `q/k/v/o_proj`, `gate/up/down_proj`. `embed_tokens` and `lm_head` stay full precision.

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
                 --lr 2e-4 --autocast-dtype bfloat16
```

Default curriculum (shrunk early, longer at low L):
`(257,300) (129,300) (65,400) (33,600) (17,1000) (9,1500) (5,2500) (3,5000)`.

Override with `--curriculum 33:200,17:200,9:300,5:500,3:1000`.

Resume mid-curriculum: `--resume ckpts/stage_03_L33.safetensors --start-stage 4`.

### 3. Stage 2: freeze T, polish scales + RMSNorms

```bash
smollmer-finalize --cache-dir cache/ \
                  --resume ckpts/stage_07_L3.safetensors \
                  --out ckpts/ --steps 1000 --lr 5e-5
```

Writes `ckpts/final_packed.safetensors` with tightly-packed ternary weights + fp32 per-row scales + bf16 embed/lm_head/norms. Packing format is chosen automatically:

- plain ternary → **1.6 bpw** base-3 (5 trits per uint8)
- sherry-constrained ternary → **1.25 bpw** (5 bits per 4-trit block)

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

With Lion + bf16 autocast, fp32 latent weights:

| component             | size |
|----------------------|------|
| latent weights (fp32) | 540 MB |
| grads (bf16)          | 270 MB |
| Lion state (fp32)     | 540 MB |
| frozen teacher        | (none — cached) |
| **subtotal**          | **~1.4 GB** |
| activations + KV      | rest |

If you OOM, add `--grad-checkpointing` and/or shrink `--batch-size`.

## Files

| file | purpose |
|------|---------|
| `smollmer/qlinear.py` | `QLinear` with mutable `levels`, generalized quantizer, STE |
| `smollmer/pack.py` | 1.6 bpw base-3 packing + 1.25 bpw sherry packing (legacy 2 bpw kept for old ckpts) |
| `smollmer/permute.py` | math-preserving free-dim permutations to align weak weights with sherry block boundaries |
| `smollmer/build_student.py` | swap projections in any HF causal LM |
| `smollmer/cache_teacher.py` | self-text generation + top-K logit cache |
| `smollmer/distill.py` | curriculum loop, Lion, KL+rest-bucket loss |
| `smollmer/finalize.py` | stage-2 freeze T + train scales/norms, write packed ckpt |
| `smollmer/chat.py` | interactive generation, auto-detects ckpt format |

## Notes / divergences from upstream Bonsai

- Bonsai's `qlinear.py` initializes `scales=ones`; the paper text describes `scales=row_L2_norm`. We follow the paper (with a `/sqrt(in_features)` normalization to keep latent weights near unit magnitude at init).
- Bonsai trains from scratch on ~3.8B tokens. We distill from the FP base — much cheaper at 135M scale.
- KL loss includes an explicit "rest mass" bucket so the student is penalized for putting probability outside the teacher's top-K (without it, ternary outliers can drift unchecked).
- Curriculum over odd L is novel to this repo (Bonsai is L=3 from step 0). The closest analog in the literature is Quant-Noise (FAIR, 2004.07320).

## Sherry mode (1.25 bpw)

Plain ternary uses 3 states per weight, so the information-theoretic floor is `log2(3) ≈ 1.585` bpw (we pack at 1.6). The **sherry constraint** forces every block of 4 weights to contain *exactly one* zero — that gives `4 zero-positions × 2³ sign choices = 32 = 2⁵` states per block, i.e. **1.25 bpw**, a ~21% size win over plain ternary at the cost of a small accuracy hit.

The constraint is folded into training so it costs ~nothing at the end:

```bash
smollmer-distill ... --sherry --permute-each-stage
smollmer-finalize ... --resume ckpts/stage_07_L3.safetensors --out ckpts/
```

- `--sherry` enables `quantize_sherry` in the QLinear forward at every L of the curriculum (not just L=3): the smallest-|w| slot in each block is forced to 0 and any other slot that would round to 0 is bumped to ±`1/half`. Gradient flows via STE as usual.
- `--permute-each-stage` permutes the *free* dimensions before each curriculum stage so the weakest columns line up with sherry block position 0. "Free" = MLP intermediate dim and per-KV-head V/O dim — these can be permuted with paired adjustments to neighbouring matrices so the forward pass is unchanged. The residual stream and RoPE-rotated Q/K dims are not free and are skipped. See `smollmer/permute.py` for the math.

Stage checkpoints record `sherry=1` in metadata so `finalize` and `chat` re-enable the constraint automatically. `finalize` then writes the 1.25 bpw packed format.
