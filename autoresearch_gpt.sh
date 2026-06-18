#!/usr/bin/env bash
# autoresearch_gpt.sh — single experiment runner for the ternary nanoGPT
# research line (vanilla decoder, no recurrence, focused on ternary speedups).
# Mirrors autoresearch.sh's structure but writes to a separate output tree
# so the HRM and GPT results stay distinct.
#
#   experiments_gpt/<RUN_NAME>/  — model, train.log, checkpoints
#   tb_gpt/<RUN_NAME>/           — TB scalars
#   autoresearch_gpt.jsonl       — JSON state
#   autoresearch_gpt.ideas.md    — ideas backlog
#
# The harness edits this file each iteration to set RUN_NAME + DESCRIPTION
# + any knob overrides, then invokes `./autoresearch_gpt.sh`. The script
# blocks until the trainer finishes (or crashes), prints METRIC lines,
# exits 0/non-0.

set -euo pipefail
cd "$(dirname "$0")"

# ---- Cluster / sandbox setup -------------------------------------------------
export LD_LIBRARY_PATH=/run/opengl-driver/lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# shellcheck disable=SC1091
source .venv/bin/activate

# ---- Per-experiment config (EDITED BY HARNESS) -------------------------------
# Always advance RUN_N + RUN_TAG for each new experiment.
RUN_N="008"
RUN_TAG="trit-embeddings-lr0p15-bs4"
DESCRIPTION="Phase 5a: TRITISE THE EMBEDDINGS. Same baseline as g005 (lr=0.15 + share-kv + bf16 m + bs=4 ga=8) + --trit-embeddings. New QEmbedding class: 25M FP embedding → 25M ternary latents + 393K FP per-(row,group) scales. Tied lm_head sees the quantised+scaled table too (no asymmetric ternary-in/FP-out). FP residual drops from 25.45M → 0.68M (-97pct of FP). Compare against g005 4.1147 (best eff=32 baseline). Pass: any val < ~4.5 means tritised embeddings work; <4.2 means competitive. Fail: model collapses or val > 5 → likely need unfrozen scales or different init scheme. ~2h ETA."

RUN_NAME="g${RUN_N}-${RUN_TAG}"
OUT_DIR="experiments_gpt/${RUN_NAME}"

# Defaults: fast-A scale (38M ternary). Override per experiment as needed.
TOTAL_STEPS="${TOTAL_STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"
NUM_HEADS="${NUM_HEADS:-8}"
INTERMEDIATE="${INTERMEDIATE:-1408}"
NUM_LAYERS="${NUM_LAYERS:-6}"
TAU_NORM="${TAU_NORM:-0.15}"
GAMMA="${GAMMA:-1e-3}"
GAMMA_V="${GAMMA_V:-1e-3}"
LR="${LR:-5e-4}"
VAL_EVERY="${VAL_EVERY:-500}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"
EMA_WARMUP="${EMA_WARMUP:-200}"
# Extra flags as a single whitespace-separated string. Baseline recipe:
EXTRA_FLAGS_STRING="${EXTRA_FLAGS_STRING:---random-scales --freeze-scales --freeze-non-embed-fp --ste-trits --c-muon --muon-lr 0.15 --muon-lr-floor 0.015 --share-kv --cmuon-state-dtype bfloat16 --trit-embeddings}"

mkdir -p "$OUT_DIR" tb_gpt experiments_gpt

# ---- Pre-checks --------------------------------------------------------------
python -c "
import smollmer.gpt_bop
import smollmer.gpt_model
import smollmer.hrm_data
import smollmer.flip_distill
import smollmer.qlinear
" >/dev/null 2>&1 || { echo 'pre-check FAIL: smollmer modules do not import'; exit 2; }

# ---- Run the trainer ---------------------------------------------------------
# shellcheck disable=SC2206
EXTRA_ARGS=( $EXTRA_FLAGS_STRING )

START_TIME=$(date +%s)

python -u -m smollmer.gpt_bop \
    --out "$OUT_DIR" \
    --run-name "$RUN_NAME" \
    --tb-dir tb_gpt \
    --total-steps "$TOTAL_STEPS" \
    --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM" \
    --hidden-size "$HIDDEN_SIZE" \
    --num-attention-heads "$NUM_HEADS" \
    --num-kv-heads "$NUM_HEADS" \
    --intermediate-size "$INTERMEDIATE" \
    --num-layers "$NUM_LAYERS" \
    --tau-norm "$TAU_NORM" --gamma "$GAMMA" --gamma-v "$GAMMA_V" \
    --lr "$LR" \
    --val-every "$VAL_EVERY" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --ema-warmup "$EMA_WARMUP" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$OUT_DIR/train.log"

EXIT=${PIPESTATUS[0]}
END_TIME=$(date +%s)
WALL=$(( END_TIME - START_TIME ))

if [ "$EXIT" -ne 0 ]; then
    echo "TRAINER EXIT=$EXIT" >&2
    exit "$EXIT"
fi

# ---- Parse metrics from log + TB --------------------------------------------
VAL_LINE=$(tr '\r' '\n' < "$OUT_DIR/train.log" | grep '^\[val\]' | tail -1 || true)
if [ -z "$VAL_LINE" ]; then
    echo "ERROR: no [val] line in $OUT_DIR/train.log" >&2
    exit 3
fi
VAL_LOSS=$(echo "$VAL_LINE" | sed -n 's/.*loss=\([0-9.]*\).*/\1/p')

EMA=$(tail -200 "$OUT_DIR/train.log" \
    | tr '\r' '\n' \
    | grep -oE 'ema=[0-9.]+' \
    | tail -1 \
    | sed 's/ema=//' \
    || true)

python <<PYEOF
import math
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
tb_dir = Path("tb_gpt/$RUN_NAME")
ea = event_accumulator.EventAccumulator(str(tb_dir),
    size_guidance={event_accumulator.SCALARS: 0})
ea.Reload()
def last(tag):
    if tag not in ea.Tags()['scalars']:
        return None
    seq = ea.Scalars(tag)
    return seq[-1].value if seq else None
def first(tag):
    if tag not in ea.Tags()['scalars']:
        return None
    seq = ea.Scalars(tag)
    return seq[0].value if seq else None
flip_rate = last('bop/flip_rate')
fz_first = first('trits/frac_zero')
fz_last  = last('trits/frac_zero')
if flip_rate is not None and math.isfinite(flip_rate):
    print(f"METRIC flip_rate={flip_rate:.6e}")
if fz_first is not None and fz_last is not None:
    print(f"METRIC frac_zero_delta={fz_last - fz_first:+.6f}")
PYEOF

echo "METRIC wall_seconds=${WALL}"
[ -n "$EMA" ]      && echo "METRIC loss_ema=${EMA}"
echo "METRIC val_loss=${VAL_LOSS}"

# Cleanup safetensors unless KEEP_SAFETENSORS is set in the per-experiment block.
LAST_STEP=$(tr '\r' '\n' < "$OUT_DIR/train.log" \
    | grep -oE 'step=[0-9]+' | tail -1 | sed 's/step=//')
if [ -n "$LAST_STEP" ] && [ "$LAST_STEP" -ge "$TOTAL_STEPS" ]; then
    rm -f "$OUT_DIR/interrupted.pt" "$OUT_DIR/interrupted.pt.tmp"
fi
if [ -z "${KEEP_SAFETENSORS:-}" ]; then
    rm -f "$OUT_DIR"/*.safetensors
else
    echo "[harness] KEEP_SAFETENSORS=1 → preserving $OUT_DIR/*.safetensors"
fi

exit 0
