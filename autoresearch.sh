#!/usr/bin/env bash
# autoresearch.sh — single experiment runner for ternary HRM fast-recipe search.
#
# The harness edits this file each iteration to set RUN_NAME + DESCRIPTION
# + any knob overrides, then invokes `./autoresearch.sh`. The script blocks
# until the trainer finishes (or crashes), prints METRIC lines, exits 0/non-0.
#
# All TB data accumulates under ./tb/<RUN_NAME>/ so they compare in one
# TensorBoard.

set -euo pipefail
cd "$(dirname "$0")"

# ---- Cluster / sandbox setup -------------------------------------------------
export LD_LIBRARY_PATH=/run/opengl-driver/lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# shellcheck disable=SC1091
source .venv/bin/activate

# ---- Per-experiment config (EDITED BY HARNESS) -------------------------------
# Always advance RUN_N + RUN_TAG for each new experiment.
RUN_N="043"
RUN_TAG="phaseB-fastA-openmath-resume-r042"
DESCRIPTION="PHASE B fine-tune from r042's fixpoint init (per ideas#H + Topological Trouble paper note). Resume r042 final.safetensors (val 4.14, c2..c24 flat at 4.13-4.14, per_loop_gap +0.011). Shift training to OpenMath-heavy mix (80pct openmath, 20pct cosmopedia for language stability). Val source = openmath: the cycle sweep at val now measures per-loop CE on MATH/REASONING tokens, not FineWeb. Lower lr (0.05 cosine to 0.005, vs Phase A's 0.20) to preserve the fixpoint while specialising. Question: does per_loop_gap re-open POSITIVELY on the OpenMath cycle sweep (c4 < c2 < c1 → depth helps reasoning), or stay flat (Topological Trouble paper prediction: depth recurrence alone insufficient)? 5000 more steps. KEEP_SAFETENSORS=1 in case r044 wants further phases."
KEEP_SAFETENSORS=1

RUN_NAME="r${RUN_N}-${RUN_TAG}"
OUT_DIR="experiments/${RUN_NAME}"

# Defaults (mirror hrm-G-bop). Override below per experiment.
TOTAL_STEPS="${TOTAL_STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
# FAST-A (38M) — surrogate for the curriculum experiment.
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"
NUM_HEADS="${NUM_HEADS:-8}"
INTERMEDIATE="${INTERMEDIATE:-1408}"
H_LAYERS="${H_LAYERS:-2}"
L_LAYERS="${L_LAYERS:-2}"
H_CYCLES="${H_CYCLES:-2}"
L_CYCLES="${L_CYCLES:-3}"
FULL_BPTT_STEPS="${FULL_BPTT_STEPS:-99999}"
TAU_NORM="${TAU_NORM:-0.15}"
GAMMA="${GAMMA:-1e-3}"
GAMMA_V="${GAMMA_V:-1e-3}"
LR="${LR:-5e-4}"
VAL_EVERY="${VAL_EVERY:-500}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"
EMA_WARMUP="${EMA_WARMUP:-200}"
# Extra flags as a single whitespace-separated string. The baseline replays
# hrm-G exactly:
# Run 43: PHASE B — resume r042 final.safetensors, OpenMath-heavy mix, fine-tune lr.
EXTRA_FLAGS_STRING="${EXTRA_FLAGS_STRING:---random-scales --freeze-scales --freeze-non-embed-fp --ste-trits --c-muon --muon-lr 0.05 --muon-lr-floor 0.005 --grad-mode full-bptt --min-h-cycles 1 --max-h-cycles 4 --eval-cycle-sweep 1,2,3,4,6,8,12,16,24,32,48,64 --resume experiments/r042-phaseA-fastA-varcycles-1to4-5000steps/final.safetensors --data-mix cosmopedia:0.2,openmath:0.8 --val-source openmath}"

mkdir -p "$OUT_DIR" tb

# ---- Pre-checks (fast: must complete in <1s) ---------------------------------
python -c "
import smollmer.hrm_bop  # importable?
import smollmer.hrm_model
import smollmer.hrm_data
import smollmer.flip_distill
import smollmer.qlinear
" >/dev/null 2>&1 || { echo 'pre-check FAIL: smollmer modules do not import'; exit 2; }

# ---- Run the trainer ---------------------------------------------------------
# shellcheck disable=SC2206
EXTRA_ARGS=( $EXTRA_FLAGS_STRING )

START_TIME=$(date +%s)

python -u -m smollmer.hrm_bop \
    --out "$OUT_DIR" \
    --run-name "$RUN_NAME" \
    --tb-dir tb \
    --total-steps "$TOTAL_STEPS" \
    --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM" \
    --hidden-size "$HIDDEN_SIZE" \
    --num-attention-heads "$NUM_HEADS" \
    --num-kv-heads "$NUM_HEADS" \
    --intermediate-size "$INTERMEDIATE" \
    --H-layers "$H_LAYERS" --L-layers "$L_LAYERS" \
    --H-cycles "$H_CYCLES" --L-cycles "$L_CYCLES" \
    --full-bptt-steps "$FULL_BPTT_STEPS" \
    --tau-norm "$TAU_NORM" --gamma "$GAMMA" --gamma-v "$GAMMA_V" \
    --lr "$LR" \
    --val-every "$VAL_EVERY" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --ema-warmup "$EMA_WARMUP" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$OUT_DIR/train.log"

EXIT=${PIPESTATUS[0]}
END_TIME=$(date +%s)
WALL=$(( END_TIME - START_TIME ))

if [ "$EXIT" -ne 0 ]; then
    echo "TRAINER EXIT=$EXIT" >&2
    exit "$EXIT"
fi

# ---- Parse metrics from log + TB --------------------------------------------
# tqdm uses CR not LF for progress updates, so [val] lines end up on the same
# physical line as the surrounding tqdm bar. Convert CR→LF before grep.
VAL_LINE=$(tr '\r' '\n' < "$OUT_DIR/train.log" | grep '^\[val\]' | tail -1 || true)
if [ -z "$VAL_LINE" ]; then
    echo "ERROR: no [val] line in $OUT_DIR/train.log" >&2
    exit 3
fi
VAL_LOSS=$(echo "$VAL_LINE" | sed -n 's/.*loss=\([0-9.]*\).*/\1/p')

# Last loss/ema postfix value:
EMA=$(tail -200 "$OUT_DIR/train.log" \
    | tr '\r' '\n' \
    | grep -oE 'ema=[0-9.]+' \
    | tail -1 \
    | sed 's/ema=//' \
    || true)

# TB-extracted final values (flip_rate, frac_zero delta, per-loop gap).
python <<PYEOF
import math
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
tb_dir = Path("tb/$RUN_NAME")
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
pl0 = last('diag/per_loop_ce_0')
pl1 = last('diag/per_loop_ce_1')
if flip_rate is not None and math.isfinite(flip_rate):
    print(f"METRIC flip_rate={flip_rate:.6e}")
if fz_first is not None and fz_last is not None:
    print(f"METRIC frac_zero_delta={fz_last - fz_first:+.6f}")
if pl0 is not None and pl1 is not None:
    print(f"METRIC per_loop_gap={pl0 - pl1:+.4f}")
PYEOF

# Primary metric LAST so anything after it (debugging echoes) doesn't
# accidentally get parsed first.
echo "METRIC wall_seconds=${WALL}"
[ -n "$EMA" ]      && echo "METRIC loss_ema=${EMA}"
echo "METRIC val_loss=${VAL_LOSS}"

# Belt-and-braces cleanup: hrm_bop.py deletes interrupted.pt on success,
# but if the parse above failed for any reason we'd leave one around.
# IMPORTANT: don't delete here if total_steps wasn't reached. The trainer
# exits success (sys.exit(0)) after SIGINT too — in that case we want to
# keep interrupted.pt for resume.
LAST_STEP=$(tr '\r' '\n' < "$OUT_DIR/train.log" \
    | grep -oE 'step=[0-9]+' | tail -1 | sed 's/step=//')
if [ -n "$LAST_STEP" ] && [ "$LAST_STEP" -ge "$TOTAL_STEPS" ]; then
    rm -f "$OUT_DIR/interrupted.pt" "$OUT_DIR/interrupted.pt.tmp"
fi

# Each run writes ~200-400 MB of final*.safetensors. We don't need them
# between experiments — TB scalars are the durable signal, and the
# safetensors can be re-created cheaply by re-running with seed. Delete
# by default to keep experiments/ from filling the disk. Set
# KEEP_SAFETENSORS=1 in this script's per-experiment block to preserve
# the safetensors for a downstream resume / phase-B fine-tune.
if [ -z "${KEEP_SAFETENSORS:-}" ]; then
    rm -f "$OUT_DIR"/*.safetensors
else
    echo "[harness] KEEP_SAFETENSORS=1 → preserving $OUT_DIR/*.safetensors"
fi

exit 0
