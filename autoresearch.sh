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
RUN_N="001"
RUN_TAG="baseline-hrm-G"
DESCRIPTION="hrm-G recipe replay: random lognormal frozen scales, frozen non-embed FP, Lion-on-embed, Bop Bet1 tau_norm=0.15"

RUN_NAME="r${RUN_N}-${RUN_TAG}"
OUT_DIR="experiments/${RUN_NAME}"

# Defaults (mirror hrm-G-bop). Override below per experiment.
TOTAL_STEPS="${TOTAL_STEPS:-2500}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
TAU_NORM="${TAU_NORM:-0.15}"
GAMMA="${GAMMA:-1e-3}"
GAMMA_V="${GAMMA_V:-1e-3}"
LR="${LR:-5e-4}"
VAL_EVERY="${VAL_EVERY:-500}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"
EMA_WARMUP="${EMA_WARMUP:-200}"
# Extra flags as a single whitespace-separated string. The baseline replays
# hrm-G exactly:
EXTRA_FLAGS_STRING="${EXTRA_FLAGS_STRING:---random-scales --freeze-scales --freeze-non-embed-fp}"

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
# but if the parse above failed for any reason we'd leave one around. Be safe.
rm -f "$OUT_DIR/interrupted.pt" "$OUT_DIR/interrupted.pt.tmp"

exit 0
