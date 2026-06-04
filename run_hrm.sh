#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/run/opengl-driver/lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source .venv/bin/activate

OUT="${OUT:-smollmer/ckpts.hrm-A-bop}"
RUN="${RUN:-hrm-A-bop}"
TOTAL="${TOTAL:-40000}"
BS="${BS:-2}"
GA="${GA:-8}"
WORKERS="${WORKERS:-1}"
TAU_NORM="${TAU_NORM:-0.5}"
GAMMA="${GAMMA:-1e-3}"
GAMMA_V="${GAMMA_V:-1e-3}"
LR="${LR:-5e-4}"
# Bop-isolation flags, off by default; pass any non-empty to enable.
RANDOM_SCALES="${RANDOM_SCALES:-}"
FREEZE_SCALES="${FREEZE_SCALES:-}"
FREEZE_NON_EMBED_FP="${FREEZE_NON_EMBED_FP:-}"
FREEZE_TRITS="${FREEZE_TRITS:-}"

mkdir -p "$OUT"
EXTRA_FLAGS=()
[ -n "$RANDOM_SCALES" ] && EXTRA_FLAGS+=("--random-scales")
[ -n "$FREEZE_SCALES" ] && EXTRA_FLAGS+=("--freeze-scales")
[ -n "$FREEZE_NON_EMBED_FP" ] && EXTRA_FLAGS+=("--freeze-non-embed-fp")
[ -n "$FREEZE_TRITS" ] && EXTRA_FLAGS+=("--freeze-trits")

exec python -u -m smollmer.hrm_bop \
    --out "$OUT" --run-name "$RUN" \
    --total-steps "$TOTAL" \
    --batch-size "$BS" --grad-accum "$GA" \
    --num-workers "$WORKERS" \
    --tau-norm "$TAU_NORM" \
    --gamma "$GAMMA" --gamma-v "$GAMMA_V" \
    --lr "$LR" \
    "${EXTRA_FLAGS[@]}"
