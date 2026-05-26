#!/usr/bin/env bash
# Regenerate cache_corpus with proper stratified cadence-60 mixing + CE filter.
#
# Why: the prior cache_corpus was generated pre-stratification and unfiltered.
# Its noisy / OCR-broken sequences pushed single-batch KL stdev to ~0.17 nats
# (n=64) — large enough to drown out any refinement signal smaller than 0.05.
# With proper 45/9/6 cadence and a CE-filter at 5.0 (well above the L_T floor
# of 1.70), we should get a quieter and more reliably-comparable eval set.
#
# --tokens 45M matches the prior cache volume so EMA-chart shapes remain
# roughly comparable across runs (just shifted by the new data distribution).
set -euo pipefail

cd /home/wmertens/Projects/smollmer

export LD_LIBRARY_PATH=/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=smollmer/cache_corpus
LOG=smollmer/cache_corpus.log

exec .venv/bin/python -u -m smollmer.cache_teacher \
  --source corpus \
  --out "$OUT" \
  --tokens 45000000 \
  --shard-seqs 240 \
  --max-mean-teacher-ce 5.0 \
  --seed 0 \
  "${@}" \
  > >(tee -a "$LOG") 2>&1
