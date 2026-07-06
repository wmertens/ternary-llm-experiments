#!/usr/bin/env bash
# run_monitor.sh RUN_NAME — liveness + progress monitor for one gpt_bop run.
#
# Emits one stdout line per event. Unlike a bare `tail -F | grep`, this also
# detects ABSENCE of compute: if the trainer process vanishes without writing
# a [done] marker, it emits TRAINER-GONE (a tail-only monitor stays silent on
# a killed/crashed process, which reads identically to "still running").
#
# Events emitted:
#   VAL <line>        — a new [val] checkpoint
#   DONE <line>       — [done]/METRIC terminal marker (clean completion)
#   ERROR <line>      — Traceback/OOM/CUDA error/assert
#   TRAINER-GONE      — python trainer PID absent AND no [done] seen (death)
#   GPU-IDLE          — GPU <5% for 3 consecutive polls while no [done] (stall)
#
# Exits after DONE or TRAINER-GONE so the Monitor watch ends cleanly.
set -uo pipefail
RUN="${1:?usage: run_monitor.sh RUN_NAME}"
LOG="experiments_gpt/${RUN}/train.log"
PATTERN="smollmer.gpt_bop.*${RUN}"
idle=0
last_val=""
seen_done=0

emit() { printf '%s\n' "$*"; }

while true; do
  # --- terminal markers in the log ---
  if [ -f "$LOG" ]; then
    done_line=$(tr '\r' '\n' < "$LOG" | grep -E '^\[done\]|^METRIC ' | tail -1)
    if [ -n "$done_line" ] && [ "$seen_done" = 0 ]; then
      seen_done=1
      emit "DONE $done_line"
    fi
    err_line=$(tr '\r' '\n' < "$LOG" | grep -E 'Traceback|CUDA error|out of memory|OOM|AcceleratorError|assert' | tail -1)
    if [ -n "$err_line" ]; then emit "ERROR $err_line"; fi
    val_line=$(tr '\r' '\n' < "$LOG" | grep -E '^\[val\]' | tail -1)
    if [ -n "$val_line" ] && [ "$val_line" != "$last_val" ]; then
      last_val="$val_line"; emit "VAL $val_line"
    fi
  fi

  # --- liveness: process present? ---
  if pgrep -f "$PATTERN" >/dev/null 2>&1; then
    alive=1
  else
    alive=0
  fi

  # --- GPU idle detection (only meaningful while training) ---
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  util=${util:-0}
  if [ "$seen_done" = 0 ] && [ "$util" -lt 5 ] 2>/dev/null; then
    idle=$((idle+1))
  else
    idle=0
  fi

  # --- exit conditions ---
  if [ "$seen_done" = 1 ]; then
    exit 0
  fi
  if [ "$alive" = 0 ]; then
    # process gone with no [done] marker → death without terminal line
    emit "TRAINER-GONE ${RUN} (no [done] marker; last: ${last_val:-none})"
    exit 0
  fi
  if [ "$idle" -ge 3 ]; then
    emit "GPU-IDLE ${RUN} (util<5% x3 polls, process alive — possible dataloader stall)"
    idle=0  # report once per stall episode, keep watching
  fi

  sleep 20
done
