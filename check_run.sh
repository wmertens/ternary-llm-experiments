#!/usr/bin/env bash
# Qualitative + quantitative check on a progressive-distill run.
# Usage: check_run.sh [ckpt_dir]
#   Defaults to most-recently-modified smollmer/ckpts.prog-* dir.
# Reads TB scalars, dumps the on-disk interrupted.pt to a temp safetensors,
# runs a fixed set of chat prompts on CPU, prints a structured report.
set -euo pipefail

cd /home/wmertens/Projects/smollmer
export LD_LIBRARY_PATH=/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
PY=.venv/bin/python

CKPT_DIR="${1:-}"
if [[ -z "$CKPT_DIR" ]]; then
  CKPT_DIR=$(ls -dt smollmer/ckpts.prog-*/ 2>/dev/null | head -1)
  CKPT_DIR="${CKPT_DIR%/}"
fi
if [[ -z "$CKPT_DIR" || ! -d "$CKPT_DIR" ]]; then
  echo "no ckpt dir found"; exit 1
fi
CKPT="$CKPT_DIR/interrupted.pt"
if [[ ! -f "$CKPT" ]]; then
  echo "no interrupted.pt in $CKPT_DIR"; exit 1
fi

echo "=========================================================="
echo "Progressive run check: $CKPT_DIR"
echo "ckpt mtime: $(stat -c '%y' "$CKPT")"
echo "=========================================================="

# ---- ckpt summary + recent log lines ----
$PY <<PY
import torch
st = torch.load("$CKPT", map_location="cpu", weights_only=False)
print(f"step={st.get('next_step')}  run_name={st.get('run_name')}")
print(f"ctrl_state={st.get('ctrl_state')}")
ss = st.get('soft_state') or {}
print(f"soft_state.round_idx={ss.get('round_idx')}  gate={ss.get('gate')}")
PY

LOG="$CKPT_DIR/log.txt"
if [[ -f "$LOG" ]]; then
  echo
  echo "--- last 5 commit/ckpt/resume lines ---"
  tr '\r' '\n' < "$LOG" | grep -E '^\[(commit|warmup|ckpt|resume)\]' | tail -5
fi

# ---- TB trend ----
echo
echo "--- TB scalars (last 10 readings) ---"
$PY <<'PY'
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob, os
# Search both legacy ('tb/...') and current ('smollmer/tb/...') locations
runs = sorted(
    glob.glob("tb/*/events.out.tfevents.*")
    + glob.glob("smollmer/tb/*/events.out.tfevents.*"),
    key=os.path.getmtime,
)
if not runs:
    print("no TB events found"); raise SystemExit
print(f"# TB events: {runs[-1]}")
ea = EventAccumulator(runs[-1], size_guidance={'scalars': 0})
ea.Reload()
tags = ea.Tags()['scalars']
def show(tag, fmt="{:.4f}"):
    if tag not in tags: return
    ev = ea.Scalars(tag)
    if not ev: return
    pts = ev[-10:] if len(ev) >= 10 else ev
    s_pts = "  ".join(f"({e.step},{fmt.format(e.value)})" for e in pts)
    print(f"{tag:38s} n={len(ev):4d}  {s_pts}")
for t in ('loss/ema','loss/gap','progressive/round','progressive/committed_frac',
          'progressive/loss_ema_fast','progressive/loss_ema_slow',
          'progressive/step_in_round','progressive/steady',
          'grad_norm','progressive/barrier','soft/latent/saturation_frac',
          'soft/latent/near_boundary_frac','weights/flip_rate'):
    show(t)
# Round transitions
rnd = ea.Scalars('progressive/round') if 'progressive/round' in tags else []
print()
print("round transitions:")
last = None
trans = []
for e in rnd:
    if e.value != last:
        trans.append((e.step, int(e.value))); last = e.value
for s, r in trans[-8:]:
    print(f"  step={s}  round={r}")
PY

# ---- Chat samples ----
TMP=/tmp/smollmer_chat_$$.safetensors
trap 'rm -f "$TMP"' EXIT
echo
echo "--- dump on-disk state to $TMP ---"
$PY -m smollmer.dump_for_chat --in "$CKPT" --out "$TMP" --scale-group-size 64 2>&1 | grep -E '^\[(load|build|save|done)\]'

echo
echo "--- chat samples (CPU, fp32, α=0, T=0.8, top_p=0.9, 60 tok) ---"
timeout 300 $PY -m smollmer.chat \
  --ckpt "$TMP" --device cpu --dtype float32 \
  --scale-group-size 64 --max-new-tokens 60 --temperature 0.8 --top-p 0.9 <<'PROMPTS' 2>&1 | grep -v "Loading weights"
The capital of France is
Once upon a time, in a small village,
def fibonacci(n):
The three laws of robotics are:
PROMPTS

echo
echo "=========================================================="
echo "done."
