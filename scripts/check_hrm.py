#!/usr/bin/env python3
"""check_hrm.py — one-look status for an hrm_bop training run.

Usage:
    python scripts/check_hrm.py                 # auto-pick newest live ckpts.hrm-*-bop
    python scripts/check_hrm.py hrm-G-bop       # specific run name
    python scripts/check_hrm.py --rows 30       # show more recent rows
    python scripts/check_hrm.py --tail-lines 5  # more raw tail context

Prints:
  - process status (alive / dead, pid)
  - last raw tqdm postfix from train.log
  - last N rows from TB: step, loss, ema, val, flip%, score_max, score_rms,
    m_max, m_rms, frac_zero, per-loop CE 0/1 (if available)
  - a tiny summary block: best EMA, best val, deltas from earliest sample
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = PROJECT_ROOT / "smollmer"


def _discover_run(name: str | None) -> tuple[str, Path]:
    """Return (run_name, ckpt_dir) for the requested run, or pick the newest
    non-archived ckpts.hrm-*-bop/ if name is None."""
    if name:
        # accept "hrm-G-bop", "ckpts.hrm-G-bop", or full path
        cand = CKPT_DIR / name
        if not cand.exists():
            cand = CKPT_DIR / f"ckpts.{name}"
        if not cand.exists() and Path(name).exists():
            cand = Path(name)
        if not cand.exists():
            sys.exit(f"no run dir matching {name!r}")
        run = cand.name.removeprefix("ckpts.")
        return run, cand
    # auto: newest mtime ckpts.hrm-*-bop, excluding archived suffixes (anything after -bop.).
    candidates = []
    for p in CKPT_DIR.glob("ckpts.hrm-*-bop"):
        if p.is_dir():
            candidates.append(p)
    if not candidates:
        sys.exit("no ckpts.hrm-*-bop directories under smollmer/")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    p = candidates[0]
    return p.name.removeprefix("ckpts."), p


def _proc_status(run: str) -> tuple[bool, str]:
    """Return (alive, info-string) by grepping ps for the trainer cmd."""
    try:
        out = subprocess.run(
            ["ps", "-ef"], capture_output=True, text=True, check=True).stdout
    except Exception as e:
        return False, f"ps failed: {e}"
    needle = f"smollmer.hrm_bop --out smollmer/ckpts.{run}"
    matching = [
        line for line in out.splitlines()
        if needle in line and "grep" not in line]
    if not matching:
        return False, "not running"
    pids = [line.split()[1] for line in matching]
    elapsed = [line.split()[6] for line in matching]
    return True, f"pid={','.join(pids)} elapsed={','.join(elapsed)}"


def _last_postfix(log_path: Path, tail_lines: int) -> list[str]:
    """Read the log and return the last `tail_lines` non-empty postfix lines
    (one per tqdm update). tqdm uses CR not LF, so split on both."""
    if not log_path.exists():
        return ["(no train.log)"]
    text = log_path.read_text(errors="replace")
    # split on either CR or LF
    parts = [p.strip() for p in text.replace("\r", "\n").splitlines()]
    parts = [p for p in parts if p]
    return parts[-tail_lines:]


def _load_tb(tb_root: Path):
    """Return event_accumulator with scalars loaded, or None if unavailable."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as e:
        print(f"tensorboard import failed: {e}", file=sys.stderr)
        return None
    # tb_root has one subdir per run_name; pick the only/newest one.
    subdirs = [p for p in tb_root.glob("*") if p.is_dir()]
    if not subdirs:
        return None
    subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    ea = event_accumulator.EventAccumulator(
        str(subdirs[0]),
        size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    return ea


def _series(ea, tag: str) -> dict[int, float]:
    if ea is None or tag not in ea.Tags()["scalars"]:
        return {}
    return {e.step: e.value for e in ea.Scalars(tag)}


def _fmt(v) -> str:
    if v is None:
        return "  -  "
    if isinstance(v, float) and math.isnan(v):
        return "NaN  "
    if isinstance(v, float) and not math.isfinite(v):
        return "Inf  "
    if isinstance(v, float):
        if abs(v) >= 1000 or (0 < abs(v) < 0.001):
            return f"{v:.2e}"
        return f"{v:.4g}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=None,
                    help="run name (hrm-X-bop). Defaults to newest live run.")
    ap.add_argument("--rows", type=int, default=12,
                    help="how many recent TB rows to show (default 12)")
    ap.add_argument("--tail-lines", type=int, default=2,
                    help="how many raw log tail lines to show (default 2)")
    args = ap.parse_args()

    run, ckpt = _discover_run(args.run)
    log = ckpt / "train.log"
    tb_root = ckpt / "tb"

    alive, info = _proc_status(run)
    status_word = "ALIVE" if alive else "DEAD "
    print(f"=== {run} [{status_word}] {info}")
    print(f"    dir: {ckpt.relative_to(PROJECT_ROOT) if ckpt.is_relative_to(PROJECT_ROOT) else ckpt}")
    if log.exists():
        sz_mb = log.stat().st_size / 1e6
        print(f"    train.log: {sz_mb:.2f} MB")

    print("--- last postfix lines:")
    for line in _last_postfix(log, args.tail_lines):
        print(f"  {line}")

    ea = _load_tb(tb_root) if tb_root.exists() else None
    if ea is None:
        print("(no TB data yet)")
        return

    tags = [
        ("loss/step",         "loss"),
        ("loss/ema",          "ema"),
        ("val/loss",          "val"),
        ("bop/flip_rate",     "flip%"),
        ("bop/score_max",     "scoreM"),
        ("bop/score_rms",     "scoreR"),
        ("bop/m_max",         "m_max"),
        ("bop/m_rms",         "m_rms"),
        ("trits/frac_zero",   "fzero"),
        ("scales/mean",       "sMean"),
        ("diag/per_loop_ce_0","pl0"),
        ("diag/per_loop_ce_1","pl1"),
        ("lion/lr",           "lr"),
    ]
    series = {tag: _series(ea, tag) for tag, _ in tags}

    # union of steps that appear in loss/step (the densest series)
    all_steps = sorted(series.get("loss/step", {}).keys())
    if not all_steps:
        print("(no loss/step scalars yet)")
        return
    show_steps = all_steps[-args.rows:]

    print(f"--- last {len(show_steps)} TB rows (of {len(all_steps)} sampled):")
    hdr = f"{'step':>6} " + " ".join(f"{lbl:>7}" for _, lbl in tags)
    print(hdr)
    for s in show_steps:
        row = [f"{s:>6}"]
        for tag, _ in tags:
            v = series[tag].get(s)
            # If exact step missing, use the closest prior sample
            if v is None and series[tag]:
                prior = [k for k in series[tag] if k <= s]
                v = series[tag][max(prior)] if prior else None
            row.append(f"{_fmt(v):>7}")
        print(" ".join(row))

    # summary
    def _best(seq: dict[int, float], how="min"):
        if not seq:
            return None, None
        if how == "min":
            step, val = min(seq.items(), key=lambda kv: kv[1])
        else:
            step, val = max(seq.items(), key=lambda kv: kv[1])
        return step, val

    print("--- summary:")
    bs, bv = _best(series["loss/ema"], "min")
    if bs is not None:
        print(f"  best loss/ema: {bv:.4f} at step {bs}")
    bs, bv = _best(series["val/loss"], "min")
    if bs is not None:
        print(f"  best val/loss: {bv:.4f} at step {bs}")
    first = series["loss/ema"].get(all_steps[0])
    last = series["loss/ema"].get(all_steps[-1])
    if first is not None and last is not None:
        print(f"  loss/ema delta: {first:.4f} → {last:.4f}  "
              f"({last-first:+.4f} over {all_steps[-1]-all_steps[0]} steps)")
    # per-loop gap
    pl0_last = series["diag/per_loop_ce_0"].get(
        max(series["diag/per_loop_ce_0"], default=0), None) \
        if series["diag/per_loop_ce_0"] else None
    pl1_last = series["diag/per_loop_ce_1"].get(
        max(series["diag/per_loop_ce_1"], default=0), None) \
        if series["diag/per_loop_ce_1"] else None
    if pl0_last is not None and pl1_last is not None:
        print(f"  per-loop gap (pl1 < pl0 = recurrence helping): "
              f"pl0={pl0_last:.4f} pl1={pl1_last:.4f}  "
              f"Δ={pl0_last-pl1_last:+.4f}")
    fz = series.get("trits/frac_zero", {})
    if fz:
        fz_steps = sorted(fz.keys())
        print(f"  trits/frac_zero: {fz[fz_steps[0]]:.4f} → {fz[fz_steps[-1]]:.4f}  "
              f"(Δ={fz[fz_steps[-1]] - fz[fz_steps[0]]:+.4f})")


if __name__ == "__main__":
    main()
