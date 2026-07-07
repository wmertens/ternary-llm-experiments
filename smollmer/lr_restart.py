"""lr_restart — pure decision logic for plateau-triggered LR warm-restart
with validation-gated rollback (P5e, user steer 2026-07-06).

The trainer owns the model/optimizer state and the LR re-anchor; this module
owns only the *decisions*, so the fiddly state-machine + plateau math can be
unit-tested without a GPU. Wire it into gpt_bop's val loop: feed each new val
loss to `RestartController.observe(step, val)`; it returns an Action telling
the trainer what to do (nothing / snapshot+rewarm / accept / rollback).

State machine:
  NORMAL     — base cosine running; watch for a plateau in the val history.
  OBSERVING  — after a rewarm; watch whether val beats the pre-restart anchor.

ponytail: pure logic + __main__ self-check; no torch, no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RestartConfig:
    plateau_tol: float = 0.005     # min improvement over the window to NOT be a plateau
    plateau_window: int = 4        # number of recent vals inspected
    spice: float = 2.5             # rewarm LR = spice × current LR (first attempt)
    observe_steps: int = 1500      # steps to judge a rewarm before accept/reject
    accept_margin: float = 0.01    # rewarm must beat anchor by this to be kept
    max_attempts: int = 3          # total rewarm attempts across the run
    spice_decay: float = 0.6       # multiply spice by this after a rejected attempt


# Action kinds returned to the trainer.
NONE = "none"            # keep going, nothing to do
REWARM = "rewarm"        # snapshot best state, re-anchor LR to `new_peak`, enter OBSERVING
ACCEPT = "accept"        # plateau escaped; adopt current state as new baseline
ROLLBACK = "rollback"    # rewarm failed; reload the anchor and resume annealing


@dataclass
class Action:
    kind: str
    new_peak: float | None = None      # for REWARM: spicier LR (= spice×cur_lr)
    spice: float | None = None         # for REWARM: the multiplier itself (apply
                                       # to every optimizer's current LR)
    observe_until: int | None = None   # for REWARM: step at which to judge
    reason: str = ""


@dataclass
class RestartController:
    cfg: RestartConfig
    state: str = "NORMAL"
    attempts_used: int = 0
    cur_spice: float = 0.0
    anchor_val: float | None = None    # val at the snapshot we can roll back to
    observe_until: int | None = None
    best_observed: float | None = None
    _hist: list[float] = field(default_factory=list)   # (step, val) vals only
    _steps: list[int] = field(default_factory=list)

    def __post_init__(self):
        if self.cur_spice == 0.0:
            self.cur_spice = self.cfg.spice

    def _is_plateau(self) -> bool:
        w = self.cfg.plateau_window
        if len(self._hist) < w:
            return False
        window = self._hist[-w:]
        # Best improvement across the window = how much the earliest-in-window
        # was beaten by the best-in-window. Flat window → tiny improvement.
        improvement = window[0] - min(window)
        return improvement < self.cfg.plateau_tol

    def observe(self, step: int, val: float, cur_lr: float) -> Action:
        """Feed a new val checkpoint. `cur_lr` is the LR that produced it
        (used to size the rewarm). Returns the Action the trainer must apply."""
        self._hist.append(val)
        self._steps.append(step)

        if self.state == "NORMAL":
            if self.attempts_used >= self.cfg.max_attempts:
                return Action(NONE, reason="attempts exhausted")
            if self._is_plateau():
                # Snapshot the current (best-in-window) state and rewarm.
                self.anchor_val = min(self._hist[-self.cfg.plateau_window:])
                new_peak = cur_lr * self.cur_spice
                self.observe_until = step + self.cfg.observe_steps
                self.best_observed = None
                self.state = "OBSERVING"
                self.attempts_used += 1
                return Action(REWARM, new_peak=new_peak, spice=self.cur_spice,
                              observe_until=self.observe_until,
                              reason=f"plateau@{step} anchor={self.anchor_val:.4f} "
                                     f"spice={self.cur_spice:.2f}→lr={new_peak:.3e}")
            return Action(NONE)

        # OBSERVING
        self.best_observed = (val if self.best_observed is None
                              else min(self.best_observed, val))
        if step < (self.observe_until or 0):
            return Action(NONE)
        # Window over: judge against the anchor.
        assert self.anchor_val is not None and self.best_observed is not None
        if self.best_observed < self.anchor_val - self.cfg.accept_margin:
            # Escaped the plateau. Adopt current trajectory as the new baseline.
            self.state = "NORMAL"
            self._hist = self._hist[-1:]
            self._steps = self._steps[-1:]
            self.cur_spice = self.cfg.spice  # reset aggressiveness after a win
            return Action(ACCEPT,
                          reason=f"escaped: best={self.best_observed:.4f} < "
                                 f"anchor={self.anchor_val:.4f}-{self.cfg.accept_margin}")
        # Failed: roll back and cool the next attempt.
        self.state = "NORMAL"
        self._hist = self._hist[-1:]
        self._steps = self._steps[-1:]
        self.cur_spice *= self.cfg.spice_decay
        return Action(ROLLBACK,
                      reason=f"no escape: best={self.best_observed:.4f} ≥ "
                             f"anchor={self.anchor_val:.4f}-{self.cfg.accept_margin}; "
                             f"spice→{self.cur_spice:.2f}")


def _demo():
    """Self-check the state machine on synthetic val trajectories."""
    # 1. Plateau detection: flat window triggers, descending window doesn't.
    c = RestartController(RestartConfig(plateau_tol=0.005, plateau_window=4))
    for step, v in [(500, 5.0), (1000, 4.5), (1500, 4.2), (2000, 4.0)]:
        a = c.observe(step, v, cur_lr=1e-2)
        assert a.kind == NONE, f"descending should not rewarm, got {a.kind}"
    # now flat → improvement < tol → plateau fires as soon as the window
    # flattens (here at step 3500, the moment last-4 improvement < tol).
    plateau_vals = [(2500, 4.00), (3000, 3.999), (3500, 4.001), (4000, 4.000)]
    acts = [c.observe(s, v, cur_lr=2e-3) for s, v in plateau_vals]
    rewarms = [a for a in acts if a.kind == REWARM]
    assert len(rewarms) == 1, f"exactly one rewarm expected, got {[a.kind for a in acts]}"
    rw = rewarms[0]
    assert abs(rw.new_peak - 2e-3 * 2.5) < 1e-9, "rewarm peak = spice×lr"
    assert rw.observe_until == rw.observe_until  # set, non-None
    assert rw.observe_until is not None

    # 2. ACCEPT path: during observation val drops below anchor-margin, and
    # the window closes (step ≥ observe_until).
    ou = rw.observe_until
    c.observe(ou - 100, 3.9, cur_lr=5e-3)      # improving, mid-window
    a = c.observe(ou + 1, 3.90, cur_lr=5e-3)   # window over
    assert a.kind == ACCEPT, f"beating anchor should accept, got {a.kind} {a.reason}"

    # 3. ROLLBACK path: force a plateau, then observation fails to beat anchor.
    c2 = RestartController(RestartConfig(plateau_window=3, plateau_tol=0.01,
                                         observe_steps=1000, accept_margin=0.01))
    for s, v in [(500, 4.00), (1000, 4.00), (1500, 4.00)]:
        last = c2.observe(s, v, cur_lr=1e-3)
    assert last.kind == REWARM
    anchor = c2.anchor_val
    c2.observe(2000, 4.05, cur_lr=3e-3)        # worse during observe
    a = c2.observe(2500, 4.02, cur_lr=3e-3)    # window over, still ≥ anchor
    assert a.kind == ROLLBACK, f"failed escape should roll back, got {a.kind}"
    assert c2.cur_spice < 2.5, "spice should cool after rejection"

    # 4. max_attempts cap: no rewarm once exhausted.
    c3 = RestartController(RestartConfig(plateau_window=2, plateau_tol=0.01,
                                         max_attempts=0))
    c3.observe(500, 4.0, cur_lr=1e-3)
    a = c3.observe(1000, 4.0, cur_lr=1e-3)
    assert a.kind == NONE, "max_attempts=0 must never rewarm"

    print("lr_restart self-check OK")


if __name__ == "__main__":
    _demo()
