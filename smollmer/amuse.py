"""AMUSE (Schedule-Free Muon) + Schedule-Free Cautious AdamW.

References:
  - AMUSE       : Jeon et al. 2026 (arxiv 2605.22432). Muon + Polyak avg.
  - Muon        : Jordan 2024. Newton-Schulz polar factor of grad momentum.
  - SF          : Defazio & Mishchenko 2024 (arxiv 2405.15682).
  - Cautious    : Liang et al. 2024 (arxiv 2411.16085). sign(m)=sign(g) mask.

Schedule-free state per parameter:
    z : base trajectory (fp32) — receives the AdamW/Muon update
    x : Polyak-averaged trajectory (fp32) — the "deploy" weight
    m : momentum, fp32
    v : second moment, fp32 (SF-CAdamW only)

During training, p.data holds y_t = (1-β1)·z_t + β1·x_t, so gradients are
evaluated at y. After each step, z and x are updated, and a fresh y is
written back into p.data for the next forward.

`.train()` and `.eval()` swap p.data between y (train) and x (deploy).
Call .eval() before any inference / deploy fold; .train() before resuming
training. The wrappers below tag their own mode so repeated calls are
idempotent.

DualOptimizer wraps two underlying schedule-free optimizers so qat_smooth
can route Linear weights → AMUSE and the rest → SF-CAdamW behind a
single .step() / .train() / .eval() / .state_dict() interface.
"""
from __future__ import annotations

from typing import Iterable

import torch


# ---------------------------------------------------------------- Muon kernel
def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Polar-factor approximation: returns U with G ≈ U·P, U^T U ≈ I.

    Coefficients (a,b,c) = (3.4445, -4.7750, 2.0315) from the Muon paper,
    tuned for fast convergence from X = G/‖G‖_F. The iteration:
        X ← a·X + (b·A + c·A²)·X    with A = X·X^T
    is contractive on the singular values toward 1.

    Computed in fp32 regardless of G's dtype; returned in fp32.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.float32)
    # Newton-Schulz is happier when rows ≤ cols (smaller A = X X^T).
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


# --------------------------------------------------- Schedule-Free Cautious AdamW
class ScheduleFreeCAdamW(torch.optim.Optimizer):
    """SF-AdamW with the cautious sign-mask, fp32 state.

    Step:
        m ← β1·m + (1-β1)·g_t        (g_t = ∇L(y_t), y_t in p.data)
        v ← β2·v + (1-β2)·g_t²
        m_hat = m / (1 - β1^t); v_hat = v / (1 - β2^t)
        upd   = m_hat / (√v_hat + eps)
        mask  = 1[sign(m)·sign(g) > 0]; mask /= mean(mask).clamp_min(1e-3)
        z ← z·(1 - lr·wd) - lr·(upd · mask)
        c_t = 1 / (t+1)
        x ← (1-c_t)·x + c_t·z
        y ← (1-β1)·z + β1·x   →   p.data

    LR is applied directly here (no external scheduler needed). Pass a
    warmup_steps>0 to ramp lr from 0 over the first `warmup_steps`. Pass
    warmup_steps=0 to use lr from step 1.
    """

    def __init__(self, params: Iterable[torch.nn.Parameter],
                 lr: float = 3e-4,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 warmup_steps: int = 0) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        warmup_steps=warmup_steps)
        super().__init__(params, defaults)
        self._train_mode = True  # p.data currently holds y, not x

    # ---- mode swap ----
    @torch.no_grad()
    def train(self) -> None:
        if self._train_mode:
            return
        # p currently = x. We want p ← y = (1-β1)·z + β1·x.
        # lerp(p, z, 1-β1) = (1-(1-β1))·p + (1-β1)·z = β1·x + (1-β1)·z ✓
        for group in self.param_groups:
            beta1 = group["betas"][0]
            for p in group["params"]:
                state = self.state.get(p, {})
                if "z" not in state:
                    continue
                p.data.lerp_(state["z"].to(p.dtype), 1 - beta1)
        self._train_mode = True

    @torch.no_grad()
    def eval(self) -> None:
        if not self._train_mode:
            return
        # p currently = y. We want p ← x.
        # x = (y - (1-β1)·z) / β1 = (1/β1)·y - ((1-β1)/β1)·z
        # lerp(p, z, w) = (1-w)·p + w·z; want (1-w)=1/β1, so w = 1 - 1/β1 = -((1-β1)/β1).
        for group in self.param_groups:
            beta1 = group["betas"][0]
            w = -((1 - beta1) / beta1)
            for p in group["params"]:
                state = self.state.get(p, {})
                if "z" not in state:
                    continue
                p.data.lerp_(state["z"].to(p.dtype), w)
        self._train_mode = False

    @property
    def train_mode(self) -> bool:
        return self._train_mode

    # ---- step ----
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            warmup = group["warmup_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "z" not in state:
                    state["step"] = 0
                    state["z"] = p.data.detach().clone().float()
                    state["x"] = p.data.detach().clone().float()
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)
                    state["v"] = torch.zeros_like(p, dtype=torch.float32)
                state["step"] += 1
                t = state["step"]
                z, x, m, v = state["z"], state["x"], state["m"], state["v"]
                g32 = grad.to(torch.float32)

                m.mul_(beta1).add_(g32, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g32, g32, value=1 - beta2)
                bc1 = 1 - beta1 ** t
                bc2 = 1 - beta2 ** t
                denom = (v / bc2).sqrt().add_(eps)
                # Cautious mask, mean-normalized (preserves effective LR).
                mask = (m * g32 > 0).to(m.dtype)
                mask.div_(mask.mean().clamp_min(1e-3))
                upd = (m * mask) / denom / bc1

                cur_lr = lr * min(1.0, t / warmup) if warmup > 0 else lr
                if wd != 0:
                    z.mul_(1 - cur_lr * wd)
                z.sub_(upd, alpha=cur_lr)

                c_t = 1.0 / (t + 1)
                x.lerp_(z, c_t)

                # y = (1-β1)·z + β1·x
                y = torch.lerp(z, x, beta1)
                p.data.copy_(y.to(p.dtype))
        return loss


# --------------------------------------------------------- AMUSE (SF-Muon)
class AMUSE(torch.optim.Optimizer):
    """Schedule-Free Muon for 2-D weight matrices.

    Step:
        m ← μ·m + g_t                                  (g_t = ∇L(y_t))
        O = NewtonSchulz(m)                            (polar factor of m)
        z ← z·(1 - lr·wd) - lr·O
        c_t = 1/(t+1)
        x ← (1-c_t)·x + c_t·z
        y ← (1-β1)·z + β1·x   →   p.data

    Per the AMUSE paper, the interpolation coefficient β1 may schedule
    upward as training progresses; this implementation keeps β1 fixed
    since we're using plateau-gated training where "progress per step"
    isn't a useful axis anyway. Set --amuse-beta1 to taste; 0.6 is the
    paper default.

    Only square-ish 2D weights are useful for orthogonalization; pass
    only QLinear.weight tensors here. Anything 1-D, embedding-shaped,
    or extreme aspect ratio should go through SF-CAdamW instead.
    """

    def __init__(self, params: Iterable[torch.nn.Parameter],
                 lr: float = 2e-3,
                 momentum: float = 0.9,
                 beta1: float = 0.6,
                 weight_decay: float = 0.0,
                 warmup_steps: int = 0,
                 ns_steps: int = 5) -> None:
        defaults = dict(lr=lr, momentum=momentum, beta1=beta1,
                        weight_decay=weight_decay,
                        warmup_steps=warmup_steps, ns_steps=ns_steps)
        super().__init__(params, defaults)
        self._train_mode = True

    @torch.no_grad()
    def train(self) -> None:
        if self._train_mode:
            return
        for group in self.param_groups:
            beta1 = group["beta1"]
            for p in group["params"]:
                state = self.state.get(p, {})
                if "z" not in state:
                    continue
                p.data.lerp_(state["z"].to(p.dtype), 1 - beta1)
        self._train_mode = True

    @torch.no_grad()
    def eval(self) -> None:
        if not self._train_mode:
            return
        for group in self.param_groups:
            beta1 = group["beta1"]
            w = -((1 - beta1) / beta1)
            for p in group["params"]:
                state = self.state.get(p, {})
                if "z" not in state:
                    continue
                p.data.lerp_(state["z"].to(p.dtype), w)
        self._train_mode = False

    @property
    def train_mode(self) -> bool:
        return self._train_mode

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            beta1 = group["beta1"]
            wd = group["weight_decay"]
            warmup = group["warmup_steps"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.dim() != 2:
                    raise ValueError(
                        f"AMUSE requires 2D params; got shape {tuple(p.shape)}. "
                        "Send non-matrix params to SF-CAdamW instead.")
                grad = p.grad
                state = self.state[p]
                if "z" not in state:
                    state["step"] = 0
                    state["z"] = p.data.detach().clone().float()
                    state["x"] = p.data.detach().clone().float()
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)
                state["step"] += 1
                t = state["step"]
                z, x, m = state["z"], state["x"], state["m"]

                m.mul_(mu).add_(grad.to(torch.float32))
                # Polar factor of momentum (size-aware Newton-Schulz).
                o = newton_schulz(m, steps=ns_steps)

                cur_lr = lr * min(1.0, t / warmup) if warmup > 0 else lr
                if wd != 0:
                    z.mul_(1 - cur_lr * wd)
                z.sub_(o, alpha=cur_lr)

                c_t = 1.0 / (t + 1)
                x.lerp_(z, c_t)

                y = torch.lerp(z, x, beta1)
                p.data.copy_(y.to(p.dtype))
        return loss


# ----------------------------------------------------- DualOptimizer wrapper
class DualOptimizer:
    """Wraps two schedule-free optimizers behind one torch.optim-ish API.

    Doesn't subclass torch.optim.Optimizer because the .state_dict()
    structure is different, and we don't need PyTorch's flatten-and-pickle
    plumbing. Forwards step / zero_grad / train / eval to both children;
    state_dict / load_state_dict return/accept {"amuse": ..., "cadamw": ...}.
    """

    def __init__(self, amuse: AMUSE,
                 cadamw: ScheduleFreeCAdamW) -> None:
        self.amuse = amuse
        self.cadamw = cadamw

    @property
    def param_groups(self):
        # Return both for consistent LR-walking from qat_smooth (each
        # group already carries its own lr; we don't merge them).
        return self.amuse.param_groups + self.cadamw.param_groups

    def step(self, closure=None):
        # CAdamW first so AMUSE sees the same g this iteration regardless;
        # they're independent since they own disjoint param sets.
        self.cadamw.step()
        self.amuse.step()
        return None

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.amuse.zero_grad(set_to_none=set_to_none)
        self.cadamw.zero_grad(set_to_none=set_to_none)

    def train(self) -> None:
        self.amuse.train()
        self.cadamw.train()

    def eval(self) -> None:
        self.amuse.eval()
        self.cadamw.eval()

    @property
    def train_mode(self) -> bool:
        # Both wrappers stay in lockstep; just read one.
        return self.amuse.train_mode

    def state_dict(self) -> dict:
        return {
            "amuse": self.amuse.state_dict(),
            "cadamw": self.cadamw.state_dict(),
            "train_mode": self._train_mode_repr(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.amuse.load_state_dict(sd["amuse"])
        self.cadamw.load_state_dict(sd["cadamw"])
        mode = sd.get("train_mode", "train")
        # Best-effort restore. The state_dict() of torch.optim doesn't
        # carry our private flag, so set it from the wrapper-level field.
        self.amuse._train_mode = (mode == "train")
        self.cadamw._train_mode = (mode == "train")

    def _train_mode_repr(self) -> str:
        return "train" if self.amuse.train_mode else "eval"


# ---------------------------------------------------------------- helpers
def split_amuse_cadamw_params(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Returns (matrix_params, other_params).

    Matrix params: 2D weights of nn.Linear-like modules (QLinear, nn.Linear).
    Other params: 1D or non-Linear (embeddings, norms, biases, learnable
    scales/codepoint_c if any).
    """
    # Anything attached to an nn.Linear / QLinear as `weight` is a matrix.
    linear_weight_ids = set()
    for m in model.modules():
        # Catches QLinear (subclass of nn.Linear) and any plain nn.Linear.
        if isinstance(m, torch.nn.Linear):
            linear_weight_ids.add(id(m.weight))
    matrix, other = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in linear_weight_ids and p.dim() == 2:
            matrix.append(p)
        else:
            other.append(p)
    return matrix, other
