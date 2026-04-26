"""Muon + AdamW optimizer helpers.

This is a compact production-oriented Muon implementation for 2D weights. It
uses Newton-Schulz orthogonalization on the momentum update and AdamW for all
non-matrix parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    if g.ndim != 2:
        raise ValueError("Muon only supports 2D tensors")
    x = g.float()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / x.norm().clamp_min(eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x
    if transposed:
        x = x.T
    return x.to(dtype=g.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise RuntimeError("Muon param group contains a non-2D parameter")
                if wd:
                    p.mul_(1 - lr * wd)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(p.grad)
                update = zeropower_via_newtonschulz5(buf)
                p.add_(update, alpha=-lr)
        return loss


class CombinedOptimizer:
    """Small wrapper around separate Muon and AdamW optimizers."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        self.optimizers = optimizers
        self.param_groups = [g for opt in optimizers for g in opt.param_groups]

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state: dict) -> None:
        for opt, opt_state in zip(self.optimizers, state["optimizers"]):
            opt.load_state_dict(opt_state)


@dataclass(frozen=True)
class OptimConfig:
    matrix_lr: float = 0.02
    adam_lr: float = 2e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)


def build_optimizer(model: torch.nn.Module, cfg: OptimConfig) -> CombinedOptimizer:
    muon_params = []
    adam_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        use_adam = p.ndim != 2 or "embed" in lname or "lm_head" in lname or "norm" in lname or "bias" in lname
        if use_adam:
            adam_params.append(p)
        else:
            muon_params.append(p)

    optimizers: list[torch.optim.Optimizer] = []
    if muon_params:
        optimizers.append(Muon(muon_params, lr=cfg.matrix_lr, momentum=0.95, weight_decay=cfg.weight_decay))
        optimizers[-1].param_groups[0]["kind"] = "muon"
    if adam_params:
        optimizers.append(torch.optim.AdamW(adam_params, lr=cfg.adam_lr, betas=cfg.betas, weight_decay=cfg.weight_decay))
        optimizers[-1].param_groups[0]["kind"] = "adamw"
    return CombinedOptimizer(optimizers)


def set_lr_and_wd(optimizer: CombinedOptimizer, lr_mult: float, weight_decay: float | None = None) -> None:
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
        group["lr"] = group["initial_lr"] * lr_mult
        if weight_decay is not None and group.get("kind") == "muon":
            group["weight_decay"] = weight_decay
