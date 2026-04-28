"""Muon (Newton–Schulz orthogonalization) for matrix params + AdamW for the rest.

Muon: see Jordan et al. 2024 (https://kellerjordan.github.io/posts/muon/) and the
nanochat implementation. Applied to 2-D parameters (excluding embeddings + lm_head).
AdamW is used for embeddings, biases, and 1-D parameters (norms, scalars).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.optim import Optimizer


@dataclass
class OptimConfig:
    matrix_lr: float = 0.02
    adam_lr: float = 2e-4
    weight_decay: float = 0.1
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    adam_betas: tuple = (0.9, 0.95)
    adam_eps: float = 1e-8


def _newton_schulz5(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton–Schulz iteration that orthogonalizes the matrix ``g``.

    Coefficients from Jordan et al. (a, b, c) = (3.4445, -4.7750, 2.0315). On
    convergence, ``X X^T ≈ I`` (or ``X^T X ≈ I``, whichever is smaller).
    Operates in float32 for numerical stability.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float()
    x = x / (x.norm() + eps)
    if x.size(0) > x.size(1):
        x = x.t()
        transposed = True
    else:
        transposed = False
    for _ in range(steps):
        a_term = x @ x.t()
        x = a * x + b * (a_term @ x) + c * (a_term @ a_term @ x)
    if transposed:
        x = x.t()
    return x.to(dtype=g.dtype)


class Muon(Optimizer):
    """Muon: SGD-momentum on the orthogonalized gradient of 2-D weight matrices."""

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.dim() != 2:
                    continue
                g = p.grad
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf
                ortho = _newton_schulz5(update, steps=ns_steps)
                scale = max(1.0, ortho.size(0) / ortho.size(1)) ** 0.5
                if wd != 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(ortho, alpha=-lr * scale)
        return loss


def _split_params(model: torch.nn.Module):
    """Matrix params (2-D, non-embedding, non-lm_head) → Muon. The rest → AdamW.

    Embedding detection is structural (``isinstance(module, nn.Embedding)``) so
    it works regardless of parameter name. ``lm_head`` is excluded by name.
    """
    embed_param_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            for p in module.parameters(recurse=False):
                embed_param_ids.add(id(p))

    matrix, other = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_embed = id(p) in embed_param_ids
        is_lm_head = "lm_head" in name
        if p.dim() == 2 and not is_embed and not is_lm_head:
            matrix.append(p)
        else:
            other.append(p)
    return matrix, other


def build_optimizer(model: torch.nn.Module, cfg: OptimConfig) -> Optimizer:
    """Return a single ``Optimizer`` whose param_groups are ``[muon-matrix, adamw-other]``.

    ``set_lr_and_wd`` updates LR/WD on both groups; the per-group base ``initial_lr``
    is preserved so the multiplier scales each group independently.
    """
    matrix_params, other_params = _split_params(model)

    class CombinedOptimizer(Optimizer):
        def __init__(self):
            param_groups = [
                {
                    "params": matrix_params,
                    "lr": cfg.matrix_lr,
                    "initial_lr": cfg.matrix_lr,
                    "weight_decay": cfg.weight_decay,
                    "momentum": cfg.momentum,
                    "nesterov": cfg.nesterov,
                    "ns_steps": cfg.ns_steps,
                    "kind": "muon",
                },
                {
                    "params": other_params,
                    "lr": cfg.adam_lr,
                    "initial_lr": cfg.adam_lr,
                    "betas": cfg.adam_betas,
                    "eps": cfg.adam_eps,
                    "weight_decay": cfg.weight_decay,
                    "kind": "adamw",
                },
            ]
            super().__init__(param_groups, defaults={})
            self._muon = Muon(matrix_params, lr=cfg.matrix_lr, momentum=cfg.momentum, nesterov=cfg.nesterov, ns_steps=cfg.ns_steps, weight_decay=cfg.weight_decay) if matrix_params else None
            self._adamw = torch.optim.AdamW(other_params, lr=cfg.adam_lr, betas=cfg.adam_betas, eps=cfg.adam_eps, weight_decay=cfg.weight_decay) if other_params else None

        @torch.no_grad()
        def step(self, closure=None):
            loss = closure() if closure is not None else None
            if self._muon is not None:
                self._muon.param_groups[0]["lr"] = self.param_groups[0]["lr"]
                self._muon.param_groups[0]["weight_decay"] = self.param_groups[0]["weight_decay"]
                self._muon.step()
            if self._adamw is not None:
                self._adamw.param_groups[0]["lr"] = self.param_groups[1]["lr"]
                self._adamw.param_groups[0]["weight_decay"] = self.param_groups[1]["weight_decay"]
                self._adamw.step()
            return loss

        def zero_grad(self, set_to_none: bool = True):
            if self._muon is not None:
                self._muon.zero_grad(set_to_none=set_to_none)
            if self._adamw is not None:
                self._adamw.zero_grad(set_to_none=set_to_none)

        def state_dict(self):
            return {
                "muon": self._muon.state_dict() if self._muon is not None else None,
                "adamw": self._adamw.state_dict() if self._adamw is not None else None,
                "groups": [{k: v for k, v in g.items() if k != "params"} for g in self.param_groups],
            }

        def load_state_dict(self, sd):
            if sd.get("muon") and self._muon is not None:
                self._muon.load_state_dict(sd["muon"])
            if sd.get("adamw") and self._adamw is not None:
                self._adamw.load_state_dict(sd["adamw"])
            for g, saved in zip(self.param_groups, sd.get("groups", [])):
                g.update({k: v for k, v in saved.items() if k != "params"})

    return CombinedOptimizer()


def set_lr_and_wd(optimizer: Optimizer, lr_mult: float, wd: float | None = None) -> None:
    """Scale each group's LR by ``lr_mult`` (relative to ``initial_lr``) and set WD."""
    for g in optimizer.param_groups:
        base = g.get("initial_lr", g["lr"])
        g["lr"] = base * lr_mult
        if wd is not None:
            g["weight_decay"] = wd
