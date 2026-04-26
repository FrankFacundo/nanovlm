"""Learning-rate and stage schedules."""

from __future__ import annotations

import math


def lr_multiplier(step: int, total_steps: int, warmup_steps: int, final_lr_frac: float = 0.05) -> float:
    if total_steps <= 0:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return final_lr_frac + (1 - final_lr_frac) * cosine


def cosine_weight_decay(step: int, total_steps: int, base: float) -> float:
    if total_steps <= 0:
        return base
    return base * 0.5 * (1 + math.cos(math.pi * step / total_steps))
