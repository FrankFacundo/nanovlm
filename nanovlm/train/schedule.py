"""Learning-rate and weight-decay schedules."""

from __future__ import annotations

import math


def lr_multiplier(step: int, total_steps: int, warmup_steps: int, min_ratio: float = 0.1) -> float:
    """Linear warmup → cosine decay to ``min_ratio`` of the peak LR."""
    if step < warmup_steps:
        return float(step + 1) / max(1, warmup_steps)
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def linear_decay(step: int, total_steps: int, warmup_steps: int, min_ratio: float = 0.0) -> float:
    """Linear warmup → linear decay to ``min_ratio`` of the peak. Used for mid-training."""
    if step < warmup_steps:
        return float(step + 1) / max(1, warmup_steps)
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return 1.0 - (1.0 - min_ratio) * progress


def cosine_weight_decay(step: int, total_steps: int, base_wd: float, min_ratio: float = 0.1) -> float:
    """Cosine WD schedule that decays to ``min_ratio * base_wd``."""
    if total_steps <= 0:
        return base_wd
    progress = min(1.0, max(0.0, step / total_steps))
    factor = min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_wd * factor
