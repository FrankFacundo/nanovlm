"""Atomic checkpointing helpers."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import torch


def rng_state() -> dict[str, Any]:
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def save_checkpoint(
    out_dir: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizer,
    meta: dict[str, Any],
    *,
    rank: int = 0,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        model_path = out_dir / f"model_{step:06d}.pt"
        tmp_model = model_path.with_suffix(".pt.tmp")
        torch.save(model.state_dict(), tmp_model)
        os.replace(tmp_model, model_path)

        meta_path = out_dir / f"meta_{step:06d}.json"
        tmp_meta = meta_path.with_suffix(".json.tmp")
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        os.replace(tmp_meta, meta_path)

    if optimizer is not None:
        opt_path = out_dir / f"optim_{step:06d}_rank{rank}.pt"
        tmp_opt = opt_path.with_suffix(".pt.tmp")
        torch.save(optimizer.state_dict(), tmp_opt)
        os.replace(tmp_opt, opt_path)


def load_checkpoint(out_dir: str | Path, step: int, device: torch.device, *, rank: int = 0, load_optimizer: bool = True):
    out_dir = Path(out_dir)
    model_state = torch.load(out_dir / f"model_{step:06d}.pt", map_location=device)
    with open(out_dir / f"meta_{step:06d}.json", encoding="utf-8") as f:
        meta = json.load(f)
    opt_state = None
    opt_path = out_dir / f"optim_{step:06d}_rank{rank}.pt"
    if load_optimizer and opt_path.exists():
        opt_state = torch.load(opt_path, map_location=device)
    return model_state, opt_state, meta
