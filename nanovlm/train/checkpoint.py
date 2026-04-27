"""Safetensors model checkpoints + torch.save for optimizer/RNG state."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file as safe_load
from safetensors.torch import save_file as safe_save


def rng_state() -> dict:
    state = {
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _strip_compile_prefix(state_dict: dict) -> dict:
    return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}


def _dedupe_shared_tensors(state_dict: dict) -> tuple[dict, dict[str, str]]:
    """Drop later occurrences of tensors that share storage (e.g., tied lm_head).

    Returns ``(deduped_state_dict, alias_map)`` where ``alias_map[dropped] = kept``.
    The model's ``__init__`` already re-ties these on load, so the missing keys
    are harmless.
    """
    seen: dict[int, str] = {}
    out: dict = {}
    aliases: dict[str, str] = {}
    for k, v in state_dict.items():
        if not hasattr(v, "data_ptr"):
            out[k] = v
            continue
        ptr = v.data_ptr()
        if ptr in seen and v.numel() > 0:
            aliases[k] = seen[ptr]
            continue
        seen[ptr] = k
        out[k] = v
    return out, aliases


def save_checkpoint(
    out_dir: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    meta: dict | None = None,
    *,
    rank: int = 0,
) -> Path:
    """Save model (safetensors), optimizer (torch.save), and ``meta_*.json``."""
    if rank != 0:
        return Path(out_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = _strip_compile_prefix(model.state_dict())
    sd = {k: v.detach().to("cpu").contiguous() for k, v in sd.items()}
    sd, aliases = _dedupe_shared_tensors(sd)
    model_path = out_dir / f"model_{step:06d}.safetensors"
    safe_save(sd, str(model_path), metadata={"tied_keys": json.dumps(aliases)})
    if optimizer is not None:
        torch.save(optimizer.state_dict(), out_dir / f"optim_{step:06d}.pt")
    meta_path = out_dir / f"meta_{step:06d}.json"
    payload = {"step": step}
    if meta:
        payload.update({k: v for k, v in meta.items() if k != "rng"})
    meta_path.write_text(json.dumps(payload, indent=2, default=str))
    if meta and "rng" in meta:
        torch.save(meta["rng"], out_dir / f"rng_{step:06d}.pt")
    return model_path


def load_checkpoint(
    model_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load model weights into ``model``; optionally load optimizer + RNG by sibling path."""
    model_path = Path(model_path)
    sd = safe_load(str(model_path), device=str(map_location))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"checkpoint mismatch: missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}"
        )
    step = int(model_path.stem.split("_")[-1])
    if optimizer is not None:
        opt_path = model_path.with_name(f"optim_{step:06d}.pt")
        if opt_path.exists():
            optimizer.load_state_dict(torch.load(opt_path, map_location=map_location))
    rng_path = model_path.with_name(f"rng_{step:06d}.pt")
    if rng_path.exists():
        restore_rng_state(torch.load(rng_path, map_location="cpu"))
    return {"step": step, "missing": list(missing), "unexpected": list(unexpected)}
