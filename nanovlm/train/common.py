"""Runtime, precision, and logging helpers for pure-PyTorch training."""

from __future__ import annotations

import contextlib
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class RuntimeContext:
    device_type: str
    device: torch.device
    dtype: torch.dtype
    ddp: bool
    rank: int
    local_rank: int
    world_size: int
    master: bool


def print0(*args, **kwargs) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(*args, **kwargs)


def default_base_dir() -> Path:
    base = os.environ.get("NANOVLM_BASE_DIR")
    if base:
        out = Path(base).expanduser()
    else:
        out = Path.home() / ".cache" / "nanovlm"
    out.mkdir(parents=True, exist_ok=True)
    return out


def detect_device_type() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(device_type: str, requested: str | None = None) -> torch.dtype:
    mapping = {"float32": torch.float32, "fp32": torch.float32,
               "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
               "float16": torch.float16, "fp16": torch.float16}
    if requested:
        return mapping[requested]
    if device_type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # MPS training is much easier to debug in fp32; users can override.
    return torch.float32


def init_runtime(device_type: str | None = None, dtype: str | None = None, seed: int = 42) -> RuntimeContext:
    device_type = device_type or detect_device_type()
    if device_type not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"unsupported device_type={device_type!r}")
    if device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if device_type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available")

    ddp_requested = all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if device_type == "cuda" and ddp_requested:
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        ddp = True
    else:
        device = torch.device(device_type)
        ddp = False
        world_size = 1
        rank = 0
        local_rank = 0

    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if device_type == "cuda":
        torch.cuda.manual_seed(seed + rank)
        torch.set_float32_matmul_precision("high")

    return RuntimeContext(
        device_type=device_type,
        device=device,
        dtype=resolve_dtype(device_type, dtype),
        ddp=ddp,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        master=rank == 0,
    )


def cleanup_runtime() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


@contextlib.contextmanager
def autocast_for(ctx: RuntimeContext) -> Iterator[None]:
    if ctx.device_type == "cuda" and ctx.dtype in {torch.float16, torch.bfloat16}:
        with torch.autocast("cuda", dtype=ctx.dtype):
            yield
    else:
        yield


def move_batch(batch: dict, device: torch.device, dtype: torch.dtype | None = None) -> dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            value = value.to(device, non_blocking=device.type == "cuda")
            if dtype is not None and torch.is_floating_point(value):
                value = value.to(dtype=dtype)
        out[key] = value
    return out


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"total": total, "trainable": trainable}


def grad_global_norm(parameters) -> float:
    total = torch.zeros((), device="cpu")
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        total += g.pow(2).sum().cpu()
    return float(total.sqrt())


def maybe_compile(model: torch.nn.Module, enabled: bool, device_type: str) -> torch.nn.Module:
    if not enabled:
        return model
    if device_type == "mps":
        print0("torch.compile disabled on MPS unless explicitly debugged upstream")
        return model
    return torch.compile(model, dynamic=False)
