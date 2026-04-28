"""Runtime context: device/dtype/DDP/MPS, batch movement, parameter helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist


@dataclass
class RuntimeContext:
    device: torch.device
    device_type: str
    dtype: torch.dtype
    ddp: bool
    rank: int
    local_rank: int
    world_size: int
    master: bool


def _autodetect_device_type() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(device_type: str, dtype: str | None) -> torch.dtype:
    if dtype is None:
        if device_type == "cuda":
            try:
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except Exception:
                pass
            return torch.float16
        return torch.float32
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]


def init_runtime(device_type: str | None = None, dtype: str | None = None) -> RuntimeContext:
    """Initialize device, dtype, and (optional) DDP from environment.

    Honors ``WORLD_SIZE``/``RANK``/``LOCAL_RANK`` (set by ``torchrun``). On CUDA
    with ``WORLD_SIZE>1`` initializes NCCL; otherwise runs single-process.
    """
    device_type = device_type or _autodetect_device_type()
    resolved_dtype = _resolve_dtype(device_type, dtype)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ddp = world_size > 1

    if ddp:
        backend = "nccl" if device_type == "cuda" else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")

    if device_type == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif device_type == "mps":
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    master = rank == 0
    return RuntimeContext(
        device=device,
        device_type=device_type,
        dtype=resolved_dtype,
        ddp=ddp,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        master=master,
    )


def cleanup_runtime() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def print0(*args, **kwargs) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(*args, **kwargs, flush=True)


def default_base_dir() -> Path:
    base = os.environ.get("NANOVLM_BASE_DIR")
    if base:
        return Path(base).expanduser()
    return Path.home() / ".cache" / "nanovlm"


def move_batch(batch: dict, device: torch.device, dtype: torch.dtype | None = None) -> dict:
    """Move tensor values of ``batch`` to ``device``; cast float tensors to ``dtype``."""
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if value.dtype.is_floating_point and dtype is not None:
                value = value.to(device=device, dtype=dtype)
            else:
                value = value.to(device=device)
        out[key] = value
    return out


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def grad_global_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    norms = [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms)).cpu())


def maybe_compile(model: torch.nn.Module, compile_flag: bool, device_type: str) -> torch.nn.Module:
    if not compile_flag:
        return model
    if device_type != "cuda":
        return model
    return torch.compile(model)


def all_reduce_mean(value: float, ctx: RuntimeContext) -> float:
    if not ctx.ddp:
        return value
    t = torch.tensor([value], device=ctx.device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / ctx.world_size)
