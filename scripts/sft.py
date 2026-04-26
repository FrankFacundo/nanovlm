from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from nanovlm.models.qwen3_5 import DEFAULT_MODEL_PATH
from nanovlm.train.checkpoint import save_checkpoint
from nanovlm.train.common import default_base_dir, grad_global_norm, init_runtime, move_batch, print0, cleanup_runtime
from nanovlm.train.data import SFTLoader
from nanovlm.train.model_factory import build_model, load_tokenizer
from nanovlm.train.optim import OptimConfig, build_optimizer, set_lr_and_wd
from nanovlm.train.report import MetricsLogger, write_html_report
from nanovlm.train.schedule import lr_multiplier


def main() -> None:
    p = argparse.ArgumentParser(description="Supervised fine-tune Qwen3.5 with assistant-only masks.")
    p.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--init", choices=["scratch", "checkpoint"], default="scratch")
    p.add_argument("--data", nargs="*", default=None)
    p.add_argument("--out-dir", default=str(default_base_dir() / "runs" / "sft"))
    p.add_argument("--device-type", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--thinking-ratio", type=float, default=0.25)
    p.add_argument("--matrix-lr", type=float, default=0.01)
    p.add_argument("--adam-lr", type=float, default=1e-4)
    args = p.parse_args()
    ctx = init_runtime(args.device_type, args.dtype)
    tokenizer = load_tokenizer(args.model_path)
    model = build_model(model_path=args.model_path, init=args.init, device=ctx.device, dtype=ctx.dtype, tiny=args.tiny, text_only=args.text_only)
    optimizer = build_optimizer(model, OptimConfig(args.matrix_lr, args.adam_lr, 0.0))
    loader = SFTLoader(tokenizer, args.data, args.batch_size, args.seq_len, rank=ctx.rank, world_size=ctx.world_size, thinking_ratio=args.thinking_ratio)
    logger = MetricsLogger(args.out_dir, "sft")
    for step in range(args.steps):
        t0 = time.time()
        batch = move_batch(next(loader), ctx.device, ctx.dtype)
        optimizer.zero_grad(set_to_none=True)
        out = model(**batch)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        set_lr_and_wd(optimizer, lr_multiplier(step, args.steps, 1))
        optimizer.step()
        metrics = {"step": step, "train_loss": float(loss.detach().cpu()), "tokens_per_sec": args.batch_size * args.seq_len / max(time.time() - t0, 1e-9), "grad_norm": grad_global_norm(model.parameters())}
        logger.log(**metrics)
        print0(f"step {step:05d} sft_loss={metrics['train_loss']:.4f}")
    if ctx.master:
        save_checkpoint(Path(args.out_dir) / "checkpoints", args.steps, model, optimizer, {"args": vars(args)}, rank=ctx.rank)
        print0(f"report: {write_html_report(args.out_dir)}")
    cleanup_runtime()


if __name__ == "__main__":
    main()
