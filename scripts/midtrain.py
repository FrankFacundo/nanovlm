"""Mid-training annealing on a Dolmino-style high-quality mixture.

Linear LR decay (vs. cosine in pretrain) over a short tail (~5–10% of pretrain
FLOPs). Same model, optimizer, and packer as pretrain; only the schedule and
mixture differ.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from nanovlm.models.qwen3_5 import DEFAULT_MODEL_PATH
from nanovlm.train.checkpoint import rng_state, save_checkpoint
from nanovlm.train.common import (
    cleanup_runtime,
    count_parameters,
    default_base_dir,
    grad_global_norm,
    init_runtime,
    maybe_compile,
    move_batch,
    print0,
)
from nanovlm.train.data.mixture import build_mixture_from_yaml
from nanovlm.train.data.packing import BestFitPacker
from nanovlm.train.losses import masked_ce_loss
from nanovlm.train.model_factory import build_model, load_tokenizer
from nanovlm.train.optim import OptimConfig, build_optimizer, set_lr_and_wd
from nanovlm.train.report import (
    EarlyStopper,
    MetricsLogger,
    WandbLogger,
    add_monitoring_args,
    write_html_report,
    write_markdown_report,
)
from nanovlm.train.schedule import linear_decay


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mid-train annealing for Qwen3.5-0.8B.")
    p.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--init", choices=["scratch", "checkpoint"], default="checkpoint")
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "midtrain_dolmino.yaml"))
    p.add_argument("--data-root", default=str(default_base_dir() / "data"))
    p.add_argument("--out-dir", default=str(default_base_dir() / "runs" / "midtrain"))
    p.add_argument("--device-type", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--total-batch-tokens", type=int, default=8192)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--matrix-lr", type=float, default=0.005)
    p.add_argument("--adam-lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--min-lr-ratio", type=float, default=0.0)
    add_monitoring_args(p, default_project="nanovlm-midtrain")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ctx = init_runtime(args.device_type, args.dtype)
    tokenizer = load_tokenizer(args.model_path)
    model = build_model(
        model_path=args.model_path, init=args.init, device=ctx.device, dtype=ctx.dtype,
        tiny=args.tiny, text_only=args.text_only,
    )
    orig_model = model
    if ctx.ddp:
        model = DDP(model, device_ids=[ctx.local_rank] if ctx.device_type == "cuda" else None)
    model = maybe_compile(model, args.compile, ctx.device_type)

    optimizer = build_optimizer(model, OptimConfig(matrix_lr=args.matrix_lr, adam_lr=args.adam_lr, weight_decay=args.weight_decay))
    stream, cfg = build_mixture_from_yaml(args.config, args.data_root, rank=ctx.rank, world_size=ctx.world_size)
    packer = BestFitPacker(stream, tokenizer, seq_len=args.seq_len, batch_size=args.batch_size)
    grad_accum = max(1, args.total_batch_tokens // (args.batch_size * args.seq_len * ctx.world_size))

    out_dir = Path(args.out_dir)
    logger = MetricsLogger(out_dir, "midtrain")
    wandb_logger = WandbLogger(
        enabled=args.wandb, project=args.wandb_project, run_name=args.wandb_run,
        entity=args.wandb_entity, mode=args.wandb_mode, config=vars(args),
        out_dir=out_dir, master=ctx.master,
    )
    early = EarlyStopper(metric=args.early_stop_metric, mode=args.early_stop_mode,
                          patience=args.early_stop_patience, min_delta=args.early_stop_min_delta,
                          max_loss=args.max_loss)

    print0(f"[midtrain] params={count_parameters(orig_model):,} grad_accum={grad_accum} mixture={cfg.get('name')}")

    step = 0
    for step in range(args.steps):
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_value = None
        for _ in range(grad_accum):
            batch = move_batch(next(packer).__dict__, ctx.device, ctx.dtype)
            out = model(input_ids=batch["input_ids"])
            logits = out["logits"][:, :-1].contiguous()
            tgt = batch["labels"][:, 1:].contiguous()
            mk = batch["loss_mask"][:, 1:].contiguous()
            loss = masked_ce_loss(logits, tgt, mk) / grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {loss}")
            loss_value = float((loss * grad_accum).detach().cpu())
            loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        lrm = linear_decay(step, args.steps, args.warmup_steps, min_ratio=args.min_lr_ratio)
        set_lr_and_wd(optimizer, lrm, args.weight_decay)
        optimizer.step()
        dt = time.time() - t0
        tokens = args.batch_size * args.seq_len * grad_accum * ctx.world_size
        if step % args.log_every == 0:
            metrics = {
                "step": step,
                "train_loss": loss_value,
                "lr": optimizer.param_groups[0]["lr"],
                "tokens_per_sec": tokens / max(dt, 1e-9),
                "grad_norm": grad_global_norm(model.parameters()),
            }
            logger.log(**metrics)
            wandb_logger.log(metrics)
            print0(f"step {step:05d} loss={loss_value:.4f} tok/s={metrics['tokens_per_sec']:.1f} lr={metrics['lr']:.2e}")
            if early.check(metrics):
                print0(f"early stop at step {step}: {early.reason}")
                break
        if ctx.master and args.save_every > 0 and step > 0 and step % args.save_every == 0:
            save_checkpoint(out_dir / "checkpoints", step, orig_model, optimizer,
                            {"args": vars(args), "rng": rng_state()}, rank=ctx.rank)
    if ctx.master:
        save_checkpoint(out_dir / "checkpoints", step + 1, orig_model, optimizer,
                        {"args": vars(args), "rng": rng_state(), "early_stop": early.reason}, rank=ctx.rank)
        print0(f"html: {write_html_report(out_dir, title='midtrain')}")
        print0(f"md:   {write_markdown_report(out_dir, title='midtrain')}")
    wandb_logger.finish()
    cleanup_runtime()


if __name__ == "__main__":
    main()
