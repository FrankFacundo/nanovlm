"""Stage-aware pretraining (S1 general / S2 STEM+code / S3 long-context).

Loads the matching ``configs/pretrain_S{N}_*.yaml`` mixture, packs records via
BOS-aligned best-fit packing, and trains with Muon+AdamW under cosine LR. On
CUDA defaults to bf16 + DDP; on MPS defaults to fp32, single-process.
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
from nanovlm.train.data.streaming import JsonlStream
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
from nanovlm.train.schedule import cosine_weight_decay, lr_multiplier

CONFIGS = {
    "S1": "pretrain_S1_general.yaml",
    "S2": "pretrain_S2_stem_code.yaml",
    "S3": "pretrain_S3_long_context.yaml",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain Qwen3.5-0.8B (text + VLM).")
    p.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--init", choices=["scratch", "checkpoint"], default="scratch")
    p.add_argument("--stage", choices=list(CONFIGS), default="S1")
    p.add_argument("--config", default=None, help="Override mixture config path.")
    p.add_argument("--data-root", default=str(default_base_dir() / "data"))
    p.add_argument("--data", nargs="*", default=None,
                   help="Optional explicit JSONL files; bypasses --stage mixture.")
    p.add_argument("--out-dir", default=str(default_base_dir() / "runs" / "pretrain"))
    p.add_argument("--device-type", default=None, choices=["cuda", "mps", "cpu"])
    p.add_argument("--dtype", default=None, choices=["float32", "bfloat16", "float16"])
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--total-batch-tokens", type=int, default=8192)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--matrix-lr", type=float, default=0.02)
    p.add_argument("--adam-lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=1)
    add_monitoring_args(p, default_project="nanovlm-pretrain")
    return p.parse_args()


def _build_loader(args, tokenizer, ctx):
    if args.data:
        stream = JsonlStream(args.data, rank=ctx.rank, world_size=ctx.world_size, loop=True)
    else:
        cfg_path = args.config or str(Path(__file__).resolve().parents[1] / "configs" / CONFIGS[args.stage])
        stream, _ = build_mixture_from_yaml(cfg_path, args.data_root, rank=ctx.rank, world_size=ctx.world_size)
    return BestFitPacker(stream, tokenizer, seq_len=args.seq_len, batch_size=args.batch_size)


def main() -> None:
    args = parse_args()
    ctx = init_runtime(args.device_type, args.dtype)
    tokenizer = load_tokenizer(args.model_path)
    model = build_model(
        model_path=args.model_path,
        init=args.init,
        device=ctx.device,
        dtype=ctx.dtype,
        tiny=args.tiny,
        text_only=args.text_only,
    )
    orig_model = model
    if ctx.ddp:
        model = DDP(model, device_ids=[ctx.local_rank] if ctx.device_type == "cuda" else None)
    model = maybe_compile(model, args.compile, ctx.device_type)

    optimizer = build_optimizer(model, OptimConfig(matrix_lr=args.matrix_lr, adam_lr=args.adam_lr, weight_decay=args.weight_decay))
    loader = _build_loader(args, tokenizer, ctx)
    grad_accum = max(1, args.total_batch_tokens // (args.batch_size * args.seq_len * ctx.world_size))

    out_dir = Path(args.out_dir)
    logger = MetricsLogger(out_dir, "pretrain")
    wandb_logger = WandbLogger(
        enabled=args.wandb, project=args.wandb_project, run_name=args.wandb_run,
        entity=args.wandb_entity, mode=args.wandb_mode, config=vars(args),
        out_dir=out_dir, master=ctx.master,
    )
    early = EarlyStopper(
        metric=args.early_stop_metric, mode=args.early_stop_mode,
        patience=args.early_stop_patience, min_delta=args.early_stop_min_delta,
        max_loss=args.max_loss,
    )

    print0(f"[pretrain] params={count_parameters(orig_model):,} grad_accum={grad_accum} dtype={ctx.dtype} stage={args.stage}")

    step = 0
    for step in range(args.steps):
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_value = None
        for _ in range(grad_accum):
            batch = move_batch(next(loader).__dict__, ctx.device, ctx.dtype)
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            mask = batch["loss_mask"]
            out = model(input_ids=input_ids)
            logits = out["logits"][:, :-1].contiguous()
            tgt = labels[:, 1:].contiguous()
            tgt_mask = mask[:, 1:].contiguous()
            loss = masked_ce_loss(logits, tgt, tgt_mask) / grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {loss}")
            loss_value = float((loss * grad_accum).detach().cpu())
            loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        lrm = lr_multiplier(step, args.steps, args.warmup_steps)
        wd = cosine_weight_decay(step, args.steps, args.weight_decay)
        set_lr_and_wd(optimizer, lrm, wd)
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
        print0(f"html: {write_html_report(out_dir, title=f'pretrain {args.stage}')}")
        print0(f"md:   {write_markdown_report(out_dir, title=f'pretrain {args.stage}', run_metadata={'stage': args.stage})}")
    wandb_logger.finish()
    cleanup_runtime()


if __name__ == "__main__":
    main()
