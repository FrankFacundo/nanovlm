from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from nanovlm.models.qwen3_5 import DEFAULT_MODEL_PATH
from nanovlm.train.checkpoint import rng_state, save_checkpoint
from nanovlm.train.common import count_parameters, default_base_dir, grad_global_norm, init_runtime, maybe_compile, move_batch, print0, cleanup_runtime
from nanovlm.train.data import PackedTextLoader, ensure_default_pretrain_jsonl, validate_data_paths
from nanovlm.train.model_factory import build_model, load_tokenizer
from nanovlm.train.optim import OptimConfig, build_optimizer, set_lr_and_wd
from nanovlm.train.report import EarlyStopper, MetricsLogger, WandbLogger, add_monitoring_args, write_html_report
from nanovlm.train.schedule import cosine_weight_decay, lr_multiplier


def parse_args():
    p = argparse.ArgumentParser(description="Pretrain Qwen3.5-0.8B VLM/text model from scratch or checkpoint.")
    default_train_data = default_base_dir() / "data" / "train.jsonl"
    p.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--init", choices=["scratch", "checkpoint"], default="scratch")
    p.add_argument("--stage", choices=["S1", "S2", "S3"], default="S1")
    p.add_argument(
        "--data",
        nargs="*",
        default=None,
        help=f"JSONL files. Default auto-creates and uses {default_train_data}.",
    )
    p.add_argument("--out-dir", default=str(default_base_dir() / "runs" / "pretrain"))
    p.add_argument("--device-type", default=None, choices=["cuda", "mps", "cpu", None])
    p.add_argument("--dtype", default=None, choices=["float32", "bfloat16", "float16", None])
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--total-batch-tokens", type=int, default=4096)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--matrix-lr", type=float, default=0.02)
    p.add_argument("--adam-lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=1)
    add_monitoring_args(p, default_project="nanovlm-pretrain")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ctx = init_runtime(args.device_type, args.dtype)
    default_train_data = default_base_dir() / "data" / "train.jsonl"
    if not args.data:
        args.data = [str(ensure_default_pretrain_jsonl(default_train_data))]
        print0(f"using default train data: {args.data[0]}")
    else:
        args.data = validate_data_paths(args.data, default_hint=default_train_data)
    tokenizer = load_tokenizer(args.model_path)
    model = build_model(
        model_path=args.model_path, init=args.init, device=ctx.device, dtype=ctx.dtype,
        tiny=args.tiny, text_only=args.text_only,
    )
    orig_model = model
    if ctx.ddp:
        model = DDP(model, device_ids=[ctx.local_rank])
    model = maybe_compile(model, args.compile, ctx.device_type)
    optimizer = build_optimizer(model, OptimConfig(args.matrix_lr, args.adam_lr, args.weight_decay))
    scaler = torch.amp.GradScaler("cuda") if ctx.device_type == "cuda" and ctx.dtype == torch.float16 else None
    loader = PackedTextLoader(tokenizer, args.data, args.batch_size, args.seq_len, rank=ctx.rank, world_size=ctx.world_size)
    grad_accum = max(1, args.total_batch_tokens // (args.batch_size * args.seq_len * ctx.world_size))
    out_dir = Path(args.out_dir)
    logger = MetricsLogger(out_dir, "pretrain")
    wandb_logger = WandbLogger(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name=args.wandb_run,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        config=vars(args),
        out_dir=out_dir,
        master=ctx.master,
    )
    early_stopper = EarlyStopper(
        metric=args.early_stop_metric,
        mode=args.early_stop_mode,
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
        max_loss=args.max_loss,
    )
    print0(f"params={count_parameters(orig_model)} grad_accum={grad_accum} dtype={ctx.dtype} stage={args.stage}")
    stop_requested = False

    for step in range(args.steps):
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_value = None
        for _ in range(grad_accum):
            batch = move_batch(next(loader), ctx.device, ctx.dtype)
            out = model(**batch)
            loss = out["loss"] / grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {loss}")
            loss_value = float(out["loss"].detach().cpu())
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        lrm = lr_multiplier(step, args.steps, args.warmup_steps)
        wd = cosine_weight_decay(step, args.steps, args.weight_decay)
        set_lr_and_wd(optimizer, lrm, wd)
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        dt = time.time() - t0
        tokens = args.batch_size * args.seq_len * grad_accum * ctx.world_size
        if step % args.log_every == 0:
            metrics = {
                "step": step, "train_loss": loss_value, "lr": optimizer.param_groups[0]["lr"],
                "tokens_per_sec": tokens / max(dt, 1e-9), "grad_norm": grad_global_norm(model.parameters()),
                "docs_seen": loader.state.docs_seen, "tokens_seen": loader.state.tokens_seen,
            }
            logger.log(**metrics)
            wandb_logger.log(metrics)
            print0(f"step {step:05d} loss={loss_value:.4f} tok/s={metrics['tokens_per_sec']:.1f} lr={metrics['lr']:.2e}")
            if early_stopper.check(metrics):
                print0(f"early stop at step {step}: {early_stopper.reason}")
                stop_requested = True
        if ctx.master and args.save_every > 0 and step > 0 and step % args.save_every == 0:
            save_checkpoint(out_dir / "checkpoints", step, orig_model, optimizer, {"args": vars(args), "rng": rng_state()}, rank=ctx.rank)
        if stop_requested:
            break

    if ctx.master:
        final_step = step + 1 if "step" in locals() else 0
        save_checkpoint(out_dir / "checkpoints", final_step, orig_model, optimizer, {"args": vars(args), "rng": rng_state(), "early_stop": early_stopper.reason}, rank=ctx.rank)
        print0(f"report: {write_html_report(out_dir)}")
    wandb_logger.finish()
    cleanup_runtime()


if __name__ == "__main__":
    main()
