"""Supervised fine-tuning with assistant-only loss masking.

Auto-detects record format:
  - ``{"messages": [...]}`` or ``{"prompt": ..., "response": ...}`` → text SFT.
  - Records with ``image``/``images`` keys, or messages with ``image`` items →
    multimodal SFT (uses ``MultimodalLoader``).

Supports a thinking-mode mix: with probability ``--thinking-ratio``, the
record's existing ``<think>...</think>`` block (if any) is preserved; otherwise
it is stripped before loss computation.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from nanovlm.models.qwen3_5 import DEFAULT_MODEL_PATH
from nanovlm.train.checkpoint import rng_state, save_checkpoint
from nanovlm.train.common import (
    cleanup_runtime,
    count_parameters,
    default_base_dir,
    grad_global_norm,
    init_runtime,
    move_batch,
    print0,
)
from nanovlm.train.data.chat import ChatLoader
from nanovlm.train.data.mixture import build_mixture_from_yaml
from nanovlm.train.data.multimodal import MultimodalLoader
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
from nanovlm.train.schedule import lr_multiplier


def _is_multimodal_record(rec: dict) -> bool:
    if "image" in rec or "images" in rec:
        return True
    msgs = rec.get("messages") or []
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list) and any(isinstance(x, dict) and (x.get("type") == "image" or "image" in x) for x in c):
            return True
    return False


def _build_loader(stream, tokenizer, *, seq_len: int, image_root: str | None, thinking_ratio: float):
    """Wrap ``stream`` such that each record is dispatched to the right loader.

    Returns an iterator of training batches (one record per ``__next__``).
    """
    text_loader = ChatLoader(iter([]), tokenizer, seq_len=seq_len, thinking_ratio=thinking_ratio)
    mm_loader = MultimodalLoader(iter([]), tokenizer, seq_len=seq_len, image_root=image_root)

    def _gen():
        for rec in stream:
            if _is_multimodal_record(rec):
                mm_loader.records = iter([rec])
                yield next(mm_loader)
            else:
                text_loader.records = iter([rec])
                yield next(text_loader)
    return _gen()


def main() -> None:
    p = argparse.ArgumentParser(description="Supervised fine-tune Qwen3.5 on chat / multimodal SFT records.")
    p.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--init", choices=["scratch", "checkpoint"], default="checkpoint")
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "sft_general.yaml"))
    p.add_argument("--data-root", default=str(default_base_dir() / "data"))
    p.add_argument("--data", nargs="*", default=None,
                   help="Optional explicit JSONL files; bypasses --config mixture.")
    p.add_argument("--image-root", default=None)
    p.add_argument("--out-dir", default=str(default_base_dir() / "runs" / "sft"))
    p.add_argument("--device-type", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--matrix-lr", type=float, default=0.005)
    p.add_argument("--adam-lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--thinking-ratio", type=float, default=0.25)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=1)
    add_monitoring_args(p, default_project="nanovlm-sft")
    args = p.parse_args()

    ctx = init_runtime(args.device_type, args.dtype)
    tokenizer = load_tokenizer(args.model_path)
    model = build_model(model_path=args.model_path, init=args.init, device=ctx.device, dtype=ctx.dtype, tiny=args.tiny, text_only=args.text_only)

    if args.data:
        stream = JsonlStream(args.data, rank=ctx.rank, world_size=ctx.world_size, loop=True)
    else:
        stream, _ = build_mixture_from_yaml(args.config, args.data_root, rank=ctx.rank, world_size=ctx.world_size)
    loader = _build_loader(stream, tokenizer, seq_len=args.seq_len, image_root=args.image_root, thinking_ratio=args.thinking_ratio)

    optimizer = build_optimizer(model, OptimConfig(matrix_lr=args.matrix_lr, adam_lr=args.adam_lr, weight_decay=args.weight_decay))

    out_dir = Path(args.out_dir)
    logger = MetricsLogger(out_dir, "sft")
    wandb_logger = WandbLogger(
        enabled=args.wandb, project=args.wandb_project, run_name=args.wandb_run,
        entity=args.wandb_entity, mode=args.wandb_mode, config=vars(args),
        out_dir=out_dir, master=ctx.master,
    )
    early = EarlyStopper(metric=args.early_stop_metric, mode=args.early_stop_mode,
                          patience=args.early_stop_patience, min_delta=args.early_stop_min_delta,
                          max_loss=args.max_loss)

    print0(f"[sft] params={count_parameters(model):,} dtype={ctx.dtype}")

    step = 0
    for step in range(args.steps):
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch = move_batch(next(loader), ctx.device, ctx.dtype)
        forward_kwargs = {k: v for k, v in batch.items() if k not in ("labels", "loss_mask")}
        out = model(**forward_kwargs)
        logits = out["logits"][:, :-1].contiguous()
        tgt = batch["labels"][:, 1:].contiguous()
        mk = batch["loss_mask"][:, 1:].contiguous()
        loss = masked_ce_loss(logits, tgt, mk)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss}")
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        lrm = lr_multiplier(step, args.steps, args.warmup_steps)
        set_lr_and_wd(optimizer, lrm, args.weight_decay)
        optimizer.step()
        dt = time.time() - t0

        loss_value = float(loss.detach().cpu())
        if step % args.log_every == 0:
            metrics = {
                "step": step,
                "train_loss": loss_value,
                "lr": optimizer.param_groups[0]["lr"],
                "tokens_per_sec": args.seq_len / max(dt, 1e-9),
                "grad_norm": grad_global_norm(model.parameters()),
            }
            logger.log(**metrics)
            wandb_logger.log(metrics)
            print0(f"step {step:05d} loss={loss_value:.4f} lr={metrics['lr']:.2e}")
            if early.check(metrics):
                print0(f"early stop at step {step}: {early.reason}")
                break
        if ctx.master and args.save_every > 0 and step > 0 and step % args.save_every == 0:
            save_checkpoint(out_dir / "checkpoints", step, model, optimizer,
                            {"args": vars(args), "rng": rng_state()}, rank=ctx.rank)

    if ctx.master:
        save_checkpoint(out_dir / "checkpoints", step + 1, model, optimizer,
                        {"args": vars(args), "rng": rng_state(), "early_stop": early.reason}, rank=ctx.rank)
        print0(f"html: {write_html_report(out_dir, title='sft')}")
        print0(f"md:   {write_markdown_report(out_dir, title='sft')}")
    wandb_logger.finish()
    cleanup_runtime()


if __name__ == "__main__":
    main()
