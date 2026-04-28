"""VLM stage-1 caption alignment.

Trainable: vision encoder + projector. The language model is frozen by
default (unfreeze with ``--unfreeze-lm``). Reads the multimodal alignment
config (``configs/vision_align.yaml``) and packs image-caption records via
``MultimodalLoader``.

This stage exists because joint multimodal pretraining from cold weights
collapses unless image embeddings first land in the LM's input subspace.
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
from nanovlm.train.data.mixture import build_mixture_from_yaml
from nanovlm.train.data.multimodal import MultimodalLoader
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


def _freeze_lm(model: torch.nn.Module) -> int:
    """Freeze the language model; return the number of frozen params."""
    n = 0
    for name, p in model.named_parameters():
        if "vision" in name.lower() or "visual" in name.lower() or "merger" in name.lower():
            continue
        p.requires_grad_(False)
        n += p.numel()
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="VLM caption alignment (vision + projector trainable).")
    p.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--init", choices=["scratch", "checkpoint"], default="checkpoint")
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "vision_align.yaml"))
    p.add_argument("--data-root", default=str(default_base_dir() / "data"))
    p.add_argument("--image-root", default=None)
    p.add_argument("--out-dir", default=str(default_base_dir() / "runs" / "vision_align"))
    p.add_argument("--device-type", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--unfreeze-lm", action="store_true")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--matrix-lr", type=float, default=0.01)
    p.add_argument("--adam-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=1)
    add_monitoring_args(p, default_project="nanovlm-vision-align")
    args = p.parse_args()

    ctx = init_runtime(args.device_type, args.dtype)
    tokenizer = load_tokenizer(args.model_path)
    model = build_model(model_path=args.model_path, init=args.init, device=ctx.device, dtype=ctx.dtype, tiny=args.tiny, text_only=False)
    if not args.unfreeze_lm:
        n_frozen = _freeze_lm(model)
        print0(f"[vision_align] froze {n_frozen:,} LM params")

    optimizer = build_optimizer(model, OptimConfig(matrix_lr=args.matrix_lr, adam_lr=args.adam_lr, weight_decay=args.weight_decay))
    stream, cfg = build_mixture_from_yaml(args.config, args.data_root, rank=ctx.rank, world_size=ctx.world_size)
    loader = MultimodalLoader(stream, tokenizer, seq_len=args.seq_len, image_root=args.image_root)

    out_dir = Path(args.out_dir)
    logger = MetricsLogger(out_dir, "vision_align")
    wandb_logger = WandbLogger(
        enabled=args.wandb, project=args.wandb_project, run_name=args.wandb_run,
        entity=args.wandb_entity, mode=args.wandb_mode, config=vars(args),
        out_dir=out_dir, master=ctx.master,
    )
    early = EarlyStopper(metric=args.early_stop_metric, mode=args.early_stop_mode,
                          patience=args.early_stop_patience, min_delta=args.early_stop_min_delta,
                          max_loss=args.max_loss)

    print0(f"[vision_align] trainable params={count_parameters(model):,} mixture={cfg.get('name')}")

    step = 0
    for step in range(args.steps):
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch = move_batch(next(loader), ctx.device, ctx.dtype)
        out = model(**{k: v for k, v in batch.items() if k != "labels" and k != "loss_mask"})
        logits = out["logits"][:, :-1].contiguous()
        tgt = batch["labels"][:, 1:].contiguous()
        mk = batch["loss_mask"][:, 1:].contiguous()
        loss = masked_ce_loss(logits, tgt, mk)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), args.grad_clip)
        lrm = lr_multiplier(step, args.steps, args.warmup_steps)
        set_lr_and_wd(optimizer, lrm, args.weight_decay)
        optimizer.step()
        dt = time.time() - t0
        if step % args.log_every == 0:
            metrics = {
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                "lr": optimizer.param_groups[0]["lr"],
                "tokens_per_sec": (args.seq_len) / max(dt, 1e-9),
                "grad_norm": grad_global_norm(p for p in model.parameters() if p.requires_grad),
            }
            logger.log(**metrics)
            wandb_logger.log(metrics)
            print0(f"step {step:05d} loss={metrics['train_loss']:.4f} lr={metrics['lr']:.2e}")
            if early.check(metrics):
                print0(f"early stop at step {step}: {early.reason}")
                break
        if ctx.master and args.save_every > 0 and step > 0 and step % args.save_every == 0:
            save_checkpoint(out_dir / "checkpoints", step, model, optimizer,
                            {"args": vars(args), "rng": rng_state()}, rank=ctx.rank)
    if ctx.master:
        save_checkpoint(out_dir / "checkpoints", step + 1, model, optimizer,
                        {"args": vars(args), "rng": rng_state(), "early_stop": early.reason}, rank=ctx.rank)
        print0(f"html: {write_html_report(out_dir, title='vision_align')}")
        print0(f"md:   {write_markdown_report(out_dir, title='vision_align')}")
    wandb_logger.finish()
    cleanup_runtime()


if __name__ == "__main__":
    main()
