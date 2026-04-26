from __future__ import annotations

import argparse
from pathlib import Path

import torch

from nanovlm.models.qwen3_5 import DEFAULT_MODEL_PATH
from nanovlm.train.checkpoint import save_checkpoint
from nanovlm.train.common import default_base_dir, init_runtime, move_batch, print0, cleanup_runtime
from nanovlm.train.data import PreferenceLoader
from nanovlm.train.losses import mpo_loss, sequence_logprobs
from nanovlm.train.model_factory import build_model, load_tokenizer
from nanovlm.train.optim import OptimConfig, build_optimizer, set_lr_and_wd
from nanovlm.train.report import EarlyStopper, MetricsLogger, WandbLogger, add_monitoring_args, write_html_report
from nanovlm.train.schedule import lr_multiplier


def main() -> None:
    p = argparse.ArgumentParser(description="DPO/MPO preference optimization for Qwen3.5.")
    p.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--init", choices=["scratch", "checkpoint"], default="scratch")
    p.add_argument("--data", nargs="*", default=None)
    p.add_argument("--out-dir", default=str(default_base_dir() / "runs" / "preference"))
    p.add_argument("--device-type", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--quality-weight", type=float, default=0.05)
    p.add_argument("--sft-weight", type=float, default=0.1)
    add_monitoring_args(p, default_project="nanovlm-preference")
    args = p.parse_args()

    ctx = init_runtime(args.device_type, args.dtype)
    tokenizer = load_tokenizer(args.model_path)
    model = build_model(model_path=args.model_path, init=args.init, device=ctx.device, dtype=ctx.dtype, tiny=args.tiny, text_only=args.text_only)
    ref = build_model(model_path=args.model_path, init="scratch" if args.tiny else args.init, device=ctx.device, dtype=ctx.dtype, tiny=args.tiny, text_only=args.text_only)
    ref.load_state_dict(model.state_dict())
    ref.eval().requires_grad_(False)
    optimizer = build_optimizer(model, OptimConfig(matrix_lr=0.005, adam_lr=5e-5, weight_decay=0.0))
    loader = PreferenceLoader(tokenizer, args.data, args.batch_size, args.seq_len)
    logger = MetricsLogger(args.out_dir, "preference")
    wandb_logger = WandbLogger(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name=args.wandb_run,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        config=vars(args),
        out_dir=args.out_dir,
        master=ctx.master,
    )
    early_stopper = EarlyStopper(
        metric=args.early_stop_metric,
        mode=args.early_stop_mode,
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
        max_loss=args.max_loss,
    )

    for step in range(args.steps):
        raw = next(loader)
        batch = move_batch(raw, ctx.device, ctx.dtype)
        optimizer.zero_grad(set_to_none=True)
        chosen = model(input_ids=batch["chosen_input_ids"], labels=batch["chosen_labels"])
        rejected = model(input_ids=batch["rejected_input_ids"], labels=batch["rejected_labels"])
        chosen_lp = sequence_logprobs(chosen["logits"], batch["chosen_labels"])
        rejected_lp = sequence_logprobs(rejected["logits"], batch["rejected_labels"])
        with torch.no_grad():
            ref_chosen = ref(input_ids=batch["chosen_input_ids"], labels=batch["chosen_labels"])
            ref_rejected = ref(input_ids=batch["rejected_input_ids"], labels=batch["rejected_labels"])
            ref_chosen_lp = sequence_logprobs(ref_chosen["logits"], batch["chosen_labels"])
            ref_rejected_lp = sequence_logprobs(ref_rejected["logits"], batch["rejected_labels"])
        loss, metrics = mpo_loss(
            chosen_lp, rejected_lp, ref_chosen_lp, ref_rejected_lp, chosen["loss"],
            beta=args.beta, quality_weight=args.quality_weight, sft_weight=args.sft_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        set_lr_and_wd(optimizer, lr_multiplier(step, args.steps, 1))
        optimizer.step()
        metrics.update({"step": step, "train_loss": float(loss.detach().cpu())})
        logger.log(**metrics)
        wandb_logger.log(metrics)
        print0(f"step {step:05d} pref_loss={metrics['train_loss']:.4f} margin={metrics['dpo_margin']:.3f}")
        if early_stopper.check(metrics):
            print0(f"early stop at step {step}: {early_stopper.reason}")
            break

    if ctx.master:
        final_step = step + 1 if "step" in locals() else 0
        save_checkpoint(Path(args.out_dir) / "checkpoints", final_step, model, optimizer, {"args": vars(args), "early_stop": early_stopper.reason}, rank=ctx.rank)
        print0(f"report: {write_html_report(args.out_dir)}")
    wandb_logger.finish()
    cleanup_runtime()


if __name__ == "__main__":
    main()
