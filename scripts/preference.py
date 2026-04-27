"""DPO and InternVL-style MPO preference optimization.

Reads ``(prompt, chosen, rejected)`` records via ``PreferenceLoader`` (text or
multimodal). A frozen copy of the policy is held in memory as the reference;
each step does four forwards (policy chosen/rejected, ref chosen/rejected) and
applies ``dpo_loss`` or ``mpo_loss``.

For larger models on MPS, prefer ``--mpo`` only after smoke-testing memory.
"""

from __future__ import annotations

import argparse
import copy
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
from nanovlm.train.data.preference import PreferenceLoader
from nanovlm.train.data.streaming import JsonlStream
from nanovlm.train.losses import dpo_loss, masked_ce_loss, mpo_loss, sequence_logprobs
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


def _seq_logp(model, ids, mask):
    """Compute summed log-prob of `mask`-selected tokens under the model.

    Uses the standard next-token shift: predict ``ids[:, 1:]`` from
    ``logits[:, :-1]``; ``mask`` is also shifted.
    """
    out = model(input_ids=ids)
    logits = out["logits"][:, :-1].contiguous()
    labels = ids[:, 1:].contiguous()
    shifted_mask = mask[:, 1:].contiguous()
    return sequence_logprobs(logits, labels, shifted_mask), logits, labels, shifted_mask


def main() -> None:
    p = argparse.ArgumentParser(description="DPO / MPO preference optimization for Qwen3.5.")
    p.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--init", choices=["scratch", "checkpoint"], default="checkpoint")
    p.add_argument("--data", nargs="+", required=False, default=None,
                   help="JSONL preference files (each line: {prompt, chosen, rejected}).")
    p.add_argument("--out-dir", default=str(default_base_dir() / "runs" / "preference"))
    p.add_argument("--device-type", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--max-prompt-len", type=int, default=1024)
    p.add_argument("--max-response-len", type=int, default=1024)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--matrix-lr", type=float, default=0.001)
    p.add_argument("--adam-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--algo", choices=["dpo", "mpo"], default="dpo")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--quality-weight", type=float, default=0.05)
    p.add_argument("--sft-weight", type=float, default=0.1)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=1)
    add_monitoring_args(p, default_project="nanovlm-preference")
    args = p.parse_args()

    ctx = init_runtime(args.device_type, args.dtype)
    tokenizer = load_tokenizer(args.model_path)
    model = build_model(model_path=args.model_path, init=args.init, device=ctx.device, dtype=ctx.dtype, tiny=args.tiny, text_only=args.text_only)

    print0(f"[preference] policy params={count_parameters(model):,}; building frozen reference …")
    ref = copy.deepcopy(model).eval()
    for p_ in ref.parameters():
        p_.requires_grad_(False)

    if not args.data:
        raise ValueError("--data is required (preference records)")
    stream = JsonlStream(args.data, rank=ctx.rank, world_size=ctx.world_size, loop=True)
    loader = PreferenceLoader(stream, tokenizer, max_prompt_len=args.max_prompt_len, max_response_len=args.max_response_len)

    optimizer = build_optimizer(model, OptimConfig(matrix_lr=args.matrix_lr, adam_lr=args.adam_lr, weight_decay=args.weight_decay))

    out_dir = Path(args.out_dir)
    logger = MetricsLogger(out_dir, "preference")
    wandb_logger = WandbLogger(
        enabled=args.wandb, project=args.wandb_project, run_name=args.wandb_run,
        entity=args.wandb_entity, mode=args.wandb_mode, config=vars(args),
        out_dir=out_dir, master=ctx.master,
    )
    early = EarlyStopper(metric=args.early_stop_metric, mode=args.early_stop_mode,
                          patience=args.early_stop_patience, min_delta=args.early_stop_min_delta,
                          max_loss=args.max_loss)

    step = 0
    for step in range(args.steps):
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch = move_batch(next(loader), ctx.device, ctx.dtype)
        chosen_ids = batch["chosen_ids"]
        rejected_ids = batch["rejected_ids"]
        chosen_mask = batch["chosen_mask"]
        rejected_mask = batch["rejected_mask"]

        chosen_logp, c_logits, c_labels, c_mask_s = _seq_logp(model, chosen_ids, chosen_mask)
        rejected_logp, _, _, _ = _seq_logp(model, rejected_ids, rejected_mask)
        with torch.no_grad():
            ref_chosen_logp, _, _, _ = _seq_logp(ref, chosen_ids, chosen_mask)
            ref_rejected_logp, _, _, _ = _seq_logp(ref, rejected_ids, rejected_mask)

        if args.algo == "mpo":
            sft_loss = masked_ce_loss(c_logits, c_labels, c_mask_s)
            loss, m = mpo_loss(
                chosen_logp, rejected_logp, ref_chosen_logp, ref_rejected_logp,
                sft_loss, beta=args.beta, quality_weight=args.quality_weight, sft_weight=args.sft_weight,
            )
        else:
            loss, m = dpo_loss(chosen_logp, rejected_logp, ref_chosen_logp, ref_rejected_logp, beta=args.beta)

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
                "step_time_s": dt,
                "grad_norm": grad_global_norm(model.parameters()),
                **m,
            }
            logger.log(**metrics)
            wandb_logger.log(metrics)
            print0(f"step {step:05d} loss={loss_value:.4f} margin={m.get('dpo_margin', 0):.3f} acc={m.get('dpo_acc', 0):.2f}")
            if early.check(metrics):
                print0(f"early stop at step {step}: {early.reason}")
                break
        if ctx.master and args.save_every > 0 and step > 0 and step % args.save_every == 0:
            save_checkpoint(out_dir / "checkpoints", step, model, optimizer,
                            {"args": vars(args), "rng": rng_state()}, rank=ctx.rank)

    if ctx.master:
        save_checkpoint(out_dir / "checkpoints", step + 1, model, optimizer,
                        {"args": vars(args), "rng": rng_state(), "early_stop": early.reason}, rank=ctx.rank)
        print0(f"html: {write_html_report(out_dir, title=args.algo)}")
        print0(f"md:   {write_markdown_report(out_dir, title=args.algo)}")
    wandb_logger.finish()
    cleanup_runtime()


if __name__ == "__main__":
    main()
