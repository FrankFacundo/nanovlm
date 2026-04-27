"""RLVR with GRPO or DAPO over verifier-rewarded prompts.

DAPO (Yu et al. 2025) defaults: decoupled clip ε_low=0.2 / ε_high=0.28,
token-level loss normalization, dynamic sampling (drop groups whose rewards
are all 0 or all 1 before backprop). Algorithm is selected via ``--algo``.

Each step:
  1. Sample a prompt and replicate it ``group_size`` times.
  2. Roll out ``group_size`` completions on policy with current weights
     (saved as ``old_logp`` by the rollout, no_grad).
  3. Score each completion via ``nanovlm.train.verifiers.reward_record``.
  4. Compute group-standardized advantages.
  5. (DAPO-only) drop the group entirely if all rewards are equal.
  6. Re-forward the (prompt + completion) batch with grad to obtain
     ``new_logp``; backprop the clipped surrogate.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from nanovlm.models.qwen3_5 import DEFAULT_MODEL_PATH
from nanovlm.models.qwen3_5.chat_template import EOS, render_chat
from nanovlm.train.checkpoint import rng_state, save_checkpoint
from nanovlm.train.common import (
    cleanup_runtime,
    count_parameters,
    default_base_dir,
    grad_global_norm,
    init_runtime,
    print0,
)
from nanovlm.train.data.rlvr import GroupSampler, RlvrRecordIter
from nanovlm.train.data.streaming import JsonlStream
from nanovlm.train.losses import (
    dapo_loss,
    grpo_kl_penalty,
    grpo_policy_loss,
    group_advantages,
    per_token_logprobs,
)
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
from nanovlm.train.rollout import group_rollout, stack_padded
from nanovlm.train.schedule import lr_multiplier
from nanovlm.train.verifiers import reward_record


def _wrap_prompt(question: str) -> str:
    return render_chat([{"role": "user", "content": question}], add_generation_prompt=True)


def _eos_id(tokenizer) -> int | None:
    eid = getattr(tokenizer, "eos_token_id", None)
    return int(eid) if eid is not None else None


def main() -> None:
    p = argparse.ArgumentParser(description="RLVR with GRPO or DAPO.")
    p.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--init", choices=["scratch", "checkpoint"], default="checkpoint")
    p.add_argument("--data", nargs="+", required=False, default=None,
                   help="JSONL prompt files: {question, answer} or {prompt, constraints} or {prompt, tests}.")
    p.add_argument("--out-dir", default=str(default_base_dir() / "runs" / "rlvr"))
    p.add_argument("--device-type", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--algo", choices=["grpo", "dapo"], default="dapo")
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-prompt-len", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--eps-low", type=float, default=0.2)
    p.add_argument("--eps-high", type=float, default=0.28)
    p.add_argument("--kl-coef", type=float, default=0.0, help="If > 0, add a KL penalty against the initial policy snapshot.")
    p.add_argument("--matrix-lr", type=float, default=0.001)
    p.add_argument("--adam-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=1)
    add_monitoring_args(p, default_project="nanovlm-rlvr")
    args = p.parse_args()

    ctx = init_runtime(args.device_type, args.dtype)
    tokenizer = load_tokenizer(args.model_path)
    model = build_model(model_path=args.model_path, init=args.init, device=ctx.device, dtype=ctx.dtype, tiny=args.tiny, text_only=args.text_only)

    ref = None
    if args.kl_coef > 0:
        import copy
        ref = copy.deepcopy(model).eval()
        for q in ref.parameters():
            q.requires_grad_(False)

    if args.data:
        records = JsonlStream(args.data, rank=ctx.rank, world_size=ctx.world_size, loop=True)
    else:
        records = RlvrRecordIter([
            {"question": "Answer with the exact number: 2+2", "answer": "4"},
            {"question": "Answer with the exact number: 6*7", "answer": "42"},
        ])
    sampler = GroupSampler(records, group_size=args.group_size)

    optimizer = build_optimizer(model, OptimConfig(matrix_lr=args.matrix_lr, adam_lr=args.adam_lr, weight_decay=args.weight_decay))

    out_dir = Path(args.out_dir)
    logger = MetricsLogger(out_dir, "rlvr")
    wandb_logger = WandbLogger(
        enabled=args.wandb, project=args.wandb_project, run_name=args.wandb_run,
        entity=args.wandb_entity, mode=args.wandb_mode, config=vars(args),
        out_dir=out_dir, master=ctx.master,
    )
    early = EarlyStopper(metric=args.early_stop_metric, mode=args.early_stop_mode,
                          patience=args.early_stop_patience, min_delta=args.early_stop_min_delta,
                          max_loss=args.max_loss)
    eos = _eos_id(tokenizer)

    print0(f"[rlvr] params={count_parameters(model):,} algo={args.algo} group_size={args.group_size}")

    step = 0
    for step in range(args.steps):
        t0 = time.time()
        group = next(sampler)
        rec = group[0]
        prompt_text = _wrap_prompt(str(rec.get("question") or rec.get("prompt") or ""))

        rollout = group_rollout(
            model, tokenizer, prompt_text,
            group_size=args.group_size,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            device=ctx.device,
        )

        rewards = []
        for txt in rollout.response_text:
            r, _ = reward_record(txt, rec)
            rewards.append(r)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=ctx.device)

        if args.algo == "dapo" and (rewards_t.min().item() == rewards_t.max().item()):
            metrics = {
                "step": step, "reward": float(rewards_t.mean().cpu()), "groups_dropped": 1,
                "lr": optimizer.param_groups[0]["lr"],
            }
            logger.log(**metrics)
            wandb_logger.log(metrics)
            print0(f"step {step:05d} dropped uniform group reward={metrics['reward']:.3f}")
            continue

        advantages = group_advantages(rewards_t, group_size=args.group_size)

        # Build a single batch [G, T_prompt + T_resp_max]
        G = args.group_size
        old_logp_seqs = rollout.response_logprobs
        resp_ids, _resp_mask = stack_padded(rollout.response_ids, pad_value=eos or 0)
        prompt_ids = rollout.prompt_ids.expand(G, -1).contiguous()
        seq_ids = torch.cat([prompt_ids, resp_ids], dim=1)
        T_p = prompt_ids.size(1)
        T_resp = resp_ids.size(1)

        # Re-forward with grad
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=seq_ids)
        logits = out["logits"]
        # Predict positions [T_p .. T_p + T_resp - 1] from positions [T_p - 1 .. T_p + T_resp - 2]
        pred_logits = logits[:, T_p - 1: T_p - 1 + T_resp, :].contiguous()
        new_logp = per_token_logprobs(pred_logits, resp_ids)
        # Mask out padding tokens (positions beyond each rollout's actual length)
        mask = torch.zeros(G, T_resp, device=ctx.device)
        for i, lp in enumerate(old_logp_seqs):
            mask[i, : lp.size(0)] = 1.0
        # Pad old_logp to [G, T_resp]
        old_logp = torch.zeros(G, T_resp, device=ctx.device)
        for i, lp in enumerate(old_logp_seqs):
            old_logp[i, : lp.size(0)] = lp.to(ctx.device)

        if args.algo == "dapo":
            loss, lm = dapo_loss(new_logp, old_logp, advantages, mask, eps_low=args.eps_low, eps_high=args.eps_high)
        else:
            loss, lm = grpo_policy_loss(new_logp, old_logp, advantages, mask, eps=args.eps_low)

        if ref is not None:
            with torch.no_grad():
                ref_out = ref(input_ids=seq_ids)
                ref_logits = ref_out["logits"][:, T_p - 1: T_p - 1 + T_resp, :].contiguous()
                ref_logp = per_token_logprobs(ref_logits, resp_ids)
            kl = grpo_kl_penalty(new_logp, ref_logp, mask)
            loss = loss + args.kl_coef * kl
            lm["kl_penalty"] = float(kl.detach().cpu())

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
                "reward": float(rewards_t.mean().cpu()),
                "pass_at_1": float((rewards_t > 0).float().mean().cpu()),
                "step_time_s": dt,
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": grad_global_norm(model.parameters()),
                **lm,
            }
            logger.log(**metrics)
            wandb_logger.log(metrics)
            print0(f"step {step:05d} loss={loss_value:.4f} reward={metrics['reward']:.3f} clip={lm.get('clip_frac', 0):.3f}")
            if early.check(metrics):
                print0(f"early stop at step {step}: {early.reason}")
                break
        if ctx.master and args.save_every > 0 and step > 0 and step % args.save_every == 0:
            save_checkpoint(out_dir / "checkpoints", step, model, optimizer,
                            {"args": vars(args), "rng": rng_state()}, rank=ctx.rank)

    if ctx.master:
        save_checkpoint(out_dir / "checkpoints", step + 1, model, optimizer,
                        {"args": vars(args), "rng": rng_state(), "early_stop": early.reason}, rank=ctx.rank)
        print0(f"html: {write_html_report(out_dir, title='rlvr')}")
        print0(f"md:   {write_markdown_report(out_dir, title='rlvr')}")
    wandb_logger.finish()
    cleanup_runtime()


if __name__ == "__main__":
    main()
