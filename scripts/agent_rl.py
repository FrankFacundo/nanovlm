"""Agentic RL: real model rollouts with tool calls + verifier rewards.

Each step:
  1. Sample a prompt from the agentic mixture (math/code/IF + tool tasks).
  2. Run ``tool_use_rollout`` ``group_size`` times to obtain trajectories.
  3. Score the final answer of each trajectory via ``reward_record``.
  4. Apply DAPO/GRPO update over the assistant tokens.

Trajectories are persisted under ``out_dir/trajectories/{step:06d}.jsonl``
for replay and debugging.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from nanovlm.models.qwen3_5 import DEFAULT_MODEL_PATH
from nanovlm.models.qwen3_5.chat_template import render_chat
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
from nanovlm.train.losses import dapo_loss, group_advantages, per_token_logprobs
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
from nanovlm.train.rollout import stack_padded, tool_use_rollout
from nanovlm.train.schedule import lr_multiplier
from nanovlm.train.tools import build_tool_registry
from nanovlm.train.verifiers import reward_record


def _eos_id(tokenizer) -> int | None:
    eid = getattr(tokenizer, "eos_token_id", None)
    return int(eid) if eid is not None else None


def _wrap(question: str) -> str:
    return render_chat([{"role": "user", "content": question}], add_generation_prompt=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Agentic RL: model rollouts with tool use.")
    p.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--init", choices=["scratch", "checkpoint"], default="checkpoint")
    p.add_argument("--data", nargs="+", required=False, default=None,
                   help="JSONL prompt files: {question, answer | constraints | tests}.")
    p.add_argument("--out-dir", default=str(default_base_dir() / "runs" / "agent_rl"))
    p.add_argument("--device-type", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-turns", type=int, default=6)
    p.add_argument("--max-new-tokens-per-turn", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--matrix-lr", type=float, default=0.001)
    p.add_argument("--adam-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--eps-low", type=float, default=0.2)
    p.add_argument("--eps-high", type=float, default=0.28)
    p.add_argument("--sandbox-root", default=None)
    p.add_argument("--no-web", action="store_true")
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=1)
    add_monitoring_args(p, default_project="nanovlm-agent-rl")
    args = p.parse_args()

    ctx = init_runtime(args.device_type, args.dtype)
    tokenizer = load_tokenizer(args.model_path)
    model = build_model(model_path=args.model_path, init=args.init, device=ctx.device, dtype=ctx.dtype, tiny=args.tiny, text_only=args.text_only)

    optimizer = build_optimizer(model, OptimConfig(matrix_lr=args.matrix_lr, adam_lr=args.adam_lr, weight_decay=args.weight_decay))

    if args.data:
        records = JsonlStream(args.data, rank=ctx.rank, world_size=ctx.world_size, loop=True)
    else:
        records = RlvrRecordIter([
            {"question": "Use the python tool to compute 17*23 and answer with the number only.", "answer": "391"},
            {"question": "Use the python tool to compute the GCD of 84 and 132 and answer with the number only.", "answer": "12"},
        ])
    sampler = GroupSampler(records, group_size=args.group_size)
    tools = build_tool_registry(sandbox_root=args.sandbox_root, enable_web=not args.no_web)

    out_dir = Path(args.out_dir)
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    logger = MetricsLogger(out_dir, "agent_rl")
    wandb_logger = WandbLogger(
        enabled=args.wandb, project=args.wandb_project, run_name=args.wandb_run,
        entity=args.wandb_entity, mode=args.wandb_mode, config=vars(args),
        out_dir=out_dir, master=ctx.master,
    )
    early = EarlyStopper(metric=args.early_stop_metric, mode=args.early_stop_mode,
                          patience=args.early_stop_patience, min_delta=args.early_stop_min_delta,
                          max_loss=args.max_loss)
    eos = _eos_id(tokenizer)

    print0(f"[agent_rl] params={count_parameters(model):,} group_size={args.group_size} max_turns={args.max_turns}")

    step = 0
    for step in range(args.steps):
        t0 = time.time()
        group = next(sampler)
        rec = group[0]
        prompt_text = _wrap(str(rec.get("question") or rec.get("prompt") or ""))

        trajectories = []
        rewards = []
        responses = []
        for _ in range(args.group_size):
            traj = tool_use_rollout(
                model, tokenizer, prompt_text, tools,
                max_turns=args.max_turns,
                max_new_tokens_per_turn=args.max_new_tokens_per_turn,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=eos,
                device=ctx.device,
            )
            trajectories.append(traj)
            r, _ = reward_record(traj.final_text, rec)
            traj.success = r > 0
            traj.reward = r
            rewards.append(r)
            # Tokenize the assistant turns for the policy update
            full_text = "".join(s.text for s in traj.steps if s.role == "assistant")
            responses.append(torch.tensor(tokenizer.encode(full_text), dtype=torch.long, device=ctx.device))

        # Persist trajectories
        with open(traj_dir / f"{step:06d}.jsonl", "w", encoding="utf-8") as f:
            for traj in trajectories:
                f.write(json.dumps({
                    "prompt": traj.prompt_text,
                    "final_text": traj.final_text,
                    "reward": traj.reward,
                    "steps": [s.__dict__ for s in traj.steps],
                }, default=str) + "\n")

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=ctx.device)
        if rewards_t.min().item() == rewards_t.max().item():
            metrics = {"step": step, "reward": float(rewards_t.mean().cpu()), "groups_dropped": 1, "lr": optimizer.param_groups[0]["lr"]}
            logger.log(**metrics)
            wandb_logger.log(metrics)
            print0(f"step {step:05d} dropped uniform group reward={metrics['reward']:.3f}")
            continue

        advantages = group_advantages(rewards_t, group_size=args.group_size)
        prompt_ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long, device=ctx.device)
        T_p = prompt_ids.size(1)
        resp_ids, resp_mask = stack_padded(responses, pad_value=eos or 0)
        seq_ids = torch.cat([prompt_ids.expand(args.group_size, -1).contiguous(), resp_ids], dim=1)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=seq_ids)
        logits = out["logits"]
        T_resp = resp_ids.size(1)
        pred_logits = logits[:, T_p - 1: T_p - 1 + T_resp, :].contiguous()
        new_logp = per_token_logprobs(pred_logits, resp_ids)
        # On-policy: old_logp ≈ new_logp.detach()
        old_logp = new_logp.detach()
        loss, lm = dapo_loss(new_logp, old_logp, advantages, resp_mask.float(), eps_low=args.eps_low, eps_high=args.eps_high)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        lrm = lr_multiplier(step, args.steps, args.warmup_steps)
        set_lr_and_wd(optimizer, lrm, args.weight_decay)
        optimizer.step()

        dt = time.time() - t0
        if step % args.log_every == 0:
            metrics = {
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                "reward": float(rewards_t.mean().cpu()),
                "pass_at_1": float((rewards_t > 0).float().mean().cpu()),
                "tool_success_rate": float(sum(1 for t in trajectories if t.success) / max(1, len(trajectories))),
                "step_time_s": dt,
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": grad_global_norm(model.parameters()),
                **lm,
            }
            logger.log(**metrics)
            wandb_logger.log(metrics)
            print0(f"step {step:05d} loss={metrics['train_loss']:.4f} reward={metrics['reward']:.3f} tool_ok={metrics['tool_success_rate']:.2f}")
            if early.check(metrics):
                print0(f"early stop at step {step}: {early.reason}")
                break
        if ctx.master and args.save_every > 0 and step > 0 and step % args.save_every == 0:
            save_checkpoint(out_dir / "checkpoints", step, model, optimizer,
                            {"args": vars(args), "rng": rng_state()}, rank=ctx.rank)

    if ctx.master:
        save_checkpoint(out_dir / "checkpoints", step + 1, model, optimizer,
                        {"args": vars(args), "rng": rng_state(), "early_stop": early.reason}, rank=ctx.rank)
        print0(f"html: {write_html_report(out_dir, title='agent_rl')}")
        print0(f"md:   {write_markdown_report(out_dir, title='agent_rl')}")
    wandb_logger.finish()
    cleanup_runtime()


if __name__ == "__main__":
    main()
