from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch

from nanovlm.models.qwen3_5 import DEFAULT_MODEL_PATH
from nanovlm.train.checkpoint import save_checkpoint
from nanovlm.train.common import default_base_dir, init_runtime, move_batch, print0, cleanup_runtime
from nanovlm.train.data import cycle_records, eos_id
from nanovlm.train.engine import generate
from nanovlm.train.losses import grpo_policy_loss, group_advantages
from nanovlm.train.model_factory import build_model, load_tokenizer
from nanovlm.train.optim import OptimConfig, build_optimizer, set_lr_and_wd
from nanovlm.train.report import MetricsLogger, write_html_report
from nanovlm.train.schedule import lr_multiplier
from nanovlm.train.verifiers import reward_record


def _prompt(rec: dict) -> str:
    return str(rec.get("prompt") or rec.get("question") or "What is 2+2?")


def _synthetic_rl_records():
    while True:
        yield {"question": "Answer with the exact number: 2+2", "answer": "4"}
        yield {"question": "Answer with the exact number: 6*7", "answer": "42"}


def main() -> None:
    p = argparse.ArgumentParser(description="GRPO/DAPO-style RLVR with programmatic verifiers.")
    p.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--init", choices=["scratch", "checkpoint"], default="scratch")
    p.add_argument("--data", nargs="*", default=None)
    p.add_argument("--out-dir", default=str(default_base_dir() / "runs" / "rlvr"))
    p.add_argument("--device-type", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-prompt-len", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--kl-coef", type=float, default=0.0)
    args = p.parse_args()

    ctx = init_runtime(args.device_type, args.dtype)
    tokenizer = load_tokenizer(args.model_path)
    model = build_model(model_path=args.model_path, init=args.init, device=ctx.device, dtype=ctx.dtype, tiny=args.tiny, text_only=args.text_only)
    ref = build_model(model_path=args.model_path, init="scratch" if args.tiny else args.init, device=ctx.device, dtype=ctx.dtype, tiny=args.tiny, text_only=args.text_only)
    ref.load_state_dict(model.state_dict())
    ref.eval().requires_grad_(False)
    optimizer = build_optimizer(model, OptimConfig(matrix_lr=0.003, adam_lr=3e-5, weight_decay=0.0))
    records = cycle_records(args.data) if args.data else _synthetic_rl_records()
    logger = MetricsLogger(args.out_dir, "rlvr")
    pad = eos_id(tokenizer)

    for step in range(args.steps):
        rec = next(records)
        prompt_ids = tokenizer.encode(_prompt(rec))[: args.max_prompt_len]
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=ctx.device).repeat(args.group_size, 1)
        with torch.no_grad():
            seq = generate(model, prompt, max_new_tokens=args.max_new_tokens, eos_token_id=pad, temperature=args.temperature)
        completions = [tokenizer.decode(row[len(prompt_ids):].tolist(), skip_special_tokens=True) for row in seq]
        rewards = torch.tensor([reward_record(rec, c) for c in completions], dtype=torch.float32, device=ctx.device)
        adv = group_advantages(rewards, args.group_size, zscore=False)

        inputs = seq[:, :-1]
        labels = seq[:, 1:].clone()
        token_mask = torch.zeros_like(labels, dtype=torch.float32)
        token_mask[:, max(0, len(prompt_ids) - 1):] = 1.0
        labels[token_mask == 0] = -100

        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=inputs, labels=labels)
        logp = torch.log_softmax(out["logits"].float(), dim=-1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        ref_logp = None
        if args.kl_coef > 0:
            with torch.no_grad():
                ref_out = ref(input_ids=inputs, labels=labels)
                ref_logp = torch.log_softmax(ref_out["logits"].float(), dim=-1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        loss, metrics = grpo_policy_loss(logp, token_mask, adv, ref_logp, args.kl_coef)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        set_lr_and_wd(optimizer, lr_multiplier(step, args.steps, 1))
        optimizer.step()
        metrics.update({"step": step, "reward": float(rewards.mean().cpu()), "train_loss": float(loss.detach().cpu()), "pass_at_1": float((rewards > 0).float().mean().cpu())})
        logger.log(**metrics)
        print0(f"step {step:05d} rl_loss={metrics['train_loss']:.4f} reward={metrics['reward']:.3f}")

    if ctx.master:
        save_checkpoint(Path(args.out_dir) / "checkpoints", args.steps, model, optimizer, {"args": vars(args)}, rank=ctx.rank)
        print0(f"report: {write_html_report(args.out_dir)}")
    cleanup_runtime()


if __name__ == "__main__":
    main()
