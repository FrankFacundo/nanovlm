"""Quick: compute val loss on a JSONL and print a sample completion."""

from __future__ import annotations

import argparse

import torch

from nanovlm.models.qwen3_5 import DEFAULT_MODEL_PATH
from nanovlm.train.common import init_runtime, move_batch, print0
from nanovlm.train.data.packing import BestFitPacker
from nanovlm.train.data.streaming import JsonlStream
from nanovlm.train.engine import generate
from nanovlm.train.losses import masked_ce_loss
from nanovlm.train.model_factory import build_model, load_tokenizer


def main() -> None:
    p = argparse.ArgumentParser(description="Quick val loss + sample for Qwen3.5.")
    p.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--init", choices=["scratch", "checkpoint"], default="checkpoint")
    p.add_argument("--data", nargs="+", required=False, default=None)
    p.add_argument("--prompt", default="Describe what a VLM does in one sentence.")
    p.add_argument("--device-type", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-steps", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=64)
    args = p.parse_args()

    ctx = init_runtime(args.device_type, args.dtype)
    tokenizer = load_tokenizer(args.model_path)
    model = build_model(model_path=args.model_path, init=args.init, device=ctx.device, dtype=ctx.dtype, tiny=args.tiny, text_only=args.text_only).eval()

    losses = []
    if args.data:
        stream = JsonlStream(args.data, loop=True)
        loader = BestFitPacker(stream, tokenizer, seq_len=args.seq_len, batch_size=args.batch_size)
        with torch.no_grad():
            for _ in range(args.eval_steps):
                batch = move_batch(next(loader).__dict__, ctx.device, ctx.dtype)
                out = model(input_ids=batch["input_ids"])
                logits = out["logits"][:, :-1].contiguous()
                tgt = batch["labels"][:, 1:].contiguous()
                mk = batch["loss_mask"][:, 1:].contiguous()
                losses.append(float(masked_ce_loss(logits, tgt, mk).cpu()))

    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=ctx.device)
    out = generate(model, prompt_ids, max_new_tokens=args.max_new_tokens, eos_token_id=getattr(tokenizer, "eos_token_id", None), temperature=0.0)
    completion = tokenizer.decode(out.sequences[0, prompt_ids.size(1):].tolist())
    if losses:
        print0(f"val_loss={sum(losses)/len(losses):.4f}")
    print0(f"prompt: {args.prompt}")
    print0(f"sample: {completion}")


if __name__ == "__main__":
    main()
