from __future__ import annotations

import argparse

import torch

from nanovlm.models.qwen3_5 import DEFAULT_MODEL_PATH
from nanovlm.train.common import init_runtime, move_batch, print0, cleanup_runtime
from nanovlm.train.data import PackedTextLoader, eos_id
from nanovlm.train.engine import generate
from nanovlm.train.model_factory import build_model, load_tokenizer


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate loss or sample from Qwen3.5.")
    p.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--init", choices=["scratch", "checkpoint"], default="checkpoint")
    p.add_argument("--data", nargs="*", default=None)
    p.add_argument("--prompt", default="Describe what a VLM does in one sentence.")
    p.add_argument("--device-type", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-steps", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=32)
    args = p.parse_args()
    ctx = init_runtime(args.device_type, args.dtype)
    tokenizer = load_tokenizer(args.model_path)
    model = build_model(model_path=args.model_path, init=args.init, device=ctx.device, dtype=ctx.dtype, tiny=args.tiny, text_only=args.text_only).eval()
    loader = PackedTextLoader(tokenizer, args.data, args.batch_size, args.seq_len)
    losses = []
    with torch.no_grad():
        for _ in range(args.eval_steps):
            batch = move_batch(next(loader), ctx.device, ctx.dtype)
            losses.append(float(model(**batch)["loss"].cpu()))
        prompt = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=ctx.device)
        out = generate(model, prompt, max_new_tokens=args.max_new_tokens, eos_token_id=eos_id(tokenizer), temperature=0.0)
    print0(f"val_loss={sum(losses)/len(losses):.4f}")
    print0(tokenizer.decode(out[0].tolist(), skip_special_tokens=True))
    cleanup_runtime()


if __name__ == "__main__":
    main()
