"""Interactive chat REPL using the pure-PyTorch Qwen3.5 model.

This replaces ``inference/qwen3_5-0_8B/qwen3_5_torch.py`` as the canonical
entry point. The model is loaded once; type messages and press Enter. Send
empty line to clear context. Ctrl-C / Ctrl-D to exit.
"""

from __future__ import annotations

import argparse
import sys

import torch

from nanovlm.models.qwen3_5 import DEFAULT_MODEL_PATH, render_chat
from nanovlm.train.common import init_runtime, print0
from nanovlm.train.engine import generate
from nanovlm.train.model_factory import build_model, load_tokenizer


def main() -> None:
    p = argparse.ArgumentParser(description="Pure-PyTorch chat REPL for Qwen3.5.")
    p.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--init", choices=["scratch", "checkpoint"], default="checkpoint")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device-type", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--system", default=None)
    p.add_argument("--enable-thinking", action="store_true")
    args = p.parse_args()

    ctx = init_runtime(args.device_type, args.dtype)
    tokenizer = load_tokenizer(args.model_path)
    model = build_model(model_path=args.model_path, init=args.init, device=ctx.device, dtype=ctx.dtype, tiny=args.tiny, text_only=args.text_only).eval()
    if args.checkpoint:
        from nanovlm.train.checkpoint import load_checkpoint
        load_checkpoint(args.checkpoint, model, strict=False, map_location=ctx.device)

    print0(f"[chat] model loaded on {ctx.device} dtype={ctx.dtype}. Empty line clears context.")

    history = []
    while True:
        try:
            user = input("\n> ").rstrip()
        except (EOFError, KeyboardInterrupt):
            print0("\n[chat] bye")
            return
        if not user:
            history = []
            print0("[chat] context cleared")
            continue
        history.append({"role": "user", "content": user})
        prompt = render_chat(
            history,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
            default_system=args.system,
        )
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=ctx.device)
        out = generate(
            model, ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            temperature=args.temperature,
            top_p=args.top_p,
            use_cache=True,
        )
        new_tokens = out.sequences[0, ids.size(1):].tolist()
        reply = tokenizer.decode(new_tokens).rstrip()
        if reply.endswith("<|im_end|>"):
            reply = reply[: -len("<|im_end|>")].rstrip()
        print(reply)
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
