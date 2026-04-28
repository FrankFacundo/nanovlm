"""Reasoning-distillation SFT with explicit ``<think>...</think>`` blocks.

Use with OpenThoughts3 / OpenR1-Math / s1K-1.1 records, where the assistant
turn includes a chain-of-thought block. Defaults to ``--thinking-ratio 0.85``
so almost all training samples preserve the thinking trace.

Records can be either:
  - ``{"messages": [...]}`` already containing ``{"role": "assistant",
    "thinking": "...", "content": "..."}``, or
  - ``{"prompt": "...", "thinking": "...", "response": "..."}``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Reasoning SFT is a trivial wrapper over scripts.sft with a different default
# config + thinking ratio. We delegate by re-execing main() with adjusted argv.

import sys

from nanovlm.train.common import default_base_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Distillation SFT with <think> blocks.")
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "sft_reasoning.yaml"))
    p.add_argument("--out-dir", default=str(default_base_dir() / "runs" / "reasoning_sft"))
    p.add_argument("--thinking-ratio", type=float, default=0.85)
    p.add_argument("--seq-len", type=int, default=4096)
    args, rest = p.parse_known_args()

    sft_argv = [
        "scripts.sft",
        "--config", args.config,
        "--out-dir", args.out_dir,
        "--thinking-ratio", str(args.thinking_ratio),
        "--seq-len", str(args.seq_len),
    ] + rest
    sys.argv = sft_argv
    from scripts.sft import main as sft_main
    sft_main()


if __name__ == "__main__":
    main()
