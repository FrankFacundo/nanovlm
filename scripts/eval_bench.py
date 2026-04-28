"""Run an arbitrary subset of the eval suite and write a benchmark report.

By default runs the simple suite. Pass ``--include-hard`` to also run the
four real tool-using benchmarks (DeepSearchQA, HLE, SWE-Multilingual,
V*-with-python). Use ``--tasks name1,name2,...`` to pick specific tasks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nanovlm.eval.runner import run_tasks
from nanovlm.eval.tasks import ALL_TASKS, HARD_TASKS, SIMPLE_TASKS
from nanovlm.models.qwen3_5 import DEFAULT_MODEL_PATH
from nanovlm.train.common import default_base_dir, init_runtime, print0
from nanovlm.train.model_factory import build_model, load_tokenizer


def _select(tasks_arg: str | None, include_hard: bool):
    if tasks_arg:
        names = {n.strip() for n in tasks_arg.split(",") if n.strip()}
        return [T() for T in ALL_TASKS if T.name in names]
    pool = list(SIMPLE_TASKS) + (list(HARD_TASKS) if include_hard else [])
    return [T() for T in pool]


def main() -> None:
    p = argparse.ArgumentParser(description="Run NanoVLM eval suite.")
    p.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--init", choices=["scratch", "checkpoint"], default="checkpoint")
    p.add_argument("--checkpoint", default=None, help="Optional path to a saved safetensors checkpoint to load.")
    p.add_argument("--tasks", default=None, help="Comma-separated task names (default: simple suite).")
    p.add_argument("--include-hard", action="store_true")
    p.add_argument("--data-root", default=str(default_base_dir() / "data"))
    p.add_argument("--out-dir", default=str(default_base_dir() / "runs" / "eval"))
    p.add_argument("--device-type", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--limit", type=int, default=200, help="Per-task example cap.")
    args = p.parse_args()

    ctx = init_runtime(args.device_type, args.dtype)
    tokenizer = load_tokenizer(args.model_path)
    model = build_model(model_path=args.model_path, init=args.init, device=ctx.device, dtype=ctx.dtype, tiny=args.tiny, text_only=args.text_only).eval()
    if args.checkpoint:
        from nanovlm.train.checkpoint import load_checkpoint
        load_checkpoint(args.checkpoint, model, strict=False, map_location=ctx.device)

    tasks = _select(args.tasks, args.include_hard)
    if not tasks:
        raise SystemExit("no tasks selected; pass --tasks or --include-hard")
    print0(f"[eval] tasks={[t.name for t in tasks]} limit={args.limit}")

    summary = run_tasks(model, tokenizer, tasks, device=ctx.device, out_dir=args.out_dir, data_root=args.data_root, limit=args.limit)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
