"""Run a sequence of tasks against a model and write a benchmark report."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import torch

from nanovlm.train.report import write_eval_report

from .task import Task


class TaskRunner:
    def __init__(self, model, tokenizer, *, device: torch.device | str = "cpu", out_dir: str | Path = "."):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, task: Task, *, data_root: str | None = None, limit: int | None = None, **eval_kwargs) -> dict:
        results = []
        per_example_path = self.out_dir / f"{task.name}.jsonl"
        with open(per_example_path, "w", encoding="utf-8") as f:
            for i, example in enumerate(task.iter_examples(data_root=data_root, limit=limit)):
                t0 = time.time()
                try:
                    res = task.evaluate_example(self.model, self.tokenizer, example, device=self.device, **eval_kwargs)
                    res["i"] = i
                    res["elapsed_s"] = time.time() - t0
                except Exception as e:
                    res = {"i": i, "error": str(e), "elapsed_s": time.time() - t0}
                results.append(res)
                f.write(json.dumps(res, default=str) + "\n")
        agg = task.aggregate(results)
        agg["n"] = len(results)
        return agg


def run_tasks(
    model,
    tokenizer,
    tasks: Sequence[Task],
    *,
    device: torch.device | str = "cpu",
    out_dir: str | Path = ".",
    data_root: str | None = None,
    limit: int | None = None,
    **eval_kwargs,
) -> dict[str, dict]:
    runner = TaskRunner(model, tokenizer, device=device, out_dir=out_dir)
    summary = {}
    for task in tasks:
        agg = runner.evaluate(task, data_root=data_root, limit=limit, **eval_kwargs)
        summary[task.name] = agg
    # Write a unified eval report
    flat_scores = {}
    for tname, agg in summary.items():
        for k, v in agg.items():
            if k == "n":
                continue
            flat_scores[f"{tname}.{k}"] = v
    write_eval_report(out_dir, flat_scores, title="NanoVLM eval report")
    (Path(out_dir) / "eval_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
