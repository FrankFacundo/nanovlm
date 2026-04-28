"""MATH-500: generative + sympy-aware equivalence."""

from __future__ import annotations

from typing import Iterator

from nanovlm.eval.data import fetch_eval_dataset, find_local, stream_parquet
from nanovlm.eval.metrics import math_equivalence_reward
from nanovlm.eval.task import GenerativeTask, TaskExample


class MATH(GenerativeTask):
    name = "math"
    max_new_tokens = 512
    temperature = 0.0

    def iter_examples(self, data_root: str | None = None, *, limit: int | None = None) -> Iterator[TaskExample]:
        paths = find_local(data_root, "eval", "math") or fetch_eval_dataset(
            "HuggingFaceH4/MATH-500", config="default", split="test",
            out_dir=(data_root or "/tmp") + "/eval/math",
        )
        if not paths:
            return
        for rec in stream_parquet(paths, limit=limit):
            yield TaskExample(
                inputs={"problem": rec.get("problem", "")},
                target=rec.get("answer") or rec.get("solution", ""),
                meta={"level": rec.get("level"), "subject": rec.get("subject")},
            )

    def format_prompt(self, example: TaskExample) -> str:
        return (
            "Solve the math problem. Put your final answer in \\boxed{}.\n"
            f"Problem: {example.inputs['problem']}\nSolution:"
        )

    def score_prediction(self, example: TaskExample, prediction: str) -> dict:
        return {"accuracy": math_equivalence_reward(prediction, str(example.target))}
