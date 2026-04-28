"""GSM8K: generative + numeric verifier."""

from __future__ import annotations

from typing import Iterator

from nanovlm.eval.data import fetch_eval_dataset, find_local, stream_parquet
from nanovlm.eval.metrics import numeric_reward
from nanovlm.eval.task import GenerativeTask, TaskExample


class GSM8K(GenerativeTask):
    name = "gsm8k"
    max_new_tokens = 256
    temperature = 0.0

    def iter_examples(self, data_root: str | None = None, *, limit: int | None = None) -> Iterator[TaskExample]:
        paths = find_local(data_root, "eval", "gsm8k") or fetch_eval_dataset(
            "openai/gsm8k", config="main", split="test", out_dir=(data_root or "/tmp") + "/eval/gsm8k"
        )
        if not paths:
            return
        for rec in stream_parquet(paths, limit=limit):
            answer_text = rec.get("answer", "")
            gold = answer_text.split("####")[-1].strip() if "####" in answer_text else answer_text
            yield TaskExample(
                inputs={"question": rec.get("question", "")},
                target=gold,
                meta={"full_answer": answer_text},
            )

    def format_prompt(self, example: TaskExample) -> str:
        return (
            "Solve the following grade-school math problem. End your answer with '#### <number>'.\n"
            f"Question: {example.inputs['question']}\nSolution:"
        )

    def score_prediction(self, example: TaskExample, prediction: str) -> dict:
        return {"accuracy": numeric_reward(prediction, str(example.target))}
