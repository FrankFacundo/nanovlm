"""HumanEval: generative + sandboxed pytest pass@1."""

from __future__ import annotations

from typing import Iterator

from nanovlm.eval.data import fetch_eval_dataset, find_local, stream_parquet
from nanovlm.eval.metrics import python_unit_test_reward
from nanovlm.eval.task import GenerativeTask, TaskExample


class HumanEval(GenerativeTask):
    name = "humaneval"
    max_new_tokens = 384
    temperature = 0.0

    def iter_examples(self, data_root: str | None = None, *, limit: int | None = None) -> Iterator[TaskExample]:
        paths = find_local(data_root, "eval", "humaneval") or fetch_eval_dataset(
            "openai/openai_humaneval", config="openai_humaneval", split="test",
            out_dir=(data_root or "/tmp") + "/eval/humaneval",
        )
        if not paths:
            return
        for rec in stream_parquet(paths, limit=limit):
            yield TaskExample(
                inputs={"prompt_code": rec.get("prompt", "")},
                target=rec.get("test", ""),
                meta={"entry_point": rec.get("entry_point"), "task_id": rec.get("task_id")},
            )

    def format_prompt(self, example: TaskExample) -> str:
        return (
            "Complete the Python function below. Return only the function body and any helpers.\n\n"
            f"{example.inputs['prompt_code']}"
        )

    def score_prediction(self, example: TaskExample, prediction: str) -> dict:
        prefix = example.inputs["prompt_code"]
        # If the model produced just the body, glue it after the prompt; if it
        # already included the def, use as-is.
        full = prediction if "def " in prediction else prefix + prediction
        # Append the ``check`` call required by HumanEval tests
        entry = example.meta.get("entry_point") or "candidate"
        test_block = str(example.target) + f"\n\ndef test_solution():\n    check({entry})\n"
        return {"pass_at_1": python_unit_test_reward(full, test_block, timeout_s=10.0)}
