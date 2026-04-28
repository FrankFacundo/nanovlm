"""HLE-with-tools: tool_use(web_search + browser + python) → exact-match.

NB: HLE is a gated dataset on Hugging Face; access is controlled via the
``HF_TOKEN`` environment variable. If access is missing this task yields
nothing (and the runner records ``n=0``).
"""

from __future__ import annotations

from typing import Iterator

from nanovlm.eval.data import fetch_eval_dataset, find_local, stream_parquet
from nanovlm.eval.metrics import exact_match
from nanovlm.eval.task import TaskExample, ToolUseTask


class HLEWithTools(ToolUseTask):
    name = "hle_with_tools"
    max_turns = 12
    max_new_tokens_per_turn = 384
    tool_names = ["web_search", "browser", "python"]

    def iter_examples(self, data_root: str | None = None, *, limit: int | None = None) -> Iterator[TaskExample]:
        paths = find_local(data_root, "eval", "hle") or fetch_eval_dataset(
            "cais/hle", config="default", split="test",
            out_dir=(data_root or "/tmp") + "/eval/hle",
        )
        if not paths:
            return
        for rec in stream_parquet(paths, limit=limit):
            yield TaskExample(
                inputs={"question": rec.get("question") or rec.get("prompt", "")},
                target=rec.get("answer") or rec.get("solution", ""),
                meta={"category": rec.get("category"), "id": rec.get("id")},
            )

    def format_prompt(self, example: TaskExample) -> str:
        return (
            "You are an expert problem solver with access to three tools:\n"
            "  - web_search(query, k=5)\n"
            "  - browser(url)\n"
            "  - python(code)\n"
            "Use them to find or compute the answer. Reply with only the final answer.\n\n"
            f"Problem: {example.inputs['question']}"
        )

    def score_prediction(self, example: TaskExample, prediction: str, trajectory: dict) -> dict:
        return {"exact_match": exact_match(prediction, str(example.target)), "n_steps": float(trajectory.get("n_steps", 0))}
