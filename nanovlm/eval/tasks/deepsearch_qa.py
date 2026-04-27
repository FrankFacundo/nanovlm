"""DeepSearchQA: tool_use(web_search + browser) → token F1 over short answers."""

from __future__ import annotations

from typing import Iterator

from nanovlm.eval.data import fetch_eval_dataset, find_local, stream_parquet
from nanovlm.eval.metrics import token_f1
from nanovlm.eval.task import TaskExample, ToolUseTask


class DeepSearchQA(ToolUseTask):
    name = "deepsearch_qa"
    max_turns = 8
    max_new_tokens_per_turn = 256
    tool_names = ["web_search", "browser"]

    def iter_examples(self, data_root: str | None = None, *, limit: int | None = None) -> Iterator[TaskExample]:
        paths = find_local(data_root, "eval", "deepsearch_qa") or fetch_eval_dataset(
            "vtllms/DeepSearch-QA", config="default", split="test",
            out_dir=(data_root or "/tmp") + "/eval/deepsearch_qa",
        )
        if not paths:
            return
        for rec in stream_parquet(paths, limit=limit):
            yield TaskExample(
                inputs={"question": rec.get("question") or rec.get("query", "")},
                target=rec.get("answer") or rec.get("ground_truth", ""),
            )

    def format_prompt(self, example: TaskExample) -> str:
        return (
            "You are a search-augmented QA agent. You have access to two tools:\n"
            "  - web_search(query, k=5)  — returns [{title,url,snippet},...]\n"
            "  - browser(url, max_chars=4000)  — fetches and returns page text\n"
            "Use them to answer the question with a short factual phrase only.\n\n"
            f"Question: {example.inputs['question']}\n"
            "When you have the answer, output it directly without using any tool."
        )

    def score_prediction(self, example: TaskExample, prediction: str, trajectory: dict) -> dict:
        gold = example.target if isinstance(example.target, str) else (example.target[0] if example.target else "")
        return {"token_f1": token_f1(prediction, gold), "n_tool_calls": float(len(trajectory.get("tools_called", [])))}
