"""ChartQA: generative + relaxed-EM (5% numeric tolerance)."""

from __future__ import annotations

from typing import Iterator

from nanovlm.eval.data import fetch_eval_dataset, find_local, stream_parquet
from nanovlm.eval.metrics import relaxed_em
from nanovlm.eval.task import GenerativeTask, TaskExample


class ChartQA(GenerativeTask):
    name = "chartqa"
    requires_image = True
    max_new_tokens = 32
    temperature = 0.0

    def iter_examples(self, data_root: str | None = None, *, limit: int | None = None) -> Iterator[TaskExample]:
        paths = find_local(data_root, "eval", "chartqa") or fetch_eval_dataset(
            "HuggingFaceM4/ChartQA", config="default", split="test",
            out_dir=(data_root or "/tmp") + "/eval/chartqa",
        )
        if not paths:
            return
        for rec in stream_parquet(paths, limit=limit):
            yield TaskExample(
                inputs={"question": rec.get("query") or rec.get("question", "")},
                target=rec.get("label") or rec.get("answer", ""),
            )

    def format_prompt(self, example: TaskExample) -> str:
        return f"Answer the chart question with a short value only.\nQuestion: {example.inputs['question']}\nAnswer:"

    def score_prediction(self, example: TaskExample, prediction: str) -> dict:
        gold = example.target
        if isinstance(gold, list):
            gold = gold[0] if gold else ""
        return {"relaxed_em": relaxed_em(prediction, str(gold))}
