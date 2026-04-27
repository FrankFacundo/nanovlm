"""DocVQA: generative + ANLS (Average Normalized Levenshtein Similarity)."""

from __future__ import annotations

from typing import Iterator

from nanovlm.eval.data import fetch_eval_dataset, find_local, stream_parquet
from nanovlm.eval.metrics import anls
from nanovlm.eval.task import GenerativeTask, TaskExample


class DocVQA(GenerativeTask):
    name = "docvqa"
    requires_image = True
    max_new_tokens = 64
    temperature = 0.0

    def iter_examples(self, data_root: str | None = None, *, limit: int | None = None) -> Iterator[TaskExample]:
        paths = find_local(data_root, "eval", "docvqa") or fetch_eval_dataset(
            "lmms-lab/DocVQA", config="DocVQA", split="validation",
            out_dir=(data_root or "/tmp") + "/eval/docvqa",
        )
        if not paths:
            return
        for rec in stream_parquet(paths, limit=limit):
            answers = rec.get("answers") or []
            if isinstance(answers, str):
                answers = [answers]
            if not answers:
                continue
            yield TaskExample(
                inputs={"question": rec.get("question", "")},
                target=list(answers),
            )

    def format_prompt(self, example: TaskExample) -> str:
        return f"Answer the question about the document with a short value.\nQuestion: {example.inputs['question']}\nAnswer:"

    def score_prediction(self, example: TaskExample, prediction: str) -> dict:
        scores = [anls(prediction, str(g)) for g in example.target]
        return {"anls": max(scores) if scores else 0.0}
