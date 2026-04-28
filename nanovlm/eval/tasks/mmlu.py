"""MMLU multiple-choice loglikelihood scoring."""

from __future__ import annotations

from typing import Iterator

from nanovlm.eval.data import fetch_eval_dataset, find_local, stream_parquet
from nanovlm.eval.task import LoglikelihoodTask, TaskExample


class MMLU(LoglikelihoodTask):
    name = "mmlu"

    def iter_examples(self, data_root: str | None = None, *, limit: int | None = None) -> Iterator[TaskExample]:
        paths = find_local(data_root, "eval", "mmlu") or fetch_eval_dataset(
            "cais/mmlu", config="all", split="test", out_dir=(data_root or "/tmp") + "/eval/mmlu"
        )
        if not paths:
            return
        for rec in stream_parquet(paths, limit=limit):
            choices = rec.get("choices") or [rec.get("A", ""), rec.get("B", ""), rec.get("C", ""), rec.get("D", "")]
            answer = rec.get("answer")
            if isinstance(answer, str):
                answer = "ABCD".index(answer.strip().upper()[0])
            prompt = (
                f"Question: {rec.get('question', '')}\n"
                f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
            )
            yield TaskExample(inputs={"prompt": prompt}, target=int(answer), meta={"subject": rec.get("subject")})

    def candidates(self, example: TaskExample) -> list[str]:
        return [" A", " B", " C", " D"]

    def correct_index(self, example: TaskExample) -> int:
        return int(example.target)
