"""AI2D multimodal MCQ loglikelihood."""

from __future__ import annotations

from typing import Iterator

from nanovlm.eval.data import fetch_eval_dataset, find_local, stream_parquet
from nanovlm.eval.task import LoglikelihoodTask, TaskExample


class AI2D(LoglikelihoodTask):
    name = "ai2d"
    requires_image = True

    def iter_examples(self, data_root: str | None = None, *, limit: int | None = None) -> Iterator[TaskExample]:
        paths = find_local(data_root, "eval", "ai2d") or fetch_eval_dataset(
            "lmms-lab/ai2d", config="default", split="test",
            out_dir=(data_root or "/tmp") + "/eval/ai2d",
        )
        if not paths:
            return
        for rec in stream_parquet(paths, limit=limit):
            options = rec.get("options") or [rec.get("A"), rec.get("B"), rec.get("C"), rec.get("D")]
            options = [o for o in options if o is not None]
            if len(options) < 2:
                continue
            answer = rec.get("answer")
            if isinstance(answer, str) and answer.strip().upper()[:1] in "ABCDEFGH":
                gold = "ABCDEFGH".index(answer.strip().upper()[0])
            else:
                try:
                    gold = int(answer)
                except (TypeError, ValueError):
                    continue
            labels = list("ABCDEFGH")[: len(options)]
            prompt = (
                f"Question: {rec.get('question', '')}\n"
                + "\n".join(f"{l}. {t}" for l, t in zip(labels, options))
                + "\nAnswer:"
            )
            yield TaskExample(inputs={"prompt": prompt}, target=gold, meta={"labels": labels})

    def candidates(self, example: TaskExample) -> list[str]:
        return [" " + l for l in example.meta["labels"]]

    def correct_index(self, example: TaskExample) -> int:
        return int(example.target)
