"""ARC-Easy / ARC-Challenge loglikelihood scoring."""

from __future__ import annotations

from typing import Iterator

from nanovlm.eval.data import fetch_eval_dataset, find_local, stream_parquet
from nanovlm.eval.task import LoglikelihoodTask, TaskExample


class _ARCBase(LoglikelihoodTask):
    config: str = "ARC-Challenge"

    def iter_examples(self, data_root: str | None = None, *, limit: int | None = None) -> Iterator[TaskExample]:
        paths = find_local(data_root, "eval", self.name) or fetch_eval_dataset(
            "allenai/ai2_arc", config=self.config, split="test",
            out_dir=(data_root or "/tmp") + f"/eval/{self.name}",
        )
        if not paths:
            return
        for rec in stream_parquet(paths, limit=limit):
            q = rec.get("question", "")
            choices = rec.get("choices") or {"text": [], "label": []}
            texts = list(choices.get("text", []))
            labels = list(choices.get("label", []))
            if not texts:
                continue
            answer_key = rec.get("answerKey") or rec.get("answer")
            if answer_key is None or str(answer_key) not in [str(l) for l in labels]:
                continue
            gold = [str(l) for l in labels].index(str(answer_key))
            prompt = "Question: " + q + "\n" + "\n".join(f"{l}. {t}" for l, t in zip(labels, texts)) + "\nAnswer:"
            yield TaskExample(
                inputs={"prompt": prompt},
                target=gold,
                meta={"labels": labels, "texts": texts},
            )

    def candidates(self, example: TaskExample) -> list[str]:
        return [" " + l for l in example.meta["labels"]]

    def correct_index(self, example: TaskExample) -> int:
        return int(example.target)


class ARCChallenge(_ARCBase):
    name = "arc_challenge"
    config = "ARC-Challenge"


class ARCEasy(_ARCBase):
    name = "arc_easy"
    config = "ARC-Easy"
