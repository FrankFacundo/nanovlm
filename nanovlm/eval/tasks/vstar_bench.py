"""V*-Bench multimodal MCQ (no tools)."""

from __future__ import annotations

from typing import Iterator

from nanovlm.eval.data import fetch_eval_dataset, find_local, stream_parquet
from nanovlm.eval.task import LoglikelihoodTask, TaskExample


class VStarBench(LoglikelihoodTask):
    name = "vstar_bench"
    requires_image = True

    def iter_examples(self, data_root: str | None = None, *, limit: int | None = None) -> Iterator[TaskExample]:
        paths = find_local(data_root, "eval", "vstar_bench") or fetch_eval_dataset(
            "craigwu/vstar_bench", config="default", split="test",
            out_dir=(data_root or "/tmp") + "/eval/vstar_bench",
        )
        if not paths:
            return
        for rec in stream_parquet(paths, limit=limit):
            options = rec.get("options") or rec.get("choices") or []
            if isinstance(options, str):
                import ast
                try:
                    options = ast.literal_eval(options)
                except Exception:
                    options = []
            if len(options) < 2:
                continue
            ans = rec.get("answer")
            if isinstance(ans, str) and ans.strip().upper()[:1] in "ABCDEFGH":
                gold = "ABCDEFGH".index(ans.strip().upper()[0])
            else:
                try:
                    gold = int(ans)
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
