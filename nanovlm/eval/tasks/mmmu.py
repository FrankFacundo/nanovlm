"""MMMU multimodal MCQ loglikelihood (text-only fallback when no image is provided)."""

from __future__ import annotations

from typing import Iterator

from nanovlm.eval.data import fetch_eval_dataset, find_local, stream_parquet
from nanovlm.eval.task import LoglikelihoodTask, TaskExample


class MMMU(LoglikelihoodTask):
    name = "mmmu"
    requires_image = True

    def iter_examples(self, data_root: str | None = None, *, limit: int | None = None) -> Iterator[TaskExample]:
        paths = find_local(data_root, "eval", "mmmu") or fetch_eval_dataset(
            "MMMU/MMMU", config="Math", split="validation",
            out_dir=(data_root or "/tmp") + "/eval/mmmu",
        )
        if not paths:
            return
        for rec in stream_parquet(paths, limit=limit):
            options = rec.get("options")
            if isinstance(options, str):
                import ast
                try:
                    options = ast.literal_eval(options)
                except Exception:
                    options = []
            if not options or len(options) < 2:
                continue
            ans = rec.get("answer")
            if isinstance(ans, str):
                gold = "ABCDEFGH".index(ans.strip().upper()[0]) if ans.strip() else 0
            else:
                gold = int(ans)
            labels = list("ABCDEFGH")[: len(options)]
            prompt = (
                f"Question: {rec.get('question', '')}\n"
                + "\n".join(f"{l}. {t}" for l, t in zip(labels, options))
                + "\nAnswer:"
            )
            yield TaskExample(inputs={"prompt": prompt}, target=gold, meta={"labels": labels, "n": len(options)})

    def candidates(self, example: TaskExample) -> list[str]:
        return [" " + l for l in example.meta["labels"]]

    def correct_index(self, example: TaskExample) -> int:
        return int(example.target)
