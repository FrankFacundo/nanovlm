"""IFEval (subset): generative + IFEval-style constraint scoring.

We support a curated subset of constraint kinds via ``instruction_reward`` in
``nanovlm.train.verifiers``. Full IFEval has dozens of constraint kinds; the
subset here covers the most common ~10 for a meaningful signal.
"""

from __future__ import annotations

from typing import Iterator

from nanovlm.eval.data import fetch_eval_dataset, find_local, stream_parquet
from nanovlm.eval.metrics import instruction_reward
from nanovlm.eval.task import GenerativeTask, TaskExample


_KW_TO_CONSTRAINT = {
    "keywords:existence": "must_contain",
    "keywords:forbidden": "must_not_contain",
    "length_constraints:max_words": "max_words",
    "length_constraints:min_words": "min_words",
    "startend:starts_with": "starts_with",
    "startend:ends_with": "ends_with",
}


def _kwargs_to_constraints(instruction_id: str, kwargs: dict) -> dict:
    key = _KW_TO_CONSTRAINT.get(instruction_id)
    if key is None:
        return {}
    if key in ("must_contain", "must_not_contain"):
        if "keywords" in kwargs:
            return {key: list(kwargs["keywords"])}
        if "forbidden_words" in kwargs:
            return {key: list(kwargs["forbidden_words"])}
    if key in ("max_words", "min_words"):
        for cand in ("num_words", "max_words", "min_words", "N"):
            if cand in kwargs:
                return {key: int(kwargs[cand])}
    if key in ("starts_with", "ends_with"):
        for cand in ("start_str", "starter", "end_phrase", "phrase"):
            if cand in kwargs:
                return {key: str(kwargs[cand])}
    return {}


class IFEval(GenerativeTask):
    name = "ifeval"
    max_new_tokens = 384
    temperature = 0.0

    def iter_examples(self, data_root: str | None = None, *, limit: int | None = None) -> Iterator[TaskExample]:
        paths = find_local(data_root, "eval", "ifeval") or fetch_eval_dataset(
            "google/IFEval", config="default", split="train",
            out_dir=(data_root or "/tmp") + "/eval/ifeval",
        )
        if not paths:
            return
        for rec in stream_parquet(paths, limit=limit):
            instructions = rec.get("instruction_id_list") or []
            kwargs_list = rec.get("kwargs") or [{}] * len(instructions)
            constraints = {}
            for inst, kw in zip(instructions, kwargs_list):
                if isinstance(kw, str):
                    import json as _json
                    try:
                        kw = _json.loads(kw)
                    except Exception:
                        kw = {}
                constraints.update(_kwargs_to_constraints(str(inst), dict(kw or {})))
            if not constraints:
                continue
            yield TaskExample(
                inputs={"prompt": rec.get("prompt", "")},
                target=constraints,
                meta={"instructions": instructions},
            )

    def format_prompt(self, example: TaskExample) -> str:
        return f"Follow the instruction precisely.\n\n{example.inputs['prompt']}"

    def score_prediction(self, example: TaskExample, prediction: str) -> dict:
        return {"strict_compliance": instruction_reward(prediction, example.target)}
