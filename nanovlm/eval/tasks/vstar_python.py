"""V*-with-python: tool_use(python + image_ops) over V*-Bench prompts.

V*-Bench is an MCQ benchmark; we re-template prompts to allow the model to
crop/zoom/slice the image with ``image_ops`` and run computations with
``python`` before answering. Score is exact-match on the chosen option.
"""

from __future__ import annotations

from typing import Iterator

from nanovlm.eval.data import fetch_eval_dataset, find_local, stream_parquet
from nanovlm.eval.metrics import exact_match
from nanovlm.eval.task import TaskExample, ToolUseTask


class VStarPython(ToolUseTask):
    name = "vstar_python"
    requires_image = True
    max_turns = 6
    max_new_tokens_per_turn = 256
    tool_names = ["image_ops", "python"]

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
                gold_label = ans.strip().upper()[0]
            else:
                try:
                    gold_label = "ABCDEFGH"[int(ans)]
                except (TypeError, ValueError):
                    continue
            labels = list("ABCDEFGH")[: len(options)]
            yield TaskExample(
                inputs={
                    "question": rec.get("question", ""),
                    "options": list(options),
                    "labels": labels,
                    "image_path": rec.get("image_path") or rec.get("image"),
                },
                target=gold_label,
            )

    def format_prompt(self, example: TaskExample) -> str:
        labels = example.inputs["labels"]
        opts = "\n".join(f"{l}. {t}" for l, t in zip(labels, example.inputs["options"]))
        img_hint = f"\n(Image at: {example.inputs.get('image_path')})" if example.inputs.get("image_path") else ""
        return (
            "Answer the visual MCQ. You may call:\n"
            "  - image_ops({op: crop|zoom|grid|info, path, bbox?, factor?, rows?, cols?})\n"
            "  - python({code})\n"
            f"Question: {example.inputs['question']}\n{opts}{img_hint}\n"
            "Reply with the single letter of the correct option."
        )

    def score_prediction(self, example: TaskExample, prediction: str, trajectory: dict) -> dict:
        # Extract the first capital letter A-H from the prediction
        for ch in prediction.strip():
            if ch.upper() in "ABCDEFGH":
                pred_label = ch.upper()
                return {"accuracy": exact_match(pred_label, str(example.target))}
        return {"accuracy": 0.0}
