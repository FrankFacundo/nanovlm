"""Eval task base classes.

Three task shapes are supported:

  - ``LoglikelihoodTask``: multiple-choice / completion scoring via per-token
    log-prob comparison (no sampling). Used for MMLU, ARC, MMMU MCQ.
  - ``GenerativeTask``: free-form generation under greedy/temperature, scored
    against gold via a metric. Used for GSM8K, MATH, HumanEval, IFEval,
    DocVQA, ChartQA, AI2D-OE.
  - ``ToolUseTask``: model rolls out with a tool registry; final answer
    scored. Used for hard benchmarks (DeepSearchQA, HLE, SWE-Multilingual,
    V*-with-python).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class TaskExample:
    inputs: dict
    target: Any
    meta: dict = field(default_factory=dict)


class Task(ABC):
    name: str = "task"
    requires_image: bool = False

    @abstractmethod
    def iter_examples(self, data_root: str | None = None, *, limit: int | None = None) -> Iterator[TaskExample]:
        ...

    @abstractmethod
    def evaluate_example(self, model, tokenizer, example: TaskExample, *, device, **kwargs) -> dict:
        ...

    def aggregate(self, results: list[dict]) -> dict[str, float]:
        if not results:
            return {}
        keys = sorted({k for r in results for k in r.keys() if isinstance(r.get(k), (int, float))})
        out = {}
        for k in keys:
            values = [r[k] for r in results if k in r and isinstance(r[k], (int, float))]
            if values:
                out[k] = sum(values) / len(values)
        return out


class LoglikelihoodTask(Task):
    """Score by per-token log-probability of each candidate completion."""

    @abstractmethod
    def candidates(self, example: TaskExample) -> list[str]:
        ...

    @abstractmethod
    def correct_index(self, example: TaskExample) -> int:
        ...

    def evaluate_example(self, model, tokenizer, example: TaskExample, *, device, **kwargs) -> dict:
        import torch
        from nanovlm.train.losses import sequence_logprobs

        prompt = example.inputs["prompt"]
        candidates = self.candidates(example)
        gold = self.correct_index(example)
        scores = []
        prompt_ids = tokenizer.encode(prompt)
        for cand in candidates:
            cand_ids = tokenizer.encode(cand)
            ids = torch.tensor([prompt_ids + cand_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(input_ids=ids)
            logits = out["logits"][:, :-1].contiguous()
            labels = ids[:, 1:].contiguous()
            mask = torch.zeros_like(labels)
            mask[:, len(prompt_ids) - 1:] = 1  # the candidate's tokens
            logp = sequence_logprobs(logits, labels, mask)
            scores.append(float(logp.cpu()))
        pred = max(range(len(scores)), key=lambda i: scores[i])
        return {"accuracy": 1.0 if pred == gold else 0.0, "pred": pred, "gold": gold, "scores": scores}


class GenerativeTask(Task):
    """Score by metric on a free-form generated answer."""

    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0

    @abstractmethod
    def format_prompt(self, example: TaskExample) -> str:
        ...

    @abstractmethod
    def score_prediction(self, example: TaskExample, prediction: str) -> dict:
        ...

    def evaluate_example(self, model, tokenizer, example: TaskExample, *, device, **kwargs) -> dict:
        import torch
        from nanovlm.train.engine import generate

        prompt = self.format_prompt(example)
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        out = generate(
            model, ids,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            temperature=self.temperature,
            top_p=self.top_p,
            use_cache=True,
        )
        new_tokens = out.sequences[0, ids.size(1):].tolist()
        prediction = tokenizer.decode(new_tokens)
        result = self.score_prediction(example, prediction)
        result["prediction"] = prediction
        return result


class ToolUseTask(Task):
    """Score by metric on the final answer of a multi-turn tool-use rollout."""

    max_turns: int = 6
    max_new_tokens_per_turn: int = 256
    tool_names: list[str] = []

    @abstractmethod
    def format_prompt(self, example: TaskExample) -> str:
        ...

    @abstractmethod
    def score_prediction(self, example: TaskExample, prediction: str, trajectory: dict) -> dict:
        ...

    def evaluate_example(self, model, tokenizer, example: TaskExample, *, device, tools=None, **kwargs) -> dict:
        from nanovlm.train.rollout import tool_use_rollout

        if tools is None:
            from nanovlm.train.tools import build_tool_registry
            tools = build_tool_registry()
        # Filter to just the requested tool names if specified
        if self.tool_names:
            tools = {k: v for k, v in tools.items() if k in self.tool_names}

        prompt = self.format_prompt(example)
        traj = tool_use_rollout(
            model, tokenizer, prompt, tools,
            max_turns=self.max_turns,
            max_new_tokens_per_turn=self.max_new_tokens_per_turn,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            device=device,
        )
        traj_dict = {
            "prompt": traj.prompt_text,
            "final_text": traj.final_text,
            "n_steps": len(traj.steps),
            "tools_called": [s.tool_name for s in traj.steps if s.tool_name],
        }
        result = self.score_prediction(example, traj.final_text, traj_dict)
        result["prediction"] = traj.final_text
        result["trajectory"] = traj_dict
        return result
