"""Smoke tests for the eval harness using mock data and the tiny model."""

import json

import pytest
import torch

from nanovlm.eval.metrics import (
    anls,
    exact_match,
    normalize_text,
    relaxed_em,
    token_f1,
)
from nanovlm.eval.runner import TaskRunner
from nanovlm.eval.task import (
    GenerativeTask,
    LoglikelihoodTask,
    TaskExample,
    ToolUseTask,
)


def test_normalize_text_strips_articles_and_punct():
    assert normalize_text("The Cat, a fox.") == "cat fox"


def test_token_f1_simple():
    assert token_f1("the cat sat", "a cat sat") == pytest.approx(1.0, rel=0.01)
    assert token_f1("foo bar", "baz qux") == 0.0


def test_anls_substring():
    assert anls("paris", "paris") == 1.0
    assert anls("paris", "paaris") > 0.5


def test_relaxed_em_numeric_tolerance():
    assert relaxed_em("47.5", "48") == 1.0   # within 5%
    assert relaxed_em("100", "200") == 0.0


class _DummyMCQ(LoglikelihoodTask):
    name = "dummy_mcq"

    def iter_examples(self, data_root=None, *, limit=None):
        yield TaskExample(inputs={"prompt": "Pick: "}, target=1)
        yield TaskExample(inputs={"prompt": "Pick: "}, target=0)

    def candidates(self, example):
        return [" A", " B"]

    def correct_index(self, example):
        return int(example.target)


def test_loglikelihood_task_runs(tmp_path):
    from nanovlm.train.model_factory import build_model, load_tokenizer

    torch.manual_seed(0)
    m = build_model(init="scratch", tiny=True, text_only=True, dtype=torch.float32, device="cpu")
    # Dummy tokenizer that just splits whitespace
    class T:
        eos_token_id = 0
        def encode(self, s):
            return [hash(t) & 0xFF for t in s.split() if t]
        def decode(self, ids):
            return " ".join(str(i) for i in ids)

    runner = TaskRunner(m, T(), device="cpu", out_dir=tmp_path)
    agg = runner.evaluate(_DummyMCQ(), limit=2)
    assert agg["n"] == 2
    assert "accuracy" in agg


class _DummyGen(GenerativeTask):
    name = "dummy_gen"
    max_new_tokens = 4

    def iter_examples(self, data_root=None, *, limit=None):
        yield TaskExample(inputs={"q": "ping"}, target="42")

    def format_prompt(self, example):
        return "answer: "

    def score_prediction(self, example, prediction):
        return {"accuracy": float("42" in prediction)}


def test_generative_task_runs(tmp_path):
    from nanovlm.train.model_factory import build_model

    torch.manual_seed(0)
    m = build_model(init="scratch", tiny=True, text_only=True, dtype=torch.float32, device="cpu")
    class T:
        eos_token_id = 0
        def encode(self, s):
            return [hash(t) & 0xFF for t in s.split() if t]
        def decode(self, ids):
            return " ".join(str(i) for i in ids)

    runner = TaskRunner(m, T(), device="cpu", out_dir=tmp_path)
    agg = runner.evaluate(_DummyGen(), limit=1)
    assert agg["n"] == 1


class _DummyTool(ToolUseTask):
    name = "dummy_tool"
    max_turns = 1
    max_new_tokens_per_turn = 4
    tool_names = ["python"]

    def iter_examples(self, data_root=None, *, limit=None):
        yield TaskExample(inputs={"q": "compute"}, target="ok")

    def format_prompt(self, example):
        return "go: "

    def score_prediction(self, example, prediction, trajectory):
        return {"got_response": 1.0, "n_steps": float(trajectory.get("n_steps", 0))}


def test_tool_use_task_runs(tmp_path):
    from nanovlm.train.model_factory import build_model

    torch.manual_seed(0)
    m = build_model(init="scratch", tiny=True, text_only=True, dtype=torch.float32, device="cpu")
    class T:
        eos_token_id = 0
        def encode(self, s):
            return [hash(t) & 0xFF for t in s.split() if t]
        def decode(self, ids):
            return " ".join(str(i) for i in ids)

    runner = TaskRunner(m, T(), device="cpu", out_dir=tmp_path)
    agg = runner.evaluate(_DummyTool(), limit=1)
    assert agg["n"] == 1


def test_all_tasks_register():
    from nanovlm.eval.tasks import ALL_TASKS, HARD_TASKS, SIMPLE_TASKS

    assert len(SIMPLE_TASKS) == 12
    assert len(HARD_TASKS) == 4
    assert len(ALL_TASKS) == len(SIMPLE_TASKS) + len(HARD_TASKS)
    names = {T.name for T in ALL_TASKS}
    assert "deepsearch_qa" in names
    assert "swe_multilingual" in names
    assert "vstar_python" in names
    assert "hle_with_tools" in names
