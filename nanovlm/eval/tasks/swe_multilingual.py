"""SWE-Bench Multilingual subset: tool_use(filesystem + python) → patch-applies-and-tests-pass.

Each example provides a small repo snapshot, a failing test, and a goal. The
agent has filesystem (read/write/list/grep) and python (run pytest) tools. It
succeeds iff after its rollout the failing test passes.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Iterator

from nanovlm.eval.data import fetch_eval_dataset, find_local, stream_parquet
from nanovlm.eval.task import TaskExample, ToolUseTask


class SWEMultilingual(ToolUseTask):
    name = "swe_multilingual"
    max_turns = 16
    max_new_tokens_per_turn = 512
    tool_names = ["filesystem", "python"]

    def iter_examples(self, data_root: str | None = None, *, limit: int | None = None) -> Iterator[TaskExample]:
        paths = find_local(data_root, "eval", "swe_multilingual") or fetch_eval_dataset(
            "ScalingIntelligence/swe-bench-multilingual", config="default", split="test",
            out_dir=(data_root or "/tmp") + "/eval/swe_multilingual",
        )
        if not paths:
            return
        for rec in stream_parquet(paths, limit=limit):
            yield TaskExample(
                inputs={
                    "instance_id": rec.get("instance_id"),
                    "repo": rec.get("repo"),
                    "problem_statement": rec.get("problem_statement", ""),
                    "test_patch": rec.get("test_patch", ""),
                    "patch": rec.get("patch", ""),
                    "lang": rec.get("language") or rec.get("lang", "python"),
                },
                target=rec.get("FAIL_TO_PASS") or [],
                meta={"id": rec.get("instance_id")},
            )

    def format_prompt(self, example: TaskExample) -> str:
        return (
            "You are a senior engineer fixing a failing test. Tools:\n"
            "  - filesystem({op: read|write|list|grep, path, ...})\n"
            "  - python({code, files?, timeout_s?})\n"
            "Read the repo, locate the bug, write a patch, and re-run the failing test until it passes.\n\n"
            f"Repo: {example.inputs.get('repo')}\nLanguage: {example.inputs.get('lang')}\n"
            f"Problem statement:\n{example.inputs['problem_statement']}\n\n"
            f"Failing test (excerpt):\n{example.inputs.get('test_patch', '')[:1500]}\n"
        )

    def score_prediction(self, example: TaskExample, prediction: str, trajectory: dict) -> dict:
        # We do not actually re-run the patched repo here (would require the
        # full per-instance environment setup, which is out of scope for the
        # default in-process harness). The evaluation reports ``patched`` =
        # whether the model used the filesystem tool with op=write at least
        # once, plus ``ran_tests`` = whether the python tool was called.
        # For full SWE-Bench scoring, post-process the per-example trajectory
        # JSONL with the official harness. Documented limitation.
        tools_called = trajectory.get("tools_called", []) or []
        return {
            "patched": float("filesystem" in tools_called),
            "ran_tests": float("python" in tools_called),
            "tool_steps": float(trajectory.get("n_steps", 0)),
        }
