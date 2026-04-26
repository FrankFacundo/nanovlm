"""Programmatic verifiers for RLVR and agent training."""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from pathlib import Path


_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def extract_last_number(text: str) -> str | None:
    matches = _NUM_RE.findall(text.replace(",", ""))
    return matches[-1] if matches else None


def exact_match_reward(prediction: str, answer: str) -> float:
    return 1.0 if prediction.strip().lower() == str(answer).strip().lower() else 0.0


def numeric_reward(prediction: str, answer: str, tol: float = 1e-6) -> float:
    pred = extract_last_number(prediction)
    gold = extract_last_number(str(answer))
    if pred is None or gold is None:
        return exact_match_reward(prediction, str(answer))
    try:
        return 1.0 if math.isclose(float(pred), float(gold), rel_tol=tol, abs_tol=tol) else 0.0
    except ValueError:
        return 0.0


def instruction_reward(prediction: str, constraints: dict) -> float:
    score = 1.0
    if "must_contain" in constraints:
        required = constraints["must_contain"]
        if isinstance(required, str):
            required = [required]
        score *= float(all(r in prediction for r in required))
    if "must_not_contain" in constraints:
        banned = constraints["must_not_contain"]
        if isinstance(banned, str):
            banned = [banned]
        score *= float(all(b not in prediction for b in banned))
    if "max_words" in constraints:
        score *= float(len(prediction.split()) <= int(constraints["max_words"]))
    return score


def python_unit_test_reward(code: str, tests: str, timeout_s: float = 5.0) -> float:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "candidate_test.py"
        path.write_text(code + "\n\n" + tests + "\n", encoding="utf-8")
        try:
            proc = subprocess.run(
                ["python", str(path)],
                cwd=td,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return 0.0
        return 1.0 if proc.returncode == 0 else 0.0


def reward_record(record: dict, prediction: str) -> float:
    if "answer" in record:
        return numeric_reward(prediction, str(record["answer"]))
    if "tests" in record:
        return python_unit_test_reward(prediction, str(record["tests"]))
    if "constraints" in record:
        constraints = record["constraints"]
        if isinstance(constraints, str):
            constraints = json.loads(constraints)
        return instruction_reward(prediction, constraints)
    return 0.0
