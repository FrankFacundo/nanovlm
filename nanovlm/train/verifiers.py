"""Programmatic verifiers for RLVR + agentic training + evaluation.

Each verifier accepts a model prediction (string) plus reference data (answer,
constraints, tests, etc.) and returns a float reward in ``[0, 1]``.
"""

from __future__ import annotations

import json
import math
import os
import re
import resource
import subprocess
import tempfile
from pathlib import Path

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def extract_last_number(text: str) -> str | None:
    matches = _NUM_RE.findall(text.replace(",", ""))
    return matches[-1] if matches else None


def extract_boxed(text: str) -> str | None:
    m = list(_BOXED_RE.finditer(text))
    return m[-1].group(1).strip() if m else None


def exact_match_reward(prediction: str, answer: str) -> float:
    return 1.0 if prediction.strip().lower() == str(answer).strip().lower() else 0.0


def numeric_reward(prediction: str, answer: str, *, tol: float = 1e-6) -> float:
    pred = extract_boxed(prediction) or extract_last_number(prediction)
    gold = extract_boxed(str(answer)) or extract_last_number(str(answer))
    if pred is None or gold is None:
        return exact_match_reward(prediction, str(answer))
    try:
        return 1.0 if math.isclose(float(pred), float(gold), rel_tol=tol, abs_tol=tol) else 0.0
    except ValueError:
        return exact_match_reward(prediction, str(answer))


def math_equivalence_reward(prediction: str, answer: str) -> float:
    """MATH-style equivalence using sympy if available, else string-normalized EM."""
    pred = extract_boxed(prediction) or prediction.strip()
    gold = extract_boxed(str(answer)) or str(answer).strip()
    try:
        import sympy
        diff = sympy.simplify(sympy.sympify(pred) - sympy.sympify(gold))
        return 1.0 if diff == 0 else 0.0
    except Exception:
        a = re.sub(r"\s+", "", pred.lower())
        b = re.sub(r"\s+", "", gold.lower())
        return 1.0 if a == b else 0.0


def instruction_reward(prediction: str, constraints: dict) -> float:
    """IFEval-style strict-constraint scoring (subset).

    Supported constraint keys: ``must_contain``, ``must_not_contain``,
    ``max_words``, ``min_words``, ``starts_with``, ``ends_with``,
    ``regex``, ``json_parsable``.
    """
    score = 1.0
    for key, val in constraints.items():
        if key == "must_contain":
            req = val if isinstance(val, list) else [val]
            score *= float(all(s in prediction for s in req))
        elif key == "must_not_contain":
            req = val if isinstance(val, list) else [val]
            score *= float(all(s not in prediction for s in req))
        elif key == "max_words":
            score *= float(len(prediction.split()) <= int(val))
        elif key == "min_words":
            score *= float(len(prediction.split()) >= int(val))
        elif key == "starts_with":
            score *= float(prediction.lstrip().startswith(str(val)))
        elif key == "ends_with":
            score *= float(prediction.rstrip().endswith(str(val)))
        elif key == "regex":
            score *= float(bool(re.search(str(val), prediction)))
        elif key == "json_parsable":
            try:
                json.loads(prediction)
                ok = True
            except json.JSONDecodeError:
                ok = False
            score *= float(ok == bool(val))
    return float(score)


def python_unit_test_reward(
    prediction: str,
    tests: str,
    *,
    timeout_s: float = 10.0,
    memory_mb: int = 512,
) -> float:
    """Write the candidate solution as ``main.py`` and run ``pytest`` against ``tests``.

    Returns 1.0 iff pytest exits 0 within the timeout. Sandboxed in a temp dir;
    network is not blocked here (use ``tools.python`` for tighter isolation).
    """
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        (tdp / "main.py").write_text(prediction)
        (tdp / "test_main.py").write_text(tests)
        try:
            proc = subprocess.run(
                ["python", "-m", "pytest", "-q", "test_main.py"],
                cwd=str(tdp),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                preexec_fn=lambda: _set_limits(memory_mb),
            )
            return 1.0 if proc.returncode == 0 else 0.0
        except subprocess.TimeoutExpired:
            return 0.0


def _set_limits(memory_mb: int) -> None:
    try:
        bytes_cap = int(memory_mb) * (1 << 20)
        resource.setrlimit(resource.RLIMIT_AS, (bytes_cap, bytes_cap))
        resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    except (ValueError, OSError):
        pass


def reward_record(prediction: str, record: dict) -> tuple[float, dict[str, float]]:
    """Dispatch verifier based on the keys present in ``record``.

    Priority: ``tests`` (python) > ``constraints`` (IFEval) > ``answer``
    (numeric or exact-match). Returns ``(reward, breakdown)``.
    """
    breakdown: dict[str, float] = {}
    if "tests" in record:
        r = python_unit_test_reward(prediction, str(record["tests"]))
        breakdown["python_tests"] = r
        return r, breakdown
    if "constraints" in record:
        r = instruction_reward(prediction, record["constraints"])
        breakdown["ifeval"] = r
        return r, breakdown
    answer = record.get("answer") or record.get("solution")
    if answer is None:
        return 0.0, breakdown
    if extract_boxed(str(answer)) is not None or _NUM_RE.search(str(answer)):
        r = numeric_reward(prediction, str(answer))
        breakdown["numeric"] = r
        return r, breakdown
    r = exact_match_reward(prediction, str(answer))
    breakdown["exact"] = r
    return r, breakdown
