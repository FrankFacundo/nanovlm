"""Eval metrics: accuracy, F1, EM, ANLS, relaxed-EM, MATH-eq, IFEval, pass@k."""

from __future__ import annotations

import re
import string
from collections import Counter

from nanovlm.train.verifiers import (
    instruction_reward,
    math_equivalence_reward,
    numeric_reward,
    python_unit_test_reward,
)


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[‘’]", "'", s)
    s = re.sub(r"[“”]", '"', s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def anls(pred: str, gold: str, *, threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity, used by DocVQA."""
    p = pred.lower().strip()
    g = gold.lower().strip()
    if not g:
        return 0.0
    n, m = len(p), len(g)
    if n == 0:
        return 0.0
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            if p[i - 1] == g[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    nls = 1.0 - dp[m] / max(n, m)
    return nls if nls >= threshold else 0.0


def relaxed_em(pred: str, gold: str, *, tol: float = 0.05) -> float:
    """ChartQA-style: numbers within 5%, otherwise EM after normalization."""
    pf = _last_number(pred)
    gf = _last_number(gold)
    if pf is not None and gf is not None:
        return 1.0 if abs(pf - gf) <= tol * max(abs(gf), 1.0) else 0.0
    return exact_match(pred, gold)


_NUM_TAIL_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*%?\s*$")


def _last_number(s: str) -> float | None:
    s = s.strip().replace(",", "")
    m = _NUM_TAIL_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def pass_at_k(scores: list[float], k: int = 1) -> float:
    """Fraction of examples with at least one correct of the top-k attempts."""
    if not scores:
        return 0.0
    return float(sum(1 for s in scores[:k] if s > 0) > 0)


__all__ = [
    "anls",
    "exact_match",
    "instruction_reward",
    "math_equivalence_reward",
    "normalize_text",
    "numeric_reward",
    "pass_at_k",
    "python_unit_test_reward",
    "relaxed_em",
    "token_f1",
]
