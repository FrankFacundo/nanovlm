"""RLVR record iterator + ``GroupSampler`` that yields N rollouts per prompt."""

from __future__ import annotations

import itertools
from typing import Iterable, Iterator


class RlvrRecordIter:
    """Cycle through RLVR records of the form::

        {"question": str, "answer": str}                 # numeric / EM
        {"prompt":   str, "constraints": dict}           # IFEval-style
        {"prompt":   str, "tests": str}                  # python unit-tests
    """

    def __init__(self, records: Iterable[dict], *, loop: bool = True):
        self.records = list(records) if not isinstance(records, list) else records
        self.loop = loop
        self.it = self._make_iter()

    def _make_iter(self):
        return itertools.cycle(self.records) if self.loop else iter(self.records)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.it)


class GroupSampler:
    """Yield ``group_size`` copies of the same prompt for GRPO/DAPO group rollout.

    Output is a list of ``group_size`` dicts (the same record, repeated).
    """

    def __init__(self, records: Iterable[dict], *, group_size: int):
        self.iter = iter(records)
        self.group_size = group_size

    def __iter__(self):
        return self

    def __next__(self) -> list[dict]:
        rec = next(self.iter)
        return [rec for _ in range(self.group_size)]
