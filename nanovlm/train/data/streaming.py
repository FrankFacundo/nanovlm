"""Lazy record streams: JSONL, parquet, round-robin, weighted mixture.

All iterators are resumable via ``state_dict`` / ``load_state_dict`` and
shard-aware: pass ``rank``/``world_size`` so each rank consumes a disjoint
slice of the input shards.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class StreamState:
    file_idx: int = 0
    record_idx: int = 0
    epoch: int = 0
    docs_seen: int = 0
    tokens_seen: int = 0


class JsonlStream:
    """Iterator over one or more JSONL files, optionally sharded across ranks.

    Records are ``dict``s. Files are visited in order; on EOF the iterator
    advances to the next file or restarts (epoch += 1) if ``loop`` is True.
    """

    def __init__(
        self,
        paths: list[str | Path],
        *,
        rank: int = 0,
        world_size: int = 1,
        loop: bool = True,
        shuffle_files: bool = False,
        seed: int = 0,
    ):
        self.paths = [Path(p).expanduser() for p in paths]
        if not self.paths:
            raise ValueError("JsonlStream requires at least one path")
        self.rank = rank
        self.world_size = max(1, world_size)
        self.loop = loop
        self.shuffle_files = shuffle_files
        self.seed = seed
        self.state = StreamState()

    def __iter__(self) -> Iterator[dict]:
        while True:
            order = list(range(len(self.paths)))
            if self.shuffle_files:
                rng = random.Random(self.seed + self.state.epoch)
                rng.shuffle(order)
            for self.state.file_idx in range(self.state.file_idx, len(order)):
                path = self.paths[order[self.state.file_idx]]
                yield from self._iter_file(path)
                self.state.record_idx = 0
            self.state.file_idx = 0
            self.state.epoch += 1
            if not self.loop:
                return

    def _iter_file(self, path: Path) -> Iterator[dict]:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < self.state.record_idx:
                    continue
                if (i % self.world_size) != self.rank:
                    self.state.record_idx = i + 1
                    continue
                line = line.strip()
                if not line:
                    self.state.record_idx = i + 1
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    self.state.record_idx = i + 1
                    continue
                self.state.record_idx = i + 1
                self.state.docs_seen += 1
                yield rec

    def state_dict(self) -> dict:
        return {
            "file_idx": self.state.file_idx,
            "record_idx": self.state.record_idx,
            "epoch": self.state.epoch,
            "docs_seen": self.state.docs_seen,
            "tokens_seen": self.state.tokens_seen,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.state = StreamState(**sd)


class ParquetStream:
    """Iterator over parquet shards using pyarrow (no Hugging Face datasets dep).

    Resumes by ``(file_idx, row_group_idx, row_in_group_idx)``. Each rank
    consumes a disjoint subset of row groups.
    """

    def __init__(
        self,
        paths: list[str | Path],
        *,
        text_field: str = "text",
        rank: int = 0,
        world_size: int = 1,
        loop: bool = True,
        seed: int = 0,
    ):
        self.paths = [Path(p).expanduser() for p in paths]
        if not self.paths:
            raise ValueError("ParquetStream requires at least one path")
        self.text_field = text_field
        self.rank = rank
        self.world_size = max(1, world_size)
        self.loop = loop
        self.seed = seed
        self.state = StreamState()

    def __iter__(self) -> Iterator[dict]:
        import pyarrow.parquet as pq

        while True:
            for self.state.file_idx in range(self.state.file_idx, len(self.paths)):
                path = self.paths[self.state.file_idx]
                pf = pq.ParquetFile(str(path))
                for rg_idx in range(pf.num_row_groups):
                    if (rg_idx % self.world_size) != self.rank:
                        continue
                    table = pf.read_row_group(rg_idx)
                    cols = table.to_pydict()
                    n_rows = len(next(iter(cols.values())))
                    for row in range(n_rows):
                        rec = {k: cols[k][row] for k in cols}
                        if self.text_field not in rec and self.text_field == "text":
                            rec.setdefault("text", "")
                        self.state.docs_seen += 1
                        yield rec
                self.state.record_idx = 0
            self.state.file_idx = 0
            self.state.epoch += 1
            if not self.loop:
                return

    def state_dict(self) -> dict:
        return {
            "file_idx": self.state.file_idx,
            "record_idx": self.state.record_idx,
            "epoch": self.state.epoch,
            "docs_seen": self.state.docs_seen,
            "tokens_seen": self.state.tokens_seen,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.state = StreamState(**sd)


class RoundRobinStream:
    """Yield from a list of streams, one record at a time, in rotation."""

    def __init__(self, streams: list):
        self.iters = [iter(s) for s in streams]

    def __iter__(self):
        return self

    def __next__(self):
        if not self.iters:
            raise StopIteration
        out_iter = self.iters.pop(0)
        try:
            rec = next(out_iter)
            self.iters.append(out_iter)
            return rec
        except StopIteration:
            return self.__next__()


class WeightedStream:
    """Sample from sub-streams in proportion to ``weights``.

    Weights need not sum to 1; they are normalized internally. Each sub-stream
    is expected to be infinite (``loop=True``).
    """

    def __init__(self, streams: list, weights: list[float], seed: int = 0):
        if len(streams) != len(weights):
            raise ValueError("streams and weights must have the same length")
        total = float(sum(weights))
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        self.weights = [w / total for w in weights]
        self.iters = [iter(s) for s in streams]
        self.rng = random.Random(seed)
        self._cum = []
        running = 0.0
        for w in self.weights:
            running += w
            self._cum.append(running)

    def __iter__(self):
        return self

    def __next__(self):
        u = self.rng.random()
        for i, c in enumerate(self._cum):
            if u <= c:
                return next(self.iters[i])
        return next(self.iters[-1])
