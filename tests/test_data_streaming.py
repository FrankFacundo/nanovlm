import json

import pytest
import torch

from nanovlm.train.data.packing import BestFitPacker
from nanovlm.train.data.streaming import (
    JsonlStream,
    RoundRobinStream,
    WeightedStream,
)


class _StubTokenizer:
    eos_token_id = 0

    def encode(self, text):
        return [hash(t) & 0xFF for t in str(text).split() if t]


def test_jsonl_stream_round_trip(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text("\n".join(json.dumps({"text": f"row {i}"}) for i in range(5)) + "\n")
    s = JsonlStream([p], loop=False)
    rows = list(s)
    assert [r["text"] for r in rows] == [f"row {i}" for i in range(5)]


def test_jsonl_stream_shards_per_rank(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text("\n".join(json.dumps({"text": f"row {i}"}) for i in range(8)) + "\n")
    s0 = JsonlStream([p], rank=0, world_size=2, loop=False)
    s1 = JsonlStream([p], rank=1, world_size=2, loop=False)
    r0 = [r["text"] for r in s0]
    r1 = [r["text"] for r in s1]
    assert r0 == [f"row {i}" for i in range(0, 8, 2)]
    assert r1 == [f"row {i}" for i in range(1, 8, 2)]


def test_jsonl_stream_resume(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text("\n".join(json.dumps({"text": f"row {i}"}) for i in range(6)) + "\n")
    s = JsonlStream([p], loop=False)
    it = iter(s)
    next(it)
    next(it)
    sd = s.state_dict()

    s2 = JsonlStream([p], loop=False)
    s2.load_state_dict(sd)
    rows = [r["text"] for r in s2]
    assert rows == [f"row {i}" for i in range(2, 6)]


def test_round_robin_alternates():
    a = iter([{"text": f"a{i}"} for i in range(3)])
    b = iter([{"text": f"b{i}"} for i in range(3)])
    rr = RoundRobinStream([a, b])
    out = [next(rr)["text"] for _ in range(6)]
    assert out == ["a0", "b0", "a1", "b1", "a2", "b2"]


def test_weighted_stream_obeys_proportions():
    import itertools
    a = iter(itertools.cycle([{"src": "a"}]))
    b = iter(itertools.cycle([{"src": "b"}]))
    ws = WeightedStream([a, b], weights=[0.8, 0.2], seed=42)
    counts = {"a": 0, "b": 0}
    for _ in range(2000):
        counts[next(ws)["src"]] += 1
    ratio = counts["a"] / sum(counts.values())
    assert 0.74 <= ratio <= 0.86  # ~0.8 with small-N noise


def test_best_fit_packer_packs_to_seq_len(tmp_path):
    tok = _StubTokenizer()
    records = [{"text": "a b c d e"}, {"text": "f g"}, {"text": "h i j k l m n o p"}]
    packer = BestFitPacker(iter(records * 100), tok, seq_len=8, batch_size=2)
    batch = next(packer)
    assert batch.input_ids.shape == (2, 8)
    assert batch.labels.shape == (2, 8)
    assert batch.loss_mask.shape == (2, 8)
    # First column is the BOS sep, mask 0
    assert (batch.loss_mask[:, 0] == 0).all()
