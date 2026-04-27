"""Dry-run tests for the dataset downloader (no network).

We monkey-patch ``list_parquet_shards`` and ``head_size`` so the tests run
offline; the goal is to validate license filtering, bucket allocation, and the
manifest payload shape — not the live HF API.
"""

from __future__ import annotations

import nanovlm.train.download as dl
from nanovlm.train.download import DEFAULT_SOURCES, plan_downloads


def test_default_registry_has_expected_buckets():
    buckets = {s.bucket for s in DEFAULT_SOURCES}
    expected = {
        "text-pretrain", "math-pretrain", "code-pretrain", "reasoning",
        "sft-text", "preference-text", "rlvr",
        "vlm-pretrain", "vlm-sft", "vlm-pref", "eval",
    }
    assert expected == buckets


def test_default_registry_only_permissive_by_default():
    for src in DEFAULT_SOURCES:
        assert src.license != "non_commercial"
        assert src.license != "openai_synthetic"


def test_license_gate_blocks_noncommercial(monkeypatch, tmp_path):
    monkeypatch.setattr(dl, "list_parquet_shards", lambda *a, **k: [])
    monkeypatch.setattr(dl, "head_size", lambda *a, **k: 0)
    fake = list(DEFAULT_SOURCES) + [
        dl.DatasetSource("nc_only", "fake/repo", "text-pretrain", "non_commercial"),
    ]
    monkeypatch.setattr(dl, "DEFAULT_SOURCES", fake)
    plan = plan_downloads(out_dir=tmp_path, max_download_gb=1.0, allow_noncommercial=False)
    names = {s["name"] for s in plan.sources}
    assert "nc_only" not in names
    plan2 = plan_downloads(out_dir=tmp_path, max_download_gb=1.0, allow_noncommercial=True)
    names2 = {s["name"] for s in plan2.sources}
    assert "nc_only" in names2


def test_plan_respects_byte_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(dl, "list_parquet_shards", lambda *a, **k: ["http://fake/0.parquet", "http://fake/1.parquet"])
    monkeypatch.setattr(dl, "head_size", lambda *a, **k: 1 << 30)  # 1 GB per shard
    plan = plan_downloads(out_dir=tmp_path, max_download_gb=2.0)
    assert plan.total_bytes <= 2 * (1 << 30) + 100  # small slack for header rounding
    assert plan.cap_bytes == 2 * (1 << 30)


def test_include_buckets_filters_sources(monkeypatch, tmp_path):
    monkeypatch.setattr(dl, "list_parquet_shards", lambda *a, **k: [])
    monkeypatch.setattr(dl, "head_size", lambda *a, **k: 0)
    plan = plan_downloads(out_dir=tmp_path, max_download_gb=1.0, include_buckets=["eval"])
    buckets = {s["bucket"] for s in plan.sources}
    assert buckets == {"eval"}
