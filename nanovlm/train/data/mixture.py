"""Load a stage mixture from YAML and build a weighted multi-source iterator.

Mixture YAML format::

    name: pretrain_S1_general
    seq_len: 4096
    batch_size: 1
    sources:
      - name: fineweb_edu
        weight: 0.5
        format: parquet
        glob: "fineweb_edu/*.parquet"
      - name: dclm
        weight: 0.25
        format: parquet
        glob: "dclm/*.parquet"
      - name: stack_edu
        weight: 0.10
        format: jsonl
        glob: "stack_edu/*.jsonl"
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .streaming import JsonlStream, ParquetStream, WeightedStream


def load_mixture_config(path: str | Path) -> dict:
    """Tiny YAML loader: only flat keys, list of dicts; no anchors / merges."""
    text = Path(path).expanduser().read_text()
    return _parse_yaml(text)


def build_mixture_from_yaml(
    config_path: str | Path,
    data_root: str | Path,
    *,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
) -> tuple[WeightedStream, dict]:
    cfg = load_mixture_config(config_path)
    sources = cfg.get("sources", [])
    streams = []
    weights = []
    for src in sources:
        paths = sorted(Path(data_root).expanduser().glob(src["glob"]))
        if not paths:
            continue
        fmt = src.get("format", "parquet")
        if fmt == "jsonl":
            stream = JsonlStream(paths, rank=rank, world_size=world_size, loop=True, seed=seed)
        elif fmt == "parquet":
            text_field = src.get("text_field", "text")
            stream = ParquetStream(paths, text_field=text_field, rank=rank, world_size=world_size, loop=True, seed=seed)
        else:
            raise ValueError(f"unknown source format: {fmt!r}")
        streams.append(stream)
        weights.append(float(src.get("weight", 1.0)))
    if not streams:
        raise FileNotFoundError(
            f"no data found for any source under {data_root}; check globs in {config_path}"
        )
    mixture = WeightedStream(streams, weights, seed=seed)
    return mixture, cfg


def _parse_yaml(text: str):
    """Tiny line-based YAML parser supporting the mixture-config subset.

    Supports: mappings (``key: value``), lists of mappings (``- key: value``),
    nested mappings, scalars (str, int, float, bool, null), and quoted strings.
    Does NOT support flow-style, anchors, multi-line scalars, comments mid-line.
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    pos = [0]

    def _indent(s: str) -> int:
        return len(s) - len(s.lstrip(" "))

    def _scalar(v: str):
        v = v.strip()
        if not v:
            return None
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            return v[1:-1]
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        if v.lower() in ("null", "~"):
            return None
        try:
            if "." in v or "e" in v.lower():
                return float(v)
            return int(v)
        except ValueError:
            return v

    def _parse_block(min_indent: int):
        out = None
        while pos[0] < len(lines):
            line = lines[pos[0]]
            ind = _indent(line)
            if ind < min_indent:
                return out
            stripped = line.lstrip(" ")
            if stripped.startswith("- "):
                if out is None:
                    out = []
                if not isinstance(out, list):
                    return out
                pos[0] += 1
                item_line = stripped[2:]
                if ":" in item_line:
                    k, _, v = item_line.partition(":")
                    item = {k.strip(): _scalar(v) if v.strip() else _parse_block(ind + 2)}
                    while pos[0] < len(lines):
                        nxt = lines[pos[0]]
                        if _indent(nxt) <= ind:
                            break
                        sub = _parse_block(ind + 2)
                        if isinstance(sub, dict):
                            item.update(sub)
                        else:
                            break
                    out.append(item)
                else:
                    out.append(_scalar(item_line))
            else:
                if out is None:
                    out = {}
                if not isinstance(out, dict):
                    return out
                k, _, v = stripped.partition(":")
                pos[0] += 1
                k = k.strip()
                v = v.strip()
                if v:
                    out[k] = _scalar(v)
                else:
                    out[k] = _parse_block(ind + 2)
        return out

    return _parse_block(0) or {}
