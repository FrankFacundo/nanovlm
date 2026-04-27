"""Eval-set loaders. Lazy-fetch parquet shards from HF if missing locally."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator


def find_local(data_root: str | None, bucket: str, name: str) -> list[Path]:
    if data_root is None:
        return []
    base = Path(data_root).expanduser() / bucket / name
    if not base.exists():
        return []
    return sorted(base.glob("*.parquet"))


def stream_parquet(paths: list[Path], *, limit: int | None = None) -> Iterator[dict]:
    import pyarrow.parquet as pq

    n = 0
    for p in paths:
        pf = pq.ParquetFile(str(p))
        for rg in range(pf.num_row_groups):
            cols = pf.read_row_group(rg).to_pydict()
            keys = list(cols.keys())
            if not keys:
                continue
            for i in range(len(cols[keys[0]])):
                yield {k: cols[k][i] for k in keys}
                n += 1
                if limit is not None and n >= limit:
                    return


def fetch_eval_dataset(repo: str, *, config: str = "default", split: str = "test", out_dir: str | Path) -> list[Path]:
    """Fetch a single eval dataset on demand using ``nanovlm.train.download``."""
    from nanovlm.train.download import head_size, list_parquet_shards
    import urllib.parse
    import urllib.request

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    try:
        urls = list_parquet_shards(repo, config, split)
    except RuntimeError:
        return []
    paths = []
    for url in urls[:1]:  # eval sets are typically a single shard
        stem = Path(urllib.parse.urlparse(url).path).name
        dest = out / stem
        if not dest.exists() or dest.stat().st_size == 0:
            with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
        paths.append(dest)
    return paths
