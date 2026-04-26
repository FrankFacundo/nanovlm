"""Dataset downloader with byte budgeting and raw HF/HTTPS access only."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSource:
    name: str
    repo: str
    kind: str
    split_hint: str
    license: str
    weight: float
    allow_noncommercial: bool = False
    allow_openai_synthetic: bool = False


DEFAULT_SOURCES = [
    DatasetSource("fineweb_edu", "HuggingFaceFW/fineweb-edu", "text_pretrain", "sample/10BT", "ODC-BY", 0.28),
    DatasetSource("dclm", "mlfoundations/dclm-baseline-1.0", "text_pretrain", "global-shard_01_of_10", "CC-BY-4.0", 0.18),
    DatasetSource("fineweb2", "HuggingFaceFW/fineweb-2", "text_pretrain", "eng_Latn", "ODC-BY", 0.08),
    DatasetSource("megamath", "LLM360/MegaMath", "text_pretrain", "default", "ODC-BY", 0.08),
    DatasetSource("stack_v2", "bigcode/the-stack-v2-dedup", "code_pretrain", "data", "per-file", 0.08),
    DatasetSource("pixmo_cap", "allenai/pixmo-cap", "vlm_pretrain", "default", "ODC-BY", 0.12),
    DatasetSource("obelics", "HuggingFaceM4/OBELICS", "vlm_pretrain", "default", "CC-BY-4.0", 0.06),
    DatasetSource("docmatix", "HuggingFaceM4/Docmatix", "vlm_pretrain", "default", "MIT", 0.05),
    DatasetSource("oasst", "OpenAssistant/oasst1", "sft", "default", "Apache-2.0", 0.04),
    DatasetSource("aya", "CohereForAI/aya_dataset", "sft", "default", "Apache-2.0", 0.03),
]


def _request_json(url: str) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": "nanovlm/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def hf_tree(repo: str, revision: str = "main") -> list[dict]:
    quoted = urllib.parse.quote(repo, safe="")
    url = f"https://huggingface.co/api/datasets/{quoted}/tree/{revision}?recursive=1"
    data = _request_json(url)
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"HF API error for {repo}: {data['error']}")
    return list(data)


def hf_resolve_url(repo: str, path: str, revision: str = "main") -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, out_path: Path, max_bytes: int | None = None) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    start = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "nanovlm/0.1", "Range": f"bytes={start}-"})
    written = start
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "ab") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            if max_bytes is not None and written + len(chunk) > max_bytes:
                chunk = chunk[: max_bytes - written]
            if chunk:
                f.write(chunk)
                written += len(chunk)
            if max_bytes is not None and written >= max_bytes:
                break
    os.replace(tmp, out_path)
    return written


def select_files(repo: str, *, exts: tuple[str, ...] = (".jsonl", ".jsonl.gz", ".parquet"), limit: int = 32) -> list[dict]:
    files = []
    for item in hf_tree(repo):
        path = item.get("path", "")
        if item.get("type") == "file" and path.endswith(exts):
            files.append(item)
    files.sort(key=lambda x: x.get("size", 0))
    return files[:limit]


def build_download_plan(
    sources: list[DatasetSource],
    max_bytes: int,
    *,
    allow_noncommercial: bool = False,
    allow_openai_synthetic: bool = False,
) -> list[dict]:
    eligible = [
        s for s in sources
        if (allow_noncommercial or not s.allow_noncommercial)
        and (allow_openai_synthetic or not s.allow_openai_synthetic)
    ]
    weight_sum = sum(s.weight for s in eligible) or 1.0
    plan = []
    for source in eligible:
        budget = int(max_bytes * source.weight / weight_sum)
        plan.append({**asdict(source), "budget_bytes": budget})
    return plan


def download_sources(out_dir: str | Path, max_gb: float, dry_run: bool = False, **policy) -> Path:
    out_dir = Path(out_dir)
    max_bytes = int(max_gb * 1024 ** 3)
    manifest = {"max_bytes": max_bytes, "sources": [], "files": []}
    remaining = max_bytes
    for source in build_download_plan(DEFAULT_SOURCES, max_bytes, **policy):
        source_files = []
        try:
            candidates = select_files(source["repo"], limit=8)
        except Exception as e:
            manifest["sources"].append({**source, "error": str(e)})
            continue
        per_source_remaining = min(source["budget_bytes"], remaining)
        for item in candidates:
            if per_source_remaining <= 0 or remaining <= 0:
                break
            size = int(item.get("size") or 0)
            if size <= 0:
                continue
            take = min(size, per_source_remaining, remaining)
            rel = Path(source["name"]) / item["path"].replace("/", "__")
            out_path = out_dir / rel
            rec = {"source": source["name"], "repo": source["repo"], "path": item["path"], "bytes": take, "local": str(out_path)}
            if not dry_run:
                url = hf_resolve_url(source["repo"], item["path"])
                rec["downloaded_bytes"] = download_file(url, out_path, max_bytes=take)
                rec["sha256"] = _sha256(out_path)
            source_files.append(rec)
            manifest["files"].append(rec)
            per_source_remaining -= take
            remaining -= take
        manifest["sources"].append({**source, "selected_files": len(source_files)})
    manifest_path = out_dir / "manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path
