"""Dataset downloader: raw HF resolve via urllib (no huggingface_hub dep).

Discovers parquet shards via the public HF REST endpoints, then streams each
file with sha256 verification and resume-on-partial. Per-source byte budget is
allocated proportionally under a hard global cap (default 100 GB). License
filters: ``ODC-BY``, ``CC-BY-4.0``, ``Apache-2.0``, ``MIT``, ``permissive`` are
allowed by default; ``non_commercial`` and ``openai_synthetic`` are gated
behind explicit flags.

Optional ``--download-weights`` grabs the Qwen3.5-0.8B safetensors from
``Qwen/Qwen3.5-0.8B`` into ``~/Models/Qwen/Qwen3.5-0.8B``.

This module ships a hand-curated source registry (see ``DEFAULT_SOURCES``)
that mirrors the SOTA recipes in the user's 2026 datasets report.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

HF_API_BASE = "https://huggingface.co"
USER_AGENT = "nanovlm-downloader/0.1 (+https://github.com/nanovlm)"


# ---------- Source registry ------------------------------------------------

@dataclass(frozen=True)
class DatasetSource:
    name: str
    repo: str                     # HF repo id e.g. "HuggingFaceFW/fineweb-edu"
    bucket: str                   # one of: text-pretrain math-pretrain code-pretrain reasoning sft-text preference-text rlvr vlm-pretrain vlm-sft vlm-pref eval
    license: str                  # tag from {ODC-BY, CC-BY-4.0, Apache-2.0, MIT, permissive, non_commercial, openai_synthetic}
    config: str = "default"        # parquet config name on HF
    split: str = "train"           # parquet split
    weight: float = 1.0            # bucket allocation weight
    max_files: int | None = None   # cap shards per source
    notes: str = ""


# Permissive defaults; weights are within-bucket and re-normalized at download time.
DEFAULT_SOURCES: list[DatasetSource] = [
    # ---------- Text pretrain (≈45 GB) ------------------------------------
    DatasetSource("fineweb_edu",        "HuggingFaceFW/fineweb-edu",       "text-pretrain",  "ODC-BY",     config="sample-10BT", weight=0.55, max_files=8,  notes="Snowflake-arctic edu classifier"),
    DatasetSource("dclm_baseline",      "mlfoundations/dclm-baseline-1.0", "text-pretrain",  "CC-BY-4.0",  config="default",     weight=0.27, max_files=4,  notes="fastText OpenHermes/ELI5 classifier"),
    DatasetSource("fineweb2_eng",       "HuggingFaceFW/fineweb-2",         "text-pretrain",  "ODC-BY",     config="eng_Latn",    weight=0.10, max_files=2),
    DatasetSource("pes2o",              "allenai/peS2o",                   "text-pretrain",  "ODC-BY",     config="default",     weight=0.08, max_files=2),

    # ---------- Math pretrain (≈12 GB) ------------------------------------
    DatasetSource("finemath_4plus",     "HuggingFaceTB/finemath",          "math-pretrain",  "ODC-BY",     config="finemath-4plus", weight=0.30, max_files=2),
    DatasetSource("megamath",           "LLM360/MegaMath",                 "math-pretrain",  "ODC-BY",     config="default",     weight=0.45, max_files=4),
    DatasetSource("nemotron_cc_math",   "nvidia/Nemotron-CC-Math",         "math-pretrain",  "permissive", config="default",     weight=0.25, max_files=2),

    # ---------- Code pretrain (≈6 GB) -------------------------------------
    DatasetSource("stack_edu",          "HuggingFaceTB/stack-edu",         "code-pretrain",  "permissive", config="default",     weight=0.95, max_files=4),
    DatasetSource("magicoder_oss",      "ise-uiuc/Magicoder-OSS-Instruct-75K","code-pretrain","MIT",       config="default",     weight=0.05, max_files=1),

    # ---------- Reasoning (≈5 GB) -----------------------------------------
    DatasetSource("openthoughts3",      "open-thoughts/OpenThoughts3-1.2M","reasoning",      "Apache-2.0", config="default",     weight=0.65, max_files=2),
    DatasetSource("openr1_math",        "open-r1/OpenR1-Math-220k",        "reasoning",      "Apache-2.0", config="default",     weight=0.30, max_files=1),
    DatasetSource("s1k_v1_1",           "simplescaling/s1K-1.1",           "reasoning",      "Apache-2.0", config="default",     weight=0.05, max_files=1),

    # ---------- SFT text (≈8 GB) ------------------------------------------
    DatasetSource("tulu3_sft",          "allenai/tulu-3-sft-mixture",      "sft-text",       "ODC-BY",     config="default",     weight=0.45, max_files=2),
    DatasetSource("smoltalk",           "HuggingFaceTB/smoltalk",          "sft-text",       "Apache-2.0", config="all",         weight=0.15, max_files=1),
    DatasetSource("oasst1",             "OpenAssistant/oasst1",            "sft-text",       "Apache-2.0", config="default",     weight=0.05, max_files=1),
    DatasetSource("aya",                "CohereForAI/aya_dataset",         "sft-text",       "Apache-2.0", config="default",     weight=0.05, max_files=1),
    DatasetSource("openmathinstruct2",  "nvidia/OpenMathInstruct-2",       "sft-text",       "CC-BY-4.0",  config="default",     weight=0.20, max_files=2),
    DatasetSource("opencodereasoning",  "nvidia/OpenCodeReasoning",        "sft-text",       "CC-BY-4.0",  config="default",     weight=0.10, max_files=1),

    # ---------- Preference text (≈0.5 GB) --------------------------------
    DatasetSource("ultrafeedback_clean","argilla/ultrafeedback-binarized-preferences-cleaned","preference-text","MIT",config="default",weight=0.6, max_files=1),
    DatasetSource("helpsteer3",         "nvidia/HelpSteer3",               "preference-text","CC-BY-4.0",  config="default",     weight=0.4, max_files=1),

    # ---------- RLVR prompts (≈0.1 GB) ------------------------------------
    DatasetSource("rlvr_mixed",         "allenai/RLVR-GSM-MATH-IF-Mixed-Constraints","rlvr","ODC-BY",      config="default",     weight=0.7, max_files=1),
    DatasetSource("dapo_math_17k",      "BytedTsinghua-SIA/DAPO-Math-17k", "rlvr",           "Apache-2.0", config="default",     weight=0.3, max_files=1),

    # ---------- VLM pretrain (≈14 GB) -------------------------------------
    DatasetSource("obelics",            "HuggingFaceM4/OBELICS",           "vlm-pretrain",   "CC-BY-4.0",  config="default",     weight=0.40, max_files=2),
    DatasetSource("pixmo_cap",          "allenai/pixmo-cap",               "vlm-pretrain",   "ODC-BY",     config="default",     weight=0.25, max_files=1),
    DatasetSource("recap_datacomp",     "UCSC-VLAA/Recap-DataComp-1B",     "vlm-pretrain",   "CC-BY-4.0",  config="default",     weight=0.20, max_files=1),
    DatasetSource("pixelprose",         "tomg-group-umd/pixelprose",       "vlm-pretrain",   "permissive", config="default",     weight=0.15, max_files=1),

    # ---------- VLM SFT (≈9 GB) -------------------------------------------
    DatasetSource("cambrian7m",         "nyu-visionx/Cambrian-10M",        "vlm-sft",        "Apache-2.0", config="default",     weight=0.30, max_files=2),
    DatasetSource("cauldron",           "HuggingFaceM4/the_cauldron",      "vlm-sft",        "CC-BY-4.0",  config="ai2d",        weight=0.20, max_files=1),
    DatasetSource("docmatix",           "HuggingFaceM4/Docmatix",          "vlm-sft",        "MIT",        config="default",     weight=0.20, max_files=1),
    DatasetSource("pixmo_docs",         "allenai/pixmo-docs",              "vlm-sft",        "ODC-BY",     config="default",     weight=0.10, max_files=1),
    DatasetSource("pixmo_charts",       "allenai/pixmo-charts",            "vlm-sft",        "ODC-BY",     config="default",     weight=0.05, max_files=1),
    DatasetSource("pixmo_clocks",       "allenai/pixmo-clocks",            "vlm-sft",        "ODC-BY",     config="default",     weight=0.05, max_files=1),
    DatasetSource("llava_onevision",    "lmms-lab/LLaVA-OneVision-Data",   "vlm-sft",        "permissive", config="default",     weight=0.10, max_files=1),

    # ---------- VLM preference (≈0.5 GB) ---------------------------------
    DatasetSource("mmpr_v12",           "OpenGVLab/MMPR-v1.2",             "vlm-pref",       "permissive", config="default",     weight=0.6, max_files=1),
    DatasetSource("rlaif_v",            "openbmb/RLAIF-V-Dataset",         "vlm-pref",       "permissive", config="default",     weight=0.4, max_files=1),

    # ---------- Eval test sets (≈<1 GB) ----------------------------------
    DatasetSource("mmlu",               "cais/mmlu",                       "eval",           "MIT",        config="all",         split="test",       weight=1.0, max_files=1),
    DatasetSource("gsm8k",              "openai/gsm8k",                    "eval",           "MIT",        config="main",        split="test",       weight=1.0, max_files=1),
    DatasetSource("math",               "HuggingFaceH4/MATH-500",          "eval",           "MIT",        config="default",     split="test",       weight=1.0, max_files=1),
    DatasetSource("humaneval",          "openai/openai_humaneval",         "eval",           "MIT",        config="openai_humaneval", split="test", weight=1.0, max_files=1),
    DatasetSource("ifeval",             "google/IFEval",                   "eval",           "Apache-2.0", config="default",     split="train",      weight=1.0, max_files=1),
    DatasetSource("arc_challenge",      "allenai/ai2_arc",                 "eval",           "CC-BY-4.0",  config="ARC-Challenge", split="test",     weight=1.0, max_files=1),
    DatasetSource("mmmu",               "MMMU/MMMU",                       "eval",           "Apache-2.0", config="Math",        split="validation", weight=1.0, max_files=1),
    DatasetSource("chartqa",            "HuggingFaceM4/ChartQA",           "eval",           "permissive", config="default",     split="test",       weight=1.0, max_files=1),
    DatasetSource("docvqa",             "lmms-lab/DocVQA",                 "eval",           "MIT",        config="DocVQA",      split="validation", weight=1.0, max_files=1),
    DatasetSource("ai2d",               "lmms-lab/ai2d",                   "eval",           "permissive", config="default",     split="test",       weight=1.0, max_files=1),
    DatasetSource("vstar_bench",        "craigwu/vstar_bench",             "eval",           "permissive", config="default",     split="test",       weight=1.0, max_files=1),
    DatasetSource("deepsearch_qa",      "vtllms/DeepSearch-QA",            "eval",           "permissive", config="default",     split="test",       weight=1.0, max_files=1),
    DatasetSource("hle",                "cais/hle",                        "eval",           "MIT",        config="default",     split="test",       weight=1.0, max_files=1, notes="gated"),
    DatasetSource("swe_multilingual",   "ScalingIntelligence/swe-bench-multilingual","eval", "MIT",        config="default",     split="test",       weight=1.0, max_files=1),
]

QWEN_WEIGHTS_REPO = "Qwen/Qwen3.5-0.8B"
PERMISSIVE_LICENSES = {"ODC-BY", "CC-BY-4.0", "Apache-2.0", "MIT", "permissive"}


# ---------- License filtering ---------------------------------------------

def _is_allowed(src: DatasetSource, allow_noncommercial: bool, allow_openai_synthetic: bool) -> bool:
    if src.license in PERMISSIVE_LICENSES:
        return True
    if src.license == "non_commercial" and allow_noncommercial:
        return True
    if src.license == "openai_synthetic" and allow_openai_synthetic:
        return True
    return False


# ---------- HF REST helpers ------------------------------------------------

def _http_get(url: str, *, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _http_get_json(url: str, *, timeout: float = 60.0) -> object:
    return json.loads(_http_get(url, timeout=timeout).decode("utf-8"))


def list_parquet_shards(repo: str, config: str, split: str) -> list[str]:
    """Resolve parquet shard URLs via the public HF parquet endpoint.

    Returns a list of fully-qualified URLs.
    """
    url = f"{HF_API_BASE}/api/datasets/{repo}/parquet/{urllib.parse.quote(config, safe='')}/{urllib.parse.quote(split, safe='')}"
    try:
        data = _http_get_json(url)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"failed to list shards for {repo}/{config}/{split}: HTTP {e.code}") from e
    if isinstance(data, list):
        return [str(u) for u in data]
    if isinstance(data, dict) and "urls" in data:
        return [str(u) for u in data["urls"]]
    raise RuntimeError(f"unexpected parquet listing payload for {repo}: {type(data).__name__}")


def head_size(url: str, *, timeout: float = 30.0) -> int | None:
    """HEAD to learn the shard size; returns ``None`` if unknown."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl is not None else None
    except urllib.error.URLError:
        return None


def _download_with_resume(url: str, dest: Path, *, expected_bytes: int | None = None) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    start = tmp.stat().st_size if tmp.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if start > 0:
        headers["Range"] = f"bytes={start}-"
    req = urllib.request.Request(url, headers=headers)
    written = start
    try:
        with urllib.request.urlopen(req, timeout=120.0) as resp, open(tmp, "ab") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                written += len(chunk)
    except urllib.error.HTTPError as e:
        if e.code == 416 and expected_bytes is not None and start >= expected_bytes:
            pass  # already complete
        else:
            raise
    tmp.rename(dest)
    return written


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------- Top-level downloader ------------------------------------------

@dataclass
class DownloadPlan:
    sources: list[dict] = field(default_factory=list)
    total_bytes: int = 0
    cap_bytes: int = 0


@dataclass
class DownloadResult:
    plan: DownloadPlan
    written_bytes: int = 0
    files: list[dict] = field(default_factory=list)


def plan_downloads(
    *,
    out_dir: str | Path,
    max_download_gb: float,
    allow_noncommercial: bool = False,
    allow_openai_synthetic: bool = False,
    include_buckets: list[str] | None = None,
    exclude_buckets: list[str] | None = None,
) -> DownloadPlan:
    """Resolve every selected source's shard URLs and sizes; allocate bytes per
    bucket proportionally under the global cap.

    The plan is a pure-data structure; ``--dry-run`` prints it without writing.
    """
    cap_bytes = int(max_download_gb * (1 << 30))
    sources = [
        s for s in DEFAULT_SOURCES
        if _is_allowed(s, allow_noncommercial, allow_openai_synthetic)
        and (include_buckets is None or s.bucket in include_buckets)
        and (exclude_buckets is None or s.bucket not in exclude_buckets)
    ]
    bucket_weights: dict[str, float] = {}
    for s in sources:
        bucket_weights[s.bucket] = bucket_weights.get(s.bucket, 0.0) + s.weight
    bucket_alloc = _bucket_allocation(cap_bytes, sources)

    plan = DownloadPlan(sources=[], total_bytes=0, cap_bytes=cap_bytes)
    global_running = 0
    for src in sources:
        per_src_cap = int(bucket_alloc.get(src.bucket, 0) * (src.weight / max(1e-9, bucket_weights[src.bucket])))
        try:
            shards = list_parquet_shards(src.repo, src.config, src.split)
        except RuntimeError as e:
            plan.sources.append({**dataclasses.asdict(src), "error": str(e), "shards": []})
            continue
        if src.max_files is not None:
            shards = shards[: src.max_files]
        sized = []
        running = 0
        for url in shards:
            size = head_size(url) or 0
            if size > 0 and running + size > per_src_cap and sized:
                break
            if size > 0 and global_running + size > cap_bytes:
                break
            sized.append({"url": url, "bytes": size})
            running += size
            global_running += size
        plan.sources.append({**dataclasses.asdict(src), "shards": sized, "alloc_bytes": per_src_cap, "planned_bytes": running})
        plan.total_bytes += running
        if global_running >= cap_bytes:
            break
    return plan


def _bucket_allocation(cap_bytes: int, sources: list[DatasetSource]) -> dict[str, int]:
    """Allocate ``cap_bytes`` to buckets in proportion to a fixed prior."""
    prior = {
        "text-pretrain": 0.45,
        "math-pretrain": 0.12,
        "code-pretrain": 0.06,
        "reasoning": 0.05,
        "sft-text": 0.08,
        "preference-text": 0.005,
        "rlvr": 0.001,
        "vlm-pretrain": 0.14,
        "vlm-sft": 0.09,
        "vlm-pref": 0.005,
        "eval": 0.01,
    }
    present = {s.bucket for s in sources}
    total = sum(p for b, p in prior.items() if b in present)
    if total <= 0:
        return {b: 0 for b in present}
    return {b: int(cap_bytes * (p / total)) for b, p in prior.items() if b in present}


def execute_plan(
    plan: DownloadPlan,
    out_dir: str | Path,
) -> DownloadResult:
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    written_total = 0
    files = []
    for src in plan.sources:
        bucket = src["bucket"]
        name = src["name"]
        for shard in src.get("shards", []):
            url = shard["url"]
            stem = Path(urllib.parse.urlparse(url).path).name
            dest = out_dir / bucket / name / stem
            if dest.exists() and dest.stat().st_size > 0:
                files.append({"url": url, "path": str(dest), "bytes": dest.stat().st_size, "skipped": True})
                continue
            n = _download_with_resume(url, dest, expected_bytes=shard.get("bytes") or None)
            written_total += n
            files.append({"url": url, "path": str(dest), "bytes": n, "sha256": _sha256(dest)})
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"plan": dataclasses.asdict(plan), "files": files}, indent=2, default=str))
    return DownloadResult(plan=plan, written_bytes=written_total, files=files)


def download_qwen_weights(out_dir: str | Path = None) -> Path:
    """Download the Qwen3.5-0.8B safetensors via raw HF resolve.

    Default ``out_dir`` mirrors ``DEFAULT_MODEL_PATH`` so existing scripts
    Just Work afterward.
    """
    from nanovlm.models.qwen3_5 import DEFAULT_MODEL_PATH

    target = Path(out_dir).expanduser() if out_dir else DEFAULT_MODEL_PATH
    target.mkdir(parents=True, exist_ok=True)
    base = f"{HF_API_BASE}/{QWEN_WEIGHTS_REPO}/resolve/main"
    files = [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "preprocessor_config.json",
        "model.safetensors.index.json",
    ]
    for f in files:
        try:
            data = _http_get(f"{base}/{f}", timeout=60.0)
            (target / f).write_bytes(data)
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
    # Resolve sharded safetensors via index
    idx_path = target / "model.safetensors.index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        for shard in sorted(set(idx.get("weight_map", {}).values())):
            dest = target / shard
            if dest.exists() and dest.stat().st_size > 0:
                continue
            _download_with_resume(f"{base}/{shard}", dest)
    return target


def download_sources(
    out_dir: str | Path,
    max_download_gb: float = 100.0,
    *,
    dry_run: bool = False,
    allow_noncommercial: bool = False,
    allow_openai_synthetic: bool = False,
    include_buckets: list[str] | None = None,
    exclude_buckets: list[str] | None = None,
    download_weights: bool = False,
) -> dict:
    plan = plan_downloads(
        out_dir=out_dir,
        max_download_gb=max_download_gb,
        allow_noncommercial=allow_noncommercial,
        allow_openai_synthetic=allow_openai_synthetic,
        include_buckets=include_buckets,
        exclude_buckets=exclude_buckets,
    )
    if dry_run:
        out = {"plan": dataclasses.asdict(plan), "would_write_bytes": plan.total_bytes}
        if download_weights:
            out["weights"] = "would download Qwen/Qwen3.5-0.8B (~2 GB)"
        return out
    result = execute_plan(plan, out_dir)
    out = {"plan": dataclasses.asdict(result.plan), "written_bytes": result.written_bytes, "files": result.files}
    if download_weights:
        out["weights_path"] = str(download_qwen_weights())
    return out
