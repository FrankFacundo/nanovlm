"""CLI for the dataset downloader. See nanovlm.train.download for details."""

from __future__ import annotations

import argparse
import json

from nanovlm.train.common import default_base_dir
from nanovlm.train.download import download_sources


def main() -> None:
    p = argparse.ArgumentParser(description="Download permissive NanoVLM training data.")
    p.add_argument("--out-dir", default=str(default_base_dir() / "data"))
    p.add_argument("--max-download-gb", type=float, default=100.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-noncommercial", action="store_true")
    p.add_argument("--allow-openai-synthetic", action="store_true")
    p.add_argument("--include", nargs="+", default=None,
                   help="Bucket names to include. Default: all permissive buckets.")
    p.add_argument("--exclude", nargs="+", default=None)
    p.add_argument("--download-weights", action="store_true",
                   help="Also fetch Qwen3.5-0.8B safetensors into ~/Models/Qwen/Qwen3.5-0.8B/.")
    args = p.parse_args()

    out = download_sources(
        args.out_dir,
        args.max_download_gb,
        dry_run=args.dry_run,
        allow_noncommercial=args.allow_noncommercial,
        allow_openai_synthetic=args.allow_openai_synthetic,
        include_buckets=args.include,
        exclude_buckets=args.exclude,
        download_weights=args.download_weights,
    )
    print(json.dumps({k: v for k, v in out.items() if k != "files"}, indent=2, default=str))
    if not args.dry_run:
        files = out.get("files", [])
        total = sum(int(f.get("bytes") or 0) for f in files)
        print(f"wrote {len(files)} files, {total/1e9:.2f} GB total")


if __name__ == "__main__":
    main()
