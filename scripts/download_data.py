from __future__ import annotations

import argparse

from nanovlm.train.common import default_base_dir
from nanovlm.train.download import download_sources


def main() -> None:
    p = argparse.ArgumentParser(description="Download permissive NanoVLM data with a byte cap.")
    p.add_argument("--out-dir", default=str(default_base_dir() / "data"))
    p.add_argument("--max-download-gb", type=float, default=100.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-noncommercial", action="store_true")
    p.add_argument("--allow-openai-synthetic", action="store_true")
    args = p.parse_args()
    manifest = download_sources(
        args.out_dir,
        args.max_download_gb,
        dry_run=args.dry_run,
        allow_noncommercial=args.allow_noncommercial,
        allow_openai_synthetic=args.allow_openai_synthetic,
    )
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
