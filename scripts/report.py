from __future__ import annotations

import argparse

from nanovlm.train.common import default_base_dir
from nanovlm.train.report import write_html_report


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a NanoVLM HTML report from JSONL metrics.")
    p.add_argument("--out-dir", default=str(default_base_dir() / "runs"))
    p.add_argument("--title", default="NanoVLM training report")
    args = p.parse_args()
    print(write_html_report(args.out_dir, args.title))


if __name__ == "__main__":
    main()
