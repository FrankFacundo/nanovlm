"""Regenerate HTML + Markdown reports from a run directory's JSONL metrics."""

from __future__ import annotations

import argparse

from nanovlm.train.common import default_base_dir
from nanovlm.train.report import write_html_report, write_markdown_report


def main() -> None:
    p = argparse.ArgumentParser(description="Regenerate HTML/Markdown reports for a run directory.")
    p.add_argument("--out-dir", default=str(default_base_dir() / "runs"))
    p.add_argument("--title", default="NanoVLM run")
    args = p.parse_args()
    print("html:", write_html_report(args.out_dir, args.title))
    print("md:  ", write_markdown_report(args.out_dir, args.title))


if __name__ == "__main__":
    main()
