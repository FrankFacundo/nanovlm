"""JSONL metrics and compact local HTML report generation."""

from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Iterable


class MetricsLogger:
    def __init__(self, out_dir: str | Path, name: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / f"{name}.jsonl"

    def log(self, **metrics) -> None:
        metrics.setdefault("time", time.time())
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, sort_keys=True) + "\n")


def read_metrics(paths: Iterable[str | Path]) -> list[dict]:
    rows = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def _svg_line(rows: list[dict], key: str, width: int = 720, height: int = 180) -> str:
    pts = [(float(r.get("step", i)), float(r[key])) for i, r in enumerate(rows) if key in r and r[key] is not None]
    if len(pts) < 2:
        return f"<p>No data for {html.escape(key)}</p>"
    xs, ys = zip(*pts)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if ymin == ymax:
        ymax = ymin + 1.0
    coords = []
    for x, y in pts:
        sx = 32 + (x - xmin) / max(1e-9, xmax - xmin) * (width - 48)
        sy = 12 + (1 - (y - ymin) / (ymax - ymin)) * (height - 32)
        coords.append(f"{sx:.1f},{sy:.1f}")
    return (
        f"<h3>{html.escape(key)}</h3>"
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"
        f"<rect width='100%' height='100%' fill='white' stroke='#ddd'/>"
        f"<polyline fill='none' stroke='#2563eb' stroke-width='2' points='{' '.join(coords)}'/>"
        f"<text x='8' y='18' font-size='11'>{ymax:.4g}</text>"
        f"<text x='8' y='{height-8}' font-size='11'>{ymin:.4g}</text>"
        "</svg>"
    )


def write_html_report(out_dir: str | Path, title: str = "NanoVLM training report") -> Path:
    out_dir = Path(out_dir)
    rows = read_metrics(out_dir.glob("*.jsonl"))
    keys = [
        "train_loss", "val_loss", "lr", "tokens_per_sec", "grad_norm",
        "reward", "pass_at_1", "dpo_margin", "rl_kl", "tool_success",
    ]
    body = "\n".join(_svg_line(rows, k) for k in keys)
    path = out_dir / "report.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "<!doctype html><meta charset='utf-8'>"
            "<style>body{font-family:system-ui;margin:32px;max-width:900px}"
            "svg{max-width:100%;height:auto}code{background:#f4f4f5;padding:2px 4px}</style>"
            f"<h1>{html.escape(title)}</h1>{body}"
        )
    return path
