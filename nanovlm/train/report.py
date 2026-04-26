"""JSONL metrics and compact local HTML report generation."""

from __future__ import annotations

import html
import json
import math
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


class WandbLogger:
    """Optional Weights & Biases logger.

    The import is intentionally lazy so the training stack still works without
    wandb installed unless ``--wandb`` is passed.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        project: str,
        run_name: str | None,
        config: dict,
        out_dir: str | Path,
        entity: str | None = None,
        mode: str | None = None,
        master: bool = True,
    ):
        self.run = None
        if not enabled or not master or mode == "disabled":
            return
        try:
            import wandb
        except ImportError as e:
            raise RuntimeError("wandb is not installed. Install requirements.txt or omit --wandb.") from e
        kwargs = {
            "project": project,
            "name": run_name,
            "config": config,
            "dir": str(Path(out_dir).expanduser()),
        }
        if entity:
            kwargs["entity"] = entity
        if mode:
            kwargs["mode"] = mode
        self.run = wandb.init(**kwargs)

    def log(self, metrics: dict) -> None:
        if self.run is not None:
            self.run.log(metrics, step=int(metrics["step"]) if "step" in metrics else None)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


class EarlyStopper:
    """Optional metric-based early stopping for smoke and debug runs."""

    def __init__(
        self,
        *,
        metric: str = "train_loss",
        patience: int = -1,
        min_delta: float = 0.0,
        max_loss: float | None = None,
        mode: str = "min",
    ):
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self.max_loss = max_loss
        if mode not in {"min", "max"}:
            raise ValueError("early-stop mode must be 'min' or 'max'")
        self.mode = mode
        self.best = math.inf if mode == "min" else -math.inf
        self.bad_steps = 0
        self.reason: str | None = None

    def check(self, metrics: dict) -> bool:
        value = metrics.get(self.metric)
        if value is None:
            return False
        value = float(value)
        if not math.isfinite(value):
            self.reason = f"{self.metric} is non-finite: {value}"
            return True
        if self.max_loss is not None and value > self.max_loss:
            self.reason = f"{self.metric}={value:.6g} exceeded --max-loss={self.max_loss:.6g}"
            return True
        if self.patience < 0:
            return False
        improved = (
            value < self.best - self.min_delta
            if self.mode == "min"
            else value > self.best + self.min_delta
        )
        if improved:
            self.best = value
            self.bad_steps = 0
        else:
            self.bad_steps += 1
        if self.bad_steps >= self.patience:
            self.reason = (
                f"{self.metric} did not improve by {self.min_delta:g} for "
                f"{self.patience} logged checks"
            )
            return True
        return False


def add_monitoring_args(parser, *, default_project: str) -> None:
    parser.add_argument("--wandb", action="store_true", help="log metrics to Weights & Biases")
    parser.add_argument("--wandb-project", default=default_project)
    parser.add_argument("--wandb-run", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--early-stop-metric", default="train_loss")
    parser.add_argument("--early-stop-mode", choices=["min", "max"], default="min")
    parser.add_argument("--early-stop-patience", type=int, default=-1, help="-1 disables metric patience stopping")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--max-loss", type=float, default=None, help="stop if early-stop metric exceeds this value")


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
