"""JSONL metrics, optional W&B, early stopping, HTML + Markdown reports."""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path

import torch

from .plots import bar_plot, line_plot


# ---------- JSONL metrics --------------------------------------------------

class MetricsLogger:
    def __init__(self, out_dir: str | Path, name: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / f"{name}.jsonl"

    def log(self, **metrics) -> None:
        metrics.setdefault("time", time.time())
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, sort_keys=True) + "\n")


# ---------- Optional W&B ---------------------------------------------------

class WandbLogger:
    def __init__(
        self,
        *,
        enabled: bool,
        project: str,
        run_name: str | None,
        config: dict,
        out_dir: str | Path,
        entity: str | None = None,
        mode: str = "online",
        master: bool = True,
    ):
        self.enabled = bool(enabled and master)
        self.run = None
        if not self.enabled:
            return
        try:
            import wandb
        except ImportError:
            self.enabled = False
            print("wandb not installed; skipping live logging.")
            return
        os.environ.setdefault("WANDB_DIR", str(Path(out_dir)))
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            mode=mode,
            config=config,
            dir=str(out_dir),
        )

    def log(self, metrics: dict) -> None:
        if not self.enabled or self.run is None:
            return
        self.run.log(metrics, step=int(metrics.get("step", 0)))

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def add_monitoring_args(parser: argparse.ArgumentParser, default_project: str = "nanovlm") -> None:
    parser.add_argument("--wandb", action="store_true", help="Enable W&B live logging.")
    parser.add_argument("--wandb-project", default=default_project)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--early-stop-metric", default="train_loss")
    parser.add_argument("--early-stop-mode", default="min", choices=["min", "max"])
    parser.add_argument("--early-stop-patience", type=int, default=0, help="0 disables.")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--max-loss", type=float, default=float("inf"))


# ---------- Early stopping -------------------------------------------------

class EarlyStopper:
    def __init__(self, *, metric: str, mode: str, patience: int, min_delta: float, max_loss: float):
        self.metric = metric
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.max_loss = max_loss
        self.best: float | None = None
        self.bad_steps = 0
        self.reason: str | None = None

    def check(self, metrics: dict) -> bool:
        loss = metrics.get("train_loss")
        if loss is not None and not _isfinite(loss):
            self.reason = "non-finite loss"
            return True
        if loss is not None and loss > self.max_loss:
            self.reason = f"loss {loss:.4g} > max {self.max_loss}"
            return True
        if self.patience <= 0:
            return False
        v = metrics.get(self.metric)
        if v is None or not _isfinite(v):
            return False
        better = (
            self.best is None
            or (self.mode == "min" and v < self.best - self.min_delta)
            or (self.mode == "max" and v > self.best + self.min_delta)
        )
        if better:
            self.best = v
            self.bad_steps = 0
        else:
            self.bad_steps += 1
            if self.bad_steps >= self.patience:
                self.reason = f"no {self.metric} improvement for {self.patience} checks"
                return True
        return False


def _isfinite(x):
    try:
        return float("-inf") < float(x) < float("inf")
    except (ValueError, TypeError):
        return False


# ---------- Reports --------------------------------------------------------

_STANDARD_CHARTS = [
    ("train_loss", "step", "loss"),
    ("val_loss", "step", "val loss"),
    ("val_bpb", "step", "bits/byte"),
    ("lr", "step", "lr"),
    ("grad_norm", "step", "grad norm"),
    ("tokens_per_sec", "step", "tokens/s"),
    ("mfu", "step", "MFU"),
    ("dpo_margin", "step", "DPO margin"),
    ("dpo_acc", "step", "DPO acc"),
    ("policy_loss", "step", "policy loss"),
    ("approx_kl", "step", "approx KL"),
    ("clip_frac", "step", "clip fraction"),
    ("reward", "step", "reward"),
    ("pass_at_1", "step", "pass@1"),
]


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _collect_runs(out_dir: Path) -> dict[str, list[dict]]:
    return {p.stem: _read_jsonl(p) for p in sorted(out_dir.glob("*.jsonl"))}


def write_html_report(out_dir: str | Path, title: str = "NanoVLM run") -> Path:
    out_dir = Path(out_dir)
    runs = _collect_runs(out_dir)
    body = [f"<h1>{html.escape(title)}</h1>", _system_html()]
    for run_name, rows in runs.items():
        if not rows:
            continue
        body.append(f"<h2>{html.escape(run_name)}</h2>")
        for metric, x_field, ylabel in _STANDARD_CHARTS:
            series = [(float(r.get(x_field, i)), float(r.get(metric))) for i, r in enumerate(rows) if r.get(metric) is not None]
            if not series:
                continue
            body.append(line_plot({metric: series}, title=metric, xlabel=x_field, ylabel=ylabel))
    out_path = out_dir / "report.html"
    out_path.write_text(_html_doc(title, "\n".join(body)))
    return out_path


def write_markdown_report(out_dir: str | Path, title: str = "NanoVLM run", run_metadata: dict | None = None) -> Path:
    """Generate a nanochat-style Markdown report card with embedded SVG plots."""
    out_dir = Path(out_dir)
    runs = _collect_runs(out_dir)
    parts = [f"# {title}", "", _system_md(run_metadata or {})]
    for run_name, rows in runs.items():
        if not rows:
            continue
        parts += ["", f"## {run_name}"]
        last = rows[-1]
        cols = sorted(k for k in last.keys() if k not in ("time",))
        parts.append("")
        parts.append("| metric | last value |")
        parts.append("|---|---:|")
        for k in cols:
            parts.append(f"| {k} | {_fmt(last[k])} |")
        parts.append("")
        for metric, x_field, ylabel in _STANDARD_CHARTS:
            series = [(float(r.get(x_field, i)), float(r.get(metric))) for i, r in enumerate(rows) if r.get(metric) is not None]
            if not series:
                continue
            svg = line_plot({metric: series}, title=metric, xlabel=x_field, ylabel=ylabel, width=600, height=220)
            data_uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")
            parts.append(f"![{metric}]({data_uri})")
            parts.append("")
    out_path = out_dir / "report.md"
    out_path.write_text("\n".join(parts))
    return out_path


def write_eval_report(out_dir: str | Path, scores: dict[str, float], title: str = "NanoVLM eval") -> Path:
    """Single-page benchmark report: bar chart + table."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    svg = bar_plot(scores, title=title)
    parts = [f"# {title}", ""]
    parts.append("| benchmark | score |")
    parts.append("|---|---:|")
    for k, v in scores.items():
        parts.append(f"| {k} | {_fmt(v)} |")
    parts.append("")
    data_uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")
    parts.append(f"![scores]({data_uri})")
    out_path = out_dir / "eval_report.md"
    out_path.write_text("\n".join(parts))
    (out_dir / "eval_report.html").write_text(_html_doc(title, svg + _scores_table_html(scores)))
    return out_path


def _scores_table_html(scores: dict[str, float]) -> str:
    rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{_fmt(v)}</td></tr>" for k, v in scores.items())
    return f"<table border=1 cellpadding=4><tr><th>benchmark</th><th>score</th></tr>{rows}</table>"


def _fmt(v) -> str:
    try:
        f = float(v)
        return f"{f:.4g}"
    except (ValueError, TypeError):
        return html.escape(str(v))


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()[:12]
    except (subprocess.SubprocessError, FileNotFoundError):
        return "n/a"


def _system_md(extra: dict) -> str:
    lines = [
        "## System",
        "",
        f"- host: {socket.gethostname()}",
        f"- platform: {platform.platform()}",
        f"- python: {platform.python_version()}",
        f"- torch: {torch.__version__}",
        f"- cuda available: {torch.cuda.is_available()}",
        f"- mps available: {bool(torch.backends.mps.is_available())}",
        f"- git: {_git_commit()}",
    ]
    for k, v in extra.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines) + "\n"


def _system_html() -> str:
    return (
        "<details><summary>System</summary><pre>"
        + html.escape(_system_md({}))
        + "</pre></details>"
    )


def _html_doc(title: str, body: str) -> str:
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:2em auto;padding:0 1em;color:#222} svg{display:block;margin:1em 0} table{border-collapse:collapse} td,th{padding:4px 8px} h2{border-bottom:1px solid #ddd;padding-bottom:4px}</style>"
        f"</head><body>{body}</body></html>"
    )
