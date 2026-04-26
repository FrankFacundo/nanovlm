"""Deterministic local sandbox tools for agentic training."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ToolResult:
    tool: str
    command: str
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float


@dataclass
class TrajectoryLog:
    path: Path
    events: list[dict] = field(default_factory=list)

    def append(self, event: dict) -> None:
        self.events.append(event)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")


class LocalSandbox:
    """A small deterministic sandbox over a temporary directory.

    It is intentionally conservative: no network helper, explicit cwd, timeout
    on every process, and every state-changing action is logged for replay.
    """

    def __init__(self, root: str | Path | None = None, log_path: str | Path | None = None, timeout_s: float = 10.0):
        self._tmp = None
        if root is None:
            self._tmp = tempfile.TemporaryDirectory()
            root = self._tmp.name
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.log = TrajectoryLog(Path(log_path) if log_path else self.root / "trajectory.jsonl")

    def close(self):
        if self._tmp is not None:
            self._tmp.cleanup()

    def write_file(self, relpath: str, content: str) -> ToolResult:
        t0 = time.time()
        path = (self.root / relpath).resolve()
        if self.root.resolve() not in path.parents and path != self.root.resolve():
            raise ValueError("write_file path escapes sandbox")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        result = ToolResult("write_file", relpath, 0, str(path), "", time.time() - t0)
        self.log.append(result.__dict__)
        return result

    def run(self, command: list[str], *, tool: str = "bash") -> ToolResult:
        t0 = time.time()
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        try:
            proc = subprocess.run(
                command,
                cwd=self.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_s,
                env=env,
            )
            result = ToolResult(tool, " ".join(command), proc.returncode, proc.stdout, proc.stderr, time.time() - t0)
        except subprocess.TimeoutExpired as e:
            result = ToolResult(tool, " ".join(command), 124, e.stdout or "", e.stderr or "timeout", time.time() - t0)
        self.log.append(result.__dict__)
        return result

    def run_python(self, code: str) -> ToolResult:
        self.write_file("main.py", code)
        return self.run(["python", "main.py"], tool="python")


def tool_success_reward(results: list[ToolResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.returncode == 0) / len(results)
