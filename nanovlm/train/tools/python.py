"""Sandboxed Python execution tool.

Runs the supplied source string in a fresh ``python -I`` subprocess inside a
bounded temporary cwd. On Linux, ``unshare -n`` is attempted first to disable
networking; on macOS/other we fall back to subprocess without the namespace
isolation (the user is expected to run untrusted code only on Linux).

The tool exposes a single name ``python`` with arguments::

    {"code": "...", "files": {"name.txt": "contents"}, "timeout_s": 10}
"""

from __future__ import annotations

import os
import resource
import shutil
import subprocess
import tempfile
from pathlib import Path


class PythonSandbox:
    def __init__(
        self,
        *,
        root: str | Path | None = None,
        timeout_s: float = 30.0,
        memory_mb: int = 1024,
        max_output_chars: int = 16384,
    ):
        self.root = Path(root).expanduser() if root else None
        self.timeout_s = timeout_s
        self.memory_mb = memory_mb
        self.max_output_chars = max_output_chars

    def __call__(self, args: dict) -> dict:
        code = str(args.get("code", ""))
        files = args.get("files") or {}
        timeout_s = float(args.get("timeout_s") or self.timeout_s)
        cwd_ctx = self._make_cwd()
        try:
            cwd = Path(cwd_ctx.__enter__())
            for name, contents in files.items():
                self._write_inside(cwd, name, contents)
            (cwd / "main.py").write_text(code, encoding="utf-8")
            cmd = ["python", "-I", "main.py"]
            if _maybe_unshare():
                cmd = ["unshare", "-n", "--"] + cmd
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(cwd),
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    preexec_fn=lambda: self._set_limits(),
                    env={"PATH": os.environ.get("PATH", ""), "HOME": str(cwd), "PYTHONNOUSERSITE": "1"},
                )
                return {
                    "stdout": proc.stdout[: self.max_output_chars],
                    "stderr": proc.stderr[: self.max_output_chars],
                    "returncode": proc.returncode,
                    "timed_out": False,
                }
            except subprocess.TimeoutExpired as e:
                return {
                    "stdout": (e.stdout or "")[: self.max_output_chars],
                    "stderr": (e.stderr or "")[: self.max_output_chars],
                    "returncode": -1,
                    "timed_out": True,
                }
        finally:
            cwd_ctx.__exit__(None, None, None)

    def _make_cwd(self):
        if self.root is None:
            return tempfile.TemporaryDirectory(prefix="nanovlm-py-")
        self.root.mkdir(parents=True, exist_ok=True)
        return _ScopedDir(self.root)

    def _set_limits(self) -> None:
        try:
            cap = int(self.memory_mb) * (1 << 20)
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
            resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
            resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 26, 1 << 26))
        except (ValueError, OSError):
            pass

    def _write_inside(self, cwd: Path, name: str, contents) -> Path:
        target = (cwd / name).resolve()
        if not str(target).startswith(str(cwd.resolve())):
            raise ValueError(f"path escapes sandbox: {name!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(contents, bytes):
            target.write_bytes(contents)
        else:
            target.write_text(str(contents), encoding="utf-8")
        return target


class _ScopedDir:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = None

    def __enter__(self) -> Path:
        self.path = Path(tempfile.mkdtemp(dir=str(self.root), prefix="run-"))
        return self.path

    def __exit__(self, *_):
        if self.path and self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)


def _maybe_unshare() -> bool:
    return os.uname().sysname == "Linux" and shutil.which("unshare") is not None
