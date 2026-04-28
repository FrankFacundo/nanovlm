"""Bounded filesystem tool: read / write / list / grep, chrooted to ``root``."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path


class FileSystemTool:
    def __init__(self, *, root: str | Path | None = None, max_bytes: int = 1 << 20):
        self.root = Path(root).expanduser() if root else Path(tempfile.mkdtemp(prefix="nanovlm-fs-"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes

    def __call__(self, args: dict) -> dict:
        op = str(args.get("op") or "list").lower()
        path = self._safe(args.get("path") or "")
        if op == "read":
            if not path.exists():
                return {"ok": False, "error": "not found"}
            data = path.read_bytes()[: self.max_bytes]
            try:
                return {"ok": True, "text": data.decode("utf-8")}
            except UnicodeDecodeError:
                return {"ok": True, "bytes_b64": data.hex()}
        if op == "write":
            content = args.get("content") or ""
            if isinstance(content, bytes):
                if len(content) > self.max_bytes:
                    return {"ok": False, "error": "too large"}
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            else:
                content = str(content)
                if len(content.encode("utf-8")) > self.max_bytes:
                    return {"ok": False, "error": "too large"}
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            return {"ok": True, "bytes": path.stat().st_size}
        if op == "list":
            base = path if path.is_dir() else self.root
            entries = []
            for p in sorted(base.iterdir()):
                entries.append({"name": p.name, "is_dir": p.is_dir(), "bytes": p.stat().st_size if p.is_file() else None})
            return {"ok": True, "entries": entries}
        if op == "grep":
            pattern = re.compile(str(args.get("pattern") or ""))
            base = path if path.is_dir() else self.root
            hits = []
            for p in base.rglob("*"):
                if not p.is_file():
                    continue
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for i, line in enumerate(text.splitlines(), start=1):
                    if pattern.search(line):
                        hits.append({"path": str(p.relative_to(self.root)), "line": i, "text": line[:200]})
                        if len(hits) >= 200:
                            return {"ok": True, "hits": hits, "truncated": True}
            return {"ok": True, "hits": hits}
        return {"ok": False, "error": f"unknown op {op!r}"}

    def _safe(self, rel: str) -> Path:
        rel = str(rel)
        # Reject explicit parent-references; only allow paths inside the root.
        if Path(rel).is_absolute():
            raise ValueError(f"absolute path not allowed: {rel!r}")
        if any(part == ".." for part in Path(rel).parts):
            raise ValueError(f"parent reference not allowed: {rel!r}")
        target = (self.root / rel).resolve()
        if not str(target).startswith(str(self.root.resolve())):
            raise ValueError(f"path escapes sandbox: {rel!r}")
        return target
