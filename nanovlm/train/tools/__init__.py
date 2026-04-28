"""Tool registry for agentic rollouts.

Each tool is a callable accepting a single ``dict`` of arguments and returning
a JSON-serializable result. Schemas are kept together with the implementations
in their respective modules. ``build_tool_registry`` returns a name → callable
map suitable for ``nanovlm.train.rollout.tool_use_rollout``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .browser import browse
from .filesystem import FileSystemTool
from .image_ops import image_op
from .python import PythonSandbox
from .web_search import web_search


def build_tool_registry(
    *,
    sandbox_root: str | Path | None = None,
    enable_web: bool = True,
    enable_python: bool = True,
    enable_filesystem: bool = True,
    enable_browser: bool = True,
    enable_image: bool = True,
) -> dict[str, Callable[[dict], object]]:
    registry: dict[str, Callable[[dict], object]] = {}
    if enable_python:
        sandbox = PythonSandbox(root=sandbox_root)
        registry["python"] = lambda args: sandbox(args)
    if enable_filesystem:
        fs = FileSystemTool(root=sandbox_root)
        registry["filesystem"] = lambda args: fs(args)
    if enable_web:
        registry["web_search"] = lambda args: web_search(**args)
    if enable_browser:
        registry["browser"] = lambda args: browse(**args)
    if enable_image:
        registry["image_ops"] = lambda args: image_op(**args)
    return registry


__all__ = [
    "build_tool_registry",
    "PythonSandbox",
    "FileSystemTool",
    "web_search",
    "browse",
    "image_op",
]
