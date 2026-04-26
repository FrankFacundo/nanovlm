"""Stable package entrypoint for the pure-torch Qwen3.5 implementation.

The original inference port lives under ``inference/qwen3_5-0_8B``. That
directory name is intentionally preserved for compatibility, while this module
gives training code a normal import path:

    from nanovlm.models.qwen3_5 import Qwen3_5ForConditionalGeneration
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_QWEN_DIR = _ROOT / "inference" / "qwen3_5-0_8B"
if str(_QWEN_DIR) not in sys.path:
    sys.path.insert(0, str(_QWEN_DIR))

from qwen3_5_torch import *  # noqa: F401,F403
from qwen3_5_torch.tokenizer import Qwen2Tokenizer

try:
    from qwen3_5_torch.image_processor import Qwen2VLImageProcessor
except ModuleNotFoundError as _image_processor_error:  # pragma: no cover - depends on optional torchvision
    class Qwen2VLImageProcessor:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "Qwen2VLImageProcessor requires torchvision. Install training IO dependencies first."
            ) from _image_processor_error

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls(*args, **kwargs)

DEFAULT_MODEL_PATH = Path("/Users/frankfacundo/Models/Qwen/Qwen3.5-0.8B")

__all__ = [name for name in globals() if name.startswith("Qwen") or name in {
    "HybridCache",
    "load_qwen3_5_weights",
    "Qwen2Tokenizer",
    "Qwen2VLImageProcessor",
    "DEFAULT_MODEL_PATH",
}]
