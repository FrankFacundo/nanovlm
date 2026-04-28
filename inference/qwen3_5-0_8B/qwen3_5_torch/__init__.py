"""Compatibility shim. The model code lives in nanovlm.models.qwen3_5."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nanovlm.models.qwen3_5 import *  # noqa: F401,F403,E402
from nanovlm.models.qwen3_5 import __all__  # noqa: F401,E402
