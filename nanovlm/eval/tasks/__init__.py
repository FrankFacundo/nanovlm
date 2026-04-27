"""Eval tasks. Importing this module registers them via ``ALL_TASKS``."""

from .ai2d import AI2D
from .arc import ARCChallenge, ARCEasy
from .chartqa import ChartQA
from .deepsearch_qa import DeepSearchQA
from .docvqa import DocVQA
from .gsm8k import GSM8K
from .hle_with_tools import HLEWithTools
from .humaneval import HumanEval
from .ifeval import IFEval
from .math import MATH
from .mmlu import MMLU
from .mmmu import MMMU
from .swe_multilingual import SWEMultilingual
from .vstar_bench import VStarBench
from .vstar_python import VStarPython

SIMPLE_TASKS = [MMLU, ARCEasy, ARCChallenge, GSM8K, MATH, HumanEval, IFEval, MMMU, ChartQA, DocVQA, AI2D, VStarBench]
HARD_TASKS = [DeepSearchQA, HLEWithTools, SWEMultilingual, VStarPython]
ALL_TASKS = SIMPLE_TASKS + HARD_TASKS

__all__ = [
    "ALL_TASKS",
    "SIMPLE_TASKS",
    "HARD_TASKS",
    "AI2D",
    "ARCChallenge",
    "ARCEasy",
    "ChartQA",
    "DeepSearchQA",
    "DocVQA",
    "GSM8K",
    "HLEWithTools",
    "HumanEval",
    "IFEval",
    "MATH",
    "MMLU",
    "MMMU",
    "SWEMultilingual",
    "VStarBench",
    "VStarPython",
]
