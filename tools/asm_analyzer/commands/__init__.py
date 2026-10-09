"""Command handlers for the asm_analyzer CLI."""

from __future__ import annotations

from .analyze import run_analyze
from .check import run_check
from .diff import run_diff
from .optimize import run_optimize
from .pmu import run_pmu
from .search import run_search
from .suggest import run_suggest
from .sweep import run_sweep

__all__ = [
    "run_analyze",
    "run_check",
    "run_diff",
    "run_optimize",
    "run_pmu",
    "run_search",
    "run_suggest",
    "run_sweep",
]
