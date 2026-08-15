"""Structured JSON serialization for assembly analysis reports."""

from __future__ import annotations

import json
from typing import List
from ..types import KernelAnalysisReport, KernelComparisonDiff


def export_reports_to_json(reports: List[KernelAnalysisReport], indent: int = 2) -> str:
    """Serialize a list of kernel analysis reports to formatted JSON."""
    data = [r.to_dict() for r in reports]
    return json.dumps(data, indent=indent)


def export_diff_to_json(diff: KernelComparisonDiff, indent: int = 2) -> str:
    """Serialize a kernel comparison diff to formatted JSON."""
    data = {
        "kernel_a": diff.kernel_a.to_dict(),
        "kernel_b": diff.kernel_b.to_dict(),
        "cycle_deltas": diff.cycle_deltas,
        "load_delta": diff.load_delta,
        "store_delta": diff.store_delta,
        "rmw_delta": diff.rmw_delta,
        "gpr_delta": diff.gpr_delta,
        "speedup_ratios": diff.speedup_ratios,
    }
    return json.dumps(data, indent=indent)
