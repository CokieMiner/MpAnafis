"""Benchmark selection, comparison policy, and reviewable execution plans."""

from __future__ import annotations

import fnmatch
import re

from .models import Benchmark, BenchmarkError


def select(catalog: list[Benchmark], patterns: list[str]) -> list[Benchmark]:
    """Select canonical full paths with shell-style patterns, never ambiguous suffix regexes."""
    selected = {}
    for pattern in patterns or ["*"]:
        matches = [item for item in catalog if fnmatch.fnmatchcase(item.path, pattern)]
        if not matches:
            raise BenchmarkError(f"benchmark selector matched nothing: {pattern!r}")
        selected.update((item.path, item) for item in matches)
    return sorted(selected.values(), key=lambda item: item.path)


def validate_catalog(catalog: list[Benchmark], *, require_comparison: bool = False) -> None:
    for item in catalog:
        parts = item.path.split("::")
        if len(parts) < 4 or parts[0] != "int" or parts[1] not in {"signed", "unsigned"}:
            raise BenchmarkError(f"benchmark is outside a numeric category: {item.path}")
        if "mp" not in item.engines:
            raise BenchmarkError(f"comparison without an Mp benchmark: {item.path}")
        if require_comparison and not item.comparisons:
            raise BenchmarkError(f"benchmark has no available comparator: {item.path}")
        if parts[-1] in {"predicates", "shl_policies", "checked_rounding", "to_size"}:
            raise BenchmarkError(f"split the combined benchmark into one method per case: {item.path}")


def benchmark_filter(path: str, engine: str, arguments: list[str]) -> str:
    # Divan matches the crate-qualified path. A group may have no arguments.
    prefix = "^public_api::" + re.escape(path + "::" + engine)
    if arguments:
        return prefix + "::(?:" + "|".join(re.escape(value) for value in arguments) + ")$"
    return prefix + r"(?:::[^:]+)?$"


def execution_plan(catalog: list[Benchmark], arguments: list[str], *, rounds: int,
                   compare: bool = True) -> list[dict]:
    """Plan paired A/B/B/A runs for each isolated function and argument selection."""
    if rounds < 1:
        raise BenchmarkError("rounds must be positive")
    plan = []
    for item in catalog:
        comparisons = item.comparisons if compare else ()
        peers = comparisons or (None,)
        for round_index in range(rounds):
            for peer_index, other in enumerate(peers):
                sequence = ("mp", other, other, "mp") if other else ("mp",)
                paired_round = round_index * len(peers) + peer_index
                for position, engine in enumerate(sequence):
                    plan.append({"path": item.path, "engine": engine,
                                 "round": paired_round, "position": position,
                                 "filter": benchmark_filter(item.path, engine, arguments)})
    return plan
