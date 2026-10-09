"""Parse Divan's tree output without losing enclosing function or scenario paths."""

from __future__ import annotations

import math
import re

from .models import ENGINES, SUITES, Benchmark, BenchmarkError, Measurement

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TREE = re.compile(r"^([ │]*)(?:├─|╰─) (.*)$")
DURATION = re.compile(r"^(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(ps|ns|[µμu]s|ms|s)$")
SCALE = {"ps": 0.001, "ns": 1.0, "us": 1_000.0, "µs": 1_000.0,
         "μs": 1_000.0, "ms": 1_000_000.0, "s": 1_000_000_000.0}


def duration_ns(value: str) -> float:
    match = DURATION.fullmatch(value.strip())
    if not match:
        raise BenchmarkError(f"invalid Divan duration: {value!r}")
    result = float(match[1]) * SCALE[match[2]]
    if not math.isfinite(result):
        raise BenchmarkError(f"non-finite Divan duration: {value!r}")
    return result


def tree_rows(text: str, *, suite: str = "public_api"):
    """Yield full node paths and optional measurement columns; reject broken trees."""
    stack: list[str] = []
    if suite not in SUITES:
        raise BenchmarkError(f"unknown benchmark suite: {suite}")
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = ANSI.sub("", raw).rstrip()
        match = TREE.match(line)
        if not match:
            # Repeated captures of one suite are valid; mixed suites need
            # separate imports so that source identity is never inferred.
            first = line.split(maxsplit=1)[:1]
            if first and first[0] in SUITES and first[0] != suite:
                raise BenchmarkError(f"unexpected suite {first[0]!r}; selected {suite!r}")
            if first == [suite]:
                stack = [suite]
            continue
        if not stack:
            raise BenchmarkError(f"Divan tree has no {suite} root at line {line_number}")
        indent = len(match[1])
        depth = indent // 3 + 1
        if indent % 3 or depth > len(stack):
            raise BenchmarkError(f"invalid Divan tree indentation at line {line_number}")
        columns = [item.strip() for item in match[2].split("│")]
        # The first timing column shares a cell with the node label.
        first = re.fullmatch(r"(.+?)\s{2,}(\d.*)", columns[0])
        values = None
        if first and len(columns) >= 6 and DURATION.fullmatch(first[2].strip()):
            label = first[1].strip()
            values = [first[2].strip(), *columns[1:6]]
        else:
            label = columns[0].strip()
        stack[depth:] = [label]
        yield tuple(stack[1:]), values, line_number


def parse_catalog(text: str) -> list[Benchmark]:
    groups: dict[str, set[str]] = {}
    for path, _values, _line in tree_rows(text):
        if path[-1] in ENGINES:
            groups.setdefault("::".join(path[:-1]), set()).add(path[-1])
    if not groups:
        raise BenchmarkError("no public API benchmarks discovered")
    return [Benchmark(path, tuple(sorted(engines))) for path, engines in sorted(groups.items())]


def parse_measurements(text: str, *, run: str = "imported", suite: str = "public_api") -> list[Measurement]:
    result: list[Measurement] = []
    seen = set()
    for path, columns, line in tree_rows(text, suite=suite):
        if columns is None:
            continue
        # Internal functions identify an engine or forced tier directly. Their
        # numeric/shape argument remains intact, including the worker budget.
        if suite == "internal_improvement":
            engine_positions = [len(path) - (2 if path[-1][:1].isdigit() else 1)]
        else:
            engine_positions = [i for i, part in enumerate(path) if part in ENGINES]
        if len(engine_positions) != 1:
            raise BenchmarkError(f"missing or ambiguous engine at line {line}")
        position = engine_positions[0]
        if len(path) > position + 2:
            raise BenchmarkError(f"unsupported nested benchmark arguments at line {line}")
        try:
            samples, iterations = (int(item.replace(",", "")) for item in columns[4:6])
        except ValueError as error:
            raise BenchmarkError(f"invalid sample counts at line {line}") from error
        if samples <= 0 or iterations < samples:
            raise BenchmarkError(f"invalid sample/iteration counts at line {line}")
        measurement = Measurement(
            path="::".join(path[:position]), engine=path[position],
            argument=path[position + 1] if len(path) > position + 1 else None,
            fastest_ns=duration_ns(columns[0]), slowest_ns=duration_ns(columns[1]),
            median_ns=duration_ns(columns[2]), mean_ns=duration_ns(columns[3]),
            samples=samples, iterations=iterations, run=run, suite=suite,
        )
        if measurement.key in seen:
            raise BenchmarkError(f"duplicate measurement at line {line}: {measurement.key}; import runs separately")
        if not measurement.fastest_ns <= measurement.median_ns <= measurement.slowest_ns:
            raise BenchmarkError(f"inconsistent timing range at line {line}")
        seen.add(measurement.key)
        result.append(measurement)
    if not result:
        raise BenchmarkError("no timing rows found; --list and --test output are not measurements")
    return result
