"""Stable benchmark identities and measured Divan summaries."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re


class BenchmarkError(ValueError):
    """Invalid benchmark input, configuration, or incomplete measurement."""


ENGINES = frozenset({"mp", "rug", "gmp", "flint"})
SUITES = frozenset({"public_api", "internal_improvement"})


@dataclass(frozen=True)
class Benchmark:
    path: str
    engines: tuple[str, ...]

    @property
    def category(self) -> str:
        return "::".join(self.path.split("::")[:3])

    @property
    def comparisons(self) -> tuple[str, ...]:
        return tuple(engine for engine in ("rug", "gmp", "flint") if engine in self.engines)


@dataclass(frozen=True)
class Measurement:
    path: str
    engine: str
    argument: str | None
    fastest_ns: float
    slowest_ns: float
    median_ns: float
    mean_ns: float
    samples: int
    iterations: int
    run: str = "imported"
    configuration: str = "unspecified"
    suite: str = "public_api"

    def __post_init__(self):
        if self.suite not in SUITES:
            raise BenchmarkError("invalid benchmark suite")
        path_pattern = (r"int::(?:signed|unsigned)(?:::[A-Za-z_][A-Za-z_0-9]*){2,}"
                        if self.suite == "public_api" else r"[A-Za-z_][A-Za-z_0-9]*(?:::[A-Za-z_][A-Za-z_0-9]*)+")
        if not isinstance(self.path, str) or not re.fullmatch(path_pattern, self.path):
            raise BenchmarkError("invalid measurement function path")
        if (self.suite == "public_api" and self.engine not in ENGINES) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", self.engine):
            raise BenchmarkError("invalid measurement engine")
        argument_pattern = r"[0-9]+" if self.suite == "public_api" else r"[0-9]+(?:x[0-9]+)?(?:-limbs/[0-9]+-workers)?"
        if self.argument is not None and (not isinstance(self.argument, str) or not re.fullmatch(argument_pattern, self.argument)):
            raise BenchmarkError("invalid measurement argument")
        for value in (self.fastest_ns, self.slowest_ns, self.median_ns, self.mean_ns):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise BenchmarkError("measurement durations must be finite and nonnegative")
        if not self.fastest_ns <= self.median_ns <= self.slowest_ns or not self.fastest_ns <= self.mean_ns <= self.slowest_ns:
            raise BenchmarkError("inconsistent timing range")
        if type(self.samples) is not int or type(self.iterations) is not int or self.samples <= 0 or self.iterations < self.samples:
            raise BenchmarkError("invalid sample/iteration counts")
        if not isinstance(self.run, str) or not self.run or not re.fullmatch(r"[A-Za-z0-9_-]+", self.configuration):
            raise BenchmarkError("invalid run or configuration identity")

    @property
    def key(self) -> tuple[str, str, str | None, str, str, str]:
        return self.path, self.engine, self.argument, self.run, self.configuration, self.suite
