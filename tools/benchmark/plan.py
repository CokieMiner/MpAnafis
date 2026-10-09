"""Validate executable benchmark plans independently of CLI and publication."""

from __future__ import annotations

import math

from .catalog import benchmark_filter
from .models import BenchmarkError


def validate_plan(plan: dict, catalog: list) -> None:
    if not isinstance(plan, dict):
        raise BenchmarkError("plan must be a JSON object")
    if plan.get("schema_version") != 1:
        raise BenchmarkError("unsupported benchmark plan schema")
    settings = plan["settings"]
    if not isinstance(settings, dict):
        raise BenchmarkError("settings must be a JSON object")
    for name in ("threads", "samples", "sample_size"):
        if type(settings[name]) is not int or settings[name] < 1:
            raise BenchmarkError(f"{name} must be a positive integer")
    timeout = settings["timeout"]
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise BenchmarkError("timeout must be finite and positive")
    cpus = settings["cpus"]
    if not isinstance(cpus, list) or any(type(cpu) is not int or cpu < 0 for cpu in cpus):
        raise BenchmarkError("cpus must be a list of nonnegative integers")
    if len(set(cpus)) != len(cpus) or (cpus and len(cpus) < settings["threads"]):
        raise BenchmarkError("CPU affinity must contain at least one distinct CPU per worker")
    if not isinstance(plan["arguments"], list) or any(not isinstance(arg, str) or not arg.isdecimal() for arg in plan["arguments"]):
        raise BenchmarkError("arguments must be decimal strings")
    if not isinstance(plan["runs"], list) or not plan["runs"]:
        raise BenchmarkError("plan contains no runs")
    known = {item.path: item for item in catalog}
    for entry in plan["runs"]:
        if not isinstance(entry, dict):
            raise BenchmarkError("each run must be a JSON object")
        item = known.get(entry["path"])
        if item is None or entry["engine"] not in item.engines:
            raise BenchmarkError(f"planned benchmark is unavailable: {entry}")
        arguments = entry.get("arguments", plan["arguments"])
        if not isinstance(arguments, list) or any(not isinstance(arg, str) or not arg.isdecimal() for arg in arguments):
            raise BenchmarkError("run arguments must be decimal strings")
        expected = benchmark_filter(entry["path"], entry["engine"], arguments)
        if entry["filter"] != expected:
            raise BenchmarkError("plan filter does not match its declared function, engine, and arguments")
