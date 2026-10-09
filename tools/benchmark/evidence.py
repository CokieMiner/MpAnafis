"""Validate measurement plans, pinning and exact raw captures before publication."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, replace
from pathlib import Path

from .catalog import validate_catalog
from .divan import parse_measurements
from .models import Benchmark, BenchmarkError, Measurement
from .plan import validate_plan


def validate_record(source: Path) -> tuple[dict, list[Measurement], list[Path]]:
    """Require complete measured runs and agreement with each preserved raw capture."""
    for filename in ("run.json", "measurements.json"):
        path = source / filename
        if not path.is_file() or path.is_symlink():
            raise BenchmarkError(f"missing regular run file: {path}")
    metadata = json.loads((source / "run.json").read_text())
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise BenchmarkError("unsupported run metadata")
    if metadata.get("status") != "complete" or metadata.get("smoke") is not False:
        raise BenchmarkError("only completed measured runs can be exported")
    if not isinstance(metadata.get("configuration"), str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", metadata["configuration"]):
        raise BenchmarkError("run has no valid configuration identity")
    plan = metadata.get("plan")
    if not isinstance(plan, dict) or not isinstance(plan.get("runs"), list) or not plan["runs"]:
        raise BenchmarkError("run has no execution plan")
    if not isinstance(metadata.get("commands"), list) or len(metadata["commands"]) != len(plan["runs"]):
        raise BenchmarkError("run command count differs from its plan")
    for field in ("binary", "binary_sha256", "git_head", "rustc", "platform", "processor"):
        if not isinstance(metadata.get(field), str) or not metadata[field]:
            raise BenchmarkError(f"run metadata is missing {field}")
    data = json.loads((source / "measurements.json").read_text())
    if not isinstance(data, list) or not data:
        raise BenchmarkError("run has no measurements")
    rows = [Measurement(**row) for row in data]
    if any(row.suite != "public_api" or row.configuration != metadata["configuration"] for row in rows):
        raise BenchmarkError("measurement suite or configuration differs from run metadata")
    expected = {row.key: asdict(row) for row in rows}
    if len(expected) != len(rows):
        raise BenchmarkError("duplicate measurements in run")
    observed = {}
    files = []
    for index, entry in enumerate(plan["runs"]):
        if not isinstance(entry, dict):
            raise BenchmarkError("invalid planned run")
        run_id = f"{index:04d}"
        for stream in ("stdout", "stderr"):
            path = source / f"{run_id}.{stream}.txt"
            if not path.is_file() or path.is_symlink():
                raise BenchmarkError(f"missing regular raw capture: {path}")
            files.append(path)
        captured = parse_measurements((source / f"{run_id}.stdout.txt").read_text(), run=run_id)
        if any(row.path != entry.get("path") or row.engine != entry.get("engine") for row in captured):
            raise BenchmarkError("raw capture differs from its planned function or engine")
        arguments = entry.get("arguments", plan.get("arguments"))
        if not isinstance(arguments, list) or any(not isinstance(arg, str) or not arg.isdecimal() for arg in arguments):
            raise BenchmarkError("invalid planned argument selection")
        if arguments and {row.argument for row in captured} != set(arguments):
            raise BenchmarkError("raw capture is missing planned arguments")
        for row in captured:
            configured = replace(row, configuration=metadata["configuration"])
            observed[configured.key] = asdict(configured)
    if observed != expected:
        raise BenchmarkError("measurements differ from the preserved raw captures")
    validate_measurement_plan(metadata, rows)
    return metadata, rows, files


def validate_measurement_plan(metadata: dict, rows: list[Measurement]) -> None:
    """Verify declared worker budgets, actual pinning commands and paired run order."""
    plan = metadata["plan"]
    engines = {}
    for row in rows:
        engines.setdefault(row.path, set()).add(row.engine)
    catalog = [Benchmark(path, tuple(sorted(names))) for path, names in sorted(engines.items())]
    validate_catalog(catalog, require_comparison=True)
    try:
        validate_plan(plan, catalog)
    except (KeyError, TypeError) as error:
        raise BenchmarkError(f"incomplete measurement plan: {error}") from error
    settings = plan["settings"]
    if not settings["cpus"]:
        raise BenchmarkError("documentation records require CPU pinning")
    configuration = metadata.get("configuration_details")
    if not isinstance(configuration, dict) or configuration.get("settings") != settings:
        raise BenchmarkError("configuration worker, sampling and affinity settings differ from the plan")
    times = [metadata.get(name) for name in ("started_unix", "finished_unix")]
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in times) or times[1] < times[0]:
        raise BenchmarkError("measurement timestamps must be finite and chronological")
    if any(row.samples != settings["samples"] for row in rows):
        raise BenchmarkError("captured sample count differs from the plan")
    cpus = ",".join(map(str, settings["cpus"]))
    for entry, command in zip(plan["runs"], metadata["commands"]):
        expected = ["taskset", "--cpu-list", cpus, metadata["binary"], "--bench", entry["filter"],
                    "--color", "never", "--sample-count", str(settings["samples"]),
                    "--sample-size", str(settings["sample_size"])]
        if command != expected:
            raise BenchmarkError("recorded benchmark command differs from the planned binary, pinning, filter or sampling")
    if len(plan["runs"]) % 4:
        raise BenchmarkError("comparison records require complete A/B/B/A rounds")
    comparisons = {item.path: item.comparisons for item in catalog}
    rounds = {}
    for start in range(0, len(plan["runs"]), 4):
        group = plan["runs"][start:start + 4]
        first = group[0]
        path = first["path"]
        comparison = group[1]["engine"]
        arguments = first.get("arguments", plan["arguments"])
        round_index = rounds.get(path, 0)
        if (comparison not in comparisons[path]
                or [entry["engine"] for entry in group] != ["mp", comparison, comparison, "mp"]
                or any(entry["path"] != path or entry.get("arguments", plan["arguments"]) != arguments
                       or type(entry.get("round")) is not int or entry["round"] != round_index
                       or type(entry.get("position")) is not int or entry["position"] != position
                       for position, entry in enumerate(group))):
            raise BenchmarkError("comparison rounds must preserve function, arguments and positions in A/B/B/A order")
        rounds[path] = round_index + 1
