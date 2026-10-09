"""Shared reports for public API and internal Divan measurements."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from .models import BenchmarkError, Measurement
from .paths import validate_output_path


def summarize(rows: list[Measurement]) -> list[dict]:
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        if row.key in seen:
            raise BenchmarkError(f"duplicate measurement: {row.key}")
        seen.add(row.key)
        groups[row.suite, row.configuration, row.path, row.argument, row.engine].append(row)
    result = []
    for (suite, configuration, path, argument, engine), values in sorted(groups.items(), key=lambda item: str(item[0])):
        medians = [row.median_ns for row in values]
        result.append({
            "suite": suite, "configuration": configuration, "path": path,
            "argument": argument, "engine": engine,
            "median_ns": statistics.median(medians),
            "min_run_median_ns": min(medians), "max_run_median_ns": max(medians),
            "runs": len(values), "samples": sum(row.samples for row in values),
            "iterations": sum(row.iterations for row in values),
        })
    return result


def write_report(rows: list[Measurement], output: Path, *, plots: bool = False) -> None:
    output = validate_output_path(output)
    if not rows:
        raise BenchmarkError("cannot report an empty measurement set")
    summary = summarize(rows)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (output / "measurements.json").write_text(json.dumps([asdict(row) for row in rows], indent=2) + "\n")
    lines = [
        "# Benchmark results", "",
        "Times are nanoseconds per benchmark iteration, including any documented operand batch.",
        "Repeated runs use the median of run medians; ranges describe runs, not confidence intervals.",
        "Function paths, suites, arguments, worker labels, and configurations remain separate.", "",
        "## Measurements", "",
        "| Suite | Configuration | Function/scenario | Argument | Engine/tier | Median ns | Run median range ns | Runs |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['suite']} | {row['configuration']} | `{row['path']}` | {row['argument'] or '—'} "
            f"| {row['engine']} | {row['median_ns']:.4g} "
            f"| {row['min_run_median_ns']:.4g}–{row['max_run_median_ns']:.4g} | {row['runs']} |"
        )
    public = [row for row in summary if row["suite"] == "public_api"]
    if public:
        lines.extend([
            "", "## Public API comparisons", "",
            "Speedup is comparator time / Mp time; values above one favor Mp.", "",
            "All measured comparators are shown. The fastest reference is selected by its measured median within each configuration and argument.", "",
            "| Configuration | Function/scenario | Argument | Comparator | Mp ns | Comparator ns | Speedup |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ])
        grouped = defaultdict(dict)
        for row in public:
            grouped[row["configuration"], row["path"], row["argument"]][row["engine"]] = row
        for (configuration, path, argument), engines in sorted(grouped.items(), key=lambda item: str(item[0])):
            mp = engines.get("mp")
            comparisons = sorted((key for key in ("rug", "gmp", "flint") if key in engines),
                                 key=lambda key: engines[key]["median_ns"])
            if mp is None or not comparisons:
                continue
            for index, comparison in enumerate(comparisons):
                peer = engines[comparison]["median_ns"]
                speedup = f"{peer / mp['median_ns']:.3f}×" if mp["median_ns"] else "—"
                label = comparison + (" (fastest measured)" if index == 0 else "")
                lines.append(f"| {configuration} | `{path}` | {argument or '—'} | {label} | {mp['median_ns']:.4g} | {peer:.4g} | {speedup} |")
    if any(row["suite"] == "internal_improvement" for row in summary):
        lines.extend([
            "", "Internal engine/tier labels and operand shapes are reported verbatim.",
            "Ratios are not inferred between different worker budgets, forced tiers, or sampling ladders.",
        ])
    if plots:
        destinations = plot_summary(summary, output)
        if destinations:
            lines.extend(["", "## Time versus size", ""])
            for destination in destinations:
                lines.append(f"- [{destination.with_suffix('').as_posix()}]({destination.as_posix()})")
        else:
            lines.extend(["", "No size-sweep plots: each curve requires at least two numeric arguments."])
    (output / "report.md").write_text("\n".join(lines) + "\n")


def plot_path(suite: str, path: str, configuration: str) -> Path:
    """Mirror the benchmark categories; keep configurations in distinct files."""
    return Path("plots", suite, *path.split("::"), configuration + ".png")


def size_sweeps(summary: list[dict]) -> dict[tuple[str, str, str], dict[str, list[dict]]]:
    """Select measured numeric curves; single points and shaped arguments stay in tables."""
    grouped = defaultdict(lambda: defaultdict(list))
    for row in summary:
        if row["argument"] is not None and row["argument"].isdigit():
            grouped[row["suite"], row["configuration"], row["path"]][row["engine"]].append(row)
    result = {}
    for key, engines in sorted(grouped.items()):
        curves = {engine: sorted(rows, key=lambda row: int(row["argument"]))
                  for engine, rows in sorted(engines.items())
                  if len({int(row["argument"]) for row in rows}) >= 2}
        if curves:
            result[key] = curves
    return result


def plot_summary(summary: list[dict], output: Path) -> list[Path]:
    sweeps = size_sweeps(summary)
    if not sweeps:
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise BenchmarkError("plotting requires matplotlib; JSON/CSV/Markdown do not") from error
    destinations = []
    for (suite, configuration, path), engines in sweeps.items():
        figure, axis = plt.subplots(figsize=(10, 6), layout="constrained")
        for engine, series in engines.items():
            medians = [row["median_ns"] for row in series]
            bounds = [[row["median_ns"] - row["min_run_median_ns"] for row in series],
                      [row["max_run_median_ns"] - row["median_ns"] for row in series]]
            axis.errorbar([int(row["argument"]) for row in series], medians, yerr=bounds,
                          marker="o", capsize=3, label=engine)
        entries = [row for series in engines.values() for row in series]
        if all(int(row["argument"]) > 0 for row in entries):
            axis.set_xscale("log", base=2)
        if all(row["min_run_median_ns"] > 0 for row in entries):
            axis.set_yscale("log")
        else:
            axis.set_ylim(bottom=0)
        axis.set(title=f"{suite}::{path}\n{configuration}",
                 xlabel="Benchmark size argument (units defined by the case)",
                 ylabel="ns / benchmark iteration")
        axis.grid(which="both", alpha=0.25)
        axis.legend()
        relative = plot_path(suite, path, configuration)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=180)
        plt.close(figure)
        destinations.append(relative)
    return destinations
