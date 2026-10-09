"""Cohesive execution of analytical backends across a CPU matrix."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

from .analyzer import Analyzer, KernelReport
from .models import CpuSpec


@dataclass(frozen=True)
class BackendFailure:
    """One supported backend/CPU cell that failed to produce a cost."""

    cpu: str
    backend: str
    reason: str

    def message(self) -> str:
        """Render a concise diagnostic for CLI and JSON reports."""
        first_line = self.reason.strip().splitlines()[0] if self.reason.strip() else "no cycle estimate"
        return f"{self.cpu}/{self.backend}: {first_line}"


@dataclass(frozen=True)
class SimulationMatrix:
    """Per-backend cycle estimates and failures for one assembly region."""

    cycles: Dict[str, Dict[str, float]]
    scheduling_costs: Dict[str, Dict[str, float]]
    scheduling_metrics: Dict[str, Dict[str, str]]
    failures: Tuple[BackendFailure, ...]
    reports: Dict[str, Dict[str, KernelReport]]

    def flattened_with_medians(self) -> Dict[str, float]:
        """Return qualified cells plus an uncalibrated median per CPU."""
        flattened: Dict[str, float] = {}
        for cpu, backend_cycles in self.cycles.items():
            for backend, cycles in backend_cycles.items():
                flattened[f"{cpu}/{backend}"] = cycles
            if backend_cycles:
                flattened[cpu] = statistics.median(backend_cycles.values())
        return flattened

    def failure_messages(self) -> Tuple[str, ...]:
        """Return stable human-readable failure diagnostics."""
        return tuple(failure.message() for failure in self.failures)

    def model_results(self) -> Dict[str, Dict[str, object]]:
        """Return JSON-ready CPU/backend execution-model details."""
        return {
            cpu: {
                backend: {
                    "cycles": report.cycles,
                    "simulated_cycles": report.simulated_cycles,
                    "uops": report.uops,
                    "dispatch_width": report.dispatch_width,
                    "resource_pressure": report.port_pressure,
                    "instructions": [
                        {
                            "line": metric.line,
                            "latency": metric.latency,
                            "throughput": metric.throughput,
                            "uops": metric.uops,
                        }
                        for metric in report.instructions
                    ],
                }
                for backend, report in backend_reports.items()
            }
            for cpu, backend_reports in self.reports.items()
        }


def simulate_backends(
    asm: str,
    cpus: Iterable[CpuSpec],
    backend_names: Iterable[str],
    backends: Mapping[str, Analyzer],
) -> SimulationMatrix:
    """Run supported backend/CPU cells and retain every failed attempt."""
    cycles: Dict[str, Dict[str, float]] = {}
    scheduling_costs: Dict[str, Dict[str, float]] = {}
    scheduling_metrics: Dict[str, Dict[str, str]] = {}
    reports: Dict[str, Dict[str, KernelReport]] = {}
    failures: List[BackendFailure] = []
    backend_names = tuple(dict.fromkeys(backend_names))

    for cpu in cpus:
        cpu_cycles: Dict[str, float] = {}
        cpu_scheduling_costs: Dict[str, float] = {}
        cpu_scheduling_metrics: Dict[str, str] = {}
        cpu_reports: Dict[str, KernelReport] = {}
        for backend_name in backend_names:
            backend = backends.get(backend_name)
            if backend is None:
                failures.append(BackendFailure(cpu.name, backend_name, "backend was not constructed"))
                continue
            if not backend.supports(cpu):
                continue
            try:
                report = backend.analyze_report(asm, cpu)
            except Exception as error:  # Backends are external process boundaries.
                failures.append(BackendFailure(cpu.name, backend_name, str(error)))
                continue
            value = report.cycles
            if report.ok and value is not None and math.isfinite(value) and value > 0:
                cpu_cycles[backend_name] = value
                cpu_reports[backend_name] = report
                scheduling_cost = report.scheduling_cost()
                if (
                    scheduling_cost is not None
                    and math.isfinite(scheduling_cost)
                    and scheduling_cost > 0
                ):
                    cpu_scheduling_costs[backend_name] = scheduling_cost
                    cpu_scheduling_metrics[backend_name] = report.scheduling_metric()
            else:
                failures.append(
                    BackendFailure(
                        cpu.name,
                        backend_name,
                        report.note or report.raw_output or "no positive finite cycle estimate",
                    )
                )
        cycles[cpu.name] = cpu_cycles
        scheduling_costs[cpu.name] = cpu_scheduling_costs
        scheduling_metrics[cpu.name] = cpu_scheduling_metrics
        reports[cpu.name] = cpu_reports

    return SimulationMatrix(
        cycles=cycles,
        scheduling_costs=scheduling_costs,
        scheduling_metrics=scheduling_metrics,
        failures=tuple(failures),
        reports=reports,
    )


__all__ = ["BackendFailure", "SimulationMatrix", "simulate_backends"]
