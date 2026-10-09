#!/usr/bin/env python3
"""OSACA backend for the assembly analyzer suite.

OSACA (RRZE-HPC) analytical execution port and critical path model.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from ..analyzer import Analyzer, KernelReport
from ..asm_util import wsl_path
from ..models import CpuSpec


class OsacaAnalyzer(Analyzer):
    """Analytical port throughput and dependency analyzer via OSACA."""

    name = "osaca"

    def __init__(self, wsl: Optional[bool] = None) -> None:
        self._wsl = bool(wsl)

    def available(self) -> bool:
        """Return True if OSACA is installed and executable."""
        return shutil.which("osaca") is not None

    def supports(self, cpu: CpuSpec) -> bool:
        """Return True if OSACA models this CPU architecture."""
        return self.available() and cpu.model_for(self.name) is not None

    def analyze(self, asm_code: str, cpu: CpuSpec, iterations: int = 200) -> Optional[float]:
        """Analyze assembly block with OSACA."""
        report = self.analyze_report(asm_code, cpu, iterations=iterations)
        return report.cycles

    def analyze_report(self, asm_code: str, cpu: CpuSpec, iterations: int = 200) -> KernelReport:
        """Analyze assembly block with OSACA and return structured report with diagnostics."""
        if not self.available():
            return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note="osaca binary not found in PATH")
        if not self.supports(cpu):
            return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note=f"CPU '{cpu.name}' not supported by OSACA")
        model = cpu.model_for(self.name)
        if not model:
            return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note=f"No OSACA model for '{cpu.name}'")

        with tempfile.NamedTemporaryFile("w", suffix=".s", delete=False) as f:
            f.write(asm_code)
            tmp = Path(f.name)
        try:
            cyc, raw = _run_osaca_cli(model, tmp, use_wsl=self._wsl)
            if cyc is None:
                missing = re.search(r"performance data for (\d+) instructions? is missing", raw)
                note = (
                    f"OSACA model {model} lacks performance data for {missing.group(1)} "
                    "instructions; no complete cycle estimate is available"
                    if missing else raw
                )
                return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note=note, raw_output=raw)
            return KernelReport(backend=self.name, cpu=cpu.name, ok=True, cycles=cyc, raw_output=raw)
        finally:
            tmp.unlink(missing_ok=True)


def _run_osaca_cli(
    arch: str,
    asm_file: Path,
    use_wsl: bool = False,
) -> Tuple[Optional[float], str]:
    """Run OSACA on an assembly file, parsing cycles and returning diagnostics."""
    file_arg = wsl_path(asm_file) if use_wsl else str(asm_file)
    cmd = ["osaca", f"--arch={arch}", "--consider-flag-deps", file_arg]
    if use_wsl:
        inner = " ".join(shlex.quote(component) for component in cmd)
        cmd = ["wsl.exe", "-e", "bash", "-lc", inner]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if result.returncode == 0:
            cycles = _parse_osaca_cycles(result.stdout)
            if cycles is not None:
                return cycles, result.stdout
            return None, f"OSACA succeeded but cycle line not found in output:\n{result.stdout}"
        error = result.stderr.strip() if result.stderr else f"OSACA exited with code {result.returncode}"
        return None, error
    except subprocess.TimeoutExpired:
        return None, "OSACA timed out after 30 seconds"
    except Exception as error:
        return None, f"OSACA execution failed: {error}"


def _parse_osaca_cycles(output: str) -> Optional[float]:
    """Parse OSACA throughput summaries from combined report rows or labelled throughput fields.

    Steady-state cost is the maximum port pressure or loop-carried dependency (LCD),
    not the single-iteration critical path.
    """
    if "No final analysis is given" in output or re.search(
        r"performance data for \d+ instructions? is missing", output,
    ):
        return None
    bounds = []
    for line in output.splitlines():
        if "Throughput (TP):" in line or "Loop-Carried Dependencies (LCD):" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    bounds.append(float(parts[1].strip().split()[0]))
                except (ValueError, IndexError):
                    pass
    if bounds:
        return max(bounds)

    in_combined_report = False
    for line in output.splitlines():
        if line.strip() == "Combined Analysis Report":
            in_combined_report = True
            continue
        if not in_combined_report or "|" in line:
            continue
        stripped = line.strip()
        if not stripped or re.fullmatch(r"[0-9.\s]+", stripped) is None:
            continue
        values = [float(value) for value in re.findall(r"\d+\.\d+", stripped)]
        if len(values) >= 2:
            port_pressures = values[:-2]
            lcd = values[-1]
            if port_pressures:
                cost = max([lcd, *port_pressures])
            else:
                cost = lcd if lcd > 0 else values[0]
            if cost > 0:
                return cost
    return None
