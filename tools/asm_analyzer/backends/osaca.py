#!/usr/bin/env python3
"""OSACA backend for the assembly analyzer suite.

OSACA (RRZE-HPC) analytical execution port and critical path model.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from ..analyzer import Analyzer, KernelReport
from ..models import CpuSpec


def _run_osaca_cli(arch: str, asm_file: Path, use_wsl: bool = False) -> Tuple[Optional[float], str]:
    """Run OSACA on an assembly file, parsing cycles and returning diagnostics."""
    from ..asm_util import wsl_path
    file_arg = wsl_path(asm_file) if use_wsl else str(asm_file)
    cmd = ["osaca", f"--arch={arch}", "--consider-flag-deps", file_arg]
    if use_wsl:
        import shlex
        inner = " ".join(shlex.quote(c) for c in cmd)
        cmd = ["wsl.exe", "-e", "bash", "-lc", inner]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30)
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if "Throughput (TP):" in line or "Loop-Carried Dependencies (LCD):" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        val = parts[1].strip().split()[0]
                        try:
                            return float(val), r.stdout
                        except ValueError:
                            pass
            return None, f"OSACA succeeded but cycle line not found in output:\n{r.stdout}"
        err = r.stderr.strip() if r.stderr else f"OSACA exited with code {r.returncode}"
        return None, err
    except subprocess.TimeoutExpired:
        return None, "OSACA timed out after 30 seconds"
    except Exception as err:
        return None, f"OSACA execution failed: {err}"


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
                return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note=raw, raw_output=raw)
            return KernelReport(backend=self.name, cpu=cpu.name, ok=True, cycles=cyc, raw_output=raw)
        finally:
            tmp.unlink(missing_ok=True)
