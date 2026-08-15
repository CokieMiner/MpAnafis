#!/usr/bin/env python3
"""uiCA backend for the assembly analyzer suite.

uiCA (IUPU) analytical execution model for modern Intel x86 microarchitectures.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from ..analyzer import Analyzer, KernelReport
from ..models import CpuSpec


def _run_uica_cli(arch: str, asm_file: Path, use_wsl: bool = False) -> Tuple[Optional[float], str]:
    """Run uiCA on an assembly file, parsing cycles and returning diagnostics."""
    from ..asm_util import wsl_path
    file_arg = wsl_path(asm_file) if use_wsl else str(asm_file)
    cmd = ["uica", f"-arch={arch}", file_arg]
    if use_wsl:
        import shlex
        inner = " ".join(shlex.quote(c) for c in cmd)
        cmd = ["wsl.exe", "-e", "bash", "-lc", inner]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30)
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if "Throughput (cycles):" in line or "Block Throughput:" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        try:
                            val = float(parts[1].strip().split()[0])
                            return val, r.stdout
                        except ValueError:
                            pass
            return None, f"uiCA succeeded but throughput line not found in output:\n{r.stdout}"
        err = r.stderr.strip() if r.stderr else f"uiCA exited with code {r.returncode}"
        return None, err
    except subprocess.TimeoutExpired:
        return None, "uiCA timed out after 30 seconds"
    except Exception as err:
        return None, f"uiCA execution failed: {err}"


class UicaAnalyzer(Analyzer):
    """Analytical throughput simulation for Intel CPUs via uiCA."""

    name = "uica"

    def __init__(self, wsl: Optional[bool] = None) -> None:
        self._wsl = bool(wsl)

    def available(self) -> bool:
        """Return True if uiCA is installed and executable."""
        return shutil.which("uica") is not None

    def supports(self, cpu: CpuSpec) -> bool:
        """Return True if uiCA models this Intel CPU."""
        return self.available() and cpu.model_for(self.name) is not None

    def analyze(self, asm_code: str, cpu: CpuSpec, iterations: int = 200) -> Optional[float]:
        """Analyze assembly block with uiCA."""
        report = self.analyze_report(asm_code, cpu, iterations=iterations)
        return report.cycles

    def analyze_report(self, asm_code: str, cpu: CpuSpec, iterations: int = 200) -> KernelReport:
        """Analyze assembly block with uiCA and return structured report with diagnostics."""
        if not self.available():
            return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note="uica binary not found in PATH")
        if not self.supports(cpu):
            return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note=f"CPU '{cpu.name}' not supported by uiCA")
        model = cpu.model_for(self.name)
        if not model:
            return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note=f"No uiCA model for '{cpu.name}'")

        with tempfile.NamedTemporaryFile("w", suffix=".s", delete=False) as f:
            f.write(asm_code)
            tmp = Path(f.name)
        try:
            cyc, raw = _run_uica_cli(model, tmp, use_wsl=self._wsl)
            if cyc is None:
                return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note=raw, raw_output=raw)
            return KernelReport(backend=self.name, cpu=cpu.name, ok=True, cycles=cyc, raw_output=raw)
        finally:
            tmp.unlink(missing_ok=True)
