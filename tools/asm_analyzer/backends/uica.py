#!/usr/bin/env python3
"""uiCA backend for the assembly analyzer suite.

uiCA (IUPU) analytical execution model for modern Intel x86 microarchitectures.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from ..analyzer import Analyzer, KernelReport
from ..asm_util import wsl_path
from ..models import CpuSpec

class UicaAnalyzer(Analyzer):
    """Analytical throughput simulation for Intel CPUs via uiCA."""

    name = "uica"

    def __init__(self, wsl: Optional[bool] = None) -> None:
        self._wsl = bool(wsl)

    def available(self) -> bool:
        """Return True if uiCA and GNU as are installed and executable."""
        return shutil.which("uica") is not None and shutil.which("as") is not None

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
            return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note="uica or as binary not found in PATH")
        if not self.supports(cpu):
            return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note=f"CPU '{cpu.name}' not supported by uiCA")
        model = cpu.model_for(self.name)
        if not model:
            return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note=f"No uiCA model for '{cpu.name}'")

        with tempfile.NamedTemporaryFile("w", suffix=".s", delete=False) as f:
            f.write(asm_code)
            tmp = Path(f.name)
        obj_file = tmp.with_suffix(".o")
        try:
            as_cmd = ["as", "-64", str(tmp), "-o", str(obj_file)]
            if self._wsl:
                as_cmd = [
                    "wsl.exe",
                    "-e",
                    "as",
                    "-64",
                    wsl_path(tmp),
                    "-o",
                    wsl_path(obj_file),
                ]
            as_res = subprocess.run(
                as_cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
            if as_res.returncode != 0:
                err_msg = as_res.stderr.strip() or f"Assembler exited with code {as_res.returncode}"
                return KernelReport(
                    backend=self.name,
                    cpu=cpu.name,
                    ok=False,
                    note=f"Assembler error: {err_msg}",
                    raw_output=as_res.stderr,
                )
            cyc, raw = _run_uica_cli(model, obj_file, use_wsl=self._wsl)
            if cyc is None:
                return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note=raw, raw_output=raw)
            return KernelReport(backend=self.name, cpu=cpu.name, ok=True, cycles=cyc, raw_output=raw)
        finally:
            tmp.unlink(missing_ok=True)
            obj_file.unlink(missing_ok=True)


def _run_uica_cli(
    arch: str,
    obj_file: Path,
    use_wsl: bool = False,
) -> Tuple[Optional[float], str]:
    """Run uiCA on an assembled object file, parsing cycles and returning diagnostics."""
    file_arg = wsl_path(obj_file) if use_wsl else str(obj_file)
    cmd = ["uica", f"-arch={arch}", file_arg]
    if use_wsl:
        inner = " ".join(shlex.quote(c) for c in cmd)
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
            for line in result.stdout.splitlines():
                if (
                    "Throughput (in cycles per iteration):" in line
                    or "Throughput (cycles):" in line
                    or "Block Throughput:" in line
                ):
                    parts = line.split(":")
                    if len(parts) >= 2:
                        try:
                            value = float(parts[1].strip().split()[0])
                            return value, result.stdout
                        except ValueError:
                            pass
            return None, (
                "uiCA succeeded but throughput line not found in output:\n"
                f"{result.stdout}"
            )
        error = (
            result.stderr.strip()
            if result.stderr
            else f"uiCA exited with code {result.returncode}"
        )
        return None, error
    except subprocess.TimeoutExpired:
        return None, "uiCA timed out after 30 seconds"
    except Exception as error:
        return None, f"uiCA execution failed: {error}"
