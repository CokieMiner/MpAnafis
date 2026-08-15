"""LLVM-MCA analyzer backend for cycle-accurate instruction throughput simulation."""

from __future__ import annotations

import os
from typing import Optional
from ..analyzer import Analyzer, KernelReport
from ..models import CpuSpec
from .mca_driver import McaDriver, discover_mca_binary

IS_WINDOWS = os.name == "nt"


class LlvmMcaAnalyzer(Analyzer):
    """Cycle-accurate out-of-order pipeline simulation via llvm-mca."""

    name = "llvm-mca"

    def __init__(self, mca: Optional[str] = None, wsl: Optional[bool] = None) -> None:
        self.wsl = IS_WINDOWS if wsl is None else wsl
        self._mca_bin = mca or discover_mca_binary(self.wsl)
        self._driver = McaDriver(self._mca_bin or "llvm-mca", self.wsl)
        self._known_cpus: dict[str, set[str]] = {}

    def available(self) -> bool:
        """Return True if llvm-mca binary was located."""
        return self._mca_bin is not None

    def _get_triple(self, family: str) -> str:
        if family in ("arm", "aarch64"):
            return "aarch64"
        if family == "arm32":
            return "arm"
        if family in ("riscv", "riscv64"):
            return "riscv64"
        if family in ("ppc", "powerpc"):
            return "powerpc64le"
        if family == "s390x":
            return "s390x"
        if family in ("sparc64", "sparc"):
            return "sparc64" if family == "sparc64" else "sparc"
        return "x86_64"

    def supports(self, cpu: CpuSpec) -> bool:
        """Return True if llvm-mca models this logical CPU."""
        if not self.available():
            return False
        model = cpu.model_for(self.name)
        if model is None:
            return False
        triple = self._get_triple(cpu.family)
        if triple not in self._known_cpus:
            self._known_cpus[triple] = set(self._driver.list_cpus(triple=triple))
        return not self._known_cpus[triple] or model in self._known_cpus[triple]

    def analyze(self, asm_code: str, cpu: CpuSpec, iterations: int = 200) -> Optional[float]:
        """Analyze assembly block with llvm-mca."""
        if not self.supports(cpu):
            return None
        model = cpu.model_for(self.name)
        if not model:
            return None
        triple = self._get_triple(cpu.family)
        return self._driver.run_on_asm(asm_code, model, iterations, triple=triple)

    def analyze_report(self, asm_code: str, cpu: CpuSpec, iterations: int = 200) -> KernelReport:
        """Analyze assembly block with llvm-mca and return rich KernelReport."""
        if not self.available():
            return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note="llvm-mca binary not found")
        if not self.supports(cpu):
            return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note=f"CPU '{cpu.name}' not supported by llvm-mca")
        model = cpu.model_for(self.name)
        if not model:
            return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note=f"No model string for '{cpu.name}'")
        triple = self._get_triple(cpu.family)
        cycles, uops, ports, raw = self._driver.run_on_asm_detailed(
            asm_code=asm_code,
            cpu=model,
            iterations=iterations,
            triple=triple,
        )
        if cycles is None:
            return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note=raw, raw_output=raw)
        return KernelReport(
            backend=self.name,
            cpu=cpu.name,
            ok=True,
            cycles=cycles,
            uops=uops,
            port_pressure=ports,
            raw_output=raw,
        )
