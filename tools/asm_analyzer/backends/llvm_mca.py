"""LLVM-MCA analytical instruction-throughput backend."""

from __future__ import annotations

import os
from typing import Optional

from ..analyzer import Analyzer, KernelReport
from ..models import CpuSpec
from .mca_driver import McaDriver, discover_mca_binary

IS_WINDOWS = os.name == "nt"


class LlvmMcaAnalyzer(Analyzer):
    """Static target scheduling and resource model via llvm-mca."""

    name = "llvm-mca"

    def __init__(self, mca: Optional[str] = None, wsl: Optional[bool] = None) -> None:
        self.wsl = IS_WINDOWS if wsl is None else wsl
        self._mca_bin = mca or discover_mca_binary(self.wsl)
        self._driver = McaDriver(self._mca_bin or "llvm-mca", self.wsl)
        self._known_cpus: dict[str, set[str]] = {}
        self._modeled_cpus: dict[tuple[str, str, tuple[str, ...]], bool] = {}

    def available(self) -> bool:
        """Return True if llvm-mca binary was located."""
        return self._mca_bin is not None

    def _get_triple(self, family: str) -> str:
        if family == "arm":
            return "aarch64"
        if family == "arm32":
            return "arm"
        if family == "x86_32":
            return "i386"
        if family == "riscv":
            return "riscv64"
        if family == "riscv32":
            return "riscv32"
        if family == "ppc":
            return "powerpc64le"
        if family == "ppc32":
            return "powerpc"
        if family == "s390x":
            return "s390x"
        if family == "mips64":
            return "mips64"
        if family == "mips32":
            return "mips"
        if family == "loongarch64":
            return "loongarch64"
        if family == "loongarch32":
            return "loongarch32"
        if family in ("amd", "intel"):
            return "x86_64"
        raise ValueError(f"unsupported LLVM target family: {family}")

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
        if self._known_cpus[triple] and model not in self._known_cpus[triple]:
            return False
        key = (triple, model, cpu.llvm_mca_features)
        if key not in self._modeled_cpus:
            smoke_instruction = {
                "arm": "add r0, r0, r1",
                "s390x": "nopr %r0",
            }.get(triple, "nop")
            self._modeled_cpus[key] = self._driver.run_on_asm(
                smoke_instruction,
                model,
                iterations=1,
                triple=triple,
                features=cpu.llvm_mca_features,
            ) is not None
        return self._modeled_cpus[key]

    def analyze(self, asm_code: str, cpu: CpuSpec, iterations: int = 200) -> Optional[float]:
        """Analyze assembly block with llvm-mca."""
        if not self.supports(cpu):
            return None
        model = cpu.model_for(self.name)
        if not model:
            return None
        triple = self._get_triple(cpu.family)
        return self._driver.run_on_asm(
            asm_code,
            model,
            iterations,
            triple=triple,
            features=cpu.llvm_mca_features,
        )

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
        result = self._driver.run_on_asm_detailed(
            asm_code=asm_code,
            cpu=model,
            iterations=iterations,
            triple=triple,
            features=cpu.llvm_mca_features,
        )
        if result.throughput_cycles is None:
            return KernelReport(
                backend=self.name,
                cpu=cpu.name,
                ok=False,
                note=result.raw_output,
                raw_output=result.raw_output,
            )
        return KernelReport(
            backend=self.name,
            cpu=cpu.name,
            ok=True,
            cycles=result.throughput_cycles,
            simulated_cycles=result.simulated_cycles,
            uops=result.uops,
            dispatch_width=result.dispatch_width,
            port_pressure=result.port_pressure,
            instructions=result.instruction_metrics,
            raw_output=result.raw_output,
        )
