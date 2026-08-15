"""Abstract base class and normalized report models for assembly analysis backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .models import CpuSpec


@dataclass
class InstrMetrics:
    """Per-instruction metrics where available."""
    line: str
    latency: Optional[float] = None
    throughput: Optional[float] = None
    uops: Optional[float] = None
    ports: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    raw: Optional[str] = None


@dataclass
class KernelReport:
    """Normalized result of analyzing one kernel on one CPU by one backend."""
    backend: str
    cpu: str
    ok: bool = True
    note: str = ""
    cycles: Optional[float] = None
    uops: Optional[float] = None
    port_pressure: Dict[str, float] = field(default_factory=dict)
    instructions: List[InstrMetrics] = field(default_factory=list)
    raw_output: str = ""

    def cost(self) -> Optional[float]:
        """Return the primary cycle latency cost metric for this report."""
        return self.cycles


class Analyzer(ABC):
    """Abstract base class for static and dynamic assembly analyzers."""

    name: str

    @abstractmethod
    def available(self) -> bool:
        """Return True if this analyzer backend is available in the environment."""
        raise NotImplementedError

    @abstractmethod
    def supports(self, cpu: CpuSpec) -> bool:
        """Return True if this analyzer supports the given CPU specification."""
        raise NotImplementedError

    @abstractmethod
    def analyze(self, asm_code: str, cpu: CpuSpec, iterations: int = 200) -> Optional[float]:
        """Analyze an assembly block on the given CPU model, returning cycles per iteration."""
        raise NotImplementedError

    def analyze_report(self, asm_code: str, cpu: CpuSpec, iterations: int = 200) -> KernelReport:
        """Analyze an assembly block on the given CPU model, returning a structured KernelReport."""
        if not self.available():
            return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note="Backend not installed or available")
        if not self.supports(cpu):
            return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note=f"CPU '{cpu.name}' not supported by backend")
        cyc = self.analyze(asm_code, cpu, iterations=iterations)
        if cyc is None:
            return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note="Simulator produced no cycle output")
        return KernelReport(backend=self.name, cpu=cpu.name, ok=True, cycles=cyc)
