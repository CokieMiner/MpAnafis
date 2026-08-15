#!/usr/bin/env python3
"""CPU model registry for the assembly analyzer suite.

Maps logical CPU names to per-backend model identifiers for llvm-mca,
OSACA, uiCA, nanoBench, and Linux perf.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

ANALYTICAL_BACKENDS = ("llvm-mca", "osaca", "uica")
ALL_BACKENDS = ANALYTICAL_BACKENDS + ("nanobench", "perf")


@dataclass(frozen=True)
class CpuSpec:
    """Mapping from a logical CPU name to each backend's model identifier."""

    name: str
    family: str
    osaca: Optional[str] = None
    uica: Optional[str] = None
    llvm_mca: Optional[str] = None
    nanobench: Optional[str] = None
    perf: Optional[str] = None
    data_available: bool = False
    notes: str = ""

    def model_for(self, backend: str) -> Optional[str]:
        """Return the model id this CPU maps to for ``backend`` (or None)."""
        if backend == "osaca":
            return self.osaca
        if backend == "uica":
            return self.uica
        if backend == "llvm-mca":
            return self.llvm_mca
        if backend == "nanobench":
            return self.nanobench
        if backend == "perf":
            return self.perf
        raise KeyError(backend)

    def supports(self, backend: str) -> bool:
        """True if this CPU has a model id for ``backend``."""
        return self.model_for(backend) is not None


def _cpu(name: str, family: str, osaca=None, uica=None, llvm_mca=None,
         nanobench=None, perf=None, data_available=False, notes=""):
    return CpuSpec(name, family, osaca, uica, llvm_mca, nanobench, perf,
                   data_available, notes)


# x86-64 AMD
ZEN1 = _cpu("znver1", "amd", osaca="ZEN1", llvm_mca="znver1", notes="Zen1.")
ZEN2 = _cpu("znver2", "amd", osaca="ZEN2", llvm_mca="znver2", notes="Zen2.")
ZEN3 = _cpu("znver3", "amd", osaca="ZEN3", llvm_mca="znver3", notes="Zen3.")
ZEN4 = _cpu("znver4", "amd", osaca="ZEN4", llvm_mca="znver4", notes="Zen4.")
ZEN5 = _cpu("znver5", "amd", osaca="ZEN5", llvm_mca="znver5", notes="Zen5.")

# x86-64 Intel
SKYLAKE = _cpu("skylake", "intel", osaca="SKX", uica="SKL", llvm_mca="skylake", notes="Intel Skylake.")
ICELAKE_SERVER = _cpu("icelake-server", "intel", osaca="ICL", uica="ICL", llvm_mca="icelake-server", notes="Intel Ice Lake.")
ALDERLAKE = _cpu("alderlake", "intel", uica="ADL", llvm_mca="alderlake", notes="Intel Alder Lake.")
COFFEE_LAKE = _cpu("coffee-lake", "intel", uica="CFL", llvm_mca="coffee-lake", notes="Intel Coffee Lake.")
ICE_LAKE = _cpu("ice-lake", "intel", uica="ICL", llvm_mca="icelake-client", notes="Intel Ice Lake client.")

# ARM / AArch64
NEOVERSE_N1 = _cpu("neoverse-n1", "arm", osaca="N1", llvm_mca="neoverse-n1", notes="Arm Neoverse N1.")
NEOVERSE_V1 = _cpu("neoverse-v1", "arm", osaca="V1", llvm_mca="neoverse-v1", notes="Arm Neoverse V1.")
NEOVERSE_V2 = _cpu("neoverse-v2", "arm", osaca="V2", llvm_mca="neoverse-v2", notes="Arm Neoverse V2.")
THUNDERX2 = _cpu("thunderx2", "arm", osaca="TX2", llvm_mca="thunderx2t99", notes="ThunderX2.")
A64FX = _cpu("a64fx", "arm", osaca="A64FX", llvm_mca="a64fx", notes="Fujitsu A64FX.")
APPLE_M1 = _cpu("apple-m1", "arm", osaca="M1", llvm_mca="apple-m1", notes="Apple M1.")
CORTEX_A72 = _cpu("cortex-a72", "arm", osaca="A72", llvm_mca="cortex-a72", notes="Cortex-A72.")

# PowerPC
POWER9 = _cpu("power9", "ppc", llvm_mca="pwr9", notes="POWER9.")
POWER10 = _cpu("power10", "ppc", llvm_mca="pwr10", notes="POWER10.")

# s390x
Z15 = _cpu("z15", "s390x", llvm_mca="z15", notes="IBM z15.")
Z16 = _cpu("z16", "s390x", llvm_mca="z16", notes="IBM z16.")

# RISC-V
RISCV64 = _cpu("riscv64", "riscv", llvm_mca="generic-rv64", notes="Generic RV64GC.")


CPUS: Dict[str, CpuSpec] = {
    spec.name: spec
    for spec in (
        ZEN1, ZEN2, ZEN3, ZEN4, ZEN5,
        SKYLAKE, ICELAKE_SERVER, ALDERLAKE, COFFEE_LAKE, ICE_LAKE,
        NEOVERSE_N1, NEOVERSE_V1, NEOVERSE_V2, THUNDERX2, A64FX,
        APPLE_M1, CORTEX_A72,
        POWER9, POWER10, Z15, Z16, RISCV64,
    )
}

DEFAULT_MATRIX: List[str] = [
    "znver2", "znver3", "znver4", "znver5",
    "skylake", "icelake-server", "alderlake",
    "neoverse-n1", "neoverse-v1",
]

DEFAULT_BACKENDS: tuple[str, ...] = ("llvm-mca", "osaca", "uica", "nanobench")


def get_cpu(name: str) -> CpuSpec:
    """Look up a logical CPU by name (raises KeyError if unknown)."""
    try:
        return CPUS[name]
    except KeyError:
        raise KeyError(f"unknown CPU '{name}'. Known: {', '.join(sorted(CPUS))}") from None


def parse_cpus(text: str) -> List[CpuSpec]:
    """Parse a comma-separated CPU list (logical names) into CpuSpec rows."""
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if tok:
            out.append(get_cpu(tok))
    return out


def parse_backends(text: str) -> List[str]:
    """Parse a comma-separated backend list; error on unknown backends."""
    out = []
    for tok in text.split(","):
        tok = tok.strip().lower()
        if tok:
            if tok not in ALL_BACKENDS:
                raise KeyError(f"unknown backend '{tok}'. Known: {', '.join(ALL_BACKENDS)}")
            out.append(tok)
    return out
