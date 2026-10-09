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
    llvm_mca_features: tuple[str, ...] = ()

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


# x86-64 AMD
ZEN1 = CpuSpec("znver1", "amd", llvm_mca="znver1", notes="Zen1.")
ZEN2 = CpuSpec("znver2", "amd", llvm_mca="znver2", notes="Zen2.")
ZEN3 = CpuSpec("znver3", "amd", llvm_mca="znver3", notes="Zen3.")
ZEN4 = CpuSpec("znver4", "amd", llvm_mca="znver4", notes="Zen4.")
ZEN5 = CpuSpec("znver5", "amd", llvm_mca="znver5", notes="Zen5.")

# x86-64 Intel
SKYLAKE = CpuSpec(
    "skylake", "intel", osaca="SKX", uica="SKL", llvm_mca="skylake",
    notes="Intel Skylake.",
)
ICELAKE_SERVER = CpuSpec(
    "icelake-server", "intel", osaca="ICL", uica="ICL",
    llvm_mca="icelake-server", notes="Intel Ice Lake.",
)
ALDERLAKE = CpuSpec("alderlake", "intel", llvm_mca="alderlake", notes="Intel Alder Lake.")
COFFEE_LAKE = CpuSpec(
    "coffee-lake", "intel", uica="CFL", llvm_mca="skylake",
    notes="Intel Coffee Lake using LLVM's compatible Skylake scheduling model.",
)
ICE_LAKE = CpuSpec("ice-lake", "intel", uica="ICL", llvm_mca="icelake-client", notes="Intel Ice Lake client.")
SKYLAKE_X86_32 = CpuSpec(
    "skylake-x86-32",
    "x86_32",
    llvm_mca="skylake",
    notes="Skylake scheduling model in 32-bit x86 mode.",
)

# ARM / AArch64
NEOVERSE_N1 = CpuSpec("neoverse-n1", "arm", osaca="N1", llvm_mca="neoverse-n1", notes="Arm Neoverse N1.")
NEOVERSE_V1 = CpuSpec("neoverse-v1", "arm", llvm_mca="neoverse-v1", notes="Arm Neoverse V1.")
NEOVERSE_V2 = CpuSpec("neoverse-v2", "arm", osaca="V2", llvm_mca="neoverse-v2", notes="Arm Neoverse V2.")
THUNDERX2 = CpuSpec("thunderx2", "arm", osaca="TX2", llvm_mca="thunderx2t99", notes="ThunderX2.")
A64FX = CpuSpec("a64fx", "arm", osaca="A64FX", llvm_mca="a64fx", notes="Fujitsu A64FX.")
APPLE_M1 = CpuSpec("apple-m1", "arm", osaca="M1", llvm_mca="apple-m1", notes="Apple M1.")
CORTEX_A72 = CpuSpec("cortex-a72", "arm", osaca="A72", llvm_mca="cortex-a72", notes="Cortex-A72.")
CORTEX_A9 = CpuSpec(
    "cortex-a9-arm32",
    "arm32",
    llvm_mca="cortex-a9",
    notes="Arm Cortex-A9 32-bit scheduling model.",
)

# PowerPC
POWER9 = CpuSpec("power9", "ppc", llvm_mca="pwr9", notes="POWER9.")
POWER10 = CpuSpec("power10", "ppc", llvm_mca="pwr10", notes="POWER10.")
POWER9_32 = CpuSpec(
    "power9-32",
    "ppc32",
    llvm_mca="pwr9",
    notes="POWER9 scheduling model in 32-bit PowerPC mode.",
)

# s390x
Z15 = CpuSpec("z15", "s390x", llvm_mca="z15", notes="IBM z15.")
Z16 = CpuSpec("z16", "s390x", llvm_mca="z16", notes="IBM z16.")

# RISC-V 64-bit representatives
ROCKET_RV64 = CpuSpec(
    "rocket-rv64",
    "riscv",
    llvm_mca="rocket-rv64",
    notes="Rocket RV64 in-order model.",
    llvm_mca_features=("+m",),
)
SIFIVE_U74 = CpuSpec(
    "sifive-u74",
    "riscv",
    llvm_mca="sifive-u74",
    notes="SiFive U74 dual-issue in-order model.",
    llvm_mca_features=("+m",),
)
SIFIVE_P550 = CpuSpec(
    "sifive-p550",
    "riscv",
    llvm_mca="sifive-p550",
    notes="SiFive P550 out-of-order model.",
    llvm_mca_features=("+m",),
)

# RISC-V 32-bit representatives
ROCKET_RV32 = CpuSpec(
    "rocket-rv32",
    "riscv32",
    llvm_mca="rocket-rv32",
    notes="Rocket RV32 in-order model.",
    llvm_mca_features=("+m", "+a", "+c"),
)

# Generic models are intentionally named after the exact LLVM scheduling
# model. They provide static coverage where no vendor-specific model is
# available and must not be presented as measurements of a physical CPU.
LOONGARCH32_GENERIC = CpuSpec(
    "loongarch32-generic",
    "loongarch32",
    llvm_mca="generic-la32",
    notes="Generic LLVM LA32 scheduling model; not a physical CPU.",
)
LOONGARCH64_LA464 = CpuSpec(
    "loongarch64-la464",
    "loongarch64",
    llvm_mca="la464",
    notes="LoongArch LA464 LLVM scheduling model.",
)
MIPS32_GENERIC_R2 = CpuSpec(
    "mips32-generic-r2",
    "mips32",
    llvm_mca="mips32r2",
    notes="Generic MIPS32r2 LLVM scheduling model; not a physical CPU.",
)
MIPS64_GENERIC_R2 = CpuSpec(
    "mips64-generic-r2",
    "mips64",
    llvm_mca="mips64r2",
    notes="Generic MIPS64r2 LLVM scheduling model; not a physical CPU.",
)


CPUS: Dict[str, CpuSpec] = {
    spec.name: spec
    for spec in (
        ZEN1, ZEN2, ZEN3, ZEN4, ZEN5,
        SKYLAKE, ICELAKE_SERVER, ALDERLAKE, COFFEE_LAKE, ICE_LAKE,
        SKYLAKE_X86_32,
        NEOVERSE_N1, NEOVERSE_V1, NEOVERSE_V2, THUNDERX2, A64FX,
        APPLE_M1, CORTEX_A72, CORTEX_A9,
        POWER9, POWER10, POWER9_32, Z15, Z16,
        ROCKET_RV64, SIFIVE_U74, SIFIVE_P550,
        ROCKET_RV32, LOONGARCH32_GENERIC, LOONGARCH64_LA464,
        MIPS32_GENERIC_R2, MIPS64_GENERIC_R2,
    )
}

DEFAULT_MATRIX: List[str] = [
    "znver2", "znver3", "znver4", "znver5",
    "skylake", "icelake-server", "alderlake",
    "neoverse-n1", "neoverse-v1",
]

DEFAULT_X86_MATRIX: List[str] = [
    name for name in DEFAULT_MATRIX if CPUS[name].family in ("amd", "intel")
]

DEFAULT_BACKENDS: tuple[str, ...] = ANALYTICAL_BACKENDS


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
            cpu = get_cpu(tok)
            if cpu not in out:
                out.append(cpu)
    return out


def parse_backends(text: str) -> List[str]:
    """Parse a comma-separated backend list; error on unknown backends."""
    out = []
    for tok in text.split(","):
        tok = tok.strip().lower()
        if tok:
            if tok not in ALL_BACKENDS:
                raise KeyError(f"unknown backend '{tok}'. Known: {', '.join(ALL_BACKENDS)}")
            if tok not in out:
                out.append(tok)
    return out
