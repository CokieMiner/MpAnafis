"""Target architecture, Rust triple, and CPU-matrix policy."""

from __future__ import annotations

import platform
import re
from pathlib import Path
from typing import Iterable, List, Optional

from .models import CPUS, CpuSpec
from .types import ArchitectureFamily

_ARCHITECTURE_MARKERS = (
    ("loongarch64", ArchitectureFamily.LOONGARCH64),
    ("loongarch32", ArchitectureFamily.LOONGARCH32),
    ("aarch64", ArchitectureFamily.AARCH64),
    ("riscv64", ArchitectureFamily.RISCV64),
    ("riscv32", ArchitectureFamily.RISCV32),
    ("powerpc64", ArchitectureFamily.POWER64),
    ("ppc64", ArchitectureFamily.POWER64),
    ("powerpc", ArchitectureFamily.POWER32),
    ("s390x", ArchitectureFamily.S390X),
    ("mips64", ArchitectureFamily.MIPS64),
    ("mips", ArchitectureFamily.MIPS32),
    ("x86_64", ArchitectureFamily.X86_64),
    ("x86", ArchitectureFamily.X86_32),
    ("arm", ArchitectureFamily.ARM32),
)
_RUST_TARGET_MARKERS = (
    ("aarch64", "aarch64-unknown-linux-gnu"),
    ("loongarch64", "loongarch64-unknown-linux-gnu"),
    ("loongarch32", "loongarch32-unknown-none"),
    ("riscv64", "riscv64gc-unknown-linux-gnu"),
    ("riscv32", "riscv32imac-unknown-none-elf"),
    ("powerpc64", "powerpc64le-unknown-linux-gnu"),
    ("powerpc", "powerpc-unknown-linux-gnu"),
    ("s390x", "s390x-unknown-linux-gnu"),
    ("mips64", "mips64-unknown-linux-gnuabi64"),
    ("mips", "mips-unknown-linux-gnu"),
    ("arm", "armv7-unknown-linux-gnueabihf"),
    ("x86.rs", "i686-unknown-linux-gnu"),
)
_CPU_FAMILIES = {
    ArchitectureFamily.X86_64: {"amd", "intel"},
    ArchitectureFamily.X86_32: {"x86_32"},
    ArchitectureFamily.AARCH64: {"arm"},
    ArchitectureFamily.ARM32: {"arm32"},
    ArchitectureFamily.RISCV64: {"riscv"},
    ArchitectureFamily.RISCV32: {"riscv32"},
    ArchitectureFamily.POWER64: {"ppc"},
    ArchitectureFamily.POWER32: {"ppc32"},
    ArchitectureFamily.S390X: {"s390x"},
    ArchitectureFamily.MIPS64: {"mips64"},
    ArchitectureFamily.MIPS32: {"mips32"},
    ArchitectureFamily.LOONGARCH64: {"loongarch64"},
    ArchitectureFamily.LOONGARCH32: {"loongarch32"},
}


def architecture_for_path(path: Path) -> ArchitectureFamily:
    """Classify an architecture kernel path without foreign-ISA fallthrough."""
    for component in reversed(path.parts):
        for marker, architecture in _ARCHITECTURE_MARKERS:
            if re.search(rf"(?:^|[_\-.]){marker}(?:$|[_\-.])", component.lower()):
                return architecture
    return ArchitectureFamily.X86_64


def rust_target_for_path(path: Path) -> Optional[str]:
    """Return the Rust target needed to validate a foreign inline-asm block."""
    name = path.name.lower()
    return next((target for marker, target in _RUST_TARGET_MARKERS if marker in name), None)


def cpu_supports_architecture(cpu: CpuSpec, architecture: ArchitectureFamily) -> bool:
    """Return whether a logical CPU model belongs to an assembly ISA."""
    return cpu.family in _CPU_FAMILIES[architecture]


def compatible_cpus(
    cpus: Iterable[CpuSpec], architecture: ArchitectureFamily
) -> List[CpuSpec]:
    """Filter a CPU list to models compatible with an assembly ISA."""
    return [cpu for cpu in cpus if cpu_supports_architecture(cpu, architecture)]


def default_cpus_for_architecture(architecture: ArchitectureFamily) -> List[CpuSpec]:
    """Return every registered CPU model compatible with an assembly ISA."""
    return compatible_cpus(CPUS.values(), architecture)


def host_architecture() -> Optional[ArchitectureFamily]:
    """Return the native ISA reported by the current machine."""
    machine = platform.machine().lower()
    aliases = {
        "x86_64": ArchitectureFamily.X86_64,
        "amd64": ArchitectureFamily.X86_64,
        "i386": ArchitectureFamily.X86_32,
        "i686": ArchitectureFamily.X86_32,
        "aarch64": ArchitectureFamily.AARCH64,
        "arm64": ArchitectureFamily.AARCH64,
        "arm": ArchitectureFamily.ARM32,
        "armv7l": ArchitectureFamily.ARM32,
        "armv8l": ArchitectureFamily.ARM32,
        "riscv64": ArchitectureFamily.RISCV64,
        "riscv32": ArchitectureFamily.RISCV32,
        "ppc64le": ArchitectureFamily.POWER64,
        "ppc64": ArchitectureFamily.POWER64,
        "ppc": ArchitectureFamily.POWER32,
        "powerpc": ArchitectureFamily.POWER32,
        "s390x": ArchitectureFamily.S390X,
        "mips64": ArchitectureFamily.MIPS64,
        "mips64el": ArchitectureFamily.MIPS64,
        "mips": ArchitectureFamily.MIPS32,
        "mipsel": ArchitectureFamily.MIPS32,
        "loongarch64": ArchitectureFamily.LOONGARCH64,
        "loongarch32": ArchitectureFamily.LOONGARCH32,
    }
    return aliases.get(machine)


__all__ = [
    "architecture_for_path",
    "compatible_cpus",
    "cpu_supports_architecture",
    "default_cpus_for_architecture",
    "host_architecture",
    "rust_target_for_path",
]
