"""Architecture semantics shared by static assembly analyses."""

from __future__ import annotations

from dataclasses import dataclass

from .types import ArchitectureFamily


@dataclass(frozen=True)
class TargetSemantics:
    """Static properties that analyses may safely assume for one target ISA."""

    architecture: ArchitectureFamily
    limb_bytes: int
    allocatable_gprs: int
    cache_line_bytes: int = 64
    has_x86_uop_cache_model: bool = False
    has_schedule_model: bool = False


_TARGET_SEMANTICS = {
    ArchitectureFamily.X86_64: TargetSemantics(
        ArchitectureFamily.X86_64,
        8,
        14,
        has_x86_uop_cache_model=True,
        has_schedule_model=True,
    ),
    ArchitectureFamily.X86_32: TargetSemantics(
        ArchitectureFamily.X86_32,
        4,
        7,
        has_x86_uop_cache_model=True,
        has_schedule_model=True,
    ),
    ArchitectureFamily.AARCH64: TargetSemantics(ArchitectureFamily.AARCH64, 8, 30),
    ArchitectureFamily.ARM32: TargetSemantics(ArchitectureFamily.ARM32, 4, 14),
    ArchitectureFamily.RISCV64: TargetSemantics(
        ArchitectureFamily.RISCV64,
        8,
        31,
    ),
    ArchitectureFamily.RISCV32: TargetSemantics(
        ArchitectureFamily.RISCV32,
        4,
        31,
    ),
    ArchitectureFamily.POWER64: TargetSemantics(ArchitectureFamily.POWER64, 8, 32),
    ArchitectureFamily.POWER32: TargetSemantics(ArchitectureFamily.POWER32, 4, 32),
    ArchitectureFamily.S390X: TargetSemantics(ArchitectureFamily.S390X, 8, 16),
    ArchitectureFamily.MIPS64: TargetSemantics(ArchitectureFamily.MIPS64, 8, 31),
    ArchitectureFamily.MIPS32: TargetSemantics(ArchitectureFamily.MIPS32, 4, 31),
    ArchitectureFamily.LOONGARCH64: TargetSemantics(ArchitectureFamily.LOONGARCH64, 8, 31),
    ArchitectureFamily.LOONGARCH32: TargetSemantics(ArchitectureFamily.LOONGARCH32, 4, 31),
}


def semantics_for(architecture: ArchitectureFamily) -> TargetSemantics:
    """Return the single source of static target assumptions for an ISA."""
    return _TARGET_SEMANTICS[architecture]


__all__ = ["TargetSemantics", "semantics_for"]
