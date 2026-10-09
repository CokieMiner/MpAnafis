#!/usr/bin/env python3
"""Dispatch compiled differential checks to architecture-specific verifiers."""

from __future__ import annotations

from typing import Callable, Dict, List

from .differential.arm import diff_test_aarch64, diff_test_arm32
from .differential.loongarch import diff_test_loongarch32, diff_test_loongarch64
from .differential.mips import diff_test_mips32, diff_test_mips64
from .differential.power import diff_test_power32, diff_test_power64
from .differential.riscv import diff_test_riscv32, diff_test_riscv64
from .differential.s390x import diff_test_s390x
from .differential.x86 import diff_test_x86_32, diff_test_x86_64
from .types import ArchitectureFamily

DifferentialVerifier = Callable[[List[str], int, bool], List[bool]]

_VERIFIERS: Dict[ArchitectureFamily, DifferentialVerifier] = {
    ArchitectureFamily.AARCH64: diff_test_aarch64,
    ArchitectureFamily.ARM32: diff_test_arm32,
    ArchitectureFamily.LOONGARCH32: diff_test_loongarch32,
    ArchitectureFamily.LOONGARCH64: diff_test_loongarch64,
    ArchitectureFamily.MIPS32: diff_test_mips32,
    ArchitectureFamily.MIPS64: diff_test_mips64,
    ArchitectureFamily.POWER32: diff_test_power32,
    ArchitectureFamily.POWER64: diff_test_power64,
    ArchitectureFamily.RISCV32: diff_test_riscv32,
    ArchitectureFamily.RISCV64: diff_test_riscv64,
    ArchitectureFamily.S390X: diff_test_s390x,
    ArchitectureFamily.X86_32: diff_test_x86_32,
    ArchitectureFamily.X86_64: diff_test_x86_64,
}


def diff_test_variants(
    bodies: List[str],
    cases: int = 150,
    use_wsl: bool = False,
    architecture: ArchitectureFamily = ArchitectureFamily.X86_64,
) -> List[bool]:
    """Compile bodies[0] and every candidate, then compare observable state."""
    if cases < 1:
        raise ValueError("differential case count must be positive")
    try:
        verifier = _VERIFIERS[architecture]
    except KeyError as error:
        raise RuntimeError(
            f"native differential execution is unavailable for {architecture.value}",
        ) from error
    return verifier(bodies, cases, use_wsl)


def supports_native_differential(architecture: ArchitectureFamily) -> bool:
    """Return whether a same-architecture native executable verifier exists."""
    return architecture in _VERIFIERS


__all__ = ["diff_test_variants", "supports_native_differential"]
