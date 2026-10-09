"""Native LoongArch differential verifiers."""

from __future__ import annotations

import platform
from typing import List

from ..types import ArchitectureFamily
from .common import compile_and_run, driver_source, native_register_plan, prepare_body_labels

_GPRS = tuple(f"r{index}" for index in range(32))


def diff_test_loongarch64(bodies: List[str], cases: int, use_wsl: bool) -> List[bool]:
    """Differentially execute LoongArch64 candidates on a LoongArch64 host."""
    if platform.machine().lower() != "loongarch64":
        raise RuntimeError(
            "LoongArch64 differential execution requires a LoongArch64 host",
        )
    return _diff_test_loongarch(
        bodies, cases, use_wsl, ArchitectureFamily.LOONGARCH64, 64,
    )


def diff_test_loongarch32(bodies: List[str], cases: int, use_wsl: bool) -> List[bool]:
    """Differentially execute LoongArch32 candidates on a LoongArch32 host."""
    if platform.machine().lower() != "loongarch32":
        raise RuntimeError(
            "LoongArch32 differential execution requires a LoongArch32 host",
        )
    return _diff_test_loongarch(
        bodies, cases, use_wsl, ArchitectureFamily.LOONGARCH32, 32,
    )


def loongarch_wrapper(
    symbol: str,
    body: str,
    r_in: str,
    r_out: str,
    word_bytes: int,
) -> str:
    """Wrap a LoongArch kernel and capture integer-register state."""
    body, exits = prepare_body_labels(body, symbol)
    suffix = "d" if word_bytes == 8 else "w"
    frame_bytes = 112 if word_bytes == 8 else 64
    saved = ("r1", "r2", "r21", "r22", *(f"r{index}" for index in range(23, 32)))
    lines = [
        ".text", f".globl {symbol}", f".type {symbol}, @function", f"{symbol}:",
        f"    addi.{suffix} $r3, $r3, -{frame_bytes}",
    ]
    for index, register in enumerate(saved):
        lines.append(f"    st.{suffix} ${register}, $r3, {index * word_bytes}")
    lines.extend(
        (
            f"    or ${r_in}, $r4, $r0",
            f"    or ${r_out}, $r5, $r0",
        ),
    )
    for index, register in enumerate(_GPRS):
        if register not in ("r0", "r3", r_in, r_out):
            lines.append(f"    ld.{suffix} ${register}, ${r_in}, {index * word_bytes}")
    lines.extend(f"    {line}" if line.strip() else "" for line in body.splitlines())
    lines.extend(f"{label}:" for label in exits)
    for index, register in enumerate(_GPRS):
        if register not in ("r3", r_in, r_out):
            lines.append(f"    st.{suffix} ${register}, ${r_out}, {index * word_bytes}")
    for register in ("r3", r_in, r_out):
        lines.append(f"    st.{suffix} $r0, ${r_out}, {_GPRS.index(register) * word_bytes}")
    for index, register in reversed(list(enumerate(saved))):
        lines.append(f"    ld.{suffix} ${register}, $r3, {index * word_bytes}")
    lines.extend(
        (
            f"    addi.{suffix} $r3, $r3, {frame_bytes}",
            "    jirl $r0, $r1, 0",
        ),
    )
    return "\n".join(lines) + "\n"


def _diff_test_loongarch(
    bodies: List[str],
    cases: int,
    use_wsl: bool,
    architecture: ArchitectureFamily,
    word_bits: int,
) -> List[bool]:
    r_in, r_out, pointers, used = native_register_plan(
        bodies,
        architecture,
        _GPRS,
        {"r0", "r1", "r2", "r3", "r21"},
    )
    if "r3" in used:
        raise ValueError("LoongArch differential execution does not accept SP operands")
    word_bytes = word_bits // 8
    wrappers = "".join(
        loongarch_wrapper(f"k_{index}", body, r_in, r_out, word_bytes)
        for index, body in enumerate(bodies)
    )
    driver = driver_source(
        len(bodies), 32, 32,
        (_GPRS.index(r_in), _GPRS.index(r_out)),
        pointers, cases, word_bits,
    )
    return compile_and_run(wrappers, driver, len(bodies), cases, use_wsl)


__all__ = [
    "diff_test_loongarch32",
    "diff_test_loongarch64",
    "loongarch_wrapper",
]
