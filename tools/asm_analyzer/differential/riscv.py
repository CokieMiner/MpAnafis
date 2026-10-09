"""Native RISC-V differential verifiers."""

from __future__ import annotations

import platform
from typing import List

from ..types import ArchitectureFamily
from .common import compile_and_run, driver_source, native_register_plan, prepare_body_labels

_GPRS = tuple(f"x{index}" for index in range(32))


def diff_test_riscv64(bodies: List[str], cases: int, use_wsl: bool) -> List[bool]:
    """Differentially execute RV64 candidates on an RV64 host."""
    if platform.machine().lower() != "riscv64":
        raise RuntimeError("RISC-V 64 differential execution requires a riscv64 host")
    return _diff_test_riscv(bodies, cases, use_wsl, ArchitectureFamily.RISCV64, 64)


def diff_test_riscv32(bodies: List[str], cases: int, use_wsl: bool) -> List[bool]:
    """Differentially execute RV32 candidates on an RV32 host."""
    if platform.machine().lower() != "riscv32":
        raise RuntimeError("RISC-V 32 differential execution requires a riscv32 host")
    return _diff_test_riscv(bodies, cases, use_wsl, ArchitectureFamily.RISCV32, 32)


def riscv64_wrapper(symbol: str, body: str, r_in: str, r_out: str) -> str:
    """Wrap an RV64 kernel and capture integer-register state."""
    return riscv_wrapper(symbol, body, r_in, r_out, 8)


def riscv_wrapper(
    symbol: str,
    body: str,
    r_in: str,
    r_out: str,
    word_bytes: int,
) -> str:
    """Wrap an RV32 or RV64 kernel while preserving its platform ABI."""
    body, exits = prepare_body_labels(body, symbol)
    saved = ("x1", "x3", "x4", "x8", "x9", *(f"x{index}" for index in range(18, 28)))
    frame_bytes = 128 if word_bytes == 8 else 64
    load = "ld" if word_bytes == 8 else "lw"
    store = "sd" if word_bytes == 8 else "sw"
    lines = [
        ".text", f".globl {symbol}", f"{symbol}:",
        f"    addi x2, x2, -{frame_bytes}",
    ]
    for index, register in enumerate(saved):
        lines.append(f"    {store} {register}, {index * word_bytes}(x2)")
    lines.extend((f"    mv {r_in}, x10", f"    mv {r_out}, x11"))
    for index, register in enumerate(_GPRS):
        if register not in ("x0", "x2", r_in, r_out):
            lines.append(f"    {load} {register}, {index * word_bytes}({r_in})")
    lines.extend(f"    {line}" if line.strip() else "" for line in body.splitlines())
    lines.extend(f"{label}:" for label in exits)
    for index, register in enumerate(_GPRS):
        if register not in ("x2", r_in, r_out):
            lines.append(f"    {store} {register}, {index * word_bytes}({r_out})")
    for register in ("x2", r_in, r_out):
        lines.append(f"    {store} x0, {_GPRS.index(register) * word_bytes}({r_out})")
    for index, register in reversed(list(enumerate(saved))):
        lines.append(f"    {load} {register}, {index * word_bytes}(x2)")
    lines.extend((f"    addi x2, x2, {frame_bytes}", "    ret"))
    return "\n".join(lines) + "\n"


def _diff_test_riscv(
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
        {"x0", "x1", "x2"},
    )
    if "x2" in used:
        raise ValueError("RISC-V differential execution does not accept SP operands")
    word_bytes = word_bits // 8
    wrappers = "".join(
        riscv_wrapper(f"k_{index}", body, r_in, r_out, word_bytes)
        for index, body in enumerate(bodies)
    )
    driver = driver_source(
        len(bodies), 32, 32,
        (_GPRS.index(r_in), _GPRS.index(r_out)),
        pointers, cases, word_bits,
    )
    return compile_and_run(wrappers, driver, len(bodies), cases, use_wsl)


__all__ = [
    "diff_test_riscv32",
    "diff_test_riscv64",
    "riscv64_wrapper",
    "riscv_wrapper",
]
