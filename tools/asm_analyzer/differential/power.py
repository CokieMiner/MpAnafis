"""Native PowerPC differential verifiers."""

from __future__ import annotations

import platform
from typing import List

from ..types import ArchitectureFamily
from .common import compile_and_run, driver_source, native_register_plan, prepare_body_labels

_GPRS = tuple(f"r{index}" for index in range(32))


def diff_test_power64(bodies: List[str], cases: int, use_wsl: bool) -> List[bool]:
    """Differentially execute POWER64 candidates on a ppc64le host."""
    if platform.machine().lower() != "ppc64le":
        raise RuntimeError("POWER64 differential execution requires a ppc64le host")
    return _diff_test_power(bodies, cases, use_wsl, ArchitectureFamily.POWER64, 64)


def diff_test_power32(bodies: List[str], cases: int, use_wsl: bool) -> List[bool]:
    """Differentially execute PowerPC32 candidates on a 32-bit PowerPC host."""
    if platform.machine().lower() not in ("ppc", "powerpc"):
        raise RuntimeError("PowerPC32 differential execution requires a 32-bit PowerPC host")
    return _diff_test_power(bodies, cases, use_wsl, ArchitectureFamily.POWER32, 32)


def power_wrapper(
    symbol: str,
    body: str,
    r_in: str,
    r_out: str,
    word_bytes: int,
) -> str:
    """Wrap a PowerPC kernel and capture GPR, CR, and XER state."""
    body, exits = prepare_body_labels(body, symbol)
    is_64_bit = word_bytes == 8
    load = "ld" if is_64_bit else "lwz"
    store = "std" if is_64_bit else "stw"
    frame_bytes = 176 if is_64_bit else 96
    saved = ("r2", *(f"r{index}" for index in range(13, 32)))
    lines = [".text"]
    if is_64_bit:
        lines.append(".abiversion 2")
    lines.extend(
        (
            f".globl {symbol}", f".type {symbol}, @function", f"{symbol}:",
            "    mflr 0",
            f"    {'stdu' if is_64_bit else 'stwu'} 1, -{frame_bytes}(1)",
            f"    {store} 0, {8 if is_64_bit else 4}(1)",
        ),
    )
    save_base = 16 if is_64_bit else 8
    for index, register in enumerate(saved):
        lines.append(f"    {store} {register[1:]}, {save_base + index * word_bytes}(1)")
    lines.extend(
        (
            f"    mr {r_in[1:]}, 3", f"    mr {r_out[1:]}, 4",
            "    li 0, 0", "    mtxer 0", "    mtcr 0",
        ),
    )
    for index, register in enumerate(_GPRS):
        if register not in ("r1", r_in, r_out):
            lines.append(f"    {load} {register[1:]}, {index * word_bytes}({r_in[1:]})")
    lines.extend(f"    {line}" if line.strip() else "" for line in body.splitlines())
    lines.extend(f"{label}:" for label in exits)
    for index, register in enumerate(_GPRS):
        if register not in ("r1", r_in, r_out):
            lines.append(f"    {store} {register[1:]}, {index * word_bytes}({r_out[1:]})")
    lines.extend(
        (
            f"    mfcr {r_in[1:]}",
            f"    {store} {r_in[1:]}, {32 * word_bytes}({r_out[1:]})",
            f"    mfxer {r_in[1:]}",
            f"    {store} {r_in[1:]}, {33 * word_bytes}({r_out[1:]})",
            f"    li {r_in[1:]}, 0",
        ),
    )
    for register in ("r1", r_in, r_out):
        lines.append(f"    {store} {r_in[1:]}, {_GPRS.index(register) * word_bytes}({r_out[1:]})")
    for index, register in reversed(list(enumerate(saved))):
        lines.append(f"    {load} {register[1:]}, {save_base + index * word_bytes}(1)")
    lines.extend(
        (
            f"    {load} 0, {8 if is_64_bit else 4}(1)",
            "    mtlr 0", f"    addi 1, 1, {frame_bytes}", "    blr",
        ),
    )
    return "\n".join(lines) + "\n"


def power64_wrapper(symbol: str, body: str, r_in: str, r_out: str) -> str:
    """Wrap a POWER64 kernel."""
    return power_wrapper(symbol, body, r_in, r_out, 8)


def _diff_test_power(
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
        {"r0", "r1", "r2", "r13"},
    )
    if "r1" in used:
        raise ValueError("PowerPC differential execution does not accept SP operands")
    word_bytes = word_bits // 8
    wrappers = "".join(
        power_wrapper(f"k_{index}", body, r_in, r_out, word_bytes)
        for index, body in enumerate(bodies)
    )
    driver = driver_source(
        len(bodies), 32, 34,
        (_GPRS.index(r_in), _GPRS.index(r_out)),
        pointers, cases, word_bits,
    )
    return compile_and_run(wrappers, driver, len(bodies), cases, use_wsl)


__all__ = [
    "diff_test_power32",
    "diff_test_power64",
    "power64_wrapper",
    "power_wrapper",
]
