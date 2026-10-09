"""Native MIPS differential verifiers."""

from __future__ import annotations

import platform
from typing import List

from ..types import ArchitectureFamily
from .common import compile_and_run, driver_source, native_register_plan, prepare_body_labels

_GPRS = tuple(f"r{index}" for index in range(32))


def diff_test_mips64(bodies: List[str], cases: int, use_wsl: bool) -> List[bool]:
    """Differentially execute MIPS64 candidates on a MIPS64 host."""
    if platform.machine().lower() not in ("mips64", "mips64el"):
        raise RuntimeError("MIPS64 differential execution requires a MIPS64 host")
    return _diff_test_mips(bodies, cases, use_wsl, ArchitectureFamily.MIPS64, 64)


def diff_test_mips32(bodies: List[str], cases: int, use_wsl: bool) -> List[bool]:
    """Differentially execute MIPS32 candidates on a MIPS32 host."""
    if platform.machine().lower() not in ("mips", "mipsel"):
        raise RuntimeError("MIPS32 differential execution requires a MIPS32 host")
    return _diff_test_mips(bodies, cases, use_wsl, ArchitectureFamily.MIPS32, 32)


def mips_wrapper(
    symbol: str,
    body: str,
    r_in: str,
    r_out: str,
    word_bytes: int,
) -> str:
    """Wrap a MIPS kernel and capture GPR plus HI/LO state."""
    body, exits = prepare_body_labels(body, symbol)
    is_64_bit = word_bytes == 8
    add = "daddiu" if is_64_bit else "addiu"
    load = "ld" if is_64_bit else "lw"
    store = "sd" if is_64_bit else "sw"
    frame_bytes = 112 if is_64_bit else 64
    saved = (*(f"r{index}" for index in range(16, 24)), "r28", "r30", "r31")
    lines = [
        ".text", ".set noreorder", f".globl {symbol}", f"{symbol}:",
        f"    {add} $29, $29, -{frame_bytes}",
    ]
    for index, register in enumerate(saved):
        lines.append(f"    {store} ${register[1:]}, {index * word_bytes}($29)")
    lines.extend(
        (
            f"    move ${r_in[1:]}, $4", f"    move ${r_out[1:]}, $5",
            "    mtlo $0", "    mthi $0",
        ),
    )
    for index, register in enumerate(_GPRS):
        if register not in ("r0", "r29", r_in, r_out):
            lines.append(f"    {load} ${register[1:]}, {index * word_bytes}(${r_in[1:]})")
    lines.extend(f"    {line}" if line.strip() else "" for line in body.splitlines())
    lines.extend(f"{label}:" for label in exits)
    for index, register in enumerate(_GPRS):
        if register not in ("r29", r_in, r_out):
            lines.append(f"    {store} ${register[1:]}, {index * word_bytes}(${r_out[1:]})")
    lines.extend(
        (
            f"    mflo ${r_in[1:]}",
            f"    {store} ${r_in[1:]}, {32 * word_bytes}(${r_out[1:]})",
            f"    mfhi ${r_in[1:]}",
            f"    {store} ${r_in[1:]}, {33 * word_bytes}(${r_out[1:]})",
            f"    move ${r_in[1:]}, $0",
        ),
    )
    for register in ("r29", r_in, r_out):
        lines.append(
            f"    {store} ${r_in[1:]}, {_GPRS.index(register) * word_bytes}(${r_out[1:]})",
        )
    for index, register in reversed(list(enumerate(saved))):
        lines.append(f"    {load} ${register[1:]}, {index * word_bytes}($29)")
    lines.extend(
        (
            f"    {add} $29, $29, {frame_bytes}",
            "    jr $31", "    nop", ".set reorder",
        ),
    )
    return "\n".join(lines) + "\n"


def _diff_test_mips(
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
        {"r0", "r1", "r26", "r27", "r28", "r29", "r31"},
    )
    if "r29" in used:
        raise ValueError("MIPS differential execution does not accept SP operands")
    word_bytes = word_bits // 8
    wrappers = "".join(
        mips_wrapper(f"k_{index}", body, r_in, r_out, word_bytes)
        for index, body in enumerate(bodies)
    )
    driver = driver_source(
        len(bodies), 32, 34,
        (_GPRS.index(r_in), _GPRS.index(r_out)),
        pointers, cases, word_bits,
    )
    return compile_and_run(wrappers, driver, len(bodies), cases, use_wsl)


__all__ = ["diff_test_mips32", "diff_test_mips64", "mips_wrapper"]
