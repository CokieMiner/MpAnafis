"""Native AArch64 and Arm32 differential verifiers."""

from __future__ import annotations

import platform
from typing import List

from ..types import ArchitectureFamily
from .common import compile_and_run, driver_source, native_register_plan, prepare_body_labels

_AARCH64_GPRS = tuple(f"x{index}" for index in range(31))
_ARM32_GPRS = tuple(f"r{index}" for index in range(16))


def diff_test_aarch64(bodies: List[str], cases: int, use_wsl: bool) -> List[bool]:
    """Differentially execute AArch64 candidates on an AArch64 host."""
    if platform.machine().lower() not in ("aarch64", "arm64"):
        raise RuntimeError("AArch64 differential execution requires an AArch64 host")
    r_in, r_out, pointers, used = native_register_plan(
        bodies,
        ArchitectureFamily.AARCH64,
        _AARCH64_GPRS,
        {"x18", "x29", "x30"},
    )
    if "sp" in used or any(register.startswith("v") for register in used):
        raise ValueError(
            "AArch64 differential execution supports scalar GPR kernels without SP operands",
        )
    wrappers = "".join(
        aarch64_wrapper(f"k_{index}", body, r_in, r_out)
        for index, body in enumerate(bodies)
    )
    driver = driver_source(
        len(bodies), 31, 32,
        (_AARCH64_GPRS.index(r_in), _AARCH64_GPRS.index(r_out)),
        pointers, cases, 64,
    )
    return compile_and_run(wrappers, driver, len(bodies), cases, use_wsl)


def diff_test_arm32(bodies: List[str], cases: int, use_wsl: bool) -> List[bool]:
    """Differentially execute Arm32 candidates on a 32-bit Arm host."""
    if platform.machine().lower() not in ("arm", "armv7l", "armv8l"):
        raise RuntimeError("Arm32 differential execution requires a 32-bit Arm host")
    r_in, r_out, pointers, used = native_register_plan(
        bodies,
        ArchitectureFamily.ARM32,
        _ARM32_GPRS,
        {"r2", "r13", "r14", "r15"},
    )
    if used.intersection({"r13", "r15"}):
        raise ValueError("Arm32 differential execution does not accept SP or PC operands")
    wrappers = "".join(
        arm32_wrapper(f"k_{index}", body, r_in, r_out)
        for index, body in enumerate(bodies)
    )
    driver = driver_source(
        len(bodies), 16, 17,
        (_ARM32_GPRS.index(r_in), _ARM32_GPRS.index(r_out)),
        pointers, cases, 32,
    )
    return compile_and_run(wrappers, driver, len(bodies), cases, use_wsl)


def aarch64_wrapper(symbol: str, body: str, r_in: str, r_out: str) -> str:
    """Wrap a scalar AArch64 kernel and capture GPR/NZCV state."""
    body, exits = prepare_body_labels(body, symbol)
    lines = [".text", f".globl {symbol}", f"{symbol}:", "    sub sp, sp, #112"]
    for offset, left in enumerate(range(18, 30, 2)):
        lines.append(f"    stp x{left}, x{left + 1}, [sp, #{offset * 16}]")
    lines.extend(("    str x30, [sp, #96]", f"    mov {r_in}, x0", f"    mov {r_out}, x1"))
    for index, register in enumerate(_AARCH64_GPRS):
        if register not in (r_in, r_out):
            lines.append(f"    ldr {register}, [{r_in}, #{index * 8}]")
    lines.append("    msr nzcv, xzr")
    lines.extend(f"    {line}" if line.strip() else "" for line in body.splitlines())
    lines.extend(f"{label}:" for label in exits)
    for index, register in enumerate(_AARCH64_GPRS):
        if register not in (r_in, r_out):
            lines.append(f"    str {register}, [{r_out}, #{index * 8}]")
    lines.extend(
        (
            f"    str xzr, [{r_out}, #{_AARCH64_GPRS.index(r_in) * 8}]",
            f"    str xzr, [{r_out}, #{_AARCH64_GPRS.index(r_out) * 8}]",
            f"    mrs {r_in}, nzcv",
            f"    str {r_in}, [{r_out}, #248]",
            "    ldr x30, [sp, #96]",
        ),
    )
    for offset, left in reversed(list(enumerate(range(18, 30, 2)))):
        lines.append(f"    ldp x{left}, x{left + 1}, [sp, #{offset * 16}]")
    lines.extend(("    add sp, sp, #112", "    ret"))
    return "\n".join(lines) + "\n"


def arm32_wrapper(symbol: str, body: str, r_in: str, r_out: str) -> str:
    """Wrap an Arm32 kernel and capture GPR/APSR state."""
    body, exits = prepare_body_labels(body, symbol)
    lines = [
        ".syntax unified", ".text", f".globl {symbol}",
        f".type {symbol}, %function", f"{symbol}:",
        "    sub sp, sp, #40", "    stmia sp, {r4-r11, lr}",
        f"    mov {r_in}, r0", f"    mov {r_out}, r1",
        "    mov r2, #0", "    msr APSR_nzcvq, r2",
    ]
    for index, register in enumerate(_ARM32_GPRS):
        if register not in ("r13", "r15", r_in, r_out):
            lines.append(f"    ldr {register}, [{r_in}, #{index * 4}]")
    lines.extend(f"    {line}" if line.strip() else "" for line in body.splitlines())
    lines.extend(f"{label}:" for label in exits)
    for index, register in enumerate(_ARM32_GPRS):
        if register not in ("r13", "r15", r_in, r_out):
            lines.append(f"    str {register}, [{r_out}, #{index * 4}]")
    for register in ("r13", "r15", r_in, r_out):
        lines.extend(
            (
                "    mov r2, #0",
                f"    str r2, [{r_out}, #{_ARM32_GPRS.index(register) * 4}]",
            ),
        )
    lines.extend(
        (
            f"    mrs {r_in}, APSR", f"    str {r_in}, [{r_out}, #64]",
            "    ldmia sp, {r4-r11, lr}", "    add sp, sp, #40", "    bx lr",
        ),
    )
    return "\n".join(lines) + "\n"


__all__ = ["aarch64_wrapper", "arm32_wrapper", "diff_test_aarch64", "diff_test_arm32"]
