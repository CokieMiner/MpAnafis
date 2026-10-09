"""Native x86-64 differential verifier."""

from __future__ import annotations

import re
from typing import List

from ..asm_util import GPR64, GPR_ALIAS_MAP, classify_regs, named_registers
from .common import compile_and_run, driver_source, prepare_body_labels, require_candidates

_GPR32 = ("eax", "ebx", "ecx", "edx", "esi", "edi", "ebp")
_GPR32_CANONICAL = ("rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp")


def diff_test_x86_64(
    bodies: List[str],
    cases: int,
    use_wsl: bool,
) -> List[bool]:
    """Differentially execute x86-64 candidates on an x86-64 host."""
    require_candidates(bodies)
    pointers, _ = classify_regs("\n".join(bodies))
    pointer_slots = sorted(GPR64.index(register) for register in pointers if register in GPR64)
    wrappers = "".join(
        x86_64_wrapper(f"k_{index}", body)
        for index, body in enumerate(bodies)
    )
    driver = driver_source(
        len(bodies),
        len(GPR64),
        len(GPR64) + 1,
        (),
        pointer_slots,
        cases,
        64,
    )
    return compile_and_run(wrappers, driver, len(bodies), cases, use_wsl)


def diff_test_x86_32(
    bodies: List[str],
    cases: int,
    use_wsl: bool,
) -> List[bool]:
    """Differentially execute i686 candidates without reserving kernel GPRs."""
    require_candidates(bodies)
    pointers, _ = classify_regs("\n".join(bodies))
    pointer_slots = sorted(
        _GPR32_CANONICAL.index(register)
        for register in pointers
        if register in _GPR32_CANONICAL
    )
    wrappers = "".join(
        x86_32_wrapper(f"k_{index}", body)
        for index, body in enumerate(bodies)
    )
    driver = driver_source(
        len(bodies), len(_GPR32), len(_GPR32) + 1,
        (), pointer_slots, cases, 32,
    )
    return compile_and_run(wrappers, driver, len(bodies), cases, use_wsl)


def x86_64_wrapper(symbol: str, body: str) -> str:
    """Capture all fifteen GPRs without reserving any kernel registers.

    The output pointer lives on the stack while the snippet executes. After
    execution, flags and every GPR are saved before two registers are reused
    to copy the snapshot. RSP-relative snippets require a separate stack ABI.
    """
    _require_scalar_registers(body)
    body, exits = prepare_body_labels(body, symbol)
    lines = [f".globl {symbol}", f"{symbol}:"]
    for register in ("rbp", "rbx", "r12", "r13", "r14", "r15"):
        lines.append(f"    push %{register}")
    lines.append("    push %rsi")
    for index, register in enumerate(GPR64):
        if register != "rdi":
            lines.append(f"    mov {index * 8}(%rdi), %{register}")
    lines.append(f"    mov {GPR64.index('rdi') * 8}(%rdi), %rdi")
    # CMP establishes every arithmetic status flag without changing a GPR.
    # CLC alone leaves OF/SF/ZF/PF/AF dependent on the caller's execution path.
    lines.extend(("    cmp %rsp, %rsp", "    cld"))
    if any(line.lstrip().startswith(("div", "idiv")) for line in body.splitlines()):
        lines.append("    xor %edx, %edx")
    lines.extend(f"    {line}" if line.strip() else "" for line in body.splitlines())
    lines.extend(f"{label}:" for label in exits)
    lines.append("    pushfq")
    lines.extend(f"    push %{register}" for register in reversed(GPR64))
    snapshot_bytes = (len(GPR64) + 1) * 8
    lines.append(f"    mov {snapshot_bytes}(%rsp), %rax")
    for index in range(len(GPR64) + 1):
        lines.extend((f"    mov {index * 8}(%rsp), %rdx", f"    mov %rdx, {index * 8}(%rax)"))
    lines.extend((f"    add ${snapshot_bytes + 8}, %rsp", "    cld"))
    for register in ("r15", "r14", "r13", "r12", "rbx", "rbp"):
        lines.append(f"    pop %{register}")
    lines.append("    ret")
    return "\n".join(lines) + "\n"


def x86_32_wrapper(symbol: str, body: str) -> str:
    """Wrap an i686 kernel while retaining all seven allocatable GPR values."""
    _require_scalar_registers(body)
    body, exits = prepare_body_labels(body, symbol)
    lines = [
        f".globl {symbol}", f"{symbol}:",
        "    push %ebp", "    push %edi", "    push %esi", "    push %ebx",
        "    mov 20(%esp), %eax",
    ]
    for index, register in enumerate(_GPR32[1:], start=1):
        lines.append(f"    mov {index * 4}(%eax), %{register}")
    lines.extend(("    mov 0(%eax), %eax", "    cmp %esp, %esp", "    cld"))
    if any(line.lstrip().startswith(("div", "idiv")) for line in body.splitlines()):
        lines.append("    xor %edx, %edx")
    lines.extend(f"    {line}" if line.strip() else "" for line in body.splitlines())
    lines.extend(f"{label}:" for label in exits)
    lines.extend(
        (
            "    pushf", "    push %ebp", "    push %edi", "    push %esi",
            "    push %edx", "    push %ecx", "    push %ebx", "    push %eax",
            "    mov 56(%esp), %ebp",
        ),
    )
    for index, register in enumerate(_GPR32):
        lines.extend((f"    mov {index * 4}(%esp), %eax", f"    mov %eax, {index * 4}(%ebp)"))
    lines.extend(
        (
            "    mov 28(%esp), %eax", "    mov %eax, 28(%ebp)",
            "    add $32, %esp", "    pop %ebx", "    pop %esi",
            "    pop %edi", "    pop %ebp", "    ret",
        ),
    )
    return "\n".join(lines) + "\n"


def _require_scalar_registers(body: str) -> None:
    """Refuse register banks absent from the scalar state snapshot."""
    instructions = "\n".join(line.split("#", 1)[0] for line in body.splitlines())
    registers = set(re.findall(r"%([A-Za-z][A-Za-z0-9]*)", instructions))
    unsupported = registers - GPR_ALIAS_MAP.keys()
    if unsupported:
        raise ValueError(
            "x86 differential execution cannot capture register state: "
            + ", ".join(sorted(unsupported)),
        )
    if "rsp" in named_registers(instructions):
        raise ValueError("x86 differential snippets cannot access the harness stack pointer")


__all__ = [
    "diff_test_x86_32",
    "diff_test_x86_64",
    "x86_32_wrapper",
    "x86_64_wrapper",
]
