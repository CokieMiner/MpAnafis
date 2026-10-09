"""Native s390x differential verifier."""

from __future__ import annotations

import platform
from typing import List

from ..types import ArchitectureFamily
from .common import compile_and_run, driver_source, native_register_plan, prepare_body_labels

_GPRS = tuple(f"r{index}" for index in range(16))


def diff_test_s390x(bodies: List[str], cases: int, use_wsl: bool) -> List[bool]:
    """Differentially execute s390x candidates on an s390x host."""
    if platform.machine().lower() != "s390x":
        raise RuntimeError("s390x differential execution requires an s390x host")
    r_in, r_out, pointers, used = native_register_plan(
        bodies,
        ArchitectureFamily.S390X,
        _GPRS,
        {"r0", "r14", "r15"},
    )
    if "r15" in used:
        raise ValueError("s390x differential execution does not accept SP operands")
    wrappers = "".join(
        s390x_wrapper(f"k_{index}", body, r_in, r_out)
        for index, body in enumerate(bodies)
    )
    driver = driver_source(
        len(bodies), 16, 17,
        (_GPRS.index(r_in), _GPRS.index(r_out)),
        pointers, cases, 64,
    )
    return compile_and_run(wrappers, driver, len(bodies), cases, use_wsl)


def s390x_wrapper(symbol: str, body: str, r_in: str, r_out: str) -> str:
    """Wrap an s390x kernel and capture GPR and condition-code state."""
    body, exits = prepare_body_labels(body, symbol)
    lines = [
        ".text", f".globl {symbol}", f".type {symbol}, @function", f"{symbol}:",
        "    lgr %r0, %r15", "    aghi %r15, -96",
        "    stmg %r6, %r14, 0(%r15)", "    stg %r0, 72(%r15)",
        f"    lgr %{r_in}, %r2", f"    lgr %{r_out}, %r3",
        "    lghi %r0, 0", "    ltgr %r0, %r0",
    ]
    for index, register in enumerate(_GPRS):
        if register not in ("r15", r_in, r_out):
            lines.append(f"    lg %{register}, {index * 8}(%{r_in})")
    lines.extend(f"    {line}" if line.strip() else "" for line in body.splitlines())
    lines.extend(f"{label}:" for label in exits)
    for index, register in enumerate(_GPRS):
        if register not in ("r15", r_in, r_out):
            lines.append(f"    stg %{register}, {index * 8}(%{r_out})")
    lines.extend(
        (
            f"    ipm %{r_in}", f"    srl %{r_in}, 28",
            f"    stg %{r_in}, 128(%{r_out})", f"    lghi %{r_in}, 0",
        ),
    )
    for register in ("r15", r_in, r_out):
        lines.append(f"    stg %{r_in}, {_GPRS.index(register) * 8}(%{r_out})")
    lines.extend(
        (
            "    lg %r0, 72(%r15)", "    lmg %r6, %r14, 0(%r15)",
            "    lgr %r15, %r0", "    br %r14",
        ),
    )
    return "\n".join(lines) + "\n"


__all__ = ["diff_test_s390x", "s390x_wrapper"]
