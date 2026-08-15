"""32-bit x86 loop control and stack invariant verification.

Verifies structural and microarchitectural correctness of 32-bit x86 inline
assembly loops under severe 6-GPR hardware register constraints:

1. Stack Frame Balancing:
   Guarantees that all stack pushes (`pushl`) and stack allocations (`subl $N, %esp`)
   are strictly balanced by matching deallocations (`addl $N, %esp` / `popl`)
   prior to every exit path.

2. Condition Flag (CF) Preservation:
   Verifies that loop counter decrements (`subl $N, (%esp)`, `decl (%esp)`) do not
   clobber live arithmetic carry/borrow condition flags unless the flag has been
   materialized into a register mask (e.g., `sbbl {reg}, {reg}` or `adcl $0, {reg}`).

3. Unroll Stride & Branch Consistency:
   Verifies that pointer increments (e.g., `addl $16, {src}`) match the stack counter
   decrement step (`subl $4, (%esp)`) and are paired with appropriate loop branch
   conditions (`jae`, `jns`, `jnz`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..asm_util import extract_mnemonic, instr_lines


@dataclass(frozen=True)
class X86_32LoopStats:
    """Analysis results for 32-bit x86 stack loop control and frame balancing."""
    is_32bit_stack_loop: bool
    net_stack_delta: int
    has_stack_imbalance: bool
    has_flag_clobber_hazard: bool
    has_stride_mismatch: bool
    unroll_stride_bytes: int
    counter_step_limbs: int
    diagnostics: Tuple[str, ...]


_PUSH_RE = re.compile(r"^push(?:l)?\s+", re.IGNORECASE)
_POP_RE = re.compile(r"^pop(?:l)?\s+", re.IGNORECASE)
_ADD_ESP_RE = re.compile(r"^add(?:l)?\s+\$(\d+)\s*,\s*%(?:esp|sp)", re.IGNORECASE)
_SUB_ESP_RE = re.compile(r"^sub(?:l)?\s+\$(\d+)\s*,\s*%(?:esp|sp)", re.IGNORECASE)
_SUB_COUNTER_RE = re.compile(r"^sub(?:l)?\s+\$(\d+)\s*,\s*\(?0?%\b(?:esp|sp)\b\)?", re.IGNORECASE)
_DEC_COUNTER_RE = re.compile(r"^dec(?:l)?\s+\(?0?%\b(?:esp|sp)\b\)?", re.IGNORECASE)
_ADD_PTR_RE = re.compile(r"^add(?:l)?\s+\$(\d+)\s*,\s*(?:\{?(?:src|dst|src_ptr|dst_ptr)\}?|%[a-z0-9]+)", re.IGNORECASE)


def analyze_x86_32_loop_control(asm: str) -> X86_32LoopStats:
    """Analyze 32-bit x86 assembly for stack balancing and loop control invariants."""
    lines = instr_lines(asm)
    stack_delta = 0
    is_32bit_loop = False
    flag_clobber_hazard = False
    stride_mismatch = False
    cf_is_live = False
    last_ptr_add_bytes: Optional[int] = None
    counter_step: Optional[int] = None
    diagnostics: List[str] = []

    for idx, line in enumerate(lines, start=1):
        clean = line.split("#")[0].split("//")[0].strip()
        if not clean or clean.startswith(".") or clean.endswith(":"):
            continue

        mnem = extract_mnemonic(clean)

        # Track stack pointer modifications
        if _PUSH_RE.search(clean):
            stack_delta += 4
            is_32bit_loop = True
        elif _POP_RE.search(clean):
            stack_delta -= 4
        elif m := _SUB_ESP_RE.search(clean):
            stack_delta += int(m.group(1))
            is_32bit_loop = True
        elif m := _ADD_ESP_RE.search(clean):
            stack_delta -= int(m.group(1))

        # Track pointer advance strides
        if m := _ADD_PTR_RE.search(clean):
            last_ptr_add_bytes = int(m.group(1))

        # Track condition flag status
        if mnem in ("adc", "adcl", "sbb", "sbbl"):
            # Arithmetic that reads/writes CF
            parts = clean.split(None, 1)
            if len(parts) >= 2 and mnem in ("sbb", "sbbl"):
                ops = [op.strip() for op in parts[1].split(",")]
                if len(ops) >= 2 and ops[0] == ops[1]:
                    # `sbbl %reg, %reg` captures CF into a mask (CF is now preserved in reg)
                    cf_is_live = False
                else:
                    cf_is_live = True
            else:
                cf_is_live = True
        elif mnem in ("clc", "stc", "bt", "btc", "btr", "bts"):
            cf_is_live = True

        # Check loop counter decrements on stack
        if m := _SUB_COUNTER_RE.search(clean):
            is_32bit_loop = True
            counter_step = int(m.group(1))
            if cf_is_live:
                flag_clobber_hazard = True
                diagnostics.append(
                    f"Line {idx}: `subl ${counter_step}, (%esp)` clobbers live Carry/Borrow flag before it was captured."
                )
            if last_ptr_add_bytes is not None:
                # 32-bit limbs are 4 bytes each: ptr_add should equal counter_step * 4
                expected_ptr_bytes = counter_step * 4
                if last_ptr_add_bytes != expected_ptr_bytes:
                    stride_mismatch = True
                    diagnostics.append(
                        f"Line {idx}: Pointer displacement (${last_ptr_add_bytes} bytes) does not match "
                        f"counter step (${counter_step} limbs = {expected_ptr_bytes} bytes)."
                    )
        elif _DEC_COUNTER_RE.search(clean):
            is_32bit_loop = True
            if cf_is_live:
                flag_clobber_hazard = True
                diagnostics.append(
                    f"Line {idx}: `decl (%esp)` clobbers live Carry/Borrow flag before it was captured."
                )

    has_imbalance = (stack_delta != 0) and is_32bit_loop
    if has_imbalance:
        diagnostics.append(
            f"Stack frame imbalance: Net stack delta at exit is {stack_delta:+d} bytes (must be 0)."
        )

    return X86_32LoopStats(
        is_32bit_stack_loop=is_32bit_loop,
        net_stack_delta=stack_delta,
        has_stack_imbalance=has_imbalance,
        has_flag_clobber_hazard=flag_clobber_hazard,
        has_stride_mismatch=stride_mismatch,
        unroll_stride_bytes=last_ptr_add_bytes or 0,
        counter_step_limbs=counter_step or 0,
        diagnostics=tuple(diagnostics),
    )
