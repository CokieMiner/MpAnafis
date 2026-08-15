"""Memory interaction and cache line straddle analysis for assembly kernels.

Classifies memory operands into pure loads, pure stores, and read-modify-writes,
and evaluates cache-line alignment characteristics.
"""

from __future__ import annotations

import re
from typing import List
from ..types import MemoryAccessStats
from ..asm_util import instr_lines, extract_mnemonic

_MEM_OPERAND_RE = re.compile(r"(-?\d*)\(([^)]*)\)")


def analyze_memory_accesses(asm: str) -> MemoryAccessStats:
    """Analyze memory accesses in an assembly block, returning structured stats."""
    loads = 0
    stores = 0
    rmw = 0
    straddles = 0

    for line in instr_lines(asm):
        mnem = extract_mnemonic(line)
        # Skip pure branches, labels, directives
        if mnem in ("jmp", "jz", "jnz", "js", "jns", "jc", "jnc", "ja", "jae", "jb", "jbe", "call", "ret", "nop"):
            continue

        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        operands = [op.strip() for op in parts[1].split(",")]

        # In AT&T syntax, destination is the last operand (unless read-only)
        if mnem in ("cmp", "test"):
            # cmp and test only read memory operands
            loads += sum(bool(_MEM_OPERAND_RE.search(op)) for op in operands)
        elif len(operands) == 1 and mnem in ("mul", "imul", "div", "idiv", "push"):
            # 1-operand arithmetic instructions read the memory operand
            if _MEM_OPERAND_RE.search(operands[0]):
                loads += 1
        else:
            has_mem_src = any(_MEM_OPERAND_RE.search(op) for op in operands[:-1])
            has_mem_dst = bool(_MEM_OPERAND_RE.search(operands[-1])) if operands else False

            if has_mem_src:
                loads += 1

            if has_mem_dst:
                if mnem.startswith("mov") or mnem.startswith("vmov") or mnem == "pop":
                    stores += 1
                else:
                    rmw += 1

        # Check cache-line straddle hazards.  The actual address depends on the
        # runtime base register value, so static displacement analysis is an
        # approximation.  We flag only guaranteed structural hazards:
        # (a) unaligned 8-byte accesses (disp % 8 != 0), or
        # (b) displacements that straddle a 64-byte boundary regardless of any
        #     8-byte-aligned base (disp % 64 in range [57..63]).
        for m in _MEM_OPERAND_RE.finditer(line):
            raw_disp = m.group(1)
            disp = int(raw_disp) if raw_disp and raw_disp not in ("-", "+") else 0
            offset_in_line = abs(disp) % 64
            if disp % 8 != 0 or offset_in_line + 8 > 64:
                straddles += 1

    return MemoryAccessStats(
        loads=loads,
        stores=stores,
        read_modify_writes=rmw,
        cache_line_straddles=straddles,
    )


def estimate_unroll_factor(asm: str) -> int:
    """Estimate loop unrolling factor based on memory offset sequences."""
    offsets: List[int] = []
    for line in instr_lines(asm):
        for m in _MEM_OPERAND_RE.finditer(line):
            raw_disp = m.group(1)
            disp = int(raw_disp) if raw_disp and raw_disp not in ("-", "+") else 0
            offsets.append(disp)

    if not offsets:
        return 1

    stride_8_counts: dict[int, int] = {}
    for off in offsets:
        if off >= 0 and off % 8 == 0:
            idx = off // 8
            stride_8_counts[idx] = stride_8_counts.get(idx, 0) + 1

    if stride_8_counts:
        max_idx = max(stride_8_counts.keys())
        if all(k in stride_8_counts for k in range(max_idx + 1)):
            return max(1, max_idx + 1)

    return 1
