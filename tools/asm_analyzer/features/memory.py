"""Memory interaction and cache line straddle analysis for assembly kernels.

Classifies memory operands into pure loads, pure stores, and read-modify-writes,
and evaluates cache-line alignment characteristics.
"""

from __future__ import annotations

import re

from ..asm_util import extract_mnemonic, instr_lines, split_asm_operands
from ..types import ArchitectureFamily, MemoryAccessStats

_MEM_OPERAND_RE = re.compile(r"(-?\d*)\(([^)]*)\)")
_BRACKET_MEM_RE = re.compile(r"\[([^\]]*)\]")
_READ_ONLY_DESTINATION_MNEMONICS = {
    "bt",
    "call",
    "cmp",
    "div",
    "idiv",
    "imul",
    "jmp",
    "mul",
    "prefetch",
    "test",
}


def analyze_memory_accesses(
    asm: str,
    limb_bytes: int = 8,
    cache_line_bytes: int = 64,
    target_arch: ArchitectureFamily = ArchitectureFamily.X86_64,
) -> MemoryAccessStats:
    """Analyze memory accesses in an assembly block, returning structured stats."""
    loads = 0
    stores = 0
    rmw = 0
    straddles = 0

    for line in instr_lines(asm):
        mnem = extract_mnemonic(line)
        if target_arch not in (ArchitectureFamily.X86_64, ArchitectureFamily.X86_32):
            logical_loads, logical_stores, logical_rmw = _load_store_isa_accesses(
                line,
                mnem,
            )
            loads += logical_loads
            stores += logical_stores
            rmw += logical_rmw
            straddles += _straddle_count(
                line,
                limb_bytes,
                cache_line_bytes,
            )
            continue
        # Skip pure branches, labels, directives
        if mnem in ("jmp", "jz", "jnz", "js", "jns", "jc", "jnc", "ja", "jae", "jb", "jbe", "call", "ret", "nop"):
            continue
        # LEA uses memory-address syntax but performs no memory access.
        if mnem == "lea":
            continue

        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        operands = split_asm_operands(parts[1])

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
        # approximation. We flag displacement alignment relative to the
        # target limb and a crossing of the configured cache-line boundary.
        for m in _MEM_OPERAND_RE.finditer(line):
            raw_disp = m.group(1)
            disp = int(raw_disp) if raw_disp and raw_disp not in ("-", "+") else 0
            offset_in_line = disp % cache_line_bytes
            if (
                disp % limb_bytes != 0
                or offset_in_line + limb_bytes > cache_line_bytes
            ):
                straddles += 1

    return MemoryAccessStats(
        loads=loads,
        stores=stores,
        read_modify_writes=rmw,
        cache_line_straddles=straddles,
    )


def estimate_unroll_factor(asm: str, limb_bytes: int = 8) -> int:
    """Estimate loop width from pointer stride or destination address lanes.

    Source loads can legally read ahead of the limbs produced by one loop
    iteration, overlapping rows can write one more output than their input
    width, and backward kernels use negative displacements.  A repeated
    pointer stride is therefore authoritative for loops.  Distinct destination
    displacements are the fallback for straight-line kernels.
    """
    lines = instr_lines(asm)
    address_scales: dict[str, set[int]] = {}
    for line in lines:
        for match in _MEM_OPERAND_RE.finditer(line):
            address_parts = [part.strip() for part in match.group(2).split(",")]
            if address_parts[0]:
                address_scales.setdefault(address_parts[0], set()).add(1)
            if len(address_parts) >= 2 and address_parts[1]:
                scale = int(address_parts[2], 0) if len(address_parts) >= 3 else 1
                address_scales.setdefault(address_parts[1], set()).add(scale)
        for match in _BRACKET_MEM_RE.finditer(line):
            base = match.group(1).split(",", 1)[0].strip()
            if base:
                address_scales.setdefault(base, set()).add(1)
    strides: set[int] = set()
    destinations: dict[str, set[int]] = {}
    for line in lines:
        mnemonic = extract_mnemonic(line)
        parts = line.split(None, 1)
        operands = split_asm_operands(parts[1]) if len(parts) == 2 else []

        if mnemonic == "lea" and len(operands) == 2:
            source = _MEM_OPERAND_RE.fullmatch(operands[0].strip())
            destination = operands[1].strip()
            if source is not None:
                raw_displacement, address = source.groups()
                base = address.split(",", 1)[0].strip()
                displacement = int(raw_displacement) if raw_displacement else 0
                if base == destination:
                    for scale in address_scales.get(base, ()):
                        effective_stride = displacement * scale
                        if effective_stride and effective_stride % limb_bytes == 0:
                            strides.add(abs(effective_stride))
            continue

        if mnemonic in ("add", "sub") and len(operands) == 2:
            immediate, destination = operands
            if immediate.startswith("$") and destination in address_scales:
                try:
                    displacement = int(immediate[1:], 0)
                except ValueError:
                    displacement = 0
                for scale in address_scales[destination]:
                    effective_stride = displacement * scale
                    if effective_stride and effective_stride % limb_bytes == 0:
                        strides.add(abs(effective_stride))

        if mnemonic in ("add", "addi", "addiw") and len(operands) == 3:
            destination, source, immediate = operands
            if destination == source and destination in address_scales:
                try:
                    displacement = int(immediate.lstrip("#$"), 0)
                except ValueError:
                    displacement = 0
                if displacement and displacement % limb_bytes == 0:
                    strides.add(abs(displacement))

        post_index = re.search(r"\]\s*,\s*#?(-?\d+)\s*$", line)
        if post_index is not None:
            displacement = int(post_index.group(1))
            if displacement and displacement % limb_bytes == 0:
                strides.add(abs(displacement))

        if (
            mnemonic in _READ_ONLY_DESTINATION_MNEMONICS
            or mnemonic.startswith("prefetch")
        ):
            continue
        if not operands:
            continue
        match = _MEM_OPERAND_RE.fullmatch(operands[-1].strip())
        if match is None:
            continue
        raw_displacement, address_stream = match.groups()
        displacement = int(raw_displacement) if raw_displacement else 0
        destinations.setdefault(address_stream.replace(" ", ""), set()).add(displacement)

    if strides:
        return max(strides) // limb_bytes
    return max((len(offsets) for offsets in destinations.values()), default=1)


def _load_store_isa_accesses(line: str, mnemonic: str) -> tuple[int, int, int]:
    has_memory = bool(_MEM_OPERAND_RE.search(line) or _BRACKET_MEM_RE.search(line))
    if not has_memory:
        return 0, 0, 0
    if mnemonic in ("ldp", "ldnp"):
        return 2, 0, 0
    if mnemonic in ("stp", "stnp"):
        return 0, 2, 0
    if mnemonic.startswith("amo") or mnemonic in ("cas", "casp", "ldadd", "swp"):
        return 0, 0, 1
    if mnemonic.startswith(("lb", "ld", "lg", "lh", "ll", "lq", "lw")) or mnemonic in (
        "lb", "lbu", "ld", "lh", "lhu", "lq", "lw", "lwu",
        "lwarx", "ldarx", "ldm",
    ):
        return 1, 0, 0
    if mnemonic.startswith("st") or mnemonic in (
        "sb", "sd", "sh", "sq", "sw", "std", "stw", "stdcx", "stwcx",
    ):
        return 0, 1, 0
    return 0, 0, 0


def _straddle_count(
    line: str,
    access_width_bytes: int,
    cache_line_bytes: int,
) -> int:
    displacements = []
    for match in _MEM_OPERAND_RE.finditer(line):
        raw_displacement = match.group(1)
        displacements.append(int(raw_displacement) if raw_displacement else 0)
    for match in _BRACKET_MEM_RE.finditer(line):
        displacement_match = re.search(r"#\s*(-?\d+)", match.group(1))
        displacements.append(
            int(displacement_match.group(1)) if displacement_match else 0
        )
    return sum(
        displacement % access_width_bytes != 0
        or displacement % cache_line_bytes + access_width_bytes > cache_line_bytes
        for displacement in displacements
    )
