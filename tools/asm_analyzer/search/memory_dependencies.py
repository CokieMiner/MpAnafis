"""Memory alias proofs and loop-carried dependency analysis for x86 schedules."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from ..asm_util import instr_lines
from ..types import MemoryDependencyStats
from .ast import Instr, Op, get_instruction_spec, parse_line


def memory_operand(instruction: Instr) -> Optional[Op]:
    """Return the instruction's first explicit memory operand, if any."""
    return next((operand for operand in instruction.ops if operand.kind == "mem"), None)


def may_alias(
    left: Optional[Op],
    right: Optional[Op],
    disjoint_bases: Optional[Set[str] | Set[Tuple[str, str]]] = None,
) -> bool:
    """Return false for byte ranges proved disjoint or explicitly disjoint pointer bases."""
    if left is None or right is None or left.addr is None or right.addr is None:
        return True
    if left.base is not None and right.base is not None and left.base != right.base:
        if disjoint_bases is not None:
            if (
                left.base in disjoint_bases
                and right.base in disjoint_bases
                and isinstance(next(iter(disjoint_bases)), str)
            ):
                return False
            if (
                (left.base, right.base) in disjoint_bases
                or (right.base, left.base) in disjoint_bases
            ):
                return False
        return True
    if (
        left.base != right.base
        or left.index != right.index
        or left.scale != right.scale
    ):
        return True
    return _ranges_overlap(
        left.displacement,
        left.width,
        right.displacement,
        right.width,
    )


def analyze_loop_memory_dependencies(
    asm: str,
    disjoint_bases: Optional[Set[str] | Set[Tuple[str, str]]] = None,
) -> MemoryDependencyStats:
    """Classify intra- and cross-iteration memory dependence conservatively."""
    instructions = [
        instruction
        for line in instr_lines(asm)
        if (instruction := parse_line(line)) is not None
    ]
    accesses: List[Tuple[int, str, Op]] = []
    strides: Dict[str, int] = {}
    for index, instruction in enumerate(instructions):
        spec = get_instruction_spec(instruction)
        operand = memory_operand(instruction)
        if spec.mem is not None and operand is not None:
            accesses.append((index, spec.mem, operand))
        stride = _pointer_stride(instruction)
        if stride is not None:
            register, delta = stride
            strides[register] = strides.get(register, 0) + delta

    proved_disjoint = 0
    may_alias_count = 0
    for left_index, left_kind, left in accesses:
        for right_index, right_kind, right in accesses:
            if right_index <= left_index or "store" not in (left_kind, right_kind):
                continue
            if may_alias(left, right, disjoint_bases=disjoint_bases):
                may_alias_count += 1
            else:
                proved_disjoint += 1

    hazards = set()
    unknown_cross_iteration = 0
    for _, left_kind, left in accesses:
        for _, right_kind, right in accesses:
            if "store" not in (left_kind, right_kind):
                continue
            if left.base is None or right.base is None or left.base != right.base:
                unknown_cross_iteration += 1
                continue
            stride = strides.get(right.base)
            if stride is None or left.index != right.index or left.scale != right.scale:
                unknown_cross_iteration += 1
                continue
            next_displacement = right.displacement + stride
            if _ranges_overlap(
                left.displacement,
                left.width,
                next_displacement,
                right.width,
            ):
                hazards.add(
                    f"{left_kind} {left.text} -> next-iteration "
                    f"{right_kind} at displacement {next_displacement:+d}"
                )

    return MemoryDependencyStats(
        memory_operations=len(accesses),
        proved_disjoint_pairs=proved_disjoint,
        may_alias_pairs=may_alias_count,
        pointer_strides=strides,
        loop_carried_dependencies=tuple(sorted(hazards)),
        unknown_cross_iteration_pairs=unknown_cross_iteration,
    )


def _pointer_stride(instruction: Instr) -> Optional[Tuple[str, int]]:
    mnemonic = instruction.mnemonic.lower()
    if not mnemonic.startswith(("add", "sub")) or len(instruction.ops) != 2:
        return None
    immediate, destination = instruction.ops
    if immediate.kind != "imm" or destination.kind != "reg" or destination.base is None:
        return None
    try:
        value = int(immediate.text[1:], 0)
    except ValueError:
        return None
    if mnemonic.startswith("sub"):
        value = -value
    return destination.base, value


def _ranges_overlap(
    left_displacement: int,
    left_width: int,
    right_displacement: int,
    right_width: int,
) -> bool:
    left_end = left_displacement + max(1, left_width // 8)
    right_end = right_displacement + max(1, right_width // 8)
    return left_displacement < right_end and right_displacement < left_end


__all__ = [
    "analyze_loop_memory_dependencies",
    "may_alias",
    "memory_operand",
]
