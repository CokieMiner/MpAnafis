"""Control-flow graph construction and repeated-region selection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .types import ArchitectureFamily
_LABEL_RE = re.compile(r"^\s*([.$A-Za-z_][\w.$]*|\d+):\s*$")
_LABEL_TOKEN_RE = re.compile(r"^(?:[.$A-Za-z_][\w.$]*|\d+[bf]?)$")
_RETURN_MNEMONICS = {"ret", "retq", "blr"}
_UNCONDITIONAL_BRANCHES = {
    "b", "ba", "br", "bra", "bx", "j", "jal", "jmp", "jr",
}
_CONDITIONAL_BRANCHES = {
    "bc", "bca", "bdz", "bdnz", "beq", "beqz", "bge", "bgeu", "bgtz",
    "blez", "blt", "bltu", "bne", "bnez", "cbnz", "cbz", "loop", "loope",
    "loopne", "loopnz", "loopz", "tbnz", "tbz",
}


@dataclass(frozen=True)
class BasicBlock:
    """One maximal instruction sequence with a single entry point."""

    index: int
    start: int
    end: int
    labels: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ControlFlowGraph:
    """Basic blocks and directed control-flow edges for an assembly block."""

    instructions: Tuple[str, ...]
    blocks: Tuple[BasicBlock, ...]
    edges: Tuple[Tuple[int, int], ...]
    back_edges: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class AssemblyRegion:
    """One assembly block or a statically identified repeated loop body."""

    asm: str
    kind: str
    label: Optional[str] = None
    block_count: int = 0
    cfg_edge_count: int = 0

    def schedulable_asm(self) -> str:
        """Return a straight-line loop iteration for schedule search."""
        if self.kind != "loop":
            return self.asm
        lines = self.asm.splitlines()
        if lines and is_branch_instruction(lines[-1]):
            lines.pop()
        return "\n".join(lines).strip()


def build_control_flow_graph(asm: str) -> ControlFlowGraph:
    """Parse labels and branch targets into a conservative basic-block CFG."""
    instructions, labels_by_position = _parse_statements(asm)
    if not instructions:
        return ControlFlowGraph((), (), (), ())

    resolved_targets: Dict[int, int] = {}
    leaders = {0} | {position for positions in labels_by_position.values() for position in positions if position < len(instructions)}
    for index, instruction in enumerate(instructions):
        target_token = _branch_target(instruction)
        target = _resolve_target(target_token, index, labels_by_position)
        if target is not None and target < len(instructions):
            resolved_targets[index] = target
            leaders.add(target)
        if (
            is_branch_instruction(instruction) or _is_return(instruction)
        ) and index + 1 < len(instructions):
            leaders.add(index + 1)

    ordered_leaders = sorted(leaders)
    labels_at_position = _invert_labels(labels_by_position)
    blocks = tuple(
        BasicBlock(
            index=block_index,
            start=start,
            end=(
                ordered_leaders[block_index + 1] - 1
                if block_index + 1 < len(ordered_leaders)
                else len(instructions) - 1
            ),
            labels=tuple(labels_at_position.get(start, ())),
        )
        for block_index, start in enumerate(ordered_leaders)
    )
    block_for_instruction = {
        instruction_index: block.index
        for block in blocks
        for instruction_index in range(block.start, block.end + 1)
    }

    edge_set = set()
    for block in blocks:
        terminator_index = block.end
        terminator = instructions[terminator_index]
        target = resolved_targets.get(terminator_index)
        if target is not None:
            edge_set.add((block.index, block_for_instruction[target]))
        if (
            not _is_return(terminator)
            and not _is_unconditional_branch(terminator)
            and block.index + 1 < len(blocks)
        ):
            edge_set.add((block.index, block.index + 1))

    edges = tuple(sorted(edge_set))
    dominators = _dominators(len(blocks), edges)
    back_edges = tuple(edge for edge in edges if edge[1] <= edge[0] and edge[1] in dominators[edge[0]])
    return ControlFlowGraph(tuple(instructions), blocks, edges, back_edges)


def select_analysis_region(asm: str, architecture: Optional[ArchitectureFamily] = None) -> AssemblyRegion:
    """Select the largest natural loop described by a CFG back edge."""
    cfg = build_control_flow_graph(asm)
    if not cfg.back_edges or architecture in (ArchitectureFamily.MIPS32, ArchitectureFamily.MIPS64):
        return AssemblyRegion(
            asm=asm.strip(),
            kind="block",
            block_count=len(cfg.blocks),
            cfg_edge_count=len(cfg.edges),
        )

    source, target = max(
        cfg.back_edges,
        key=lambda edge: (
            sum(block.end - block.start + 1 for block in cfg.blocks[edge[1]:edge[0] + 1]),
            edge[0] - edge[1],
        ),
    )
    loop_blocks = cfg.blocks[target:source + 1]
    first = loop_blocks[0].start
    last = loop_blocks[-1].end
    label = loop_blocks[0].labels[0] if loop_blocks[0].labels else None
    # Preserve directives, comments, and labels in their original positions.
    # Reconstructing from instruction-only CFG nodes can drop encoded bytes.
    raw_lines = asm.splitlines()
    instruction_positions = [
        index for index, raw_line in enumerate(raw_lines)
        if (line := _strip_comment(raw_line))
        and not line.startswith(".") and _LABEL_RE.fullmatch(line) is None
    ]
    start = instruction_positions[first]
    while start > 0 and start - 1 not in instruction_positions:
        start -= 1
    loop_lines = raw_lines[start:instruction_positions[last] + 1]
    return AssemblyRegion(
        asm="\n".join(loop_lines).strip(),
        kind="loop",
        label=label,
        block_count=len(loop_blocks),
        cfg_edge_count=len(cfg.edges),
    )


def is_branch_instruction(instruction: str) -> bool:
    """Return whether an instruction changes control flow on a supported ISA."""
    mnemonic = _mnemonic(instruction)
    return (
        mnemonic.startswith("j")
        or mnemonic.startswith(("bct", "brc", "brct"))
        or mnemonic in _UNCONDITIONAL_BRANCHES
        or mnemonic.startswith("b.")
        or mnemonic in _CONDITIONAL_BRANCHES
    )


def _parse_statements(asm: str) -> Tuple[List[str], Dict[str, List[int]]]:
    instructions: List[str] = []
    labels: Dict[str, List[int]] = {}
    for raw_line in asm.splitlines():
        line = _strip_comment(raw_line)
        if not line or line.startswith(".") and not line.endswith(":"):
            continue
        label_match = _LABEL_RE.match(line)
        if label_match is not None:
            labels.setdefault(label_match.group(1), []).append(len(instructions))
            continue
        instructions.append(line)
    return instructions, labels


def _strip_comment(line: str) -> str:
    if line.lstrip().startswith(("#", "//")):
        return ""
    without_slashes = line.split("//", 1)[0]
    without_hash_comment = re.split(r"\s+#(?![-+]?\d)", without_slashes, maxsplit=1)[0]
    return without_hash_comment.strip()


def _branch_target(instruction: str) -> Optional[str]:
    if not is_branch_instruction(instruction):
        return None
    operands = instruction.split(None, 1)
    if len(operands) != 2:
        return None
    token = operands[1].rsplit(",", 1)[-1].strip().split()[-1]
    token = token.lstrip("*")
    return token if _LABEL_TOKEN_RE.fullmatch(token) else None


def _resolve_target(
    token: Optional[str],
    instruction_index: int,
    labels: Dict[str, List[int]],
) -> Optional[int]:
    if token is None:
        return None
    if token[-1:] in ("b", "f") and token[:-1].isdigit():
        positions = labels.get(token[:-1], ())
        if token.endswith("b"):
            return max(
                (position for position in positions if position <= instruction_index),
                default=None,
            )
        return min(
            (position for position in positions if position > instruction_index),
            default=None,
        )
    positions = labels.get(token, ())
    return positions[0] if positions else None


def _invert_labels(labels: Dict[str, List[int]]) -> Dict[int, List[str]]:
    by_position: Dict[int, List[str]] = {}
    for label, positions in labels.items():
        for position in positions:
            by_position.setdefault(position, []).append(label)
    return by_position


def _mnemonic(instruction: str) -> str:
    parts = instruction.split(None, 1)
    return parts[0].lower() if parts else ""


def _dominators(count: int, edges: Sequence[Tuple[int, int]]) -> Dict[int, set[int]]:
    """Compute entry-reachable dominators for natural-loop recognition."""
    reachable = {0}
    while True:
        expanded = reachable | {target for source, target in edges if source in reachable}
        if expanded == reachable:
            break
        reachable = expanded
    predecessors = {index: {source for source, target in edges if target == index and source in reachable} for index in range(count)}
    dominators = {index: ({0} if index == 0 else reachable.copy() if index in reachable else set()) for index in range(count)}
    changed = True
    while changed:
        changed = False
        for index in sorted(reachable - {0}):
            incoming = predecessors[index]
            value = {index} | set.intersection(*(dominators[pred] for pred in incoming))
            if value != dominators[index]:
                dominators[index] = value
                changed = True
    return dominators


def _is_unconditional_branch(instruction: str) -> bool:
    return _mnemonic(instruction) in _UNCONDITIONAL_BRANCHES


def _is_return(instruction: str) -> bool:
    return _mnemonic(instruction) in _RETURN_MNEMONICS


__all__ = [
    "AssemblyRegion",
    "BasicBlock",
    "ControlFlowGraph",
    "build_control_flow_graph",
    "is_branch_instruction",
    "select_analysis_region",
]
