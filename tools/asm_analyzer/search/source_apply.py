"""Fail-closed mapping from confirmed emitted schedules to Rust ``asm!`` lines."""

from __future__ import annotations

from collections import defaultdict, deque
import re
from typing import DefaultDict, Deque, List, Sequence

from ..extraction_parser import find_asm_blocks, is_string_literal, split_args


def rewrite_confirmed_schedule(
    source: str,
    source_line: int,
    emitted_asm: str,
    original_region: str,
    confirmed_region: str,
) -> str:
    """Return source with one exactly mapped inline-assembly region reordered."""
    block_start, block_end = _block_at_line(source, source_line)
    source_lines = source.splitlines(keepends=True)
    block_start_line = source.count("\n", 0, block_start)
    block_end_line = source.count("\n", 0, block_end) + 1
    instruction_lines = [
        index
        for index in range(block_start_line, min(block_end_line, len(source_lines)))
        if _is_single_line_instruction(source_lines[index])
    ]
    arguments = split_args(source[source.index("(", block_start) + 1:block_end - 1])
    templates = [argument for argument in arguments if is_string_literal(argument)]
    if len(templates) != len(instruction_lines) or any(
        argument.startswith("concat!") for argument in arguments
    ):
        raise ValueError("source application requires standalone single-line string templates")
    for index in instruction_lines:
        _validate_template_line(source_lines[index])
    emitted_lines = _lines(emitted_asm)
    original_lines = _lines(original_region)
    confirmed_lines = _lines(confirmed_region)
    if len(instruction_lines) != len(emitted_lines):
        raise ValueError(
            "source application requires one source string line per emitted instruction",
        )
    region_start = _unique_subsequence_start(emitted_lines, original_lines)
    permutation = _permutation(original_lines, confirmed_lines)
    scheduled_source_lines = instruction_lines[
        region_start:region_start + len(original_lines)
    ]
    _reject_interleaved_comments(source_lines, scheduled_source_lines)
    originals = [source_lines[index] for index in scheduled_source_lines]
    for destination, original_index in zip(scheduled_source_lines, permutation):
        source_lines[destination] = originals[original_index]
    return "".join(source_lines)


def _block_at_line(source: str, source_line: int) -> tuple[int, int]:
    matches = [
        (start, end)
        for start, end in find_asm_blocks(source)
        if source.count("\n", 0, start) + 1 == source_line
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one asm! block beginning at source line {source_line}",
        )
    return matches[0]


def _is_single_line_instruction(line: str) -> bool:
    stripped = line.strip()
    return is_string_literal(stripped) or stripped.startswith("concat!(")


def _validate_template_line(line: str) -> None:
    """Require one nonempty assembly statement per complete physical line."""
    match = re.fullmatch(
        r'\s*(?:"(?P<normal>[^"\\\r\n]*)"|r(?P<hashes>\#*)"(?P<raw>[^\r\n]*)"(?P=hashes))'
        r'\s*,\s*(?://[^\r\n]*)?\s*',
        line,
    )
    if match is None:
        raise ValueError("source application requires one unescaped string per line")
    statement = (match.group("normal") if match.group("normal") is not None else match.group("raw")).strip()
    if not statement or statement.startswith(("//", "#", "/*")) or ";" in statement or '"' in statement:
        raise ValueError("source application requires one nonempty assembly statement per string")


def _lines(assembly: str) -> List[str]:
    return [line.strip() for line in assembly.splitlines() if line.strip()]


def _unique_subsequence_start(full: Sequence[str], region: Sequence[str]) -> int:
    starts = [
        start
        for start in range(len(full) - len(region) + 1)
        if list(full[start:start + len(region)]) == list(region)
    ]
    if len(starts) != 1:
        raise ValueError(
            "scheduled region does not map uniquely into emitted assembly",
        )
    return starts[0]


def _permutation(original: Sequence[str], candidate: Sequence[str]) -> List[int]:
    if len(original) != len(candidate):
        raise ValueError("confirmed schedule changed the instruction count")
    positions: DefaultDict[str, Deque[int]] = defaultdict(deque)
    for index, line in enumerate(original):
        positions[line].append(index)
    permutation: List[int] = []
    for line in candidate:
        if not positions[line]:
            raise ValueError("confirmed schedule changed an instruction")
        permutation.append(positions[line].popleft())
    if any(indices for indices in positions.values()):
        raise ValueError("confirmed schedule omitted an instruction")
    return permutation


def _reject_interleaved_comments(
    source_lines: Sequence[str],
    instruction_lines: Sequence[int],
) -> None:
    if not instruction_lines:
        raise ValueError("scheduled region contains no source instruction lines")
    instruction_set = set(instruction_lines)
    for index in range(instruction_lines[0] + 1, instruction_lines[-1]):
        stripped = source_lines[index].strip()
        if index not in instruction_set and stripped:
            raise ValueError(
                "standalone comments inside the scheduled region require manual review",
            )


__all__ = ["rewrite_confirmed_schedule"]
