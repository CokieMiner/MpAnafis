"""Instruction-parser and dependency-spec adapters for schedule search."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

from ..types import ArchitectureFamily
from .ast import Instr, Spec, get_instruction_spec, parse_line
from .native_ast import native_instruction_spec, parse_native_line


@dataclass(frozen=True)
class ScheduleAdapter:
    """Architecture-specific parsing and dependency semantics."""

    architecture: ArchitectureFamily
    parse: Callable[[str], Instr | None]
    spec: Callable[[Instr], Spec]


def adapter_for(architecture: ArchitectureFamily) -> ScheduleAdapter:
    """Return strict dependency semantics for a supported target ISA."""
    if architecture in (ArchitectureFamily.X86_64, ArchitectureFamily.X86_32):
        return ScheduleAdapter(architecture, parse_line, get_instruction_spec)
    return ScheduleAdapter(
        architecture,
        partial(parse_native_line, architecture=architecture),
        partial(native_instruction_spec, architecture=architecture),
    )


__all__ = ["ScheduleAdapter", "adapter_for"]
