"""Public facade for Rust inline-assembly extraction."""

from __future__ import annotations

from .emitted_asm import (
    extract_asm_region,
    extract_asm_regions,
    extract_named_asm_regions,
)
from .extraction_harness import (
    compile_snippet,
    real_asm_for_block,
    real_asm_for_source,
    real_asm_regions_for_source,
    render_snippet,
)
from .extraction_parser import (
    AsmBlock,
    Operand,
    extract_asm_blocks,
    find_asm_blocks,
    is_string_literal,
    parse_operand,
    split_args,
)

__all__ = [
    "AsmBlock",
    "Operand",
    "compile_snippet",
    "extract_asm_blocks",
    "extract_asm_region",
    "extract_asm_regions",
    "extract_named_asm_regions",
    "find_asm_blocks",
    "is_string_literal",
    "parse_operand",
    "real_asm_for_block",
    "real_asm_for_source",
    "real_asm_regions_for_source",
    "render_snippet",
    "split_args",
]
