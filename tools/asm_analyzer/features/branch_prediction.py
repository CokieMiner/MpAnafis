"""Branch target buffer (BTB) density and loop entry alignment analyzer.

Evaluates branch instruction density within 64-byte code fetch windows and
verifies that loop entry points are properly aligned to prevent instruction
decoder boundary bubbles.
"""

from __future__ import annotations

import re
from ..asm_util import extract_mnemonic, instr_lines
from ..types import BranchStats

_BRANCH_MNEMONICS = {
    "jmp", "jz", "jnz", "je", "jne", "js", "jns", "jc", "jnc",
    "ja", "jae", "jb", "jbe", "jg", "jge", "jl", "jle", "call", "ret",
}

_LABEL_RE = re.compile(r"^[0-9a-zA-Z_.]+:\s*$")
_ALIGN_RE = re.compile(r"^\s*\.(?:p2align|align)\b")


def analyze_branch_patterns(asm: str) -> BranchStats:
    """Analyze branch instructions, BTB density, and loop head alignment."""
    raw_lines = asm.splitlines()
    cleaned = instr_lines(asm)

    branch_count = 0
    for line in cleaned:
        mnem = extract_mnemonic(line)
        if mnem in _BRANCH_MNEMONICS:
            branch_count += 1

    # Approximate 64-byte fetch windows (assuming ~4 bytes per instruction)
    num_instructions = max(len(cleaned), 1)
    estimated_code_bytes = num_instructions * 4
    num_windows = max(estimated_code_bytes / 64.0, 1.0)
    branches_per_64b = branch_count / num_windows

    # High BTB density hazard occurs when more than 3 branches reside in a 64B window
    has_density_hazard = branches_per_64b > 3.0

    # Check loop alignment: check if inner loop labels (like `1:`, `2:`) have an alignment directive
    has_unaligned_loop = False
    for idx, raw in enumerate(raw_lines):
        line = raw.strip()
        if _LABEL_RE.match(line) and not line.startswith(".L"):
            # Check if previous non-empty line had .p2align or .align
            prev_aligned = False
            for prev_idx in range(idx - 1, -1, -1):
                prev_line = raw_lines[prev_idx].strip()
                if not prev_line or prev_line.startswith("#") or prev_line.startswith("//"):
                    continue
                if _ALIGN_RE.match(prev_line):
                    prev_aligned = True
                break
            if not prev_aligned and idx > 0:
                has_unaligned_loop = True

    return BranchStats(
        branch_count=branch_count,
        branches_per_64_bytes=round(branches_per_64b, 2),
        has_btb_density_hazard=has_density_hazard,
        has_unaligned_loop_head=has_unaligned_loop,
    )
