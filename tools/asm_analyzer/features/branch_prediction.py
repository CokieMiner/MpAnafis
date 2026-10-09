"""Branch target buffer (BTB) density and loop entry alignment analyzer.

Estimates branch density within code-fetch windows and identifies loop labels
without an adjacent alignment directive. Density and alignment impact remain
heuristic because exact instruction sizes and CPU front ends vary.
"""

from __future__ import annotations

import re

from ..asm_util import extract_mnemonic, instr_lines
from ..regions import is_branch_instruction
from ..types import BranchStats

_LABEL_RE = re.compile(r"^[0-9a-zA-Z_.]+:\s*$")
_ALIGN_RE = re.compile(r"^\s*\.(?:p2align|align)\b")


def analyze_branch_patterns(asm: str, encoded_bytes: int | None = None) -> BranchStats:
    """Analyze branch instructions, BTB density, and loop head alignment."""
    raw_lines = asm.splitlines()
    cleaned = instr_lines(asm)

    branch_count = 0
    for line in cleaned:
        if is_branch_instruction(line) or extract_mnemonic(line) in ("call", "ret"):
            branch_count += 1

    num_instructions = max(len(cleaned), 1)
    code_bytes = encoded_bytes or num_instructions * 4
    num_windows = max(code_bytes / 64.0, 1.0)
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
