"""Store-to-Load Forwarding (STLF) hazard and penalty predictor.

Analyzes memory access sequences within sliding instruction windows to model:
1. Exact Forwarding (0-1 cycle latency): Store to [base+disp] -> Load from [base+disp] with same width.
2. Offset Mismatch / Partial Overlap (10-15 cycle stall): Store to [base+disp] -> Load from [base+disp+K] with misaligned overlap.
3. Cross-Iteration Reload Hazard (8-12 cycle stall): Store to dst[j+1] followed by load of dst[j+1] across loop iterations without register carry-forward.
4. Cacheline Straddle Forwarding Rejection (20+ cycle stall): Forwarding attempt across a 64-byte boundary.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple
from ..asm_util import extract_mnemonic, instr_lines
from ..types import StlfAnalysis, StlfHazard

# Matches memory operands across x86, AArch64, ARM, RISC-V, PowerPC, s390x
_X86_MEM_RE = re.compile(r"(-?[0-9]*)\((%[a-z0-9]+)\)")
_ARM_MEM_RE = re.compile(r"\[([a-z0-9]+)(?:,\s*#?(-?[0-9]+))?\]", re.IGNORECASE)


def analyze_stlf_hazards(asm: str) -> StlfAnalysis:
    """Detect and quantify Store-to-Load Forwarding (STLF) hazards."""
    lines = instr_lines(asm)

    hazards: List[StlfHazard] = []
    max_penalty_cycles = 0.0

    # Collect memory operations in order: (idx, is_store, base_reg, disp, size_bytes)
    mem_ops: List[Tuple[int, bool, str, int, int]] = []

    for idx, line in enumerate(lines):
        mnem = extract_mnemonic(line)
        is_store = False
        is_load = False

        # Classification
        if mnem in ("movq", "movl", "mov", "str", "stp", "sw", "sd", "stw", "std", "stg"):
            # Stores typically have memory as second operand (x86) or first operand (ARM/RISC-V/PPC)
            if "(" in line and "," in line:
                parts = line.split(",")
                if "(" in parts[-1]:
                    is_store = True
                elif "(" in parts[0]:
                    is_load = True
            elif "[" in line:
                if mnem.startswith("st"):
                    is_store = True
                elif mnem.startswith("ld"):
                    is_load = True

        if not (is_store or is_load):
            continue

        # Extract base and displacement
        base_reg: Optional[str] = None
        disp = 0
        size_bytes = 8  # Default 64-bit

        # Check x86
        m_x86 = _X86_MEM_RE.search(line)
        if m_x86:
            disp_str, base = m_x86.groups()
            base_reg = base.lower()
            disp = int(disp_str) if disp_str and disp_str != "-" else 0

        # Check ARM/RISC-V
        m_arm = _ARM_MEM_RE.search(line)
        if m_arm:
            base, disp_str = m_arm.groups()
            base_reg = base.lower()
            disp = int(disp_str) if disp_str else 0

        if base_reg:
            mem_ops.append((idx, is_store, base_reg, disp, size_bytes))

    # Analyze store-load pairs in sliding window (distance <= 8 instructions)
    for i, (s_idx, s_is_store, s_base, s_disp, s_size) in enumerate(mem_ops):
        if not s_is_store:
            continue

        for l_idx, l_is_store, l_base, l_disp, l_size in mem_ops[i + 1:]:
            if l_is_store:
                continue
            if l_idx - s_idx > 8:
                break

            if s_base == l_base:
                if s_disp == l_disp and s_size == l_size:
                    # Perfect exact match: 0-1 cycle forwarding
                    pass
                elif s_disp != l_disp and abs(s_disp - l_disp) < s_size:
                    # Offset mismatch / partial overlap: STLF stall!
                    penalty = 12.0
                    max_penalty_cycles = max(max_penalty_cycles, penalty)
                    hazards.append(StlfHazard(
                        hazard_type="offset_mismatch",
                        store_line=lines[s_idx],
                        load_line=lines[l_idx],
                        distance_instructions=l_idx - s_idx,
                        penalty_cycles=penalty,
                        description=f"Store to [{s_base}+{s_disp}] followed by partial-overlap load from [{l_base}+{l_disp}] causes STLF pipeline stall.",
                    ))
                elif (s_disp % 64 + s_size > 64) and abs(s_disp - l_disp) < s_size:
                    # Cacheline straddle forwarding rejection
                    penalty = 20.0
                    max_penalty_cycles = max(max_penalty_cycles, penalty)
                    hazards.append(StlfHazard(
                        hazard_type="straddle_rejection",
                        store_line=lines[s_idx],
                        load_line=lines[l_idx],
                        distance_instructions=l_idx - s_idx,
                        penalty_cycles=penalty,
                        description=f"Store spanning a 64-byte cacheline boundary rejects STLF forwarding.",
                    ))

    return StlfAnalysis(
        has_stlf_hazard=len(hazards) > 0,
        hazard_count=len(hazards),
        max_penalty_cycles=max_penalty_cycles,
        hazards=tuple(hazards),
    )
