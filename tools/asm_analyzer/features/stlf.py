"""Store-to-Load Forwarding (STLF) hazard and penalty predictor.

Screens nearby store/load address forms for exact matches, partial overlaps, and
cache-line straddles. Reported penalty magnitudes are generic heuristics rather
than target-specific cycle predictions.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..asm_util import extract_mnemonic, instr_lines, split_asm_operands
from ..types import StlfAnalysis, StlfHazard

# Matches memory operands across x86, AArch64, ARM, RISC-V, PowerPC, s390x
_X86_MEM_RE = re.compile(r"([+-]?(?:0x[\da-fA-F]+|\d+))?\((%[a-z0-9]+)\)")
_ARM_MEM_RE = re.compile(r"\[([a-z0-9]+)(?:,\s*#?([+-]?(?:0x[\da-fA-F]+|\d+)))?\]", re.IGNORECASE)


def analyze_stlf_hazards(
    asm: str,
    access_width_bytes: int = 8,
    cache_line_bytes: int = 64,
) -> StlfAnalysis:
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

        _STORE_MNEMS = ("movq", "movl", "mov", "str", "stp", "stur", "sw", "sd", "stw", "std", "stg")
        _LOAD_MNEMS = ("movq", "movl", "mov", "ldr", "ldp", "ldur", "lw", "ld", "lwz", "lg", "lgr")

        # Classification
        if mnem in _STORE_MNEMS or mnem in _LOAD_MNEMS:
            if "(" in line and "," in line:
                parts = split_asm_operands(line.split(None, 1)[1])
                if "(" in parts[-1] and mnem in _STORE_MNEMS:
                    is_store = True
                elif "(" in parts[0] and mnem in _LOAD_MNEMS:
                    is_load = True
                elif mnem in _STORE_MNEMS and mnem not in _LOAD_MNEMS:
                    is_store = True
                elif mnem in _LOAD_MNEMS and mnem not in _STORE_MNEMS:
                    is_load = True
            elif "[" in line:
                if mnem.startswith("st") and mnem in _STORE_MNEMS:
                    is_store = True
                elif mnem.startswith("ld") and mnem in _LOAD_MNEMS:
                    is_load = True
            elif "(" in line:
                if mnem in _STORE_MNEMS:
                    is_store = True
                elif mnem in _LOAD_MNEMS:
                    is_load = True

        if not (is_store or is_load):
            continue

        # Extract base and displacement
        base_reg: Optional[str] = None
        disp = 0
        size_bytes = access_width_bytes
        raw_mnemonic = line.split(None, 1)[0].lower()
        if raw_mnemonic in ("movb", "movw", "movl", "movq"):
            size_bytes = {"b": 1, "w": 2, "l": 4, "q": 8}[raw_mnemonic[-1]]
        elif "[" in line:
            register = line.split(None, 1)[1].split(",", 1)[0].strip()
            size_bytes = 16 if register.startswith("q") else 4 if register.startswith(("w", "r")) else 8
            if mnem in ("ldp", "stp"):
                size_bytes *= 2

        # Check x86
        m_x86 = _X86_MEM_RE.search(line)
        if m_x86:
            disp_str, base = m_x86.groups()
            base_reg = base.lower()
            disp = int(disp_str, 16 if "x" in disp_str.lower() else 10) if disp_str else 0

        # Check ARM/RISC-V
        m_arm = _ARM_MEM_RE.search(line)
        if m_arm:
            base, disp_str = m_arm.groups()
            base_reg = base.lower()
            disp = int(disp_str, 16 if "x" in disp_str.lower() else 10) if disp_str else 0

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

            # Check if the base register was modified between the store and the load
            # (e.g., post-increment `str x0, [x1], #8` or `add x1, x1, #8`).
            base_modified = False
            for intervening_idx in range(s_idx, l_idx):
                intervening_line = lines[intervening_idx]
                # A simple check: if the base register appears as a destination or in a post-increment
                if "]" in intervening_line and s_base in intervening_line:
                    suffix = intervening_line.split("]", 1)[1].strip()
                    if suffix.startswith(",") or suffix.startswith("!"):
                        base_modified = True
                        break
                # Check for explicit ALU modifications to the base register
                mnem = extract_mnemonic(intervening_line)
                if mnem in ("add", "sub", "lea", "inc", "dec", "adds", "subs"):
                    parts = intervening_line.split(None, 1)
                    if len(parts) > 1 and re.search(rf'(?<!\w){re.escape(s_base)}(?!\w)', parts[1].split(",")[0]):
                        base_modified = True
                        break
            if base_modified:
                break

            if s_base == l_base and s_disp < l_disp + l_size and l_disp < s_disp + s_size:
                if s_disp == l_disp and s_size == l_size:
                    # Perfect exact match: 0-1 cycle forwarding
                    pass
                elif (
                    s_disp % cache_line_bytes + s_size > cache_line_bytes
                ):
                    penalty = 20.0
                    max_penalty_cycles = max(max_penalty_cycles, penalty)
                    hazards.append(StlfHazard(
                        hazard_type="straddle_rejection",
                        store_line=lines[s_idx],
                        load_line=lines[l_idx],
                        distance_instructions=l_idx - s_idx,
                        penalty_cycles=penalty,
                        description=(
                            f"Store spanning a {cache_line_bytes}-byte cache-line "
                            "boundary may reject forwarding."
                        ),
                    ))
                else:
                    # Offset mismatch / partial overlap: STLF stall!
                    penalty = 12.0
                    max_penalty_cycles = max(max_penalty_cycles, penalty)
                    hazards.append(StlfHazard(
                        hazard_type="offset_mismatch" if s_disp != l_disp else "size_mismatch",
                        store_line=lines[s_idx],
                        load_line=lines[l_idx],
                        distance_instructions=l_idx - s_idx,
                        penalty_cycles=penalty,
                        description=(
                            f"Store to [{s_base}+{s_disp}] followed by a "
                            f"partial-overlap load from [{l_base}+{l_disp}] may "
                            "stall store-to-load forwarding."
                        ),
                    ))

    return StlfAnalysis(
        has_stlf_hazard=len(hazards) > 0,
        hazard_count=len(hazards),
        max_penalty_cycles=max_penalty_cycles,
        hazards=tuple(hazards),
    )
