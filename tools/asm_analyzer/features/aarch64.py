"""AArch64 (ARMv8-A / ARMv9-A) instruction and execution model.

Analyzes 64-bit pair loads/stores (LDP/STP), 64x64->128 multiplication (MUL/UMULH),
and condition code carry chains (ADCS/SBCS) on ARM Neoverse and Apple Silicon cores.
"""

from __future__ import annotations

import re
from typing import Dict, Set

from ..asm_util import extract_mnemonic, instr_lines

_AARCH64_GPR_RE = re.compile(r"\b([xw](?:[0-9]|[12][0-9]|30))\b", re.IGNORECASE)
_AARCH64_MEM_RE = re.compile(r"\[([^\]]+)\]")


def analyze_aarch64_instructions(asm: str) -> Dict[str, object]:
    """Extract AArch64-specific memory, arithmetic, and register metrics."""
    loads = 0
    stores = 0
    pair_loads = 0
    pair_stores = 0
    muls = 0
    umulhs = 0
    carry_adds = 0
    gprs: Set[str] = set()

    for line in instr_lines(asm):
        mnem = extract_mnemonic(line)

        # Track registers X0..X30
        for reg in _AARCH64_GPR_RE.findall(line):
            gprs.add(reg.lower())

        # Memory classification
        if mnem == "ldp":
            pair_loads += 1
            loads += 2
        elif mnem == "stp":
            pair_stores += 1
            stores += 2
        elif mnem.startswith("ldr"):
            loads += 1
        elif mnem.startswith("str"):
            stores += 1

        # Multiplication
        if mnem == "mul":
            muls += 1
        elif mnem == "umulh":
            umulhs += 1

        # Carry arithmetic
        if mnem in ("adcs", "adc", "sbcs", "sbc"):
            carry_adds += 1

    return {
        "target_arch": "aarch64",
        "gprs_used": len(gprs),
        "loads": loads,
        "stores": stores,
        "pair_loads_ldp": pair_loads,
        "pair_stores_stp": pair_stores,
        "mul_count": muls,
        "umulh_count": umulhs,
        "carry_instructions": carry_adds,
    }
