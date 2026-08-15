"""Decode width and µOp cache (DSB / Op-Cache) saturation analyzer.

Evaluates whether an unrolled loop body fits comfortably within the target CPU's
µOp cache (Intel DSB / AMD Op-Cache) or risks overflowing into the legacy MITE
instruction decoder (which caps throughput at 4-6 instructions per cycle).
"""

from __future__ import annotations

from typing import Dict
from ..asm_util import extract_mnemonic, instr_lines
from ..types import UopCacheStats

# Approximate average instruction byte lengths by mnemonic on x86_64
_APPROX_BYTE_SIZES: Dict[str, int] = {
    "mov": 4, "add": 4, "sub": 4, "adc": 4, "sbb": 4, "adcx": 5, "adox": 5,
    "mul": 4, "mulx": 5, "imul": 4, "xor": 3, "and": 4, "or": 4,
    "shl": 4, "shr": 4, "sar": 4, "ror": 4, "rol": 4,
    "leaq": 4, "lea": 4, "dec": 3, "inc": 3, "cmp": 4, "test": 4,
    "jmp": 2, "jz": 2, "jnz": 2, "js": 2, "jns": 2, "jc": 2, "jnc": 2,
}

# Approximate µOp count per instruction on Intel Core / AMD Zen.
# These are averages across microarchitectures:
#   mul/mulx:  Zen3+ = 1 µOp, Zen2 = 2, Skylake = 3 µOps (uses 2 as average)
#   adcx/adox: 1 µOp on all modern x86
_APPROX_UOP_COUNTS: Dict[str, int] = {
    "mov": 1, "add": 1, "sub": 1, "adc": 1, "sbb": 1, "adcx": 1, "adox": 1,
    "mul": 2, "mulx": 2, "imul": 2, "xor": 1, "and": 1, "or": 1,
    "shl": 1, "shr": 1, "sar": 1, "ror": 1, "rol": 1,
    "shlx": 1, "shrx": 1, "sarx": 1,
    "leaq": 1, "lea": 1, "dec": 1, "inc": 1, "cmp": 1, "test": 1,
    "jmp": 1, "jz": 1, "jnz": 1, "js": 1, "jns": 1, "jc": 1, "jnc": 1,
    "neg": 1, "not": 1, "shld": 2, "shrd": 2,
}

# Microarchitecture DSB / Op-Cache capacities (modern Golden Cove / Zen 3-5: 4096 µOps)
INTEL_DSB_UOP_LIMIT = 256  # Target limit for a tight innermost loop body
AMD_OPCACHE_UOP_LIMIT = 512


def analyze_uop_cache(asm: str) -> UopCacheStats:
    """Analyze assembly body to estimate byte size, µOp count, and cache saturation."""
    lines = instr_lines(asm)
    instruction_count = len(lines)

    total_bytes = 0
    total_uops = 0

    for line in lines:
        mnem = extract_mnemonic(line)
        base_size = _APPROX_BYTE_SIZES.get(mnem, 4)
        base_uops = _APPROX_UOP_COUNTS.get(mnem, 1)

        # Memory operands add byte displacement overhead and load/store µOps
        if "(" in line and ")" in line:
            base_size += 2
            if not (mnem.startswith("mov") or mnem.startswith("lea")):
                base_uops += 1  # Memory-folded ALU arithmetic generates an extra load µOp

        total_bytes += base_size
        total_uops += base_uops

    fits_intel = total_uops <= INTEL_DSB_UOP_LIMIT
    fits_amd = total_uops <= AMD_OPCACHE_UOP_LIMIT

    if total_uops <= 32:
        rec_unroll = 16
    elif total_uops <= 64:
        rec_unroll = 8
    elif total_uops <= 128:
        rec_unroll = 4
    else:
        rec_unroll = 2

    return UopCacheStats(
        instruction_count=instruction_count,
        estimated_uops=total_uops,
        estimated_bytes=total_bytes,
        fits_intel_dsb=fits_intel,
        fits_amd_op_cache=fits_amd,
        recommended_max_unroll=rec_unroll,
    )
