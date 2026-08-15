"""SIMD (AVX2 / AVX-512 IFMA) vectorization feasibility analyzer.

Evaluates whether arithmetic or bitwise assembly blocks can be mapped to 256-bit
AVX2 vector lanes or 512-bit AVX-512 IFMA (Integer Fused Multiply-Add) lanes using
52-bit redundant limb representations.
"""

from __future__ import annotations

from ..asm_util import extract_mnemonic, instr_lines
from ..types import VectorizationFeasibility


def analyze_vectorization_feasibility(asm: str) -> VectorizationFeasibility:
    """Evaluate whether an assembly kernel can benefit from AVX2 or AVX-512 IFMA vectorization."""
    lines = instr_lines(asm)

    mul_count = 0
    bitwise_count = 0
    shift_count = 0
    add_sub_count = 0

    for line in lines:
        mnem = extract_mnemonic(line)
        if mnem in ("mul", "mulx", "imul"):
            mul_count += 1
        elif mnem in ("xor", "and", "or", "not"):
            bitwise_count += 1
        elif mnem in ("shl", "shr", "sar", "rol", "ror"):
            shift_count += 1
        elif mnem in ("add", "sub", "adc", "sbb", "adcx", "adox"):
            add_sub_count += 1

    total_ops = len(lines)

    # AVX-512 IFMA is advantageous when multiplication density is large (>= 16 mul operations)
    # where 8 concurrent 52x52->104 bit multiply-accumulates per 512-bit ZMM vector amortize
    # the Radix-2^52 format conversion and carry assimilation overhead.
    is_ifma_candidate = mul_count >= 16 or any("vpmadd52" in ln or "%zmm" in ln for ln in lines)

    # AVX2 candidate when parallel bitwise / shift operations dominate
    is_avx2_candidate = (bitwise_count + shift_count >= 4) or (add_sub_count >= 8 and mul_count == 0)

    reasons = []
    if is_ifma_candidate:
        reasons.append(f"High multiplication density ({mul_count} muls) suitable for 52-bit radix AVX-512 IFMA (8 lanes/ZMM)")
    if is_avx2_candidate:
        reasons.append(f"Parallel data paths ({bitwise_count + shift_count} shifts/logic) suitable for 256-bit AVX2 (4 lanes/YMM)")
    if not reasons:
        reasons.append("Scalar sequential carry chains optimal for current operand width")

    return VectorizationFeasibility(
        is_avx2_candidate=is_avx2_candidate,
        is_avx512_ifma_candidate=is_ifma_candidate,
        lane_count_256=4,
        lane_count_512=8,
        rationale="; ".join(reasons),
    )
