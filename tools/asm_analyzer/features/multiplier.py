"""Multiplier latency slack and paired pipelining detection.

Measures the instruction distance between a multiplier definition and its first
dependent use, and detects parallel multi-register pipelining across x86,
AArch64, ARM32, RISC-V, PowerPC, and s390x kernels.

The ``min_slack`` field reports instruction count (not estimated cycles) between
a multiply and its first consumer. A slack of 0 means the next instruction
reads a product register, indicating an out-of-order execution latency stall.

Paired pipelining is detected when multiply throughput approaches 1 per cycle,
either through consecutive multiplies or through an interleaved multiply+carry
pattern (e.g., mulx / adcx / adox / mulx on x86, or unrolled mul/umulh on AArch64).
"""

from __future__ import annotations

import re
from typing import List, Optional, Set
from ..types import MultiplierStats
from ..asm_util import instr_lines, extract_mnemonic
from ..asm_util import GPR_ALIAS_MAP as _GPR_MAP

_X86_REG_RE = re.compile(r"%([a-z0-9]+)\b", re.IGNORECASE)
_AARCH64_REG_RE = re.compile(r"\b([xw][0-9]+)\b", re.IGNORECASE)
_ARM_REG_RE = re.compile(r"\b(r[0-9]+|lr|ip|fp)\b", re.IGNORECASE)
_RISCV_REG_RE = re.compile(r"\b([a-t][0-9]|x[0-9]+|zero|ra|sp|gp|tp)\b", re.IGNORECASE)
_PPC_REG_RE = re.compile(r"\b(r[0-9]+)\b", re.IGNORECASE)
_S390X_REG_RE = re.compile(r"%(r[0-9]+)\b", re.IGNORECASE)

_ALL_REG_RES = [_X86_REG_RE, _AARCH64_REG_RE, _ARM_REG_RE, _RISCV_REG_RE, _PPC_REG_RE, _S390X_REG_RE]

# All multiply mnemonics across supported architectures
_MUL_MNEMS = {
    # x86
    "mulx", "mulxq", "mul", "mulq", "imul", "imulq", "mull",
    # AArch64
    "umulh", "smulh", "madd", "msub",
    # ARM 32-bit
    "umull", "umlal", "umaal", "smull", "smlal",
    # RISC-V
    "mulh", "mulhu", "mulhsu", "mulw",
    # PowerPC
    "mulld", "mulhdu", "mullw", "mulhwu",
    # s390x
    "mlgr", "msgr", "msgfr",
}

# Instructions that form the multiply pipeline pattern when interleaved across architectures
_CARRY_CHAIN_MNEMS = {
    # x86
    "adcx", "adcxq", "adox", "adoxq", "adc", "adcq", "addq", "add", "sbb", "sbbq",
    "mov", "movq", "movl", "lea", "leaq",
    # AArch64
    "adds", "adcs", "adc", "sub", "subs", "sbcs", "sbc", "mov", "ldr", "str", "ldp", "stp",
    # ARM32
    "umaal", "umlal", "mov", "ldr", "str",
    # RISC-V
    "sltu", "mv", "ld", "sd", "lw", "sw",
    # PowerPC
    "addc", "adde", "addze", "subfc", "subfe", "mr", "ld", "std",
    # s390x
    "algr", "alcgr", "slgr", "slbgr", "lgr", "lg", "stg", "la",
}


def _extract_all_regs(text: str) -> Set[str]:
    regs: Set[str] = set()
    for rx in _ALL_REG_RES:
        for r in rx.findall(text):
            r_lower = r.lower()
            base = _GPR_MAP.get(r_lower, r_lower)
            regs.add(base)
    return regs


def analyze_multiplier(asm: str) -> MultiplierStats:
    """Analyze multiplier instructions, slack distance, and pipelining across ISAs."""
    lines = instr_lines(asm)
    mul_indices: List[int] = []
    min_slack: Optional[int] = None
    mul_count = 0

    for idx, line in enumerate(lines):
        mnem = extract_mnemonic(line)
        # Check if line contains any multiplication instruction
        is_mul = mnem in _MUL_MNEMS or (mnem == "mul" and not line.strip().startswith("//"))
        if is_mul:
            mul_indices.append(idx)
            mul_count += 1

            parts = line.split(None, 1)
            operands: List[str] = []
            if len(parts) >= 2:
                operands = [op.strip() for op in parts[1].split(",")]

            defined_regs: Set[str] = set()

            # Architecture-specific destination register extraction
            if mnem.startswith("mulx"):
                # x86 BMI2 mulx src, dst_lo, dst_hi
                if len(operands) >= 3:
                    for op in operands[-2:]:
                        defined_regs.update(_extract_all_regs(op))
            elif mnem in ("mul", "mulq", "mull") and "%" in line:
                # x86 1-operand mul implicit rax:rdx
                defined_regs.add("rax")
                defined_regs.add("rdx")
            elif mnem in ("umull", "umlal", "umaal", "smull", "smlal"):
                # ARM 32-bit: umull RdLo, RdHi, Rn, Rm
                if len(operands) >= 2:
                    defined_regs.update(_extract_all_regs(operands[0]))
                    defined_regs.update(_extract_all_regs(operands[1]))
            elif mnem in ("mlgr",):
                # s390x: mlgr R1, R2 defines R1 and R1+1
                if operands:
                    defined_regs.update(_extract_all_regs(operands[0]))
            else:
                # Standard 3-operand destination in operands[0] (AArch64, RISC-V, PowerPC)
                if operands:
                    defined_regs.update(_extract_all_regs(operands[0]))

            # Look forward for the first consumer of defined registers
            slack = 0
            found_consumer = False
            for forward_idx in range(idx + 1, len(lines)):
                forward_line = lines[forward_idx]
                forward_mnem = extract_mnemonic(forward_line)
                if forward_mnem in ("jmp", "jz", "jnz", "js", "jns", "b", "b.ne", "b.eq", "b.lt", "b.ge", "cbz", "cbnz"):
                    continue

                forward_regs = _extract_all_regs(forward_line)
                if any(r in defined_regs for r in forward_regs):
                    found_consumer = True
                    break
                slack += 1

            if found_consumer:
                if min_slack is None or slack < min_slack:
                    min_slack = slack

    # Detect paired pipelining across ISAs
    is_paired = False
    for i in range(len(mul_indices) - 1):
        gap = mul_indices[i + 1] - mul_indices[i]
        if gap == 1:
            # Consecutive multiplies (e.g. back-to-back mul/umulh or mulx)
            is_paired = True
            break
        if gap <= 8:
            # Check if intervening instructions are all carry-chain/forwarding ops
            intervening = lines[mul_indices[i] + 1:mul_indices[i + 1]]
            if all(extract_mnemonic(ln) in _CARRY_CHAIN_MNEMS for ln in intervening):
                is_paired = True
                break

    return MultiplierStats(
        mul_count=mul_count,
        min_slack=min_slack,
        is_paired_pipeline=is_paired,
    )
