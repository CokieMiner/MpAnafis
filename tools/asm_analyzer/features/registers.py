"""Register usage, pressure, and condition flag analysis.

Tracks unique 64-bit GPRs, SIMD vector registers, and condition flag
dependencies across assembly blocks.
"""

from __future__ import annotations

import re
from typing import Optional, Set, Tuple

from ..asm_util import GPR_ALIAS_MAP, instr_lines, split_asm_operands
from ..semantics import TargetSemantics, semantics_for
from ..types import ArchitectureFamily, RegisterStats

_REG_OPERAND_RE = re.compile(r"%([a-z0-9]+)\b", re.IGNORECASE)
_SIMD_REG_RE = re.compile(r"%([xyz]mm\d+)\b", re.IGNORECASE)
_AARCH_GPR_RE = re.compile(r"\b[wx](?:[12]?\d|30)\b", re.IGNORECASE)
_AARCH_SIMD_RE = re.compile(r"\b[vdq](?:[12]?\d|3[01])\b", re.IGNORECASE)
_ARM_GPR_RE = re.compile(r"\b(?:r(?:1[0-5]|\d)|lr|sp)\b", re.IGNORECASE)
_NUMBERED_GPR_RE = re.compile(r"(?:%|\$)?r(?:[12]?\d|3[01])\b", re.IGNORECASE)
_RISCV_GPR_RE = re.compile(
    r"\b(?:x(?:[12]?\d|3[01])|a[0-7]|s(?:[0-9]|1[01])|t[0-6]|fp|ra|sp|gp|tp)\b",
    re.IGNORECASE,
)
_MIPS_GPR_RE = re.compile(
    r"\$(?:\d+|a[0-3]|v[01]|t[0-9]|s[0-7]|k[01]|gp|sp|fp|ra)\b",
    re.IGNORECASE,
)
_LOONGARCH_GPR_RE = re.compile(r"\$r(?:[12]?\d|3[01])\b", re.IGNORECASE)
_BARE_NUMBER_RE = re.compile(r"(?<![$%\w])(?:[12]?\d|3[01])(?![\w])")
_RISCV_ALIAS_MAP = {
    "ra": "x1", "sp": "x2", "gp": "x3", "tp": "x4",
    "t0": "x5", "t1": "x6", "t2": "x7", "s0": "x8", "fp": "x8",
    "s1": "x9", "a0": "x10", "a1": "x11", "a2": "x12", "a3": "x13",
    "a4": "x14", "a5": "x15", "a6": "x16", "a7": "x17",
    "s2": "x18", "s3": "x19", "s4": "x20", "s5": "x21",
    "s6": "x22", "s7": "x23", "s8": "x24", "s9": "x25",
    "s10": "x26", "s11": "x27", "t3": "x28", "t4": "x29",
    "t5": "x30", "t6": "x31",
}


def analyze_registers(
    asm: str,
    semantics: Optional[TargetSemantics] = None,
) -> RegisterStats:
    """Extract distinct GPR and SIMD register statistics."""
    target = semantics or semantics_for(ArchitectureFamily.X86_64)
    gprs: Set[str] = set()
    simds: Set[str] = set()
    flags_read: Set[str] = set()
    flags_written: Set[str] = set()

    for line in instr_lines(asm):
        line_gprs, line_simds = _registers_in_line(line, target.architecture)
        gprs.update(line_gprs)
        simds.update(line_simds)

        mnem = line.split(None, 1)[0].lower().split(".")[0]
        if mnem in ("adc", "adcq", "adcl", "sbb", "sbbq", "sbbl"):
            flags_read.add("CF")
            flags_written.update({"CF", "OF", "ZF", "SF"})
        elif mnem in ("jc", "jnc", "jb", "jae"):
            flags_read.add("CF")
        elif mnem in ("adcx", "adcxq"):
            flags_read.add("CF")
            flags_written.add("CF")
        elif mnem in ("adox", "adoxq"):
            flags_read.add("OF")
            flags_written.add("OF")
        elif mnem in ("add", "addq", "addl", "sub", "subq", "subl"):
            flags_written.update({"CF", "OF", "ZF", "SF"})
        elif mnem in ("adcs", "sbcs"):
            flags_read.add("NZCV")
            flags_written.add("NZCV")
        elif mnem in ("adds", "subs"):
            flags_written.add("NZCV")

    return RegisterStats(
        gprs_used=len(gprs),
        gpr_names=tuple(sorted(gprs)),
        simds_used=len(simds),
        simd_names=tuple(sorted(simds)),
        flags_read=tuple(sorted(flags_read)),
        flags_written=tuple(sorted(flags_written)),
        allocatable_gprs=target.allocatable_gprs,
    )


def _registers_in_line(
    line: str,
    architecture: ArchitectureFamily,
) -> Tuple[Set[str], Set[str]]:
    if architecture in (ArchitectureFamily.X86_64, ArchitectureFamily.X86_32):
        gprs = {
            base
            for register in _REG_OPERAND_RE.findall(line)
            if (base := GPR_ALIAS_MAP.get(register.lower())) and base != "rsp"
        }
        simds = {register.lower() for register in _SIMD_REG_RE.findall(line)}
        return gprs, simds

    if architecture is ArchitectureFamily.AARCH64:
        gprs = {
            "x" + register.lower()[1:]
            for register in _AARCH_GPR_RE.findall(line)
        }
        gprs.discard("x31")
        return gprs, {
            register.lower() for register in _AARCH_SIMD_RE.findall(line)
        }

    if architecture is ArchitectureFamily.ARM32:
        gprs = {
            "r14" if register.lower() == "lr" else register.lower()
            for register in _ARM_GPR_RE.findall(line)
        }
        gprs.discard("r13")
        gprs.discard("sp")
        return gprs, set()

    if architecture in (ArchitectureFamily.RISCV64, ArchitectureFamily.RISCV32):
        gprs = {
            _RISCV_ALIAS_MAP.get(register.lower(), register.lower())
            for register in _RISCV_GPR_RE.findall(line)
        }
        gprs.difference_update(("x0", "x2"))
        return gprs, set()

    if architecture in (ArchitectureFamily.MIPS64, ArchitectureFamily.MIPS32):
        gprs = {register.lower() for register in _MIPS_GPR_RE.findall(line)}
        gprs.difference_update(("$0", "$sp"))
        return gprs, set()

    if architecture in (
        ArchitectureFamily.LOONGARCH64,
        ArchitectureFamily.LOONGARCH32,
    ):
        gprs = {register.lower() for register in _LOONGARCH_GPR_RE.findall(line)}
        gprs.discard("$r0")
        return gprs, set()

    if architecture in (ArchitectureFamily.POWER64, ArchitectureFamily.POWER32):
        return _power_registers(line), set()

    if architecture is ArchitectureFamily.S390X:
        gprs = {register.lower() for register in _NUMBERED_GPR_RE.findall(line)}
        gprs.discard("%r15")
        return gprs, set()

    gprs = {register.lower() for register in _NUMBERED_GPR_RE.findall(line)}
    gprs.difference_update(("r0", "%r0", "$r0", "r1", "%r1"))
    return gprs, set()


def _power_registers(line: str) -> Set[str]:
    registers = {
        f"r{register}"
        for register in _BARE_NUMBER_RE.findall(line)
    }
    registers.update(
        register.lower().lstrip("%")
        for register in _NUMBERED_GPR_RE.findall(line)
    )
    operands = split_asm_operands(line.split(None, 1)[1]) if " " in line else []
    if operands and line.split(None, 1)[0].lower().startswith(("addi", "subi")):
        immediate = operands[-1].strip()
        if immediate.isdigit():
            registers.discard(f"r{immediate}")
    registers.discard("r1")
    return registers
