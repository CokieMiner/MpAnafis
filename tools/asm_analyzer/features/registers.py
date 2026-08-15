"""Register usage, pressure, and condition flag analysis.

Tracks unique 64-bit GPRs, SIMD vector registers, and condition flag
dependencies across assembly blocks.
"""

from __future__ import annotations

import re
from typing import Set
from ..types import RegisterStats
from ..asm_util import GPR_ALIAS_MAP, instr_lines

_REG_OPERAND_RE = re.compile(r"%([a-z0-9]+)\b", re.IGNORECASE)
_SIMD_REG_RE = re.compile(r"%([xyz]mm\d+)\b", re.IGNORECASE)


def analyze_registers(asm: str) -> RegisterStats:
    """Extract distinct GPR and SIMD register statistics."""
    gprs: Set[str] = set()
    simds: Set[str] = set()
    flags_read: Set[str] = set()
    flags_written: Set[str] = set()

    for line in instr_lines(asm):
        for reg in _REG_OPERAND_RE.findall(line):
            base = GPR_ALIAS_MAP.get(reg.lower())
            if base and base != "rsp":
                gprs.add(base)

        for simd in _SIMD_REG_RE.findall(line):
            simds.add(simd.lower())

        mnem = line.split(None, 1)[0].lower().split(".")[0]
        if mnem in ("adc", "adcq", "adcl", "sbb", "sbbq", "sbbl", "jc", "jnc", "jb", "jae"):
            flags_read.add("CF")
            flags_written.add("CF")
        elif mnem in ("adcx", "adcxq"):
            flags_read.add("CF")
            flags_written.add("CF")
        elif mnem in ("adox", "adoxq"):
            flags_read.add("OF")
            flags_written.add("OF")
        elif mnem in ("add", "addq", "sub", "subq"):
            flags_written.add("CF")
            flags_written.add("OF")
            flags_written.add("ZF")
            flags_written.add("SF")

    return RegisterStats(
        gprs_used=len(gprs),
        gpr_names=tuple(sorted(gprs)),
        simds_used=len(simds),
        simd_names=tuple(sorted(simds)),
        flags_read=tuple(sorted(flags_read)),
        flags_written=tuple(sorted(flags_written)),
    )
