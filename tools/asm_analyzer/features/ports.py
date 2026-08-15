"""Execution port pressure and ALU saturation model for Intel, AMD, and ARM CPUs.

Maps assembly instructions to execution port dispatch units:
- Intel Skylake / Alder Lake: Port 0 (ADCX/ALU), Port 1 (MULX/ALU), Port 5 (ALU/Shift), Port 6 (ADOX/ALU)
- AMD Zen 2 / 3 / 4 / 5: ALU0 (Branch/Int), ALU1 (Mul/Int), ALU2, ALU3, ALU4/5 (Zen4+), AGU0-3
- Apple Silicon (M1-M4 / Firestorm): ALU0-5 (6 integer ALUs), M-Pipes (P2/P3), L/S Pipes (P6-P8)
- ARM Neoverse (V1/V2 / Cortex-X): ALU0-3 (4 ALUs), Mul0/1, Load/Store Units
"""

from __future__ import annotations

from typing import Dict
from ..asm_util import extract_mnemonic, instr_lines
from ..types import PortPressureStats

_HAS_MEM = "("


def _is_memory_operand(line: str) -> bool:
    """Return True if the instruction line contains a memory operand."""
    return _HAS_MEM in line or "[" in line


def analyze_port_pressure(asm: str) -> PortPressureStats:
    """Calculate port binding pressure and execution bottlenecks across target architectures."""
    intel_ports: Dict[str, float] = {
        "P0 (ADCX/ALU)": 0.0,
        "P1 (MULX/ALU)": 0.0,
        "P5 (Shift/ALU)": 0.0,
        "P6 (ADOX/ALU)": 0.0,
        "P2/P3 (Load)": 0.0,
        "P4/P7 (Store)": 0.0,
    }

    amd_alus: Dict[str, float] = {
        "ALU0 (Branch/Int)": 0.0,
        "ALU1 (Mul/Int)": 0.0,
        "ALU2 (Int)": 0.0,
        "ALU3 (Int)": 0.0,
        "AGU (Load/Store)": 0.0,
    }

    arm_units: Dict[str, float] = {
        "ALU0-3 (Int/Branch)": 0.0,
        "Mul0/1 (P2/P3 Multiplier)": 0.0,
        "L/S0-2 (Load/Store AGU)": 0.0,
    }

    for line in instr_lines(asm):
        mnem = extract_mnemonic(line)
        has_mem = _is_memory_operand(line)

        # 1. Carry-chain instructions
        if mnem in ("adcx", "adcxq"):
            intel_ports["P0 (ADCX/ALU)"] += 1.0
            amd_alus["ALU0 (Branch/Int)"] += 0.5
            amd_alus["ALU1 (Mul/Int)"] += 0.5
            arm_units["ALU0-3 (Int/Branch)"] += 1.0
        elif mnem in ("adox", "adoxq"):
            intel_ports["P6 (ADOX/ALU)"] += 1.0
            amd_alus["ALU0 (Branch/Int)"] += 0.5
            amd_alus["ALU1 (Mul/Int)"] += 0.5
            arm_units["ALU0-3 (Int/Branch)"] += 1.0
        elif mnem in ("adc", "adcq", "adcl", "sbb", "sbbq", "sbbl", "adcs", "sbcs"):
            intel_ports["P0 (ADCX/ALU)"] += 0.5
            intel_ports["P6 (ADOX/ALU)"] += 0.5
            amd_alus["ALU0 (Branch/Int)"] += 0.25
            amd_alus["ALU1 (Mul/Int)"] += 0.25
            amd_alus["ALU2 (Int)"] += 0.25
            amd_alus["ALU3 (Int)"] += 0.25
            arm_units["ALU0-3 (Int/Branch)"] += 1.0

        # 2. Multiplications
        elif mnem in ("mulx", "mulxq", "mul", "mulq", "imul", "imulq", "umulh", "smulh", "umaal", "umlal"):
            intel_ports["P1 (MULX/ALU)"] += 1.0
            amd_alus["ALU1 (Mul/Int)"] += 1.0
            arm_units["Mul0/1 (P2/P3 Multiplier)"] += 1.0

        # 3. Shifts
        elif mnem in ("shld", "shldq", "shrd", "shrdq", "shl", "shr", "shlx", "shrx", "lsl", "lsr", "asr", "ror"):
            intel_ports["P5 (Shift/ALU)"] += 1.0
            amd_alus["ALU0 (Branch/Int)"] += 1.0
            arm_units["ALU0-3 (Int/Branch)"] += 1.0

        # 4. Moves and Loads/Stores
        elif mnem.startswith("mov") or mnem.startswith("lea") or mnem in ("ldr", "str", "ldp", "stp"):
            if has_mem and not mnem.startswith("lea"):
                if mnem in ("str", "stp") or (line.count(",") >= 1 and "(" in line.split(",")[-1]):
                    intel_ports["P4/P7 (Store)"] += 1.0
                    amd_alus["AGU (Load/Store)"] += 1.0
                    arm_units["L/S0-2 (Load/Store AGU)"] += 1.0
                else:
                    intel_ports["P2/P3 (Load)"] += 0.5
                    amd_alus["AGU (Load/Store)"] += 0.5
                    arm_units["L/S0-2 (Load/Store AGU)"] += 0.5
            else:
                intel_ports["P0 (ADCX/ALU)"] += 0.25
                intel_ports["P1 (MULX/ALU)"] += 0.25
                intel_ports["P5 (Shift/ALU)"] += 0.25
                intel_ports["P6 (ADOX/ALU)"] += 0.25
                amd_alus["ALU0 (Branch/Int)"] += 0.25
                amd_alus["ALU1 (Mul/Int)"] += 0.25
                amd_alus["ALU2 (Int)"] += 0.25
                amd_alus["ALU3 (Int)"] += 0.25
                arm_units["ALU0-3 (Int/Branch)"] += 0.25

        # 5. Generic ALUs
        elif mnem in ("add", "addq", "adds", "sub", "subq", "subs", "xor", "xorl", "dec", "decq", "inc", "incq",
                      "and", "andq", "or", "orq", "orr", "eor", "neg", "negq", "not", "notq"):
            intel_ports["P0 (ADCX/ALU)"] += 0.25
            intel_ports["P1 (MULX/ALU)"] += 0.25
            intel_ports["P5 (Shift/ALU)"] += 0.25
            intel_ports["P6 (ADOX/ALU)"] += 0.25
            amd_alus["ALU0 (Branch/Int)"] += 0.25
            amd_alus["ALU1 (Mul/Int)"] += 0.25
            amd_alus["ALU2 (Int)"] += 0.25
            amd_alus["ALU3 (Int)"] += 0.25
            arm_units["ALU0-3 (Int/Branch)"] += 0.25

        if has_mem and not mnem.startswith("mov") and not mnem.startswith("lea") and mnem not in ("ldr", "str", "ldp", "stp"):
            if mnem not in ("push", "pop", "call", "ret", "jmp", "b", "bl", "br"):
                intel_ports["P2/P3 (Load)"] += 0.5
                amd_alus["AGU (Load/Store)"] += 0.5
                arm_units["L/S0-2 (Load/Store AGU)"] += 0.5

    bottleneck_port = None
    max_cycles = 0.0
    for port, count in intel_ports.items():
        if count > max_cycles:
            max_cycles = count
            bottleneck_port = port

    return PortPressureStats(
        intel_ports=intel_ports,
        amd_alus=amd_alus,
        arm_units=arm_units,
        bottleneck_port=bottleneck_port,
        bottleneck_cycles=max_cycles,
    )
