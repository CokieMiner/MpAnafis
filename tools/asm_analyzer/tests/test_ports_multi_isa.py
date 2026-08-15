"""Unit tests for multi-ISA execution port pressure modeling."""

from __future__ import annotations

import unittest
from asm_analyzer.features.ports import analyze_port_pressure


class TestMultiISAPorts(unittest.TestCase):
    def test_x86_adx_port_pressure(self):
        asm = """
        mulxq (%rsi), %rax, %r8
        adcxq %rax, %r9
        adoxq %r8, %r10
        """
        stats = analyze_port_pressure(asm)
        self.assertGreater(stats.intel_ports["P1 (MULX/ALU)"], 0.0)
        self.assertGreater(stats.intel_ports["P0 (ADCX/ALU)"], 0.0)
        self.assertGreater(stats.intel_ports["P6 (ADOX/ALU)"], 0.0)
        self.assertGreater(stats.amd_alus["ALU1 (Mul/Int)"], 0.0)

    def test_arm_aarch64_port_pressure(self):
        asm = """
        ldp x2, x3, [x0]
        mul x4, x2, x1
        umulh x5, x2, x1
        adds x6, x6, x4
        adcs x7, x7, x5
        """
        stats = analyze_port_pressure(asm)
        self.assertGreater(stats.arm_units["Mul0/1 (P2/P3 Multiplier)"], 0.0)
        self.assertGreater(stats.arm_units["ALU0-3 (Int/Branch)"], 0.0)
        self.assertGreater(stats.arm_units["L/S0-2 (Load/Store AGU)"], 0.0)


if __name__ == "__main__":
    unittest.main()
