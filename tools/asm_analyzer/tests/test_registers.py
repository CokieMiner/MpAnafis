"""Unit tests for register pressure and condition flag tracking."""

from __future__ import annotations

import unittest
from asm_analyzer.features.registers import analyze_registers


class TestRegisterAnalysis(unittest.TestCase):
    def test_gpr_counting(self):
        asm = """
        movq %rax, %rcx
        addq %rdx, %r8
        movq %r9, %r10
        """
        stats = analyze_registers(asm)
        self.assertEqual(stats.gprs_used, 6)
        self.assertIn("rax", stats.gpr_names)
        self.assertIn("rcx", stats.gpr_names)
        self.assertIn("r8", stats.gpr_names)
        self.assertFalse(stats.is_gpr_pressure_high)

    def test_adx_flag_tracking(self):
        asm = """
        adcxq %r8, %rax
        adoxq %r9, %rdx
        """
        stats = analyze_registers(asm)
        self.assertIn("CF", stats.flags_read)
        self.assertIn("OF", stats.flags_read)


if __name__ == "__main__":
    unittest.main()
