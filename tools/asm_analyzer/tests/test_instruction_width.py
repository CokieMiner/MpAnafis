"""Tests for exact target instruction encoding widths."""

from __future__ import annotations

import unittest

from asm_analyzer.features.instruction_width import analyze_instruction_widths
from asm_analyzer.types import ArchitectureFamily


class TestInstructionWidth(unittest.TestCase):
    def test_x86_variable_width_encodings_are_exact(self):
        stats = analyze_instruction_widths(
            "nop\nmulxq (%rdi), %r8, %r9",
            ArchitectureFamily.X86_64,
        )
        self.assertTrue(stats.exact, stats.error)
        self.assertEqual(stats.instruction_bytes, (1, 5))
        self.assertEqual(stats.total_bytes, 6)

    def test_aarch64_fixed_width_encodings_are_exact(self):
        stats = analyze_instruction_widths(
            "add x0, x0, x1\nret",
            ArchitectureFamily.AARCH64,
        )
        self.assertTrue(stats.exact, stats.error)
        self.assertEqual(stats.instruction_bytes, (4, 4))


if __name__ == "__main__":
    unittest.main()
