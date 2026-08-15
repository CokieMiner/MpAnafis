"""Unit tests for AArch64 instruction and register analysis."""

from __future__ import annotations

import unittest
from asm_analyzer.features.aarch64 import analyze_aarch64_instructions


class TestAArch64Analysis(unittest.TestCase):
    def test_pair_memory_operations(self):
        asm = """
        ldp x2, x3, [x0, #16]
        stp x4, x5, [x1, #16]
        """
        stats = analyze_aarch64_instructions(asm)
        self.assertEqual(stats["pair_loads_ldp"], 1)
        self.assertEqual(stats["pair_stores_stp"], 1)
        self.assertEqual(stats["loads"], 2)
        self.assertEqual(stats["stores"], 2)

    def test_multiply_and_carry(self):
        asm = """
        mul x4, x2, x0
        umulh x5, x2, x0
        adcs x6, x4, x1
        """
        stats = analyze_aarch64_instructions(asm)
        self.assertEqual(stats["mul_count"], 1)
        self.assertEqual(stats["umulh_count"], 1)
        self.assertEqual(stats["carry_instructions"], 1)


if __name__ == "__main__":
    unittest.main()
