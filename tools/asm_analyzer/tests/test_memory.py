"""Unit tests for memory interaction and cache straddle feature analysis."""

from __future__ import annotations

import unittest
from asm_analyzer.features.memory import analyze_memory_accesses, estimate_unroll_factor


class TestMemoryAnalysis(unittest.TestCase):
    def test_pure_loads_and_stores(self):
        asm = """
        movq 0(%rsi), %rax
        movq 8(%rsi), %rdx
        movq %rax, 0(%rdi)
        movq %rdx, 8(%rdi)
        """
        stats = analyze_memory_accesses(asm)
        self.assertEqual(stats.loads, 2)
        self.assertEqual(stats.stores, 2)
        self.assertEqual(stats.read_modify_writes, 0)
        self.assertFalse(stats.has_rmw_hazard)

    def test_read_modify_write_detection(self):
        asm = """
        movq 0(%rsi), %rax
        addq %rax, 0(%rdi)
        adcq %rdx, 8(%rdi)
        """
        stats = analyze_memory_accesses(asm)
        self.assertEqual(stats.loads, 1)
        self.assertEqual(stats.stores, 0)
        self.assertEqual(stats.read_modify_writes, 2)
        self.assertTrue(stats.has_rmw_hazard)

    def test_unroll_factor_estimation(self):
        asm = """
        movq 0(%rsi), %r8
        movq 8(%rsi), %r9
        movq 16(%rsi), %r10
        movq 24(%rsi), %r11
        """
        unroll = estimate_unroll_factor(asm)
        self.assertEqual(unroll, 4)


if __name__ == "__main__":
    unittest.main()
