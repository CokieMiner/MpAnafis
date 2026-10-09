"""Unit tests for multiplier slack and paired pipelining detection."""

from __future__ import annotations

import unittest
from asm_analyzer.features.multiplier import analyze_multiplier


class TestMultiplierAnalysis(unittest.TestCase):
    def test_paired_pipeline_detection(self):
        asm = """
        mulxq 0(%rsi), %r8, %r9
        mulxq 8(%rsi), %r10, %r11
        adcxq 0(%rdi), %r8
        adoxq %rcx, %r8
        """
        stats = analyze_multiplier(asm)
        self.assertEqual(stats.mul_count, 2)
        self.assertTrue(stats.is_paired_pipeline)

    def test_multiplier_stall_detection(self):
        asm = """
        mulxq 0(%rsi), %rax, %rdx
        addq %rax, %rcx
        """
        stats = analyze_multiplier(asm)
        self.assertEqual(stats.mul_count, 1)
        self.assertEqual(stats.min_slack, 0)
        self.assertTrue(stats.has_multiplier_stall)


if __name__ == "__main__":
    unittest.main()
