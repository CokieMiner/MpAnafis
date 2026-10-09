"""Unit tests for decode width and µOp cache analyzer."""

from __future__ import annotations

import unittest
from asm_analyzer.features.uop_cache import analyze_uop_cache


class TestUopCacheAnalysis(unittest.TestCase):
    def test_small_loop_fits_dsb(self) -> None:
        asm = """
        1:
            movq 0(%rsi), %rax
            addq %rax, 0(%rdi)
            leaq 8(%rsi), %rsi
            leaq 8(%rdi), %rdi
            decq %rcx
            jnz 1b
        """
        stats = analyze_uop_cache(asm)
        self.assertTrue(stats.fits_intel_dsb)
        self.assertTrue(stats.fits_amd_op_cache)
        self.assertGreater(stats.estimated_bytes, 0)
        self.assertGreater(stats.estimated_uops, 0)
        self.assertEqual(stats.instruction_count, 6)

    def test_large_unrolled_body_sizing(self) -> None:
        # Create an oversized body with 300 instructions (> 256 limit)
        body = "\n".join(["movq (%rsi), %rax\naddq %rax, (%rdi)"] * 150)
        stats = analyze_uop_cache(body)
        self.assertFalse(stats.fits_intel_dsb)
        self.assertLessEqual(stats.recommended_max_unroll, 4)


if __name__ == "__main__":
    unittest.main()
