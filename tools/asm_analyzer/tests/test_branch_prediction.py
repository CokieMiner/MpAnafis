"""Unit tests for branch target buffer density and loop alignment analyzer."""

from __future__ import annotations

import unittest
from asm_analyzer.features.branch_prediction import analyze_branch_patterns


class TestBranchPredictionAnalysis(unittest.TestCase):
    def test_clean_single_loop_branch(self) -> None:
        asm = """
        .p2align 4
        1:
            movq (%rsi), %rax
            addq %rax, (%rdi)
            decq %rcx
            jnz 1b
        """
        stats = analyze_branch_patterns(asm)
        self.assertEqual(stats.branch_count, 1)
        self.assertFalse(stats.has_btb_density_hazard)
        self.assertFalse(stats.has_unaligned_loop_head)

    def test_unaligned_loop_head_detection(self) -> None:
        asm = """
        1:
            movq (%rsi), %rax
            addq %rax, (%rdi)
            decq %rcx
            jnz 1b
        """
        stats = analyze_branch_patterns(asm)
        self.assertTrue(stats.has_unaligned_loop_head)

    def test_dense_branch_hazard_detection(self) -> None:
        asm = """
            jz 1f
            jnz 2f
            jc 3f
            jnc 4f
            1:
            2:
            3:
            4:
        """
        stats = analyze_branch_patterns(asm)
        self.assertGreaterEqual(stats.branch_count, 4)
        self.assertTrue(stats.has_btb_density_hazard)


if __name__ == "__main__":
    unittest.main()
