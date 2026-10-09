"""Unit tests for SIMD (AVX2 / AVX-512 IFMA) vectorization feasibility analyzer."""

from __future__ import annotations

import unittest
from asm_analyzer.features.vectorization import analyze_vectorization_feasibility


class TestVectorizationFeasibility(unittest.TestCase):
    def test_high_density_multiplication_ifma_candidate(self) -> None:
        asm = "\n".join(f"mulxq {i*8}(%rsi), %r8, %r9" for i in range(16))
        feasibility = analyze_vectorization_feasibility(asm)
        self.assertTrue(feasibility.is_avx512_ifma_candidate)
        self.assertEqual(feasibility.lane_count_512, 8)

    def test_shift_and_bitwise_avx2_candidate(self) -> None:
        asm = """
            shrq $4, %rax
            shlq $60, %rdx
            orq %rdx, %rax
            shrq $4, %rbx
            shlq $60, %rcx
            orq %rcx, %rbx
        """
        feasibility = analyze_vectorization_feasibility(asm)
        self.assertTrue(feasibility.is_avx2_candidate)
        self.assertEqual(feasibility.lane_count_256, 4)


if __name__ == "__main__":
    unittest.main()
