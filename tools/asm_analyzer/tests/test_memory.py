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
        self.assertFalse(stats.has_rmw)

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
        self.assertTrue(stats.has_rmw)

    def test_unroll_factor_estimation(self):
        asm = """
        movq %r8, 0(%rdi)
        movq %r9, 8(%rdi)
        movq %r10, 16(%rdi)
        movq %r11, 24(%rdi)
        """
        unroll = estimate_unroll_factor(asm)
        self.assertEqual(unroll, 4)

    def test_lea_pointer_advance_sets_width_but_is_not_memory_access(self):
        asm = """
        movq 0(%rsi), %r8
        movq 8(%rsi), %r9
        movq 16(%rsi), %r10
        movq 24(%rsi), %r11
        leaq 32(%rsi), %rsi
        """
        stats = analyze_memory_accesses(asm)
        self.assertEqual(stats.loads, 4)
        self.assertEqual(stats.stores, 0)
        self.assertEqual(estimate_unroll_factor(asm), 4)

    def test_unroll_factor_handles_backward_destination_offsets(self):
        asm = """
        movq %r8, 0(%rdi)
        movq %r9, -8(%rdi)
        movq %r10, -16(%rdi)
        movq %r11, -24(%rdi)
        """
        self.assertEqual(estimate_unroll_factor(asm), 4)

    def test_source_read_ahead_does_not_inflate_unroll_factor(self):
        asm = """
        movq 0(%rsi), %r8
        movq 8(%rsi), %r9
        movq 16(%rsi), %r10
        movq 24(%rsi), %r11
        movq 32(%rsi), %rax
        movq %r8, 0(%rdi)
        movq %r9, 8(%rdi)
        movq %r10, 16(%rdi)
        movq %r11, 24(%rdi)
        """
        self.assertEqual(estimate_unroll_factor(asm), 4)

    def test_separate_destination_streams_do_not_double_width(self):
        asm = """
        movq %r8, 0(%rdi)
        movq %r9, 8(%rdi)
        movq %r10, 0(%rdx)
        movq %r11, 8(%rdx)
        """
        self.assertEqual(estimate_unroll_factor(asm), 2)

    def test_pointer_stride_overrides_overlapping_output_lane(self):
        asm = """
        movq %r8, 0(%rdi)
        movq %r9, 8(%rdi)
        movq %r10, 16(%rdi)
        movq %r11, 24(%rdi)
        movq %rax, 32(%rdi)
        addq $32, %rsi
        addq $32, %rdi
        """
        self.assertEqual(estimate_unroll_factor(asm), 4)

    def test_prefetch_is_not_a_destination_lane(self):
        asm = """
        prefetcht0 128(%rdi)
        movq %r8, 0(%rdi)
        movq %r9, 8(%rdi)
        movq %r10, 16(%rdi)
        movq %r11, 24(%rdi)
        """
        self.assertEqual(estimate_unroll_factor(asm), 4)

    def test_scaled_index_stride_is_converted_to_bytes(self):
        asm = """
        adcq %rax, 0(%rdi,%rsi,8)
        adcq %rcx, 8(%rdi,%rsi,8)
        adcq %rdx, 16(%rdi,%rsi,8)
        adcq %r8, 24(%rdi,%rsi,8)
        adcq %rax, 32(%rdi,%rsi,8)
        adcq %rcx, 40(%rdi,%rsi,8)
        adcq %rdx, 48(%rdi,%rsi,8)
        adcq %r8, 56(%rdi,%rsi,8)
        leaq 8(%rsi), %rsi
        """
        self.assertEqual(estimate_unroll_factor(asm), 8)

    def test_indexed_addresses_are_single_memory_operands(self):
        asm = """
        movq 8(%rsi,%rcx,8), %rax
        adcq %rax, 8(%rdi,%rcx,8)
        """
        stats = analyze_memory_accesses(asm)
        self.assertEqual(stats.loads, 1)
        self.assertEqual(stats.read_modify_writes, 1)


if __name__ == "__main__":
    unittest.main()
