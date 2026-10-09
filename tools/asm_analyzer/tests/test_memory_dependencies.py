"""Tests for alias proofs and loop-carried memory dependencies."""

from __future__ import annotations

import unittest

from asm_analyzer.search.memory_dependencies import analyze_loop_memory_dependencies


class TestMemoryDependencies(unittest.TestCase):
    def test_stride_proves_adjacent_iterations_disjoint(self):
        stats = analyze_loop_memory_dependencies(
            "\n".join(
                (
                    "movq 0(%rdi), %rax",
                    "movq %rax, 0(%rsi)",
                    "addq $8, %rdi",
                    "addq $8, %rsi",
                )
            )
        )
        self.assertEqual(stats.pointer_strides, {"rdi": 8, "rsi": 8})
        self.assertEqual(stats.loop_carried_dependencies, ())
        self.assertGreater(stats.unknown_cross_iteration_pairs, 0)

    def test_short_stride_reports_loop_carried_overlap(self):
        stats = analyze_loop_memory_dependencies(
            "\n".join(
                (
                    "movq %rax, 0(%rdi)",
                    "addq $4, %rdi",
                )
            )
        )
        self.assertEqual(stats.pointer_strides, {"rdi": 4})
        self.assertEqual(len(stats.loop_carried_dependencies), 1)
        self.assertIn("next-iteration store", stats.loop_carried_dependencies[0])

    def test_within_iteration_counts_proved_disjoint_ranges(self):
        stats = analyze_loop_memory_dependencies(
            "\n".join(
                (
                    "movq %rax, 0(%rdi)",
                    "movq 8(%rdi), %rcx",
                    "movl 4(%rdi), %edx",
                )
            )
        )
        self.assertEqual(stats.proved_disjoint_pairs, 1)
        self.assertEqual(stats.may_alias_pairs, 1)


if __name__ == "__main__":
    unittest.main()
