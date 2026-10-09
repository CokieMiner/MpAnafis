"""Tests for memory disjointness proof relaxation and no-alias pointer analysis."""

from __future__ import annotations

import unittest

from asm_analyzer.search.ast import parse_line
from asm_analyzer.search.dag import build_dag
from asm_analyzer.search.memory_dependencies import (
    analyze_loop_memory_dependencies,
    may_alias,
    memory_operand,
)


class TestNoAliasRelaxation(unittest.TestCase):
    def test_may_alias_default_conservative(self):
        # Different base registers conservatively alias by default
        op1 = memory_operand(parse_line("movq 0(%rsi), %rax"))
        op2 = memory_operand(parse_line("movq %rax, 0(%rdi)"))
        self.assertTrue(may_alias(op1, op2))

    def test_may_alias_with_disjoint_bases(self):
        # When disjoint_bases contains {"rsi", "rdi"}, they do not alias!
        op1 = memory_operand(parse_line("movq 0(%rsi), %rax"))
        op2 = memory_operand(parse_line("movq %rax, 0(%rdi)"))
        self.assertFalse(may_alias(op1, op2, disjoint_bases={"rsi", "rdi"}))

    def test_build_dag_with_disjoint_bases_eliminates_store_load_edge(self):
        asm = """
        movq %rax, 0(%rdi)
        movq 0(%rsi), %rbx
        """
        lines = [line.strip() for line in asm.strip().splitlines()]
        instructions = [parse_line(l) for l in lines if parse_line(l)]

        # Default: store to (%rdi) creates a memory dependency edge to load from (%rsi)
        nodes_default = build_dag(instructions)
        self.assertIn(0, nodes_default[1].preds)

        # With disjoint bases {"rdi", "rsi"}: load from (%rsi) does not depend on store to (%rdi)
        nodes_disjoint = build_dag(instructions, disjoint_bases={"rdi", "rsi"})
        self.assertNotIn(0, nodes_disjoint[1].preds)

    def test_analyze_loop_memory_dependencies_with_disjoint_bases(self):
        asm = """
        movq 0(%rsi), %rax
        movq %rax, 0(%rdi)
        addq $8, %rsi
        addq $8, %rdi
        """
        stats_default = analyze_loop_memory_dependencies(asm)
        self.assertGreater(stats_default.may_alias_pairs, 0)

        stats_disjoint = analyze_loop_memory_dependencies(asm, disjoint_bases={"rsi", "rdi"})
        self.assertEqual(stats_disjoint.may_alias_pairs, 0)
        self.assertGreater(stats_disjoint.proved_disjoint_pairs, 0)


if __name__ == "__main__":
    unittest.main()

