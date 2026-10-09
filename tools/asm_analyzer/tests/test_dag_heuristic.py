"""Unit tests for critical-path and heuristic list schedulers."""

from __future__ import annotations

import unittest

from asm_analyzer.search.ast import parse_line
from asm_analyzer.search.dag import (
    build_dag,
    compute_critical_paths,
    heuristic_topological_schedule,
    topological_permutations,
)


class TestDagHeuristic(unittest.TestCase):
    def test_critical_path_calculation(self):
        raw_lines = [
            "mulxq %rdx, %rax, %r8",  # latency = 3
            "adcxq %rax, %r9",        # latency = 1 (depends on mulx)
            "adoxq %r8, %r10",        # latency = 1 (depends on mulx)
            "movq %r9, (%rdi)",       # latency = 1 (depends on adcxq)
        ]
        insts = [parse_line(ln) for ln in raw_lines]
        valid_insts = [i for i in insts if i is not None]
        nodes = build_dag(valid_insts)
        depths = compute_critical_paths(nodes)

        # Node 0 (mulx) -> Node 1 (adcx) -> Node 3 (movq store) = 3 + 1 + 1 = 5 cycles depth
        self.assertEqual(depths[0], 5)
        self.assertEqual(depths[1], 2)
        self.assertEqual(depths[3], 1)

    def test_heuristic_scheduler_order(self):
        raw_lines = [
            "movq %r11, %r12",
            "mulxq %rdx, %rax, %r8",
            "adcxq %rax, %r9",
        ]
        insts = [parse_line(ln) for ln in raw_lines]
        valid_insts = [i for i in insts if i is not None]
        nodes = build_dag(valid_insts)

        schedule = heuristic_topological_schedule(nodes)
        self.assertEqual(len(schedule), len(valid_insts))
        # Node 1 (mulxq, depth = 4) should be scheduled ahead of independent low-latency ops
        self.assertEqual(schedule[0], 1)

    def test_permutations_include_heuristic(self):
        raw_lines = [
            "movq %r11, %r12",
            "mulxq %rdx, %rax, %r8",
            "adcxq %rax, %r9",
        ]
        insts = [parse_line(ln) for ln in raw_lines]
        valid_insts = [i for i in insts if i is not None]
        nodes = build_dag(valid_insts)

        perms = topological_permutations(nodes, count=10)
        self.assertGreater(len(perms), 0)
        heuristic = heuristic_topological_schedule(nodes)
        self.assertEqual(perms[0], heuristic)

    def test_register_write_cannot_move_before_prior_read(self):
        insts = [
            parse_line("movq %rax, %rcx"),
            parse_line("movq %rdx, %rax"),
        ]
        nodes = build_dag([inst for inst in insts if inst is not None])
        self.assertIn(0, nodes[1].preds)

    def test_flag_write_cannot_move_before_prior_flag_read(self):
        insts = [
            parse_line("cmpq %rax, %rbx"),
            parse_line("cmovcq %rcx, %rdx"),
            parse_line("addq %r8, %r9"),
        ]
        nodes = build_dag([inst for inst in insts if inst is not None])
        self.assertIn(1, nodes[2].preds)

    def test_unknown_instruction_is_an_ordering_barrier(self):
        insts = [
            parse_line("movq %rax, %rcx"),
            parse_line("mysteryq %rdx, %r8"),
            parse_line("movq %r9, %r10"),
        ]
        nodes = build_dag([inst for inst in insts if inst is not None])
        self.assertIn(0, nodes[1].preds)
        self.assertIn(1, nodes[2].preds)

    def test_memory_writes_preserve_all_prior_memory_observations(self):
        insts = [
            parse_line("movq (%rsi), %rax"),
            parse_line("movq (%rdx), %rcx"),
            parse_line("addq %r8, (%rdi)"),
            parse_line("movq (%r9), %r10"),
        ]
        nodes = build_dag([inst for inst in insts if inst is not None])
        self.assertIn(0, nodes[2].preds)
        self.assertIn(1, nodes[2].preds)
        self.assertIn(2, nodes[3].preds)

    def test_same_base_disjoint_ranges_can_reorder(self):
        insts = [
            parse_line("movq %rax, 0(%rdi)"),
            parse_line("movq 8(%rdi), %rcx"),
        ]
        nodes = build_dag([inst for inst in insts if inst is not None])
        self.assertNotIn(0, nodes[1].preds)

    def test_same_base_overlapping_ranges_remain_ordered(self):
        insts = [
            parse_line("movq %rax, 0(%rdi)"),
            parse_line("movl 4(%rdi), %ecx"),
        ]
        nodes = build_dag([inst for inst in insts if inst is not None])
        self.assertIn(0, nodes[1].preds)

    def test_different_pointer_registers_remain_may_alias(self):
        insts = [
            parse_line("movq %rax, 0(%rdi)"),
            parse_line("movq 64(%rsi), %rcx"),
        ]
        nodes = build_dag([inst for inst in insts if inst is not None])
        self.assertIn(0, nodes[1].preds)


if __name__ == "__main__":
    unittest.main()
