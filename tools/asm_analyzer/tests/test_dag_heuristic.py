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


if __name__ == "__main__":
    unittest.main()
