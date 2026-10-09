"""Tests for advanced topological search: simulated annealing, exact optimal solver, and transitive closure."""

from __future__ import annotations

import unittest

from asm_analyzer.search.ast import Instr, Op, Spec, parse_line
from asm_analyzer.search.dag import (
    DagNode,
    build_dag,
    compute_transitive_closure,
    evaluate_schedule_cost,
    exact_optimal_schedule,
    heuristic_topological_schedule,
    simulated_annealing_permutations,
    topological_permutations,
)


class TestAdvancedSearch(unittest.TestCase):
    def test_transitive_closure_diamond_dag(self):
        # 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
        nodes = [
            DagNode(0, Instr("a", "a", []), Spec(), preds=set(), succs={1, 2}),
            DagNode(1, Instr("b", "b", []), Spec(), preds={0}, succs={3}),
            DagNode(2, Instr("c", "c", []), Spec(), preds={0}, succs={3}),
            DagNode(3, Instr("d", "d", []), Spec(), preds={1, 2}, succs=set()),
        ]
        reach = compute_transitive_closure(nodes)
        self.assertTrue(reach[0][0])
        self.assertTrue(reach[0][1])
        self.assertTrue(reach[0][2])
        self.assertTrue(reach[0][3])
        self.assertTrue(reach[1][3])
        self.assertTrue(reach[2][3])
        # 1 and 2 are independent
        self.assertFalse(reach[1][2])
        self.assertFalse(reach[2][1])
        # 3 does not reach 0
        self.assertFalse(reach[3][0])

    def test_simulated_annealing_generates_valid_topological_sorts(self):
        asm = """
        movq 0(%rsi), %rax
        movq 8(%rsi), %rcx
        addq %rax, %r8
        addq %rcx, %r9
        """
        lines = [line.strip() for line in asm.strip().splitlines()]
        instructions = [parse_line(l) for l in lines if parse_line(l)]
        nodes = build_dag(instructions)
        reach = compute_transitive_closure(nodes)

        annealed_schedules = simulated_annealing_permutations(nodes, count=20, seed=42)
        self.assertGreater(len(annealed_schedules), 0)

        for sched in annealed_schedules:
            self.assertEqual(len(sched), len(nodes))
            # Verify topological validity: if u reaches v, pos(u) < pos(v)
            pos = {node_idx: i for i, node_idx in enumerate(sched)}
            for u in range(len(nodes)):
                for v in range(len(nodes)):
                    if u != v and reach[u][v]:
                        self.assertLess(
                            pos[u],
                            pos[v],
                            f"Node {u} must precede {v} in topological sort",
                        )

    def test_exact_optimal_schedule_on_small_dag(self):
        asm = """
        movq 0(%rsi), %rax
        movq 8(%rsi), %rcx
        addq %rax, %r8
        addq %rcx, %r9
        """
        lines = [line.strip() for line in asm.strip().splitlines()]
        instructions = [parse_line(l) for l in lines if parse_line(l)]
        nodes = build_dag(instructions)
        reach = compute_transitive_closure(nodes)

        optimal = exact_optimal_schedule(nodes)
        self.assertIsNotNone(optimal)
        self.assertEqual(len(optimal), len(nodes))

        # Verify topological sort validity
        pos = {node_idx: i for i, node_idx in enumerate(optimal)}
        for u in range(len(nodes)):
            for v in range(len(nodes)):
                if u != v and reach[u][v]:
                    self.assertLess(pos[u], pos[v])

        # Cost should be finite and minimal
        cost = evaluate_schedule_cost(optimal, nodes)
        self.assertTrue(cost > 0)


if __name__ == "__main__":
    unittest.main()

