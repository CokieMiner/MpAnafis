"""Tests for structured LLVM-MCA execution-model parsing."""

from __future__ import annotations

import unittest

from asm_analyzer.backends.mca_driver import (
    _parse_instruction_metrics,
    _parse_resource_pressure,
)
from asm_analyzer.backends.llvm_mca import LlvmMcaAnalyzer
from asm_analyzer.models import CPUS


class TestLlvmMcaDetails(unittest.TestCase):
    def test_report_separates_throughput_from_order_sensitive_simulation(self):
        report = LlvmMcaAnalyzer().analyze_report(
            "addq %rax, %rbx\naddq %rcx, %rdx",
            CPUS["znver3"],
            iterations=10,
        )
        if not report.ok:
            self.skipTest(report.note)
        self.assertIsNotNone(report.cycles)
        self.assertIsNotNone(report.simulated_cycles)
        self.assertEqual(report.scheduling_metric(), "simulated_cycles")

    def test_resource_names_are_bound_to_per_iteration_pressure(self):
        lines = """
Resources:
[0]   - ALU0
[1.0] - Load

Resource pressure per iteration:
[0]    [1.0]
2.50   1.00
""".splitlines()
        self.assertEqual(
            _parse_resource_pressure(lines),
            {"ALU0": 2.5, "Load": 1.0},
        )

    def test_instruction_latency_throughput_and_uops_are_retained(self):
        lines = """
[1]    [2]    [3]    [4]    [5]    [6]    Instructions:
 2      8     1.00    *                   mulxq (%rsi), %rax, %rcx
 1      1     0.25                        addq %rax, %rcx

Resources:
""".splitlines()
        metrics = _parse_instruction_metrics(lines)
        self.assertEqual(len(metrics), 2)
        self.assertEqual(metrics[0].latency, 8.0)
        self.assertEqual(metrics[0].throughput, 1.0)
        self.assertEqual(metrics[0].uops, 2.0)
        self.assertTrue(metrics[0].line.startswith("mulxq"))


if __name__ == "__main__":
    unittest.main()
