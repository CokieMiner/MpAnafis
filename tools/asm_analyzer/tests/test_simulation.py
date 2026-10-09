"""Tests for cohesive backend matrix execution."""

from __future__ import annotations

import unittest

from asm_analyzer.analyzer import KernelReport
from asm_analyzer.models import CPUS
from asm_analyzer.simulation import simulate_backends


class FakeBackend:
    def __init__(self, cycles=None, note=""):
        self.cycles = cycles
        self.note = note

    def supports(self, _cpu):
        return True

    def analyze_report(self, _asm, cpu):
        return KernelReport(
            backend="fake",
            cpu=cpu.name,
            ok=self.cycles is not None,
            cycles=self.cycles,
            note=self.note,
        )


class TestSimulationMatrix(unittest.TestCase):
    def test_prefers_order_sensitive_cost_for_schedule_search(self):
        class SimulatedBackend(FakeBackend):
            def analyze_report(self, _asm, cpu):
                return KernelReport(
                    backend="fake",
                    cpu=cpu.name,
                    cycles=4.0,
                    simulated_cycles=7.0,
                )

        matrix = simulate_backends(
            "addq %rax, %rbx",
            [CPUS["znver3"]],
            ["fake"],
            {"fake": SimulatedBackend(4.0)},
        )
        self.assertEqual(matrix.cycles, {"znver3": {"fake": 4.0}})
        self.assertEqual(matrix.scheduling_costs, {"znver3": {"fake": 7.0}})
        self.assertEqual(
            matrix.scheduling_metrics,
            {"znver3": {"fake": "simulated_cycles"}},
        )

    def test_retains_qualified_values_and_cpu_median(self):
        matrix = simulate_backends(
            "addq %rax, %rbx",
            [CPUS["znver3"]],
            ["a", "b"],
            {"a": FakeBackend(4.0), "b": FakeBackend(6.0)},
        )
        self.assertEqual(
            matrix.flattened_with_medians(),
            {"znver3/a": 4.0, "znver3/b": 6.0, "znver3": 5.0},
        )
        self.assertEqual(matrix.failures, ())

    def test_supported_backend_failure_is_not_silently_dropped(self):
        matrix = simulate_backends(
            "mulxq (%rax), %r8, %r9",
            [CPUS["znver3"]],
            ["broken"],
            {"broken": FakeBackend(note="instruction model missing\ndetails")},
        )
        self.assertEqual(matrix.flattened_with_medians(), {})
        self.assertEqual(
            matrix.failure_messages(),
            ("znver3/broken: instruction model missing",),
        )


if __name__ == "__main__":
    unittest.main()
