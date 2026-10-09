"""Unit tests for instruction search fail-closed safety and differential testing."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from asm_analyzer.analyzer import KernelReport
from asm_analyzer.models import CPUS
from asm_analyzer.search.engine import _rank_candidates, search_kernel
from asm_analyzer.search.hardware import evaluate_hardware_candidates
from asm_analyzer.search.results import CandidateResult
from asm_analyzer.types import ArchitectureFamily


class TestSearchSafety(unittest.TestCase):
    def test_search_prefers_order_sensitive_backend_cost(self):
        asm = "movq %rax, %rbx\nmovq %rcx, %rdx"

        class FakeBackend:
            def supports(self, _cpu):
                return True

            def analyze_report(self, body, cpu):
                return KernelReport(
                    backend="llvm-mca",
                    cpu=cpu.name,
                    cycles=4.0,
                    simulated_cycles=9.0 if body != asm else 10.0,
                )

        with patch(
            "asm_analyzer.search.engine.diff_test_variants",
            return_value=[True] * 9,
        ), patch(
            "asm_analyzer.search.engine.make_backends",
            return_value={"llvm-mca": FakeBackend()},
        ):
            results, error = search_kernel(
                asm,
                [CPUS["znver3"]],
                backend_names=["llvm-mca"],
                candidates_count=8,
            )
        self.assertEqual(error, "")
        self.assertNotEqual(results[0].idx, 0)
        self.assertEqual(results[0].cycles["znver3"]["llvm-mca"], 9.0)
        self.assertEqual(
            results[0].provenance["static_cost_metrics"],
            {"znver3": {"llvm-mca": "simulated_cycles"}},
        )

    def test_search_fails_closed_when_diff_test_fails(self):
        asm = """
        movq %rax, %rcx
        addq %rdx, %rcx
        """
        cpus = [CPUS["znver3"]]

        # Mock diff_test_variants to simulate a compiler or runtime failure
        with patch("asm_analyzer.search.engine.diff_test_variants", side_effect=RuntimeError("Compilation failed in as")):
            results, err = search_kernel(asm, cpus, candidates_count=5, run_diff_test=True)
            self.assertIn("Differential testing error", err)
            # All candidates should have been marked invalid (not verified)
            self.assertEqual(len(results), 0)

    def test_search_succeeds_when_diff_test_passes(self):
        asm = """
        movq %rax, %rcx
        addq %rdx, %rcx
        """
        cpus = [CPUS["znver3"]]

        class FakeBackend:
            def supports(self, _cpu):
                return True

            def analyze_report(self, body, cpu):
                return KernelReport(
                    backend="llvm-mca",
                    cpu=cpu.name,
                    cycles=float(len(body)),
                )

        with patch("asm_analyzer.search.engine.diff_test_variants", return_value=[True, True, True]), \
             patch("asm_analyzer.search.engine.make_backends", return_value={"llvm-mca": FakeBackend()}):
            results, err = search_kernel(
                asm,
                cpus,
                backend_names=["llvm-mca"],
                candidates_count=2,
                run_diff_test=True,
            )
            self.assertEqual(err, "")
            self.assertGreater(len(results), 0)
            self.assertTrue(all(r.is_valid for r in results))

    def test_candidates_are_ranked_by_normalized_regret_and_coverage(self):
        results = [
            CandidateResult(0, "original", True, {"znver3": {"a": 10.0, "b": 20.0}}),
            CandidateResult(1, "faster", True, {"znver3": {"a": 8.0, "b": 18.0}}),
            CandidateResult(2, "partial", True, {"znver3": {"a": 1.0}}),
        ]
        _rank_candidates(results)
        self.assertEqual([result.idx for result in results], [1, 0, 2])
        self.assertEqual(results[0].coverage, 2)
        self.assertEqual(results[2].missing, 1)

    def test_hardware_measurements_follow_abba_order(self):
        calls = []

        class FakeHardwareBackend:
            def available(self):
                return True

            def supports(self, _cpu):
                return True

            def analyze_report(self, body, cpu):
                calls.append(body)
                return KernelReport(
                    backend="nanobench",
                    cpu=cpu.name,
                    note="counter=RDTSC; nanoBench normalized per unrolled copy",
                    cycles=10.0 if body == "A" else 9.0,
                )

        results, failures = evaluate_hardware_candidates(
            ["A", "B"],
            [True, True],
            [CPUS["znver3"]],
            FakeHardwareBackend(),
            screening_rounds=1,
            holdout_rounds=1,
        )
        self.assertEqual(calls, ["A", "B", "B", "A"])
        self.assertEqual(failures, set())
        self.assertEqual(results[0].samples["znver3"]["nanobench"], [10.0, 10.0])
        self.assertEqual(results[1].samples["znver3"]["nanobench"], [9.0, 9.0])
        self.assertEqual(results[0].provenance["measurement_order"], "A/B/B/A")
        self.assertEqual(
            results[0].provenance["measurement_metrics"],
            ["counter=RDTSC; nanoBench normalized per unrolled copy"],
        )

    def test_hardware_ranking_accepts_stable_paired_improvement(self):
        class StableHardwareBackend:
            def available(self):
                return True

            def supports(self, _cpu):
                return True

            def analyze_report(self, body, cpu):
                return KernelReport(
                    backend="nanobench",
                    cpu=cpu.name,
                    cycles=10.0 if body == "A" else 9.0,
                )

        results, failures = evaluate_hardware_candidates(
            ["A", "B"],
            [True, True],
            [CPUS["znver3"]],
            StableHardwareBackend(),
            screening_rounds=9,
            holdout_rounds=9,
        )
        self.assertEqual(failures, set())
        self.assertEqual(results[0].idx, 1)
        self.assertAlmostEqual(results[0].score, 0.9)
        self.assertTrue(results[0].comparisons["znver3"]["confirmed"])
        self.assertTrue(
            results[0].comparisons["znver3"]["holdout"]["passed"],
        )

    def test_hardware_ranking_retains_original_for_unstable_result(self):
        values = iter(
            value
            for ratio in (0.9, 1.1, 0.9, 1.1, 0.9, 1.1, 0.9, 1.1, 0.9)
            for value in (10.0, 10.0 * ratio, 10.0 * ratio, 10.0)
        )

        class UnstableHardwareBackend:
            def available(self):
                return True

            def supports(self, _cpu):
                return True

            def analyze_report(self, _body, cpu):
                return KernelReport(
                    backend="nanobench",
                    cpu=cpu.name,
                    cycles=next(values),
                )

        results, failures = evaluate_hardware_candidates(
            ["A", "B"],
            [True, True],
            [CPUS["znver3"]],
            UnstableHardwareBackend(),
            screening_rounds=9,
            holdout_rounds=9,
        )
        self.assertEqual(failures, set())
        self.assertEqual(results[0].idx, 0)
        candidate = next(result for result in results if result.idx == 1)
        self.assertFalse(candidate.comparisons["znver3"]["confirmed"])
        self.assertIsNone(candidate.comparisons["znver3"]["holdout"])
        self.assertIsNone(candidate.score)

    def test_hardware_tolerates_a_transient_sample_when_enough_pairs_remain(self):
        class MostlyStableHardwareBackend:
            calls = 0

            def available(self):
                return True

            def supports(self, _cpu):
                return True

            def analyze_report(self, body, cpu):
                self.calls += 1
                if self.calls == 1:
                    return KernelReport(
                        backend="nanobench",
                        cpu=cpu.name,
                        ok=False,
                        note="transient counter read",
                    )
                return KernelReport(
                    backend="nanobench",
                    cpu=cpu.name,
                    cycles=10.0 if body == "A" else 9.0,
                )

        results, failures = evaluate_hardware_candidates(
            ["A", "B"],
            [True, True],
            [CPUS["znver3"]],
            MostlyStableHardwareBackend(),
            screening_rounds=10,
            holdout_rounds=10,
        )
        self.assertEqual(failures, set())
        self.assertEqual(results[0].idx, 1)
        screening = results[0].comparisons["znver3"]["screening"]
        self.assertEqual(screening["measurement_errors"], ["znver3/nanobench: transient counter read"])
        self.assertIsNotNone(screening["holm_adjusted_p_value"])

    def test_aarch64_search_requires_native_differential_host(self):
        results, error = search_kernel(
            "add x0, x0, x1\nadd x2, x2, x3",
            [CPUS["neoverse-n1"]],
            architecture=ArchitectureFamily.AARCH64,
        )
        self.assertEqual(results, [])
        self.assertIn("AArch64 host", error)

    def test_hardware_measurement_explains_host_mismatch(self):
        class UnsupportedHardwareBackend:
            def available(self):
                return True

            def supports(self, _cpu):
                return False

        results, failures = evaluate_hardware_candidates(
            ["A", "B"],
            [True, True],
            [CPUS["znver3"]],
            UnsupportedHardwareBackend(),
            screening_rounds=1,
            holdout_rounds=1,
        )
        self.assertTrue(all(result.score is None for result in results))
        self.assertEqual(len(failures), 1)
        self.assertIn("exactly match", next(iter(failures)))

    def test_search_schedules_inside_control_flow_blocks(self):
        asm = """
.Lentry:
movq %rax, %rcx
movq %r8, %r9
jrcxz .Lexit
addq %rdx, %rcx
.Lexit:
movq %rcx, %r10
"""

        class FakeBackend:
            def supports(self, _cpu):
                return True

            def analyze_report(self, body, cpu):
                return KernelReport(
                    backend="llvm-mca",
                    cpu=cpu.name,
                    cycles=float(len(body)),
                )

        with patch(
            "asm_analyzer.search.engine.diff_test_variants",
            return_value=[True] * 8,
        ), patch(
            "asm_analyzer.search.engine.make_backends",
            return_value={"llvm-mca": FakeBackend()},
        ):
            results, error = search_kernel(
                asm,
                [CPUS["znver3"]],
                backend_names=["llvm-mca"],
                candidates_count=4,
            )
        self.assertEqual(error, "")
        self.assertGreaterEqual(len(results), 2)
        self.assertTrue(all("jrcxz .Lexit" in result.body for result in results))

    def test_search_rejects_unparsed_source_lines(self):
        results, error = search_kernel(
            "movq %rax, %rbx\n.byte 0",
            [CPUS["znver3"]],
        )
        self.assertEqual(results, [])
        self.assertIn("parsed instructions and labels", error)

    def test_search_keeps_directives_fixed(self):
        asm = "movq %rax, %rbx\n.p2align 4\nmovq %rcx, %rdx"
        with patch(
            "asm_analyzer.search.engine.diff_test_variants",
            return_value=[True] * 4,
        ):
            results, error = search_kernel(
                asm,
                [CPUS["znver3"]],
                backend_names=["llvm-mca"],
                candidates_count=2,
            )
        self.assertTrue(results)
        self.assertTrue(all(".p2align 4" in result.body for result in results))


if __name__ == "__main__":
    unittest.main()
