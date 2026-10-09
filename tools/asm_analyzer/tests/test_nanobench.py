"""Unit tests for nanoBench analyzer backend registration and support."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from asm_analyzer.backends import make_backends
from asm_analyzer.backends.nanobench import (
    NanobenchAnalyzer,
    _blob_is_idempotent,
    _discover_nanobench_binary,
    _initialization_asm,
    _measurement_shape,
    _parse_cycle_measurement,
    _select_shape,
    _validate_measurable,
)
from asm_analyzer.asm_util import _cpu_name_from_model, _model_name_from_cpuinfo, classify_regs
from asm_analyzer.models import CPUS


class TestNanobenchBackend(unittest.TestCase):
    def test_nanobench_backend_registration(self):
        backends = make_backends(["nanobench"])
        self.assertIn("nanobench", backends)
        self.assertIsInstance(backends["nanobench"], NanobenchAnalyzer)

    def test_nanobench_report_handles_missing_binary(self):
        analyzer = NanobenchAnalyzer()
        # When binary is missing, analyze_report should return ok=False with actionable note
        analyzer._bin = None
        report = analyzer.analyze_report("addq %rax, %rcx", CPUS["znver3"])
        self.assertFalse(report.ok)
        self.assertIn("nanoBench executable not found", report.note)

    def test_nanobench_only_supports_exact_detected_host(self):
        analyzer = NanobenchAnalyzer()
        analyzer._bin = Path("/fake/nanoBench")
        analyzer._host = "znver3"
        self.assertTrue(analyzer.supports(CPUS["znver3"]))
        self.assertFalse(analyzer.supports(CPUS["znver4"]))

    def test_nanobench_rejects_unknown_host(self):
        analyzer = NanobenchAnalyzer()
        analyzer._bin = Path("/fake/nanoBench")
        analyzer._host = "unknown"
        self.assertFalse(analyzer.supports(CPUS["znver3"]))

    def test_cycle_parser_accepts_amd_rdtsc_without_dividing_by_unroll(self):
        self.assertEqual(_parse_cycle_measurement("RDTSC: 18.25\n"), ("RDTSC", 18.25))

    def test_cycle_parser_prefers_core_cycles(self):
        output = "RDTSC: 19.0\nCORE_CYCLES: 17.5\n"
        self.assertEqual(_parse_cycle_measurement(output), ("CORE_CYCLES", 17.5))

    def test_cycle_parser_rejects_unrelated_counters(self):
        self.assertIsNone(_parse_cycle_measurement("INST_RETIRED: 12.0\n"))

    def test_initialization_preserves_nanobench_distinct_memory_bases(self):
        init = _initialization_asm("mulxq (%rdi), %r8, %r9\nmovq %r8, (%rsi)")
        self.assertNotIn("%rdi, %rdi", init)
        self.assertNotIn("%rsi, %rsi", init)
        self.assertIn("movq $2, %rdx", init)
        self.assertIn("testq %rsp, %rsp", init)

    def test_initialization_seeds_scalars_deterministically(self):
        init = _initialization_asm("decq %rbx\njns .Lloop\n.Lloop:\naddq %rbx, %rax")
        self.assertIn("movq $2, %rbx", init)
        self.assertIn("movq $2, %rax", init)
        self.assertNotIn("xorq", init)

    def test_initialization_assigns_other_pointers_to_separate_offsets(self):
        init = _initialization_asm("movq (%rax), %r8\nmovq %r8, (%rcx)")
        self.assertIn("leaq 4096(%r14), %rax", init)
        self.assertIn("leaq 8192(%r14), %rcx", init)

    def test_initialization_normalizes_gpr_aliases_and_ignores_simd_registers(self):
        init = _initialization_asm("xorl %r8d, %r8d\npxor %xmm0, %xmm0")
        self.assertIn("movq $2, %r8", init)
        self.assertNotIn("%r8d", init)
        self.assertNotIn("%xmm0", init)

    def test_host_model_detection_is_independent_of_lscpu_labels(self):
        cpuinfo = "processor: 0\nmodel name: AMD Ryzen AI 7 350 w/ Radeon 860M\n"
        model_name = _model_name_from_cpuinfo(cpuinfo)
        self.assertEqual(model_name, "AMD Ryzen AI 7 350 w/ Radeon 860M")
        self.assertEqual(_cpu_name_from_model(model_name), "znver5")

    def test_measurement_shape_amortizes_noise_without_using_r15(self):
        self.assertEqual(_measurement_shape("addq %rax, %rbx", 200), (64, 16))
        self.assertEqual(_measurement_shape("addq %rax, %r15", 200), (200, 0))

    def test_discover_nanobench_binary(self):
        with patch("asm_analyzer.backends.nanobench.shutil.which", return_value="/opt/nanoBench"):
            self.assertEqual(_discover_nanobench_binary(), Path("/opt/nanoBench"))


class TestMeasurabilityValidation(unittest.TestCase):
    def _check(self, body: str) -> str | None:
        pointers, scalars = classify_regs(body)
        return _validate_measurable(body, pointers, scalars)

    def test_straight_line_block_is_measurable(self):
        self.assertIsNone(self._check("mulxq (%rdi), %r8, %r9\nmovq %r8, (%rsi)"))

    def test_scalar_seeded_loop_counter_is_measurable(self):
        body = "movq %r14, %rcx\n.Ltmp0:\naddq %rax, %rbx\nloop .Ltmp0"
        self.assertIsNone(self._check(body))

    def test_immediate_loop_counter_is_measurable(self):
        body = "movl $4, %ecx\n.Ltmp0:\naddq %rax, %rbx\nloop .Ltmp0"
        self.assertIsNone(self._check(body))

    def test_loop_counter_from_pointer_is_refused(self):
        body = "movq (%rdi), %rax\nmovq %rdi, %rcx\n.Ltmp0:\naddq %rax, %rbx\nloop .Ltmp0"
        note = self._check(body)
        self.assertIsNotNone(note)
        self.assertIn("%rdi", note)

    def test_loop_counter_from_memory_is_refused(self):
        body = "movq (%rdi), %rcx\n.Ltmp0:\naddq %rax, %rbx\nloop .Ltmp0"
        note = self._check(body)
        self.assertIsNotNone(note)
        self.assertIn("(%rdi)", note)

    def test_scalar_dec_loop_is_measurable(self):
        body = ".Ltmp1:\naddq %rax, %rdx\ndecq %rbx\njns .Ltmp1"
        self.assertIsNone(self._check(body))

    def test_pointer_dec_loop_is_refused(self):
        body = ".Ltmp1:\naddq (%rsi), %rax\ndecq %rsi\njnz .Ltmp1"
        note = self._check(body)
        self.assertIsNotNone(note)
        self.assertIn("%rsi", note)

    def test_memory_reloaded_counter_is_refused(self):
        body = "movq (%rdi), %rbx\n.Ltmp1:\naddq %rax, %rbx\ndecq %rbx\njnz .Ltmp1"
        note = self._check(body)
        self.assertIsNotNone(note)
        self.assertIn("%rbx", note)

    def test_forward_branches_are_not_loops(self):
        body = "decq %rbx\njs .Ldone\naddq %rax, %rbx\n.Ldone:"
        self.assertIsNone(self._check(body))


class TestIdempotencyClassification(unittest.TestCase):
    def _idempotent(self, body: str) -> bool:
        _, scalars = classify_regs(body)
        return _blob_is_idempotent(body, scalars)

    def test_straight_line_block_is_idempotent(self):
        self.assertTrue(self._idempotent("mulxq (%rdi), %r8, %r9\nmovq %r8, (%rsi)"))

    def test_loop_reseeded_from_invariant_scalar_is_idempotent(self):
        body = "movq %r14, %rcx\n.Ltmp0:\naddq %rax, %rbx\nloop .Ltmp0"
        self.assertTrue(self._idempotent(body))

    def test_loop_reseeded_from_clobbered_reg_is_one_shot(self):
        # r8 is clobbered by nanoBench's counter reads between copies, so a
        # counter re-derived from it replays init state only on the first copy.
        body = "movq (%rdi), %rax\nmovq %r8, %rcx\n.Ltmp0:\naddq %rax, %rbx\nloop .Ltmp0"
        self.assertFalse(self._idempotent(body))

    def test_mulx_high_output_counts_as_written(self):
        # r14 is mulx's low output here: written in-blob, so reseeding from it
        # is not per-execution invariant even though r14 itself is unclobbered.
        body = "mulxq %rdx, %r14, %r9\nmovq %r14, %rcx\n.Ltmp0:\naddq %rax, %rbx\nloop .Ltmp0"
        self.assertFalse(self._idempotent(body))

    def test_dec_counters_without_reseed_are_one_shot(self):
        body = "decq %rbx\njs .Ldone\n.Ltmp1:\naddq %rax, %rbx\ndecq %rbx\njns .Ltmp1\n.Ldone:\nret"
        self.assertFalse(self._idempotent(body))

    def test_dec_counter_with_immediate_reseed_is_idempotent(self):
        body = "movq $8, %rbx\n.Ltmp1:\naddq %rax, %rbx\ndecq %rbx\njnz .Ltmp1"
        self.assertTrue(self._idempotent(body))

    def test_one_shot_blob_selects_single_execution_shape(self):
        body = "decq %rbx\njs .Ldone\n.Ltmp1:\naddq %rax, %rbx\ndecq %rbx\njns .Ltmp1\n.Ldone:\nret"
        _, scalars = classify_regs(body)
        self.assertEqual(_select_shape(body, 200, scalars), (1, 0))

    def test_idempotent_loop_keeps_averaged_shape(self):
        body = "movq %r14, %rcx\n.Ltmp0:\naddq %rax, %rbx\nloop .Ltmp0"
        _, scalars = classify_regs(body)
        unroll, loop_count = _select_shape(body, 200, scalars)
        self.assertGreater(unroll, 1)
        self.assertGreater(loop_count, 0)


if __name__ == "__main__":
    unittest.main()
