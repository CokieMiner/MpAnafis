"""External process failures, hardware provenance, and CLI validation."""

import contextlib
import io
import platform
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from asm_analyzer.__main__ import main
from asm_analyzer.analyzer import KernelReport
from asm_analyzer.asm_util import GPR64, _cpu_name_from_model
from asm_analyzer.backends.mca_driver import McaDriver
from asm_analyzer.backends.nanobench import _initialization_asm, _select_shape
from asm_analyzer.backends.perf_runner import PerfAnalyzer, _create_bench_assembly
from asm_analyzer.differential.common import compile_and_run, driver_source
from asm_analyzer.differential.x86 import diff_test_x86_64, x86_64_wrapper
from asm_analyzer.features.stlf import analyze_stlf_hazards
from asm_analyzer.models import CPUS
from asm_analyzer.search.hardware import evaluate_hardware_candidates
from asm_analyzer.simulation import simulate_backends
from asm_analyzer.targets import architecture_for_path
from asm_analyzer.types import ArchitectureFamily


class TestExecutionBoundaries(unittest.TestCase):
    @unittest.skipUnless(
        platform.machine().lower() in ("x86_64", "amd64")
        and shutil.which("as") and shutil.which("rustc"),
        "native x86-64 toolchain is required",
    )
    def test_differential_flags_do_not_depend_on_callers_overflow_state(self):
        wrappers = (
            ".globl k_0\nk_0:\n xor %eax, %eax\n jmp wrapped_0\n"
            ".globl k_1\nk_1:\n mov $127, %eax\n addb $1, %al\n jmp wrapped_1\n"
        )
        for index in range(2):
            wrappers += x86_64_wrapper(
                f"wrapped_{index}", "seto %al",
            )
        driver = driver_source(
            2, len(GPR64), len(GPR64) + 1,
            (), (), 10, 64,
        )
        self.assertEqual(compile_and_run(wrappers, driver, 2, 10, False), [True, True])

    @unittest.skipUnless(
        platform.machine().lower() in ("x86_64", "amd64")
        and shutil.which("as") and shutil.which("rustc"),
        "native x86-64 toolchain is required",
    )
    def test_x86_verifier_detects_changes_with_every_register_allocation(self):
        # Each register must be observed, including the incoming ABI pointers.
        for register in GPR64:
            baseline = "\n".join(f"movq $7, %{reg}" for reg in GPR64)
            candidate = baseline + f"\nmovq $9, %{register}"
            with self.subTest(register=register):
                self.assertEqual(diff_test_x86_64([baseline, candidate, baseline], 2, False),
                                 [True, False, True])

    @unittest.skipUnless(
        platform.machine().lower() in ("x86_64", "amd64")
        and shutil.which("as") and shutil.which("rustc"),
        "native x86-64 toolchain is required",
    )
    def test_x86_verifier_preserves_distinct_input_and_output_pointers(self):
        self.assertEqual(diff_test_x86_64([
            "movq %rax, %rbx\naddq %rcx, %rbx",
            "movq %rax, %rbx\nsubq %rcx, %rbx",
        ], 50, False), [True, False])

    def test_x86_verifier_refuses_state_outside_its_snapshot(self):
        for body in ("pxor %xmm0, %xmm0", "vpxor %ymm1, %ymm1, %ymm1",
                     "kmovq %rax, %k1", "fst %st(1)", "mov %rax, (%rsp)"):
            with self.subTest(body=body), self.assertRaises(ValueError):
                x86_64_wrapper("candidate", body)

    def test_differential_protocol_requires_every_candidate_exactly_once(self):
        for output in ("", "CAND 1 OK extra", "CAND 1 OK\nCAND 1 OK", "CAND 99 OK"):
            with self.subTest(output=output), patch(
                "asm_analyzer.differential.common.run",
                return_value=subprocess.CompletedProcess([], 0, output, ""),
            ), self.assertRaises(RuntimeError):
                compile_and_run("", "", 2, 1, False)

    def test_differential_protocol_preserves_explicit_failure(self):
        with patch("asm_analyzer.differential.common.run", return_value=(
            subprocess.CompletedProcess([], 0, "CAND 1 FAIL\nCAND 2 OK", "")
        )):
            self.assertEqual(compile_and_run("", "", 3, 1, False), [True, False, True])

    def test_wsl_arguments_are_not_interpreted_by_a_shell(self):
        arguments = ["llvm-mca", "/mnt/c/kernel with spaces;bad.s"]
        with patch("asm_analyzer.backends.mca_driver.subprocess.run") as run:
            McaDriver("llvm-mca", True)._run(arguments, Path("/tmp"))
        self.assertEqual(run.call_args.args[0], ["wsl", "-e", *arguments])

    def test_backend_generator_is_reused_for_every_cpu(self):
        class Backend:
            def supports(self, cpu):
                return True

            def analyze_report(self, body, cpu):
                return KernelReport("fake", cpu.name, cycles=2.0)

        matrix = simulate_backends("nop", [CPUS["znver3"], CPUS["znver4"]],
                                   iter(["fake"]), {"fake": Backend()})
        self.assertEqual(matrix.cycles, {"znver3": {"fake": 2.0}, "znver4": {"fake": 2.0}})

    def test_invalid_cli_values_are_usage_errors(self):
        for arguments in (
            ["check", "--cpu", "not-a-cpu"], ["check", "--backend", "unknown"],
            ["check", "--cpu", ","], ["check", "--backend", ""],
            ["search", "kernel.s", "--candidates", "0"],
            ["optimize", "--hardware-shortlist", "-1"],
        ):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(arguments)
            self.assertEqual(error.exception.code, 2)

    def test_architecture_uses_nearest_complete_path_marker(self):
        self.assertEqual(architecture_for_path(Path("/tmp/arm/x86_64.rs")), ArchitectureFamily.X86_64)
        self.assertEqual(architecture_for_path(Path("/tmp/farm/kernel.s")), ArchitectureFamily.X86_64)
        self.assertEqual(architecture_for_path(Path("/tmp/aarch64/kernel.s")), ArchitectureFamily.AARCH64)

    def test_ambiguous_brand_names_do_not_identify_an_exact_host(self):
        for name in ("AMD Ryzen 7 5700U", "AMD Ryzen 5 7520U", "Intel Core i7-10710U"):
            self.assertEqual(_cpu_name_from_model(name), "unknown")

    def test_perf_rejects_other_cpus_on_the_same_isa(self):
        analyzer = PerfAnalyzer()
        analyzer._perf = "/fake/perf"
        analyzer._host_name = "znver3"
        analyzer._host_arch = ArchitectureFamily.X86_64
        self.assertTrue(analyzer.supports(CPUS["znver3"]))
        self.assertFalse(analyzer.supports(CPUS["znver4"]))
        analyzer._host_name = "unknown"
        self.assertFalse(analyzer.supports(CPUS["znver3"]))

    def test_hardware_rejects_an_unvalidated_original(self):
        results, failures = evaluate_hardware_candidates(["a", "b"], [False, True], [CPUS["znver3"]], object())
        self.assertEqual(results, [])
        self.assertTrue(failures)

    def test_hardware_rejects_uncalibrated_perf_confirmation(self):
        results, failures = evaluate_hardware_candidates(["a", "b"], [True, True], [CPUS["znver3"]], PerfAnalyzer())
        self.assertEqual(results, [])
        self.assertIn("harness overhead", next(iter(failures)))

    def test_hardware_timeout_is_retained_as_a_failed_measurement(self):
        class Backend:
            def available(self):
                return True

            def supports(self, cpu):
                return True

            def analyze_report(self, body, cpu):
                raise subprocess.TimeoutExpired("benchmark", 1)

        with patch("asm_analyzer.search.hardware.os.sched_getaffinity", side_effect=OSError("unsupported")):
            results, failures = evaluate_hardware_candidates(
                ["a", "b"], [True, True], [CPUS["znver3"]], Backend(), 1, 1,
            )
        self.assertTrue(failures)
        self.assertTrue(all(result.score is None for result in results))

    def test_nanobench_seeds_implicit_multiply_input(self):
        self.assertIn("movq $2, %rax", _initialization_asm("mulq %rbx"))

    def test_nanobench_pointer_drift_selects_one_execution(self):
        self.assertEqual(_select_shape("movq (%rdi), %rax\naddq $4096, %rdi", 200, {"rax"}), (1, 0))

    @unittest.skipUnless(shutil.which("llvm-mc"), "llvm-mc is required")
    def test_riscv_perf_harness_uses_encodable_immediates(self):
        assembly = _create_bench_assembly("add a0, a1, a2", 300, ArchitectureFamily.RISCV64)
        result = subprocess.run(
            ["llvm-mc", "-triple=riscv64", "-filetype=obj", "-o", "/dev/null"],
            input=assembly, text=True, capture_output=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_stlf_uses_actual_access_widths_and_interval_overlap(self):
        for assembly in (
            "movl %eax, (%rdi)\nmovq (%rdi), %rax",
            "movl %eax, 8(%rdi)\nmovq 4(%rdi), %rax",
            "movq %rax, 0x38(%rdi)\nmovq 0x3c(%rdi), %rax",
        ):
            self.assertTrue(analyze_stlf_hazards(assembly).has_stlf_hazard)
        self.assertFalse(analyze_stlf_hazards("movl %eax, (%rdi)\nmovl 4(%rdi), %eax").has_stlf_hazard)


if __name__ == "__main__":
    unittest.main()
