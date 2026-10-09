"""Tests for PerfAnalyzer backend and multi-ISA benchmark harness generation."""

from __future__ import annotations

import unittest

from asm_analyzer.backends.perf_runner import (
    PerfAnalyzer,
    _create_bench_assembly,
    _parse_perf_cycles,
)
from asm_analyzer.types import ArchitectureFamily


class TestPerfRunner(unittest.TestCase):
    def test_parse_perf_cycles(self):
        sample_stderr = """
# Performance counter stats for '/tmp/bench':

       1,250,400,,cycles,1250400,100.00,,

       0.000451230 seconds time elapsed
"""
        # Test clean comma-delimited perf output parsing
        parsed = _parse_perf_cycles("1250400,,cycles,1250400,100.00,,")
        self.assertEqual(parsed, 1250400.0)

    def test_create_bench_assembly_x86(self):
        code = _create_bench_assembly("addq %rax, %rbx", loops=100, arch=ArchitectureFamily.X86_64)
        self.assertIn("mov $100, %rcx", code)
        self.assertIn("addq %rax, %rbx", code)
        self.assertIn("jnz 1b", code)
        self.assertIn("syscall", code)

    def test_create_bench_assembly_aarch64(self):
        code = _create_bench_assembly("add x0, x1, x2", loops=500, arch=ArchitectureFamily.AARCH64)
        self.assertIn("mov x19, #500", code)
        self.assertIn("add x0, x1, x2", code)
        self.assertIn("b.ne 1b", code)
        self.assertIn("svc #0", code)

    def test_create_bench_assembly_riscv(self):
        code = _create_bench_assembly("add a0, a1, a2", loops=300, arch=ArchitectureFamily.RISCV64)
        self.assertIn("li s0, 300", code)
        self.assertIn("add a0, a1, a2", code)
        self.assertIn("bnez s0, 1b", code)
        self.assertIn("ecall", code)

    def test_perf_analyzer_instantiation(self):
        analyzer = PerfAnalyzer()
        self.assertEqual(analyzer.name, "perf")


if __name__ == "__main__":
    unittest.main()

