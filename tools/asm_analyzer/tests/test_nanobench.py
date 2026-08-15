"""Unit tests for nanoBench analyzer backend registration and support."""

from __future__ import annotations

import unittest
from asm_analyzer.backends import make_backends
from asm_analyzer.backends.nanobench import NanobenchAnalyzer
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


if __name__ == "__main__":
    unittest.main()
