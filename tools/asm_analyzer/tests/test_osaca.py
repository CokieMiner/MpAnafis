"""OSACA output parsing regressions."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from asm_analyzer.backends.osaca import OsacaAnalyzer, _parse_osaca_cycles
from asm_analyzer.models import CPUS


class TestOsacaParsing(unittest.TestCase):
    def test_labelled_summary_uses_largest_steady_state_bound(self):
        self.assertEqual(_parse_osaca_cycles("Throughput (TP): 2.0\nLoop-Carried Dependencies (LCD): 8.0"), 8.0)
        self.assertEqual(_parse_osaca_cycles("Loop-Carried Dependencies (LCD): 2.0\nThroughput (TP): 8.0"), 8.0)

    def test_current_combined_summary_uses_ports_and_lcd_not_critical_path(self):
        output = """
Combined Analysis Report
------------------------
   1 | instruction row

       0.25          0.50                    12.0    1.0

Loop-Carried Dependencies Analysis Report
"""
        self.assertEqual(_parse_osaca_cycles(output), 1.0)

    def test_labelled_summary_remains_supported(self):
        output = "Throughput (TP): 3.50 cycles"
        self.assertEqual(_parse_osaca_cycles(output), 3.5)

    def test_missing_summary_fails_closed(self):
        self.assertIsNone(_parse_osaca_cycles("No final analysis is given"))

    def test_incomplete_model_cannot_supply_a_cycle_estimate(self):
        output = (
            "WARNING: The performance data for 16 instructions is missing.\n"
            "No final analysis is given.\nThroughput (TP): 2.0\n"
        )
        self.assertIsNone(_parse_osaca_cycles(output))
        with patch.object(OsacaAnalyzer, "available", return_value=True), patch(
            "asm_analyzer.backends.osaca._run_osaca_cli", return_value=(None, output),
        ):
            report = OsacaAnalyzer().analyze_report("mulxq (%rdi), %r8, %r9", CPUS["skylake"])
        self.assertFalse(report.ok)
        self.assertIsNone(report.cycles)
        self.assertIn("lacks performance data for 16 instructions", report.note)
        self.assertEqual(report.raw_output, output)


if __name__ == "__main__":
    unittest.main()
