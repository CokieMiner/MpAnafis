"""CLI argument-routing regressions."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from asm_analyzer.__main__ import _build_parser, main


class TestCli(unittest.TestCase):
    def test_subcommand_options_are_not_overwritten(self):
        with patch("asm_analyzer.__main__.run_analyze", return_value=0) as analyze:
            rc = main([
                "analyze",
                "kernel.s",
                "--cpu",
                "znver4",
                "--backend",
                "llvm-mca",
            ])
        self.assertEqual(rc, 0)
        self.assertEqual([cpu.name for cpu in analyze.call_args.kwargs["cpus"]], ["znver4"])
        self.assertEqual(analyze.call_args.kwargs["backend_names"], ["llvm-mca"])

    def test_common_options_before_subcommand_are_rejected_not_ignored(self):
        parser = _build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--cpu", "znver4", "analyze", "kernel.s"])

    def test_removed_calibrate_subcommand_is_rejected(self):
        parser = _build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["calibrate"])

    def test_check_cpu_default_does_not_leak_to_other_commands(self):
        parser = _build_parser()
        args = parser.parse_args(["check"])
        self.assertIsNotNone(args.cpu)

    def test_suggest_uses_the_normal_cpu_default(self):
        with patch("asm_analyzer.__main__.run_suggest", return_value=0) as suggest:
            rc = main(["suggest", "kernel.s"])
        self.assertEqual(rc, 0)
        suggest.assert_called_once()

    def test_optimize_routes_batch_and_hardware_options(self):
        with patch("asm_analyzer.__main__.run_optimize", return_value=0) as optimize:
            rc = main([
                "optimize",
                "kernels",
                "--backend",
                "llvm-mca",
                "--candidates",
                "12",
                "--hardware-shortlist",
                "3",
                "--hardware",
                "--apply-confirmed",
            ])
        self.assertEqual(rc, 0)
        self.assertEqual(optimize.call_args.kwargs["target_path"], "kernels")
        self.assertEqual(optimize.call_args.kwargs["backend_names"], ["llvm-mca"])
        self.assertEqual(optimize.call_args.kwargs["candidates"], 12)
        self.assertEqual(optimize.call_args.kwargs["hardware_shortlist"], 3)
        self.assertTrue(optimize.call_args.kwargs["hardware"])
        self.assertTrue(optimize.call_args.kwargs["apply_confirmed"])


if __name__ == "__main__":
    unittest.main()
