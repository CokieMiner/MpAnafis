"""Execution failures retain evidence and cannot produce successful reports."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.benchmark.catalog import execution_plan
from tools.benchmark.models import Benchmark, BenchmarkError
from tools.benchmark.runner import discover, execute, measurement_environment, run_plan


TIMINGS = """public_api
╰─ int
   ╰─ unsigned
      ╰─ arithmetic
         ╰─ add
            ╰─ mp
               ╰─ 256  1 ns │ 9 ns │ 3 ns │ 4 ns │ 3 │ 6
"""


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.binary = self.root / "benchmark"
        self.binary.write_bytes(b"benchmark executable fixture")
        self.output = self.root / "results"
        self.plan = {
            "schema_version": 1, "features": "std", "arguments": ["256"],
            "settings": {"cpus": [], "threads": 1, "samples": 3, "sample_size": 2, "timeout": 10},
            "runs": execution_plan([Benchmark("int::unsigned::arithmetic::add", ("mp",))],
                                   ["256"], rounds=1),
        }

    def run_mock(self, result, smoke=False):
        with patch("tools.benchmark.runner.command", return_value=subprocess.CompletedProcess([], 0, "fixture", "")), \
             patch("tools.benchmark.runner.execute", return_value=result):
            return run_plan(self.binary, self.plan, self.output, smoke=smoke)

    def test_success_records_command_configuration_and_measurements(self):
        metadata = self.run_mock(subprocess.CompletedProcess([], 0, TIMINGS, ""))
        self.assertEqual(metadata["status"], "complete")
        self.assertIn("--bench", metadata["commands"][0])
        rows = json.loads((self.output / "measurements.json").read_text())
        self.assertEqual(rows[0]["configuration"], metadata["configuration"])
        self.assertEqual((self.output / "0000.stdout.txt").read_text(), TIMINGS)

    def test_listing_and_measurements_use_explicit_sampling_and_worker_environment(self):
        inherited = {"DIVAN_TEST": "1", "DIVAN_SAMPLE_COUNT": "999", "NEXTEST": "1", "RAYON_NUM_THREADS": "99", "PATH": "fixture-path"}
        with patch.dict(os.environ, inherited, clear=True):
            self.assertEqual(measurement_environment(3), {"PATH": "fixture-path", "RAYON_NUM_THREADS": "3"})
            with patch("tools.benchmark.runner.command", return_value=subprocess.CompletedProcess([], 0, TIMINGS, "")) as listing:
                self.assertTrue(discover(self.binary, timeout=1))
                self.assertEqual(listing.call_args.kwargs["env"], {"PATH": "fixture-path", "RAYON_NUM_THREADS": "1"})
            self.plan["settings"]["threads"] = 3
            with patch("tools.benchmark.runner.command", return_value=subprocess.CompletedProcess([], 0, "fixture", "")), \
                 patch("tools.benchmark.runner.execute", return_value=subprocess.CompletedProcess([], 0, TIMINGS, "")) as execution:
                run_plan(self.binary, self.plan, self.output)
                self.assertEqual(execution.call_args.kwargs["env"], {"PATH": "fixture-path", "RAYON_NUM_THREADS": "3"})


    def test_failed_process_preserves_error_output(self):
        with self.assertRaises(BenchmarkError):
            self.run_mock(subprocess.CompletedProcess([], 7, "", "arithmetic mismatch"))
        metadata = json.loads((self.output / "run.json").read_text())
        self.assertEqual(metadata["status"], "failed")
        self.assertEqual((self.output / "0000.stderr.txt").read_text(), "arithmetic mismatch")

    def test_timing_run_rejects_missing_argument(self):
        with self.assertRaisesRegex(BenchmarkError, "arguments"):
            self.run_mock(subprocess.CompletedProcess([], 0, TIMINGS.replace("256", "512"), ""))

    def test_smoke_run_also_rejects_missing_argument(self):
        smoke = TIMINGS.split("  1 ns")[0].replace("256", "512") + "\n"
        with self.assertRaisesRegex(BenchmarkError, "arguments"):
            self.run_mock(subprocess.CompletedProcess([], 0, smoke, ""), smoke=True)

    def test_timing_run_uses_its_argument_selection(self):
        self.plan["arguments"] = ["512"]
        self.plan["runs"][0]["arguments"] = ["256"]
        metadata = self.run_mock(subprocess.CompletedProcess([], 0, TIMINGS, ""))
        self.assertEqual(metadata["status"], "complete")

    def test_smoke_run_uses_its_argument_selection(self):
        self.plan["arguments"] = ["512"]
        self.plan["runs"][0]["arguments"] = ["256"]
        smoke = TIMINGS.split("  1 ns")[0] + "\n"
        metadata = self.run_mock(subprocess.CompletedProcess([], 0, smoke, ""), smoke=True)
        self.assertEqual(metadata["status"], "complete")

    def test_timing_run_rejects_missing_run_argument(self):
        self.plan["runs"][0]["arguments"] = ["256", "512"]
        with self.assertRaisesRegex(BenchmarkError, "arguments"):
            self.run_mock(subprocess.CompletedProcess([], 0, TIMINGS, ""))

    def test_smoke_run_rejects_missing_run_argument(self):
        self.plan["runs"][0]["arguments"] = ["256", "512"]
        smoke = TIMINGS.split("  1 ns")[0] + "\n"
        with self.assertRaisesRegex(BenchmarkError, "arguments"):
            self.run_mock(subprocess.CompletedProcess([], 0, smoke, ""), smoke=True)

    def test_nonempty_directory_is_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "existing"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(BenchmarkError, "not empty"):
            self.run_mock(subprocess.CompletedProcess([], 0, TIMINGS, ""))
        self.assertEqual(sentinel.read_text(), "keep")

    def test_timeout_retains_partial_stdout_and_failure_state(self):
        timeout = subprocess.TimeoutExpired(["fixture"], 10, output=b"partial", stderr=b"waiting")
        with patch("tools.benchmark.runner.command", return_value=subprocess.CompletedProcess([], 0, "", "")), \
             patch("tools.benchmark.runner.execute", side_effect=timeout), \
             self.assertRaisesRegex(BenchmarkError, "timed out"):
            run_plan(self.binary, self.plan, self.output)
        self.assertEqual((self.output / "0000.stdout.txt").read_text(), "partial")
        self.assertEqual(json.loads((self.output / "run.json").read_text())["status"], "failed")

    def test_execution_timeout_stops_child_and_retains_output(self):
        with self.assertRaises(subprocess.TimeoutExpired) as caught:
            execute([sys.executable, "-c", "import time; print('started', flush=True); time.sleep(60)"], timeout=0.2)
        self.assertIn("started", caught.exception.stdout)


if __name__ == "__main__":
    unittest.main()
