"""Unit tests for PMU command and profile parser."""

from __future__ import annotations

import io
import json
import subprocess
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from asm_analyzer.commands.pmu import PmuProfile, profile_command, run_pmu


class TestPmuProfile(unittest.TestCase):
    def test_profile_formats_measured_and_missing_counters(self):
        profile = PmuProfile(
            cycles=75195739,
            instructions=143277897,
            ipc=143277897 / 75195739,
            branches=30836919,
            branch_misses=625041,
            branch_miss_rate=625041 / 30836919,
            cache_references=3859315,
            cache_misses=416297,
            cache_miss_rate=416297 / 3859315,
            stalled_cycles_frontend=24233847,
            stalled_cycles_backend=None,
        )
        md = profile.to_markdown()
        self.assertIn("75,195,739", md)
        self.assertIn("143,277,897", md)
        self.assertIn("1.91", md)
        self.assertIn("2.03%", md)
        self.assertIn("| Backend Stalled Cycles | N/A |", md)
        self.assertEqual(profile.to_dict()["cycles"], 75195739)
        self.assertIsNone(profile.to_dict()["stalled_cycles_backend"])

    def test_command_parses_counters_and_preserves_arguments(self):
        default_events = (
            "cycles,instructions,branches,branch-misses,cache-references,"
            "cache-misses,stalled-cycles-frontend,stalled-cycles-backend"
        )
        cases = [
            (
                "all counters",
                None,
                "# perf stat\nmalformed\n"
                "100,,cycles:u,1000,100.00\n200,,instructions:k,1000,100.00\n"
                "50,,branches,1000,100.00\n5,,branch-misses,1000,100.00\n"
                "20,,cache-references,1000,100.00\n2,,cache-misses,1000,100.00\n"
                "10,,stalled-cycles-frontend,1000,100.00\n"
                "3,,stalled-cycles-backend,1000,100.00\n",
                PmuProfile(
                    cycles=100, instructions=200, ipc=2.0,
                    branches=50, branch_misses=5, branch_miss_rate=0.1,
                    cache_references=20, cache_misses=2, cache_miss_rate=0.1,
                    stalled_cycles_frontend=10, stalled_cycles_backend=3,
                ),
            ),
            (
                "partial counters and zero instructions",
                ["cycles:u", "instructions:k", "cache-references", "cache-misses"],
                "50,,cycles:u\n0,,instructions:k\n"
                "<not supported>,,cache-references\n<not counted>,,cache-misses\n",
                PmuProfile(cycles=50, instructions=0, ipc=0.0),
            ),
            (
                "zero denominators",
                None,
                "0,,cycles\n0,,instructions\n0,,branches\n0,,branch-misses\n"
                "0,,cache-references\n0,,cache-misses\n",
                PmuProfile(
                    cycles=0, instructions=0, branches=0, branch_misses=0,
                    cache_references=0, cache_misses=0,
                ),
            ),
            (
                "one measured counter",
                ["branch-misses"],
                "7,,branch-misses\n",
                PmuProfile(branch_misses=7),
            ),
        ]
        command = ["echo", "argument with spaces; $literal"]
        for name, events, stderr, expected in cases:
            with self.subTest(name=name), patch(
                "asm_analyzer.commands.pmu.shutil.which", return_value="/usr/bin/perf"
            ), patch(
                "asm_analyzer.commands.pmu.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, stdout="hello\n", stderr=stderr),
            ) as run:
                self.assertEqual(profile_command(command, events), expected)
                run.assert_called_once()
                args, kwargs = run.call_args
                self.assertEqual(args, ([
                    "perf", "stat", "-x,", "-e",
                    default_events if events is None else ",".join(events),
                    "--", *command,
                ],))
                self.assertEqual(kwargs["env"]["LC_ALL"], "C")
                self.assertEqual(kwargs["stdout"], subprocess.PIPE)
                self.assertEqual(kwargs["stderr"], subprocess.PIPE)
                self.assertTrue(kwargs["text"])
                self.assertFalse(kwargs["check"])

    def test_command_rejects_failed_runs_and_unmeasured_counters(self):
        command = ["echo", "hello"]
        with patch("asm_analyzer.commands.pmu.shutil.which", return_value=None), patch(
            "asm_analyzer.commands.pmu.subprocess.run"
        ) as run:
            self.assertIsNone(profile_command(command))
            run.assert_not_called()

        cases = [
            ("missing executable", FileNotFoundError("perf")),
            ("launch permission denied", PermissionError("perf")),
            ("perf permission denied", subprocess.CompletedProcess(
                [], 255, stderr="No permission to enable cycles event.\n"
            )),
            ("failed measured command", subprocess.CompletedProcess(
                [], 1, stderr="100,,cycles\n200,,instructions\n"
            )),
            ("empty output", subprocess.CompletedProcess([], 0, stderr="")),
            ("unsupported counters", subprocess.CompletedProcess(
                [], 0, stderr="<not supported>,,cycles\n<not counted>,,instructions\n"
            )),
            ("malformed counters", subprocess.CompletedProcess(
                [], 0, stderr="# perf stat\ninvalid\ninvalid,,cycles\n-1,,instructions\n"
            )),
            ("unrepresented counter", subprocess.CompletedProcess(
                [], 0, stderr="100,,cpu-clock\n"
            )),
        ]
        for name, result in cases:
            with self.subTest(name=name), patch(
                "asm_analyzer.commands.pmu.shutil.which", return_value="/usr/bin/perf"
            ), patch("asm_analyzer.commands.pmu.subprocess.run") as run:
                if isinstance(result, OSError):
                    run.side_effect = result
                else:
                    run.return_value = result
                self.assertIsNone(profile_command(command))

    def test_cli_status_and_output_follow_measurement_results(self):
        profile = PmuProfile(cycles=10, instructions=20, ipc=2.0)
        for as_json in (False, True):
            for command, result in (([], None), (["echo", "hello"], None), (["echo", "hello"], profile)):
                with self.subTest(as_json=as_json, command=command, result=result), patch(
                    "asm_analyzer.commands.pmu.profile_command", return_value=result
                ) as measure, redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()) as stderr:
                    status = run_pmu(command, as_json=as_json)
                    if command:
                        measure.assert_called_once_with(command)
                    else:
                        measure.assert_not_called()
                    if result is None:
                        self.assertEqual(status, 1)
                        self.assertEqual(stdout.getvalue(), "")
                        self.assertIn("Error:", stderr.getvalue())
                    else:
                        self.assertEqual(status, 0)
                        self.assertEqual(stderr.getvalue(), "")
                        if as_json:
                            self.assertEqual(json.loads(stdout.getvalue()), result.to_dict())
                        else:
                            self.assertEqual(stdout.getvalue(), result.to_markdown() + "\n")


if __name__ == "__main__":
    unittest.main()

