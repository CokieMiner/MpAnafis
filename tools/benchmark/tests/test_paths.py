"""Working outputs and publication roots cannot write benchmark evidence into source trees."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.benchmark.cli import main
from tools.benchmark.models import BenchmarkError
from tools.benchmark.paths import ROOT, validate_output_path
from tools.benchmark.publish import publish_run
from tools.benchmark.report import write_report
from tools.benchmark.runner import run_plan


class OutputPathTests(unittest.TestCase):
    def test_placement_covers_working_records_external_paths_and_symlink_resolution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            root.mkdir()
            for path, documentation, accepted in (
                (root / "target/bench-results/run", False, True),
                (root / "docs/int/benchmarks", True, True),
                (root / "src/raw.txt", False, False),
                (root / "docs/int/benchmarks/raw.txt", False, False),
                (root / "target/bench-results/../raw.txt", False, False),
                (root / "docs/other", True, False),
                (root.parent / "external-results", False, True),
                (root.parent / "external-records", True, True),
            ):
                with self.subTest(path=path, documentation=documentation):
                    if accepted:
                        self.assertEqual(validate_output_path(path, documentation=documentation, root=root), path.resolve())
                    else:
                        with self.assertRaises(BenchmarkError):
                            validate_output_path(path, documentation=documentation, root=root)
            allowed = root / "target/bench-results"
            allowed.mkdir(parents=True)
            (allowed / "link").symlink_to(root / "src", target_is_directory=True)
            with self.assertRaises(BenchmarkError):
                validate_output_path(allowed / "link/raw.txt", root=root)
            documentation = root / "docs/int/benchmarks"
            documentation.mkdir(parents=True)
            (documentation / "public_api").symlink_to(root / "src", target_is_directory=True)
            with self.assertRaises(BenchmarkError):
                validate_output_path(documentation / "public_api/record", documentation=True, root=root)

    def test_cli_and_package_boundaries_reject_source_outputs_before_building_or_writing(self):
        output = ROOT / "tools/benchmark/tests/rejected-output"
        self.assertFalse(output.exists())
        with contextlib.redirect_stderr(io.StringIO()), patch("tools.benchmark.cli.build_binary") as build:
            self.assertEqual(main(["run", "--output", str(output)]), 1)
            build.assert_not_called()
        for operation in (
            lambda: run_plan(Path("missing-binary"), {}, output),
            lambda: write_report([], output),
            lambda: publish_run(Path("missing-run"), output, "fixture", "fixture"),
        ):
            with self.assertRaisesRegex(BenchmarkError, "repository benchmark output"):
                operation()
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
