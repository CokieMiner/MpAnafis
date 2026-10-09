"""Curve exports validate local evidence without copying raw runs into Git."""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from tools.benchmark.catalog import execution_plan
from tools.benchmark.divan import parse_measurements
from tools.benchmark.evidence import validate_measurement_plan
from tools.benchmark.models import Benchmark, BenchmarkError
from tools.benchmark.publish import publish_run
from tools.benchmark.report import plot_path
from tools.benchmark.tests.test_runner import TIMINGS


class PublishTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "run"
        self.source.mkdir()
        self.output = self.root / "docs"
        self.timings = TIMINGS + TIMINGS.replace("256", "1024")
        path = "int::unsigned::arithmetic::add"
        settings = {"cpus": [0], "threads": 1, "samples": 3, "sample_size": 2, "timeout": 10}
        runs = execution_plan([Benchmark(path, ("mp", "rug"))], ["256", "1024"], rounds=1)
        self.metadata = {
            "schema_version": 1, "status": "complete", "smoke": False,
            "configuration": "fixture", "binary": "benchmark", "binary_sha256": "a" * 64,
            "git_head": "b" * 40, "git_status": " M src/example.rs",
            "rustc": "fixture", "platform": "fixture", "processor": "fixture CPU",
            "started_unix": 1.0, "finished_unix": 2.0,
            "configuration_details": {"settings": dict(settings)},
            "commands": [["taskset", "--cpu-list", "0", "benchmark", "--bench", entry["filter"],
                          "--color", "never", "--sample-count", "3", "--sample-size", "2"]
                         for entry in runs],
            "plan": {"schema_version": 1, "features": "std", "settings": settings,
                     "arguments": ["256", "1024"], "runs": runs},
        }
        (self.source / "run.json").write_text(json.dumps(self.metadata))
        self.write_captures(self.timings)
        plotter = patch("tools.benchmark.publish.plot_summary", side_effect=self.draw_fixture)
        plotter.start()
        self.addCleanup(plotter.stop)

    def write_captures(self, timings):
        rows = []
        for index, entry in enumerate(self.metadata["plan"]["runs"]):
            run_id = f"{index:04d}"
            capture = timings.replace("╰─ mp", f"╰─ {entry['engine']}")
            rows.extend(asdict(row) | {"configuration": "fixture"}
                        for row in parse_measurements(capture, run=run_id))
            (self.source / f"{run_id}.stdout.txt").write_text(capture)
            (self.source / f"{run_id}.stderr.txt").write_text("")
        (self.source / "measurements.json").write_text(json.dumps(rows))

    @staticmethod
    def draw_fixture(summary, output):
        relative = plot_path("public_api", summary[0]["path"], "fixture")
        (output / relative).parent.mkdir(parents=True)
        (output / relative).write_bytes(b"plot fixture")
        return [relative]

    def publish(self, **kwargs):
        return publish_run(self.source, self.output, "example", "Harness validation.", **kwargs)

    def test_record_contains_only_curves_and_compact_context(self):
        destination = self.publish()
        self.assertEqual(destination, self.output / "public_api/example")
        self.assertEqual((self.source / "0000.stdout.txt").read_text(), self.timings)
        self.assertFalse((destination / "raw").exists())
        self.assertFalse((destination / "measurements.json").exists())
        manifest = json.loads((destination / "manifest.json").read_text())
        self.assertEqual(manifest["measurement_details"]["plan"], self.metadata["plan"])
        self.assertIn("0000.stdout.txt", manifest["source_files"])
        self.assertEqual(set(manifest["files"]), {"README.md", "plots/public_api/int/unsigned/arithmetic/add/fixture.png"})
        readme = (destination / "README.md").read_text()
        self.assertIn("revision alone does not reproduce", readme)
        self.assertIn("--output target/bench-results/example-report", readme)
        self.assertIn(str(self.source / "measurements.json"), readme)

    def test_single_size_validation_is_not_documentation(self):
        self.metadata["plan"]["arguments"] = ["256"]
        for entry in self.metadata["plan"]["runs"]:
            entry["filter"] = entry["filter"].replace("256|1024", "256")
        for command in self.metadata["commands"]:
            command[5] = command[5].replace("256|1024", "256")
        (self.source / "run.json").write_text(json.dumps(self.metadata))
        self.write_captures(TIMINGS)
        with self.assertRaisesRegex(BenchmarkError, "requires a size sweep"):
            self.publish()
        self.assertFalse(self.output.exists())

    def test_existing_record_is_preserved(self):
        destination = self.publish()
        with self.assertRaisesRegex(BenchmarkError, "already exists"):
            self.publish()
        self.assertTrue((destination / "manifest.json").exists())

    def test_export_uses_each_runs_argument_selection(self):
        self.metadata["plan"]["arguments"] = ["512"]
        for entry in self.metadata["plan"]["runs"]:
            entry["arguments"] = ["256", "1024"]
        (self.source / "run.json").write_text(json.dumps(self.metadata))
        destination = self.publish()
        manifest = json.loads((destination / "manifest.json").read_text())
        self.assertEqual(manifest["measurement_details"]["plan"], self.metadata["plan"])

    def test_export_rejects_missing_run_argument(self):
        self.metadata["plan"]["runs"][0]["arguments"] = ["256", "512"]
        (self.source / "run.json").write_text(json.dumps(self.metadata))
        with self.assertRaisesRegex(BenchmarkError, "planned arguments"):
            self.publish()

    def test_export_rejects_invalid_run_argument_selection(self):
        self.metadata["plan"]["runs"][0]["arguments"] = [256]
        (self.source / "run.json").write_text(json.dumps(self.metadata))
        with self.assertRaisesRegex(BenchmarkError, "argument selection"):
            self.publish()

    def test_failed_or_smoke_run_is_not_exported(self):
        for updates in ({"status": "failed"}, {"smoke": True}):
            (self.source / "run.json").write_text(json.dumps(self.metadata | updates))
            with self.assertRaisesRegex(BenchmarkError, "completed measured"):
                self.publish()
        self.assertFalse(self.output.exists())

    def test_export_requires_pinning_sampling_and_measurement_metadata(self):
        pristine = json.dumps(self.metadata)
        for section, field, value, message in (
            ("settings", "cpus", [], "CPU pinning"),
            ("settings", "threads", 2, "CPU affinity"),
            ("settings", "sample_size", 3, "configuration"),
            ("settings", "samples", 4, "sample count"),
            ("metadata", "processor", "", "processor"),
            ("metadata", "finished_unix", 0, "timestamps"),
        ):
            metadata = json.loads(pristine)
            if section == "settings":
                metadata["plan"]["settings"][field] = value
                if field == "samples":
                    metadata["configuration_details"]["settings"][field] = value
            else:
                metadata[field] = value
            (self.source / "run.json").write_text(json.dumps(metadata))
            with self.subTest(field=field), self.assertRaisesRegex(BenchmarkError, message):
                self.publish()
        self.assertFalse(self.output.exists())

    def test_export_checks_actual_commands_and_round_positions(self):
        for section in ("pinning", "sampling", "binary", "filter", "order", "round"):
            metadata = json.loads(json.dumps(self.metadata))
            if section == "pinning":
                metadata["commands"][0][2] = "1"
            elif section == "sampling":
                metadata["commands"][0][-1] = "99"
            elif section == "binary":
                metadata["commands"][0][3] = "other-binary"
            elif section == "filter":
                metadata["commands"][0][5] = "other-filter"
            elif section == "order":
                metadata["plan"]["runs"][1]["position"] = 2
            else:
                metadata["plan"]["runs"][0]["round"] = 1
            (self.source / "run.json").write_text(json.dumps(metadata))
            with self.subTest(section=section), self.assertRaisesRegex(BenchmarkError, "pinning|A/B/B/A"):
                self.publish()
        self.assertFalse(self.output.exists())

    def test_multiple_references_require_complete_unmixed_comparison_rounds(self):
        path = "int::unsigned::arithmetic::add"
        runs = execution_plan([Benchmark(path, ("mp", "rug", "flint"))], ["256", "1024"], rounds=2)
        self.metadata["plan"]["runs"] = runs
        self.metadata["commands"] = [
            ["taskset", "--cpu-list", "0", "benchmark", "--bench", entry["filter"],
             "--color", "never", "--sample-count", "3", "--sample-size", "2"]
            for entry in runs
        ]
        self.write_captures(self.timings)
        rows = [row for index, entry in enumerate(runs)
                for row in parse_measurements(self.timings.replace("╰─ mp", f"╰─ {entry['engine']}"), run=str(index))]
        validate_measurement_plan(self.metadata, rows)
        runs[2]["engine"] = "flint"
        runs[2]["filter"] = runs[2]["filter"].replace("::rug::", "::flint::")
        self.metadata["commands"][2][5] = runs[2]["filter"]
        with self.assertRaisesRegex(BenchmarkError, "A/B/B/A"):
            validate_measurement_plan(self.metadata, rows)

    def test_missing_capture_is_rejected(self):
        (self.source / "0000.stderr.txt").unlink()
        with self.assertRaisesRegex(BenchmarkError, "raw capture"):
            self.publish()

    def test_edited_summary_is_rejected(self):
        path = self.source / "measurements.json"
        rows = json.loads(path.read_text())
        rows[0]["median_ns"] = 5
        path.write_text(json.dumps(rows))
        with self.assertRaisesRegex(BenchmarkError, "preserved raw"):
            self.publish()

    def test_wrong_planned_case_is_rejected(self):
        self.metadata["plan"]["runs"][0]["path"] = "int::unsigned::arithmetic::other"
        (self.source / "run.json").write_text(json.dumps(self.metadata))
        with self.assertRaisesRegex(BenchmarkError, "planned function"):
            self.publish()

    def test_bad_names_cannot_escape_documentation_root(self):
        for name in ("../outside", "/absolute", "nested/name", ""):
            with self.subTest(name=name), self.assertRaises(BenchmarkError):
                publish_run(self.source, self.output, name, "Scope.")

    def test_failed_plot_does_not_leave_partial_documentation(self):
        with patch("tools.benchmark.publish.plot_summary", side_effect=BenchmarkError("plot failed")):
            with self.assertRaisesRegex(BenchmarkError, "plot failed"):
                self.publish()
        self.assertFalse((self.output / "public_api/example").exists())
        self.assertEqual(list((self.output / "public_api").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
