"""Internal reports preserve geometry, thread count, and engine/tier identity."""

import tempfile
import unittest
from pathlib import Path

from tools.benchmark.divan import parse_measurements
from tools.benchmark.models import BenchmarkError
from tools.benchmark.report import plot_path, summarize, write_report


def internal(engine="mp", argument="256-limbs/1-workers"):
    return f"""internal_improvement
╰─ compare
   ╰─ production
      ╰─ {engine}
         ╰─ {argument}  1 ns │ 9 ns │ 3 ns │ 4 ns │ 3 │ 6
"""


class InternalReportTests(unittest.TestCase):
    def test_worker_counts_remain_distinct(self):
        rows = parse_measurements(internal() + internal(argument="256-limbs/8-workers"), suite="internal_improvement")
        self.assertEqual(len(summarize(rows)), 2)
        self.assertEqual(rows[0].path, "compare::production")
        self.assertEqual(rows[1].argument, "256-limbs/8-workers")

    def test_huge_ladders_never_overwrite_standard_ladders(self):
        rows = parse_measurements(internal() + internal(engine="mp_huge"), suite="internal_improvement")
        self.assertEqual({row.engine for row in rows}, {"mp", "mp_huge"})

    def test_shapes_and_serial_engines_are_preserved(self):
        text = internal(argument="512x256-limbs/8-workers") + internal("gmp_serial", "256") + internal("flint_parallel", "256-limbs/8-workers")
        rows = parse_measurements(text, suite="internal_improvement")
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0].argument, "512x256-limbs/8-workers")

    def test_wrong_suite_is_rejected(self):
        with self.assertRaises(BenchmarkError):
            parse_measurements(internal())

    def test_mixed_suites_are_rejected_after_a_valid_root(self):
        with self.assertRaisesRegex(BenchmarkError, "unexpected suite"):
            parse_measurements(internal() + "public_api\n", suite="internal_improvement")

    def test_internal_reports_do_not_infer_thread_policy_ratios(self):
        rows = parse_measurements(internal() + internal("gmp_serial", "256"), suite="internal_improvement")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_report(rows, output)
            report = (output / "report.md").read_text()
            self.assertIn("Ratios are not inferred", report)
            self.assertIn("256-limbs/1-workers", report)
            self.assertNotIn("Speedup", report)

    def test_plot_folders_mirror_suites_and_functions(self):
        path = plot_path("public_api", "int::unsigned::arithmetic::add", "config")
        self.assertEqual(path, Path("plots/public_api/int/unsigned/arithmetic/add/config.png"))


if __name__ == "__main__":
    unittest.main()
