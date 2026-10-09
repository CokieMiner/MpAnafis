"""Figures represent size sweeps, without inventing curves from isolated points."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.benchmark.divan import parse_measurements
from tools.benchmark.report import plot_summary, size_sweeps, summarize, write_report
from tools.benchmark.tests.test_bench import output
from tools.benchmark.tests.test_internal_report import internal


class PlotTests(unittest.TestCase):
    def test_single_size_comparison_does_not_make_a_graph(self):
        rows = parse_measurements(output() + output(engine="rug"))
        with tempfile.TemporaryDirectory() as directory:
            write_report(rows, Path(directory), plots=True)
            self.assertFalse(list(Path(directory).rglob("*.png")))
            text = (Path(directory) / "report.md").read_text()
            self.assertIn("No size-sweep plots", text)
            self.assertIn("| rug |", text)

    def test_isolated_sizes_across_engines_do_not_form_a_curve(self):
        rows = parse_measurements(output() + output(engine="rug", argument="1024"))
        self.assertFalse(size_sweeps(summarize(rows)))

    def test_curves_sort_numerically_and_exclude_singletons(self):
        rows = parse_measurements(output(argument="1024") + output() + output(engine="rug"))
        sweeps = size_sweeps(summarize(rows))
        curves = next(iter(sweeps.values()))
        self.assertEqual(set(curves), {"mp"})
        self.assertEqual([row["argument"] for row in curves["mp"]], ["256", "1024"])

    def test_workers_and_shapes_do_not_become_numeric_sizes(self):
        rows = parse_measurements(internal() + internal(argument="512-limbs/8-workers"), suite="internal_improvement")
        self.assertFalse(size_sweeps(summarize(rows)))

    def test_table_only_results_do_not_require_matplotlib(self):
        summary = summarize(parse_measurements(output()))
        with patch.dict("sys.modules", {"matplotlib": None}):
            self.assertEqual(plot_summary(summary, Path("unused")), [])


if __name__ == "__main__":
    unittest.main()
