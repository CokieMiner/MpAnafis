"""Measurement integrity and execution-plan regression tests."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tools.benchmark.catalog import benchmark_filter, execution_plan, select, validate_catalog
from tools.benchmark.cli import validate_plan
from tools.benchmark.divan import duration_ns, parse_catalog, parse_measurements
from tools.benchmark.models import Benchmark, BenchmarkError
from tools.benchmark.report import summarize, write_report
from tools.benchmark.runner import measurement_environment


def output(function="add", engine="mp", argument="256", median="3 ns"):
    return f"""Timer precision: 10 ns
public_api              fastest │ slowest │ median │ mean │ samples │ iters
╰─ int                          │         │        │      │         │
   ╰─ unsigned
      ╰─ arithmetic
         ╰─ {function}
            ╰─ {engine}
               ╰─ {argument}  1 ns │ 9 ns │ {median} │ 4 ns │ 3 │ 6
"""


class DivanTests(unittest.TestCase):
    def test_complete_identity_survives_multiple_functions(self):
        rows = parse_measurements(output() + output("sub"))
        self.assertEqual([row.path for row in rows], ["int::unsigned::arithmetic::add", "int::unsigned::arithmetic::sub"])
        self.assertEqual(len(summarize(rows)), 2)

    def test_all_duration_units(self):
        for unit, scale in (("ps", .001), ("ns", 1), ("us", 1000), ("µs", 1000), ("μs", 1000), ("ms", 1e6), ("s", 1e9)):
            with self.subTest(unit=unit):
                self.assertEqual(duration_ns("2 " + unit), 2 * scale)

    def test_invalid_duration_rejected(self):
        for value in ("NaN ns", "-1 ns", "inf s", "23", "1e999 s"):
            with self.subTest(value=value), self.assertRaises(BenchmarkError):
                duration_ns(value)

    def test_duplicate_is_not_silently_overwritten(self):
        with self.assertRaisesRegex(BenchmarkError, "duplicate"):
            parse_measurements(output() + output())

    def test_ansi_and_unicode_units(self):
        text = "\x1b[32m" + output(median="3 ns") + "\x1b[0m"
        self.assertEqual(parse_measurements(text)[0].median_ns, 3)

    def test_bad_counts_and_ranges_rejected(self):
        for text in (output().replace("3 │ 6", "0 │ 0"), output(median="10 ns")):
            with self.assertRaises(BenchmarkError):
                parse_measurements(text)

    def test_list_is_not_a_measurement(self):
        with self.assertRaisesRegex(BenchmarkError, "no timing rows"):
            parse_measurements("public_api\n╰─ int\n")

    def test_tree_must_have_a_root_and_no_skipped_levels(self):
        for text in ("╰─ int\n", "public_api\n      ╰─ int\n"):
            with self.assertRaises(BenchmarkError):
                parse_catalog(text)

    def test_engine_leaf_without_argument(self):
        text = output().replace("            ╰─ mp\n               ╰─ 256", "            ╰─ mp")
        row = parse_measurements(text)[0]
        self.assertIsNone(row.argument)
        self.assertEqual(row.engine, "mp")

    def test_catalog_and_paired_engines(self):
        catalog = parse_catalog(output() + output(engine="rug"))
        self.assertEqual(catalog[0].engines, ("mp", "rug"))
        validate_catalog(catalog, require_comparison=True)

    def test_report_retains_runs_and_separates_scenarios(self):
        rows = parse_measurements(output(), run="A") + parse_measurements(output(median="5 ns"), run="B")
        summary = summarize(rows)
        self.assertEqual(summary[0]["median_ns"], 4)
        self.assertEqual(summary[0]["runs"], 2)
        with tempfile.TemporaryDirectory() as directory:
            write_report(rows, Path(directory))
            saved = json.loads((Path(directory) / "measurements.json").read_text())
            self.assertEqual(len(saved), 2)
            self.assertTrue((Path(directory) / "summary.csv").is_file())

    def test_reports_never_merge_distinct_configurations(self):
        row = parse_measurements(output())[0]
        self.assertEqual(len(summarize([replace(row, configuration="serial"), replace(row, configuration="parallel")])), 2)

    def test_report_selects_the_fastest_measured_reference_and_retains_peers(self):
        rows = (parse_measurements(output(median="6 ns"), run="mp")
                + parse_measurements(output(engine="rug", median="5 ns"), run="rug")
                + parse_measurements(output(engine="flint", median="3 ns"), run="flint"))
        with tempfile.TemporaryDirectory() as directory:
            write_report(rows, Path(directory))
            report = (Path(directory) / "report.md").read_text()
        self.assertIn("| flint (fastest measured) | 6 | 3 | 0.500× |", report)
        self.assertIn("| rug | 6 | 5 | 0.833× |", report)
        self.assertNotIn("rug (fastest measured)", report)

    def test_json_rows_validate_measurements_and_safe_paths(self):
        row = parse_measurements(output())[0]
        for fields in ({"median_ns": float("nan")}, {"path": "../../output"},
                       {"samples": 0}, {"mean_ns": 100}, {"configuration": "../bad"}):
            with self.subTest(fields=fields), self.assertRaises(BenchmarkError):
                replace(row, **fields)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.case = Benchmark("int::unsigned::arithmetic::add", ("mp", "rug"))

    def test_abba_order_and_exact_filter(self):
        plan = execution_plan([self.case], ["256"], rounds=2)
        self.assertEqual([item["engine"] for item in plan], ["mp", "rug", "rug", "mp"] * 2)
        self.assertTrue(plan[0]["filter"].endswith("::(?:256)$"))

    def test_selectors_match_full_paths_and_fail_closed(self):
        self.assertEqual(select([self.case], ["*::arithmetic::add"]), [self.case])
        with self.assertRaises(BenchmarkError):
            select([self.case], ["missing"])

    def test_all_comparators_get_independent_complete_rounds(self):
        case = Benchmark(self.case.path, ("mp", "rug", "gmp", "flint"))
        validate_catalog([case], require_comparison=True)
        plan = execution_plan([case], ["256"], rounds=2)
        expected = ["mp", "rug", "rug", "mp", "mp", "gmp", "gmp", "mp",
                    "mp", "flint", "flint", "mp"] * 2
        self.assertEqual([entry["engine"] for entry in plan], expected)
        self.assertEqual([entry["round"] for entry in plan], [index // 4 for index in range(24)])
        self.assertEqual([entry["position"] for entry in plan], list(range(4)) * 6)
        self.assertEqual([entry["engine"] for entry in execution_plan([case], [], rounds=2, compare=False)], ["mp", "mp"])

    def test_no_comparator_can_be_required(self):
        with self.assertRaises(BenchmarkError):
            validate_catalog([Benchmark(self.case.path, ("mp",))], require_comparison=True)

    def test_mixed_functions_are_forbidden(self):
        for name in ("predicates", "shl_policies", "to_size", "checked_rounding"):
            with self.assertRaises(BenchmarkError):
                validate_catalog([Benchmark("int::unsigned::bitwise::" + name, ("mp", "rug"))])

    def test_plan_cannot_substitute_an_arbitrary_filter(self):
        plan = {"schema_version": 1, "arguments": ["256"],
                "settings": {"threads": 1, "samples": 3, "sample_size": 2, "timeout": 30, "cpus": []},
                "runs": execution_plan([self.case], ["256"], rounds=1)}
        validate_plan(plan, [self.case])
        plan["runs"][0]["filter"] = ".*"
        with self.assertRaises(BenchmarkError):
            validate_plan(plan, [self.case])

    def test_engine_regex_is_escaped(self):
        self.assertIn(r"a\+b", benchmark_filter("int::unsigned::arithmetic::a+b", "mp", ["256"]))

    def test_plan_accepts_each_runs_exact_argument_selection(self):
        plan = {"schema_version": 1, "arguments": ["512"],
                "settings": {"threads": 1, "samples": 3, "sample_size": 2, "timeout": 30, "cpus": []},
                "runs": execution_plan([self.case], ["256"], rounds=1)}
        for entry in plan["runs"]:
            entry["arguments"] = ["256"]
        validate_plan(plan, [self.case])
        plan["runs"][0]["arguments"] = ["1024"]
        with self.assertRaisesRegex(BenchmarkError, "filter"):
            validate_plan(plan, [self.case])

    def test_plan_rejects_invalid_run_argument_selection(self):
        plan = {"schema_version": 1, "arguments": [],
                "settings": {"threads": 1, "samples": 3, "sample_size": 2, "timeout": 30, "cpus": []},
                "runs": execution_plan([self.case], [], rounds=1)}
        for arguments in (None, "256", [256], ["invalid"], ["256.*"]):
            plan["runs"][0]["arguments"] = arguments
            with self.subTest(arguments=arguments), self.assertRaisesRegex(BenchmarkError, "arguments"):
                validate_plan(plan, [self.case])

    def test_plan_requires_an_object(self):
        for plan in ([], 0, "plan"):
            with self.subTest(plan=plan), self.assertRaises(BenchmarkError):
                validate_plan(plan, [self.case])

    def test_divan_environment_is_sanitized(self):
        from unittest.mock import patch
        with patch.dict("os.environ", {"DIVAN_SAMPLE_SIZE": "1", "RAYON_NUM_THREADS": "99"}):
            env = measurement_environment(2)
        self.assertNotIn("DIVAN_SAMPLE_SIZE", env)
        self.assertEqual(env["RAYON_NUM_THREADS"], "2")


if __name__ == "__main__":
    unittest.main()
