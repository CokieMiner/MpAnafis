"""Benchmark symmetry audit fixtures."""

import unittest

from tools.benchmark.source_audit import check_source


class SourceTests(unittest.TestCase):
    def test_symmetric_pair(self):
        source = 'mod add { #[divan::bench(args = [256], sample_size = 32)] fn mp() {} #[divan::bench(args = [256], sample_size = 32)] fn rug() {} }'
        self.assertEqual(check_source(source), [])

    def test_mismatched_sampling_and_widths(self):
        source = 'mod add { #[divan::bench(args = [256])] fn mp() {} #[divan::bench(args = [512])] fn rug() {} }'
        self.assertEqual(check_source(source)[0]["kind"], "asymmetric_benchmark_configuration")

    def test_flint_and_rug_share_the_same_configuration_and_timing(self):
        source = 'mod add { #[divan::bench()] fn mp() {} #[divan::bench()] fn rug() {} #[divan::bench()] fn flint() {} }'
        self.assertEqual(check_source(source), [])
        mismatched = source.replace('fn flint() {}', 'fn flint() { b.bench_local(|| f()); }')
        self.assertEqual(check_source(mismatched)[0]["kind"], "asymmetric_benchmark_timing")

    def test_labeled_engine_is_not_silently_ignored(self):
        source = 'mod add { #[divan::bench()] fn mp() {} #[divan::bench()] fn flint_cost_reference() {} }'
        self.assertEqual(check_source(source)[0]["kind"], "nonstandard_benchmark_engine")

    def test_reset_strategy_must_match(self):
        source = 'mod add { #[divan::bench()] fn mp() { b.bench_local(|| f()); } #[divan::bench()] fn rug() { b.bench_local_values(|x| f(x)); } }'
        self.assertEqual(check_source(source)[0]["kind"], "asymmetric_benchmark_timing")

    def test_duplicate_engine_is_not_overwritten(self):
        source = 'mod add { #[divan::bench()] fn mp() {} #[divan::bench()] fn mp() {} }'
        self.assertEqual(check_source(source)[0]["kind"], "duplicate_benchmark_engine")


if __name__ == "__main__":
    unittest.main()
