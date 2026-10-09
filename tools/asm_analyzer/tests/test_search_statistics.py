"""Tests for exact paired hardware-selection statistics."""

from __future__ import annotations

import unittest

from asm_analyzer.search.statistics import (
    holm_adjusted_p_values,
    holm_bonferroni,
    one_sided_sign_p_value,
    paired_evidence,
)


class TestSearchStatistics(unittest.TestCase):
    def test_exact_one_sided_sign_probability(self):
        self.assertEqual(one_sided_sign_p_value(7, 7), 1 / 128)
        self.assertEqual(one_sided_sign_p_value(6, 7), 8 / 128)
        self.assertEqual(one_sided_sign_p_value(0, 0), 1.0)

    def test_paired_evidence_ignores_ties_for_sign_test(self):
        evidence = paired_evidence((0.95, 1.0, 0.97, 1.04))
        self.assertEqual(evidence.effective_rounds, 3)
        self.assertEqual(evidence.wins, 2)
        self.assertEqual(evidence.losses, 1)
        self.assertEqual(evidence.ties, 1)
        self.assertAlmostEqual(evidence.median_ratio, 0.985)

    def test_holm_bonferroni_stops_after_first_rejection_failure(self):
        decisions = holm_bonferroni({1: 0.001, 2: 0.02, 3: 0.021})
        self.assertEqual(decisions, {1: True, 2: True, 3: True})
        decisions = holm_bonferroni({1: 0.001, 2: 0.03, 3: 0.031})
        self.assertEqual(decisions, {1: True, 2: False, 3: False})

    def test_holm_rejects_invalid_alpha(self):
        with self.assertRaises(ValueError):
            holm_bonferroni({1: 0.01}, alpha=1.0)

    def test_holm_adjusted_values_are_monotone(self):
        adjusted = holm_adjusted_p_values({1: 0.01, 2: 0.03, 3: 0.04})
        self.assertEqual(adjusted, {1: 0.03, 2: 0.06, 3: 0.06})


if __name__ == "__main__":
    unittest.main()
