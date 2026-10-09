"""Consensus scoring rejects invalid costs and incomplete evidence."""

import unittest

from asm_analyzer.consensus.score import Cell, score_cpu


class TestConsensus(unittest.TestCase):
    def test_invalid_costs_do_not_divide_by_zero_or_win(self):
        cell = Cell("model", "cpu", {"zero": 0.0, "negative": -1.0, "nan": float("nan"), "inf": float("inf"), "valid": 2.0})
        result = score_cpu([cell], list(cell.costs))
        self.assertEqual(cell.winners, ["valid"])
        self.assertEqual(result.ranking[0], "valid")

    def test_partial_coverage_cannot_beat_complete_coverage(self):
        cells = [Cell("a", "cpu", {"partial": 1.0, "complete": 2.0}),
                 Cell("b", "cpu", {"partial": None, "complete": 2.0})]
        result = score_cpu(cells, ["partial", "complete"])
        self.assertEqual(result.ranking[0], "complete")
        self.assertEqual(result.consensus_winners, [])

    def test_agreement_on_tied_winners_is_not_disagreement(self):
        result = score_cpu([Cell("a", "cpu", {"x": 1.0, "y": 1.0}),
                            Cell("b", "cpu", {"x": 1.0, "y": 1.0})], ["x", "y"])
        self.assertFalse(result.disagreement)
        self.assertEqual(result.consensus_winners, ["x", "y"])


if __name__ == "__main__":
    unittest.main()
