"""Unit tests for instruction search fail-closed safety and differential testing."""

from __future__ import annotations

import unittest
from unittest.mock import patch
from asm_analyzer.models import CPUS
from asm_analyzer.search.engine import search_kernel


class TestSearchSafety(unittest.TestCase):
    def test_search_fails_closed_when_diff_test_fails(self):
        asm = """
        movq %rax, %rcx
        addq %rdx, %rcx
        """
        cpus = [CPUS["znver3"]]

        # Mock diff_test_variants to simulate a compiler or runtime failure
        with patch("asm_analyzer.search.engine.diff_test_variants", side_effect=RuntimeError("Compilation failed in as")):
            results, err = search_kernel(asm, cpus, candidates_count=5, run_diff_test=True)
            self.assertIn("Differential testing error", err)
            # All candidates should have been marked invalid (not verified)
            self.assertEqual(len(results), 0)

    def test_search_succeeds_when_diff_test_passes(self):
        asm = """
        movq %rax, %rcx
        addq %rdx, %rcx
        """
        cpus = [CPUS["znver3"]]

        with patch("asm_analyzer.search.engine.diff_test_variants", return_value=[True, True, True]):
            results, err = search_kernel(asm, cpus, candidates_count=2, run_diff_test=True)
            self.assertEqual(err, "")
            self.assertGreater(len(results), 0)
            self.assertTrue(all(r.is_valid for r in results))


if __name__ == "__main__":
    unittest.main()
