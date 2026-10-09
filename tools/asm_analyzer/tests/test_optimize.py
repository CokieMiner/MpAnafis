"""Batch optimizer source-update regressions."""

from __future__ import annotations

import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asm_analyzer.commands.optimize import _atomic_write, _search_variant
from asm_analyzer.search.results import CandidateResult
from asm_analyzer.types import ArchitectureFamily


class TestOptimize(unittest.TestCase):
    def test_batch_record_discloses_original_winner_metric_and_ties(self):
        winner = CandidateResult(
            idx=1,
            body="winner",
            is_valid=True,
            cycles={"cpu": {"llvm-mca": 9.0}},
            score=1.0,
            coverage=1,
            provenance={"static_cost_metrics": {"cpu": {"llvm-mca": "simulated_cycles"}}},
        )
        original = CandidateResult(
            idx=0,
            body="original",
            is_valid=True,
            cycles={"cpu": {"llvm-mca": 10.0}},
            score=10.0 / 9.0,
            coverage=1,
        )
        with patch(
            "asm_analyzer.commands.optimize.search_kernel",
            return_value=([winner, original], ""),
        ):
            record, confirmed = _search_variant(
                Path("kernel.rs"),
                "kernel",
                ArchitectureFamily.X86_64,
                "body",
                [],
                ["llvm-mca"],
                2,
                1,
                42,
                False,
                False,
                False,
            )
        self.assertIsNone(confirmed)
        self.assertEqual(record["static_original"]["index"], 0)
        self.assertEqual(record["static_winner"]["index"], 1)
        self.assertEqual(record["static_tie_count"], 1)
        self.assertEqual(
            record["static_winner"]["provenance"]["static_cost_metrics"],
            {"cpu": {"llvm-mca": "simulated_cycles"}},
        )

    def test_atomic_source_update_preserves_file_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "kernel.rs"
            path.write_text("before", encoding="utf-8")
            path.chmod(0o640)
            _atomic_write(path, "after")
            self.assertEqual(path.read_text(encoding="utf-8"), "after")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o640)


if __name__ == "__main__":
    unittest.main()
