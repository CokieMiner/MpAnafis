"""uiCA output parsing regressions."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from asm_analyzer.backends.uica import _run_uica_cli


class TestUicaParsing(unittest.TestCase):
    """Verifies that uiCA throughput lines are correctly parsed across formats."""

    @patch("subprocess.run")
    def test_parses_cycles_per_iteration(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Throughput (in cycles per iteration): 6.50\nBottleneck: Port 0\n",
            stderr="",
        )
        cycles, raw = _run_uica_cli("SKL", Path("dummy.o"))
        self.assertEqual(cycles, 6.50)

    @patch("subprocess.run")
    def test_parses_throughput_cycles_legacy(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Throughput (cycles): 4.00\n",
            stderr="",
        )
        cycles, raw = _run_uica_cli("SKL", Path("dummy.o"))
        self.assertEqual(cycles, 4.00)

    @patch("subprocess.run")
    def test_parses_block_throughput(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Block Throughput: 1.25\n",
            stderr="",
        )
        cycles, raw = _run_uica_cli("SKL", Path("dummy.o"))
        self.assertEqual(cycles, 1.25)

    @patch("subprocess.run")
    def test_returns_none_when_throughput_missing(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Some unexpected output without throughput header\n",
            stderr="",
        )
        cycles, raw = _run_uica_cli("SKL", Path("dummy.o"))
        self.assertIsNone(cycles)


if __name__ == "__main__":
    unittest.main()

