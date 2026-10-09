"""Unit tests for terminal diff rendering."""

from __future__ import annotations

import unittest

from asm_analyzer.report.terminal import render_terminal_diff
from asm_analyzer.types import (
    ArchitectureFamily,
    BranchStats,
    KernelAnalysisReport,
    KernelComparisonDiff,
    MemoryAccessStats,
    MultiplierStats,
    PortPressureStats,
    RegisterStats,
    UopCacheStats,
)


def _dummy_report(name: str, cycles: dict) -> KernelAnalysisReport:
    return KernelAnalysisReport(
        kernel_name=name,
        target_arch=ArchitectureFamily.X86_64,
        unroll_factor=4,
        memory=MemoryAccessStats(loads=8, stores=4, read_modify_writes=0),
        registers=RegisterStats(gprs_used=10, gpr_names=("rax", "rbx")),
        multiplier=MultiplierStats(mul_count=4, min_slack=2),
        port_pressure=PortPressureStats(bottleneck_port="P1"),
        uop_cache=UopCacheStats(instruction_count=32, estimated_uops=40),
        branch=BranchStats(branch_count=1),
        cpu_cycles=cycles,
    )


class TestTerminalDiff(unittest.TestCase):
    def test_render_terminal_diff_completes_without_error(self):
        """Verify render_terminal_diff runs without NameError or AttributeError."""
        rep_a = _dummy_report("baseline", {"znver3": 10.0, "skylake": 12.0})
        rep_b = _dummy_report("optimized", {"znver3": 8.0, "skylake": 11.0})
        diff = KernelComparisonDiff(
            kernel_a=rep_a,
            kernel_b=rep_b,
            cycle_deltas={"znver3": -2.0, "skylake": -1.0},
            load_delta=0,
            store_delta=0,
            rmw_delta=0,
            gpr_delta=0,
            speedup_ratios={"znver3": 0.25, "skylake": 0.09},
        )
        output = render_terminal_diff(diff, enable_color=False)
        self.assertIn("baseline", output)
        self.assertIn("optimized", output)
        self.assertIn("znver3", output)
        self.assertIn("skylake", output)
        self.assertIn("Memory Loads", output)
        self.assertIn("Port Bottleneck", output)

    def test_render_terminal_diff_with_color(self):
        rep_a = _dummy_report("kernel_a", {"znver3": 10.0})
        rep_b = _dummy_report("kernel_b", {"znver3": 8.0})
        diff = KernelComparisonDiff(
            kernel_a=rep_a,
            kernel_b=rep_b,
            cycle_deltas={"znver3": -2.0},
            load_delta=-1,
            store_delta=0,
            rmw_delta=0,
            gpr_delta=0,
        )
        output = render_terminal_diff(diff, enable_color=True)
        # Should contain ANSI escape codes
        self.assertIn("\033[", output)


if __name__ == "__main__":
    unittest.main()
