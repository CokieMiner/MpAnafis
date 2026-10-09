"""Golden structural regressions over compiler-emitted repository kernels."""

from __future__ import annotations

import unittest
from pathlib import Path

from asm_analyzer.features import extract_kernel_report
from asm_analyzer.kernel_source import extract_kernel_variants
from asm_analyzer.regions import select_analysis_region
from asm_analyzer.types import ArchitectureFamily

_ARCH_ROOT = (
    Path(__file__).resolve().parents[3]
    / "src/int/logic/unsigned/math/arch"
)


class TestRealKernelGoldens(unittest.TestCase):
    def test_adx_add_mul_hot_loop_shape(self):
        path = _ARCH_ROOT / "add_mul_limbs_unchecked/x86_64_adx.rs"
        variants, error = extract_kernel_variants(path)
        self.assertEqual(error, "")
        self.assertEqual(len(variants), 1)
        region = select_analysis_region(variants[0].asm)
        report = extract_kernel_report(
            region.asm,
            target_arch=ArchitectureFamily.X86_64,
            analysis_scope=region.kind,
        )
        self.assertEqual(region.kind, "loop")
        self.assertEqual(report.unroll_factor, 8)
        self.assertEqual((report.memory.loads, report.memory.stores), (16, 8))

    def test_propagate_carry_keeps_internal_exit_block(self):
        path = _ARCH_ROOT / "propagate_carry_unchecked/x86_64.rs"
        variants, error = extract_kernel_variants(path)
        self.assertEqual(error, "")
        self.assertEqual(len(variants), 1)
        region = select_analysis_region(variants[0].asm)
        report = extract_kernel_report(
            region.asm,
            target_arch=ArchitectureFamily.X86_64,
            analysis_scope=region.kind,
        )
        self.assertEqual(region.kind, "loop")
        self.assertEqual(region.block_count, 2)
        self.assertEqual(report.unroll_factor, 1)
        self.assertEqual((report.memory.loads, report.memory.stores), (1, 1))


if __name__ == "__main__":
    unittest.main()
