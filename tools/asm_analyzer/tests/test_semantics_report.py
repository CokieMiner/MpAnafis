"""Target-semantics and unified-report regression tests."""

from __future__ import annotations

import unittest

from asm_analyzer.features import extract_kernel_report
from asm_analyzer.features.memory import estimate_unroll_factor
from asm_analyzer.features.registers import analyze_registers
from asm_analyzer.semantics import semantics_for
from asm_analyzer.types import AnalysisConfidence, ArchitectureFamily


class TestTargetSemantics(unittest.TestCase):
    def test_32_bit_pointer_stride_uses_four_byte_limbs(self):
        asm = "movl (%esi), %eax\naddl $16, %esi"
        semantics = semantics_for(ArchitectureFamily.X86_32)
        self.assertEqual(estimate_unroll_factor(asm, semantics.limb_bytes), 4)

    def test_aarch64_register_scan_uses_target_registers_and_budget(self):
        semantics = semantics_for(ArchitectureFamily.AARCH64)
        stats = analyze_registers("add x2, x3, x4\nldr x5, [x6]", semantics)
        self.assertEqual(stats.gpr_names, ("x2", "x3", "x4", "x5", "x6"))
        self.assertEqual(stats.allocatable_gprs, 30)

    def test_report_integrates_applicable_and_unavailable_analyses(self):
        report = extract_kernel_report(
            "ldp x2, x3, [x0, #16]\nstp x4, x5, [x1, #16]\nmul x4, x2, x1",
            target_arch=ArchitectureFamily.AARCH64,
        )
        self.assertIsNotNone(report.aarch64)
        self.assertIsNone(report.vectorization)
        self.assertEqual(
            report.assessments["aarch64"].confidence,
            AnalysisConfidence.STRUCTURAL,
        )
        self.assertFalse(report.assessments["uop_cache"].applicable)
        encoded = report.to_dict()
        self.assertEqual(encoded["limb_bytes"], 8)
        self.assertEqual((encoded["memory"]["loads"], encoded["memory"]["stores"]), (2, 2))
        self.assertEqual(encoded["assessments"]["uop_cache"]["confidence"], "unavailable")

    def test_riscv_store_is_not_misclassified_as_rmw(self):
        report = extract_kernel_report(
            "ld a0, 0(a1)\nsd a0, 8(a2)\naddi a1, a1, 32",
            target_arch=ArchitectureFamily.RISCV64,
        )
        self.assertEqual(report.memory.loads, 1)
        self.assertEqual(report.memory.stores, 1)
        self.assertEqual(report.memory.read_modify_writes, 0)
        self.assertEqual(report.unroll_factor, 4)


if __name__ == "__main__":
    unittest.main()
