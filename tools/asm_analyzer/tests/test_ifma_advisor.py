"""Unit tests for AVX-512 IFMA redundant radix suggestion rule."""

from __future__ import annotations

import unittest
from asm_analyzer.features.suggestions import generate_suggestions


class TestIfmaAdvisor(unittest.TestCase):
    def test_ifma_redundant_radix_rule_triggers_on_multiplications(self):
        asm_lines = []
        for i in range(16):
            asm_lines.extend([
                f"mulxq {i*8}(%rsi), %rax, %r8",
                f"adcxq %rax, %r{i+8}",
                f"adoxq %r8, %r{i+9}",
            ])
        asm = "\n".join(asm_lines)
        suggestions = generate_suggestions(asm, kernel_name="add_mul_16_limbs")
        rule_ids = [s.rule_id for s in suggestions]
        self.assertIn("OPT009-IFMA-REDUNDANT-RADIX", rule_ids)


if __name__ == "__main__":
    unittest.main()
