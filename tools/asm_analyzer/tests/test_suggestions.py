"""Unit tests for the optimization suggestion advisory engine."""

from __future__ import annotations

import unittest
from asm_analyzer.features.suggestions import generate_suggestions, SuggestionSeverity


class TestSuggestions(unittest.TestCase):
    def test_rmw_hazard_warning(self):
        asm = """
        movq 0(%rsi), %rax
        addq %rax, 0(%rdi)
        """
        suggestions = generate_suggestions(asm)
        self.assertTrue(any(s.rule_id == "OPT001-RMW-HAZARD" for s in suggestions))
        rmw_s = next(s for s in suggestions if s.rule_id == "OPT001-RMW-HAZARD")
        self.assertEqual(rmw_s.severity, SuggestionSeverity.CRITICAL)

    def test_p2align_fallthrough_warning(self):
        asm = """
        decq %rcx
        js 1f
        .p2align 4
        2:
        movq 0(%rsi), %rax
        """
        suggestions = generate_suggestions(asm)
        self.assertTrue(any(s.rule_id == "OPT003-ALIGN-FALLTHROUGH" for s in suggestions))

    def test_clean_assembly_no_warnings(self):
        asm = """
        movq 0(%rsi), %r8
        movq 8(%rsi), %r9
        movq 0(%rdi), %rax
        addq %r8, %rax
        movq %rax, 0(%rdi)
        """
        suggestions = generate_suggestions(asm)
        self.assertEqual(len(suggestions), 0)

    def test_stlf_hazard_warning(self):
        asm = """
        movq %rax, 0(%rsi)
        movq 4(%rsi), %rdx
        """
        suggestions = generate_suggestions(asm)
        self.assertTrue(any(s.rule_id == "OPT008-STLF-FORWARDING-HAZARD" for s in suggestions))


if __name__ == "__main__":
    unittest.main()
