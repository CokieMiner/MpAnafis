"""Unit tests for 32-bit x86 stack loop control and frame balancing."""

from __future__ import annotations

import unittest
from asm_analyzer.features.x86_32_loop import analyze_x86_32_loop_control
from asm_analyzer.features.suggestions import generate_suggestions, SuggestionSeverity


class TestX86_32LoopControl(unittest.TestCase):
    def test_clean_32bit_sub_mul_balanced_loop(self):
        asm = """
        pushl %ebx
        pushl %esi
        cmpl $4, (%esp)
        jb 2f
        1:
        movl 4(%esp), %eax
        mull 0({src})
        addl %ebx, %eax
        adcl $0, %edx
        movl %edx, %ebx
        addl $1, %ecx
        sbbl %eax, 0({dst})
        sbbl %ecx, %ecx
        addl $16, {src}
        addl $16, {dst}
        subl $4, (%esp)
        cmpl $4, (%esp)
        jae 1b
        2:
        negl %ecx
        addl $8, %esp
        """
        stats = analyze_x86_32_loop_control(asm)
        self.assertTrue(stats.is_32bit_stack_loop)
        self.assertFalse(stats.has_stack_imbalance)
        self.assertFalse(stats.has_flag_clobber_hazard)
        self.assertFalse(stats.has_stride_mismatch)
        self.assertEqual(stats.net_stack_delta, 0)

        suggs = generate_suggestions(asm, kernel_name="x86")
        critical_or_warnings = [s for s in suggs if s.severity in (SuggestionSeverity.CRITICAL, SuggestionSeverity.WARNING)]
        self.assertEqual(len(critical_or_warnings), 0)

    def test_stack_imbalance_detection(self):
        asm = """
        pushl %ebx
        pushl %esi
        subl $4, (%esp)
        # Missing addl $8, %esp before exit!
        """
        stats = analyze_x86_32_loop_control(asm)
        self.assertTrue(stats.has_stack_imbalance)
        self.assertEqual(stats.net_stack_delta, 8)

        suggs = generate_suggestions(asm)
        self.assertTrue(any(s.rule_id == "OPT010-STACK-IMBALANCE" for s in suggs))

    def test_live_flag_clobber_hazard_detection(self):
        asm = """
        pushl %ebx
        sbbl %eax, 0(%edi)
        # Live CF is NOT captured into mask!
        subl $4, (%esp)
        addl $4, %esp
        """
        stats = analyze_x86_32_loop_control(asm)
        self.assertTrue(stats.has_flag_clobber_hazard)

        suggs = generate_suggestions(asm)
        self.assertTrue(any(s.rule_id == "OPT011-FLAG-CLOBBER-LOOP-CONTROL" for s in suggs))

    def test_stride_mismatch_detection(self):
        asm = """
        pushl %ebx
        # Pointer advances 8 bytes (2 limbs), but counter decrements by 4 limbs!
        addl $8, %esi
        subl $4, (%esp)
        addl $4, %esp
        """
        stats = analyze_x86_32_loop_control(asm)
        self.assertTrue(stats.has_stride_mismatch)

        suggs = generate_suggestions(asm)
        self.assertTrue(any(s.rule_id == "OPT012-LOOP-STRIDE-MISMATCH" for s in suggs))


if __name__ == "__main__":
    unittest.main()
