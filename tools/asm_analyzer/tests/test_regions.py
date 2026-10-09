"""Tests for hot-region selection."""

from __future__ import annotations

import unittest

from asm_analyzer.regions import build_control_flow_graph, select_analysis_region


class TestAnalysisRegions(unittest.TestCase):
    def test_loop_selection_preserves_encoded_bytes_and_alignment(self):
        asm = "xorq %rax, %rax\n.p2align 4\n1:\n.byte 0x90\naddq %rax, %rbx\njnz 1b"
        region = select_analysis_region(asm)
        self.assertEqual(region.kind, "loop")
        self.assertIn(".byte 0x90", region.asm)
        self.assertIn(".p2align 4", region.asm)

    def test_multiple_entry_cycle_is_not_a_natural_loop(self):
        asm = "jz right\nleft:\naddq %rax, %rbx\njmp right\nright:\naddq %rcx, %rdx\njnz left"
        self.assertEqual(build_control_flow_graph(asm).back_edges, ())
        self.assertEqual(select_analysis_region(asm).kind, "block")

    def test_full_line_comments_are_not_cfg_instructions(self):
        cfg = build_control_flow_graph("# comment\nmovq %rax, %rbx\n// comment")
        self.assertEqual(cfg.instructions, ("movq %rax, %rbx",))

    def test_straight_line_block_is_unchanged(self):
        asm = "movq %rax, %rbx\naddq %rcx, %rbx"
        region = select_analysis_region(asm)
        self.assertEqual(region.kind, "block")
        self.assertEqual(region.asm, asm)

    def test_largest_backward_loop_wins_over_tail_loop(self):
        asm = """
setup:
    xorl %eax, %eax
.Lmain:
    movq (%rsi), %r8
    addq %r8, (%rdi)
    leaq 8(%rsi), %rsi
    leaq 8(%rdi), %rdi
    decq %rcx
    jnz .Lmain
.Ltail:
    movq (%rsi), %r8
    decq %rdx
    jnz .Ltail
done:
    retq
"""
        region = select_analysis_region(asm)
        self.assertEqual(region.kind, "loop")
        self.assertEqual(region.label, ".Lmain")
        self.assertIn("addq %r8, (%rdi)", region.asm)
        self.assertNotIn("retq", region.asm)
        self.assertNotIn("jnz .Lmain", region.schedulable_asm())

    def test_aarch64_backward_branch_is_recognized(self):
        asm = ".Lloop:\nadd x0, x0, x1\nsubs x2, x2, #1\nb.ne .Lloop"
        region = select_analysis_region(asm)
        self.assertEqual(region.kind, "loop")
        self.assertEqual(region.label, ".Lloop")

    def test_numeric_local_label_is_resolved_directionally(self):
        asm = "1:\naddq %rax, %rbx\ndecq %rcx\njnz 1b\n2:\nretq"
        region = select_analysis_region(asm)
        self.assertEqual(region.kind, "loop")
        self.assertEqual(region.label, "1")
        self.assertNotIn("jnz", region.schedulable_asm())

    def test_riscv_multi_operand_branch_builds_back_edge(self):
        asm = ".Lloop:\nadd a0, a0, a1\naddi a2, a2, -1\nbne a2, zero, .Lloop"
        cfg = build_control_flow_graph(asm)
        region = select_analysis_region(asm)
        self.assertEqual(cfg.back_edges, ((0, 0),))
        self.assertEqual(region.kind, "loop")
        self.assertEqual(region.label, ".Lloop")

    def test_cfg_retains_internal_conditional_path(self):
        asm = """
.Lloop:
    cbz x0, .Lskip
    add x1, x1, x2
.Lskip:
    subs x3, x3, #1
    b.ne .Lloop
"""
        cfg = build_control_flow_graph(asm)
        region = select_analysis_region(asm)
        self.assertEqual(len(cfg.blocks), 3)
        self.assertEqual(region.block_count, 3)
        self.assertIn("cbz x0, .Lskip", region.asm)
        self.assertIn(".Lskip:", region.asm)


if __name__ == "__main__":
    unittest.main()
