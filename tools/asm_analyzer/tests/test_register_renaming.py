"""Tests for live range analysis and virtual register renaming anti-dependency relaxation."""

from __future__ import annotations

import unittest

from asm_analyzer.search.ast import (
    clone_instruction_with_renaming,
    format_x86_reg,
    parse_line,
)
from asm_analyzer.search.dag import (
    compute_live_ranges,
    relax_anti_dependencies_with_renaming,
)
from asm_analyzer.search.native_ast import (
    clone_native_instruction_with_renaming,
    parse_native_line,
)
from asm_analyzer.types import ArchitectureFamily


class TestRegisterRenaming(unittest.TestCase):
    def test_x86_address_renaming_preserves_index_only_and_symbolic_forms(self):
        renamed = clone_instruction_with_renaming(parse_line("movq 8(,%rcx,4), %rax"), {"rcx": "r10"})
        self.assertIsNone(renamed.ops[0].base)
        self.assertEqual(renamed.ops[0].index, "r10")
        self.assertIn("8(,%r10,4)", renamed.line)
        renamed = clone_instruction_with_renaming(parse_line("movq symbol(%rcx), %rax"), {"rcx": "r10"})
        self.assertIsNone(renamed.ops[0].addr)
        self.assertEqual(renamed.ops[0].regs, ["r10"])

    def test_x86_high_byte_renaming_preserves_high_byte(self):
        instruction = parse_line("movb %ah, %cl")
        self.assertIn("%bh", clone_instruction_with_renaming(instruction, {"rax": "rbx"}).line)
        with self.assertRaises(ValueError):
            clone_instruction_with_renaming(instruction, {"rax": "r8"})

    def test_aarch64_index_alias_is_reparsed_after_renaming(self):
        instruction = parse_native_line("ldr x0, [x1, w2, uxtw]", ArchitectureFamily.AARCH64)
        renamed = clone_native_instruction_with_renaming(instruction, {"x2": "x10"}, ArchitectureFamily.AARCH64)
        self.assertIn("w10", renamed.line)
        self.assertIn("x10", renamed.ops[1].regs)
        self.assertIsNone(renamed.ops[1].addr)

    def test_x86_reg_formatting(self):
        self.assertEqual(format_x86_reg("rax", 64), "rax")
        self.assertEqual(format_x86_reg("rax", 32), "eax")
        self.assertEqual(format_x86_reg("rax", 16), "ax")
        self.assertEqual(format_x86_reg("rax", 8), "al")
        self.assertEqual(format_x86_reg("r8", 64), "r8")
        self.assertEqual(format_x86_reg("r8", 32), "r8d")
        self.assertEqual(format_x86_reg("r8", 16), "r8w")
        self.assertEqual(format_x86_reg("r8", 8), "r8b")

    def test_clone_x86_instruction_renaming(self):
        instr = parse_line("    movq %rax, 8(%rsi, %rcx, 8)")
        self.assertIsNotNone(instr)
        renamed = clone_instruction_with_renaming(instr, {"rax": "r10", "rcx": "r11"})
        self.assertEqual(renamed.ops[0].text, "%r10")
        self.assertEqual(renamed.ops[1].text, "8(%rsi, %r11, 8)")
        self.assertIn("%r10", renamed.line)
        self.assertIn("%r11", renamed.line)

    def test_clone_native_instruction_renaming_aarch64(self):
        instr = parse_native_line("    add x0, x1, x2", ArchitectureFamily.AARCH64)
        self.assertIsNotNone(instr)
        renamed = clone_native_instruction_with_renaming(
            instr,
            {"x1": "x10", "x2": "x11"},
            ArchitectureFamily.AARCH64,
        )
        self.assertEqual(renamed.ops[1].text, "x10")
        self.assertEqual(renamed.ops[2].text, "x11")
        self.assertIn("x10", renamed.line)
        self.assertIn("x11", renamed.line)

    def test_compute_live_ranges(self):
        asm = """
        movq $1, %r8
        addq %r8, %rax
        movq $2, %r8
        addq %r8, %rbx
        """
        lines = [line.strip() for line in asm.strip().splitlines()]
        instructions = [parse_line(l) for l in lines if parse_line(l)]
        live_ranges = compute_live_ranges(instructions)
        self.assertIn("r8", live_ranges)
        self.assertEqual(len(live_ranges["r8"]), 2)
        # Range 1: def at 0, use at 1
        self.assertEqual(live_ranges["r8"][0], (0, 1))
        # Range 2: def at 2, use at 3
        self.assertEqual(live_ranges["r8"][1], (2, 3))

    def test_relax_anti_dependencies_with_renaming(self):
        asm = """
        movq $1, %r8
        addq %r8, %rax
        movq $2, %r8
        addq %r8, %rbx
        """
        lines = [line.strip() for line in asm.strip().splitlines()]
        instructions = [parse_line(l) for l in lines if parse_line(l)]
        renamed_instructions = relax_anti_dependencies_with_renaming(
            instructions,
            architecture=ArchitectureFamily.X86_64,
            available_spares=["r10", "r11"],
            live_out={"rax", "rbx"},
        )
        # The second %r8 def-use chain should be renamed to %r10
        self.assertIn("%r10", renamed_instructions[2].line)
        self.assertIn("%r10", renamed_instructions[3].line)
        # The first %r8 chain should remain untouched
        self.assertIn("%r8", renamed_instructions[0].line)
        self.assertIn("%r8", renamed_instructions[1].line)


if __name__ == "__main__":
    unittest.main()
