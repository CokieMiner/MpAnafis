"""Bounded loop contracts and relocatable-code rejection."""

import shutil
import tempfile
import unittest
from pathlib import Path

from asm_analyzer.asm_util import classify_regs
from asm_analyzer.backends.nanobench import _assemble_binary
from asm_analyzer.backends.nanobench_flow import asm_flow, validate_measurable


class TestNanobenchFlow(unittest.TestCase):
    def check(self, body):
        pointers, scalars = classify_regs(body)
        return validate_measurable(body, pointers, scalars)

    def test_counter_reset_inside_loop_is_rejected(self):
        self.assertIsNotNone(self.check("1:\nmovl $4, %ecx\nloop 1b"))

    def test_counter_modified_before_decrement_is_rejected(self):
        self.assertIsNotNone(self.check("1:\naddq %rax, %rbx\ndecq %rbx\njnz 1b"))

    def test_positive_bounded_initial_count_is_required(self):
        for value in ("0", "-1", "4097", "0xffffffff"):
            with self.subTest(value=value):
                self.assertIsNotNone(self.check(f"movq ${value}, %rcx\n1:\nnop\nloop 1b"))
        for value in ("1", "4096", "0x10"):
            self.assertIsNone(self.check(f"movq ${value}, %rcx\n1:\nnop\nloop 1b"))

    def test_external_transfers_and_unbounded_back_edges_are_rejected(self):
        for body in ("1:\njmp 1b", "ret", "call external", "jmp *%rax", "jmp missing", ".byte 0xc3"):
            with self.subTest(body=body):
                self.assertIsNotNone(self.check(body))

    def test_repeated_numeric_labels_resolve_in_the_requested_direction(self):
        _, resolve = asm_flow("1:\nnop\njnz 1b\n1:\nnop\njnz 1b")
        self.assertEqual(resolve("1b", 2), 0)
        self.assertEqual(resolve("1f", 2), 3)
        self.assertEqual(resolve("1b", 5), 3)
        self.assertIsNone(resolve("1f", 5))

    def test_counter_source_modified_in_preheader_is_rejected(self):
        self.assertIsNotNone(self.check("xorq %r14, %r14\nmovq %r14, %rcx\n1:\nnop\nloop 1b"))

    @unittest.skipUnless(all(shutil.which(tool) for tool in ("as", "objcopy", "objdump")), "binutils required")
    def test_binary_extraction_rejects_unresolved_relocations(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source = work / "kernel.s"
            source.write_text(".text\nmovq external(%rip), %rax\n", encoding="utf-8")
            self.assertIn("relocations", _assemble_binary(source, work / "kernel.o", work / "kernel.bin"))
            self.assertFalse((work / "kernel.bin").exists())


if __name__ == "__main__":
    unittest.main()
