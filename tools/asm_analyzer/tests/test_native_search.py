"""Dependency and fail-closed scheduling tests for non-x86 ISA adapters."""

from __future__ import annotations

import shutil
import subprocess
import unittest

from asm_analyzer.diff_test import supports_native_differential
from asm_analyzer.differential.arm import aarch64_wrapper, arm32_wrapper
from asm_analyzer.differential.common import native_register_plan
from asm_analyzer.differential.loongarch import loongarch_wrapper
from asm_analyzer.differential.mips import mips_wrapper
from asm_analyzer.differential.power import power64_wrapper, power_wrapper
from asm_analyzer.differential.riscv import riscv64_wrapper, riscv_wrapper
from asm_analyzer.differential.s390x import s390x_wrapper
from asm_analyzer.differential.x86 import x86_32_wrapper
from asm_analyzer.models import CPUS
from asm_analyzer.search.adapters import adapter_for
from asm_analyzer.search.dag import build_dag
from asm_analyzer.search.engine import search_kernel
from asm_analyzer.types import ArchitectureFamily


class TestNativeSearch(unittest.TestCase):
    def test_pointer_arguments_survive_the_native_wrapper_copy_order(self):
        arguments = {
            ArchitectureFamily.AARCH64: ("x0", "x1"),
            ArchitectureFamily.ARM32: ("r0", "r1"),
            ArchitectureFamily.RISCV32: ("x10", "x11"),
            ArchitectureFamily.RISCV64: ("x10", "x11"),
            ArchitectureFamily.LOONGARCH32: ("r4", "r5"),
            ArchitectureFamily.LOONGARCH64: ("r4", "r5"),
            ArchitectureFamily.MIPS32: ("r4", "r5"),
            ArchitectureFamily.MIPS64: ("r4", "r5"),
            ArchitectureFamily.POWER32: ("r3", "r4"),
            ArchitectureFamily.POWER64: ("r3", "r4"),
            ArchitectureFamily.S390X: ("r2", "r3"),
        }
        for architecture, registers in arguments.items():
            with self.subTest(architecture=architecture):
                r_in, r_out, _, _ = native_register_plan(
                    ["nop", "nop"], architecture, registers, set(),
                )
                state = dict(zip(registers, ("input", "output")))
                state[r_in] = state[registers[0]]
                state[r_out] = state[registers[1]]
                self.assertEqual((state[r_in], state[r_out]), ("input", "output"))

    def test_every_supported_isa_has_a_native_host_verifier(self):
        for architecture in ArchitectureFamily:
            self.assertTrue(supports_native_differential(architecture), architecture)

    def test_aarch64_post_index_and_carry_dependencies(self):
        adapter = adapter_for(ArchitectureFamily.AARCH64)
        instructions = [
            adapter.parse("ldr x15, [x10], #8"),
            adapter.parse("mul x18, x15, x13"),
            adapter.parse("adds x18, x18, x8"),
            adapter.parse("adc x0, x0, xzr"),
        ]
        nodes = build_dag(instructions, spec_provider=adapter.spec)
        self.assertEqual(nodes[0].spec.defs, {"x10", "x15"})
        self.assertEqual(nodes[1].preds, {0})
        self.assertIn(2, nodes[3].preds)

    def test_mips_hi_lo_dependencies(self):
        adapter = adapter_for(ArchitectureFamily.MIPS64)
        instructions = [
            adapter.parse("dmultu $9, $5"),
            adapter.parse("mflo $12"),
            adapter.parse("mfhi $13"),
        ]
        nodes = build_dag(instructions, spec_provider=adapter.spec)
        self.assertEqual(nodes[0].spec.defs, {"hi", "lo"})
        self.assertEqual(nodes[1].preds, {0})
        self.assertEqual(nodes[2].preds, {0})

    def test_s390_register_move_is_not_misclassified_as_memory(self):
        adapter = adapter_for(ArchitectureFamily.S390X)
        move = adapter.parse("lgr %r9, %r5")
        load = adapter.parse("lg %r9, 0(%r5)")
        self.assertIsNone(adapter.spec(move).mem)
        self.assertEqual(adapter.spec(load).mem, "load")

    def test_power_word_store_has_32_bit_width(self):
        adapter = adapter_for(ArchitectureFamily.POWER32)
        store = adapter.parse("stw 9, 0(5)")
        self.assertEqual(store.ops[1].width, 32)

    def test_unknown_native_instruction_is_an_ordering_barrier(self):
        adapter = adapter_for(ArchitectureFamily.RISCV64)
        instructions = [
            adapter.parse("add a0, a1, a2"),
            adapter.parse("mystery a3, a4"),
            adapter.parse("add a5, a6, a7"),
        ]
        nodes = build_dag(instructions, spec_provider=adapter.spec)
        self.assertTrue(nodes[1].spec.unknown)
        self.assertEqual(nodes[1].preds, {0})
        self.assertIn(1, nodes[2].preds)

    def test_aarch64_static_search_generates_safe_candidates(self):
        body = "\n".join(
            (
                "mul x10, x0, x1",
                "mul x11, x2, x3",
                "add x12, x10, x4",
                "add x13, x11, x5",
            ),
        )
        results, error = search_kernel(
            body,
            [CPUS["neoverse-n1"]],
            backend_names=["llvm-mca"],
            candidates_count=4,
            run_diff_test=False,
            architecture=ArchitectureFamily.AARCH64,
        )
        self.assertEqual(error, "")
        self.assertGreaterEqual(len(results), 2)

    @unittest.skipUnless(shutil.which("llvm-mc"), "llvm-mc is required")
    def test_aarch64_differential_wrapper_assembles(self):
        wrapper = aarch64_wrapper(
            "candidate",
            "mul x10, x0, x1\nadds x11, x10, x2",
            "x28",
            "x27",
        )
        assembled = subprocess.run(
            ["llvm-mc", "-triple=aarch64", "-filetype=obj", "-o", "/dev/null"],
            input=wrapper,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(assembled.returncode, 0, assembled.stderr)

    @unittest.skipUnless(shutil.which("llvm-mc"), "llvm-mc is required")
    def test_arm32_differential_wrapper_assembles(self):
        wrapper = arm32_wrapper(
            "candidate",
            "umull r8, r9, r0, r1\nadds r10, r8, r2",
            "r12",
            "r11",
        )
        assembled = subprocess.run(
            [
                "llvm-mc",
                "-triple=armv7-linux-gnueabihf",
                "-filetype=obj",
                "-o",
                "/dev/null",
            ],
            input=wrapper,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(assembled.returncode, 0, assembled.stderr)

    @unittest.skipUnless(shutil.which("llvm-mc"), "llvm-mc is required")
    def test_riscv64_differential_wrapper_assembles(self):
        wrapper = riscv64_wrapper(
            "candidate",
            "mul x12, x10, x11\nadd x13, x12, x14",
            "x31",
            "x30",
        )
        assembled = subprocess.run(
            [
                "llvm-mc",
                "-triple=riscv64",
                "-mattr=+m",
                "-filetype=obj",
                "-o",
                "/dev/null",
            ],
            input=wrapper,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(assembled.returncode, 0, assembled.stderr)

    @unittest.skipUnless(shutil.which("llvm-mc"), "llvm-mc is required")
    def test_riscv32_differential_wrapper_assembles(self):
        wrapper = riscv_wrapper(
            "candidate",
            "mul x12, x10, x11\nadd x13, x12, x14",
            "x31",
            "x30",
            4,
        )
        assembled = subprocess.run(
            [
                "llvm-mc",
                "-triple=riscv32",
                "-mattr=+m",
                "-filetype=obj",
                "-o",
                "/dev/null",
            ],
            input=wrapper,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(assembled.returncode, 0, assembled.stderr)

    @unittest.skipUnless(shutil.which("llvm-mc"), "llvm-mc is required")
    def test_power64_differential_wrapper_assembles(self):
        wrapper = power64_wrapper(
            "candidate",
            "mulld 10, 3, 4\naddc 11, 10, 5",
            "r12",
            "r11",
        )
        assembled = subprocess.run(
            [
                "llvm-mc",
                "-triple=powerpc64le-linux-gnu",
                "-filetype=obj",
                "-o",
                "/dev/null",
            ],
            input=wrapper,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(assembled.returncode, 0, assembled.stderr)

    @unittest.skipUnless(shutil.which("llvm-mc"), "llvm-mc is required")
    def test_x86_32_differential_wrapper_assembles(self):
        wrapper = x86_32_wrapper("candidate", "addl %ecx, %eax")
        assembled = subprocess.run(
            ["llvm-mc", "-triple=i686-linux-gnu", "-filetype=obj", "-o", "/dev/null"],
            input=wrapper,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(assembled.returncode, 0, assembled.stderr)

    @unittest.skipUnless(shutil.which("llvm-mc"), "llvm-mc is required")
    def test_power32_differential_wrapper_assembles(self):
        wrapper = power_wrapper("candidate", "mullw 10, 3, 4\naddc 11, 10, 5", "r12", "r11", 4)
        assembled = subprocess.run(
            ["llvm-mc", "-triple=powerpc-linux-gnu", "-filetype=obj", "-o", "/dev/null"],
            input=wrapper,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(assembled.returncode, 0, assembled.stderr)

    @unittest.skipUnless(shutil.which("llvm-mc"), "llvm-mc is required")
    def test_mips_wrappers_assemble(self):
        for triple, word_bytes, body in (
            ("mips64-unknown-linux-gnu", 8, "dmultu $9, $5\nmflo $12"),
            ("mips-unknown-linux-gnu", 4, "multu $9, $5\nmflo $12"),
        ):
            with self.subTest(triple=triple):
                wrapper = mips_wrapper("candidate", body, "r25", "r24", word_bytes)
                assembled = subprocess.run(
                    ["llvm-mc", f"-triple={triple}", "-filetype=obj", "-o", "/dev/null"],
                    input=wrapper,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(assembled.returncode, 0, assembled.stderr)

    @unittest.skipUnless(shutil.which("llvm-mc"), "llvm-mc is required")
    def test_s390x_differential_wrapper_assembles(self):
        wrapper = s390x_wrapper("candidate", "algr %r9, %r5", "r13", "r12")
        assembled = subprocess.run(
            ["llvm-mc", "-triple=s390x-linux-gnu", "-filetype=obj", "-o", "/dev/null"],
            input=wrapper,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(assembled.returncode, 0, assembled.stderr)

    @unittest.skipUnless(shutil.which("llvm-mc"), "llvm-mc is required")
    def test_loongarch_wrappers_assemble(self):
        for triple, word_bytes, body in (
            ("loongarch64-linux-gnu", 8, "mul.d $a0, $a1, $a2"),
            ("loongarch32-linux-gnu", 4, "mul.w $a0, $a1, $a2"),
        ):
            with self.subTest(triple=triple):
                wrapper = loongarch_wrapper("candidate", body, "r31", "r30", word_bytes)
                assembled = subprocess.run(
                    ["llvm-mc", f"-triple={triple}", "-filetype=obj", "-o", "/dev/null"],
                    input=wrapper,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(assembled.returncode, 0, assembled.stderr)


if __name__ == "__main__":
    unittest.main()
