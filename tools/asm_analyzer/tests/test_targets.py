"""Tests for target and CPU matrix policy."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from asm_analyzer.models import CPUS
from asm_analyzer.targets import (
    architecture_for_path,
    compatible_cpus,
    default_cpus_for_architecture,
    host_architecture,
    rust_target_for_path,
)
from asm_analyzer.types import ArchitectureFamily


class TestTargetPolicy(unittest.TestCase):
    def test_foreign_architectures_do_not_fall_through_to_x86(self):
        self.assertEqual(
            architecture_for_path(Path("kernel/loongarch32.rs")),
            ArchitectureFamily.LOONGARCH32,
        )
        self.assertEqual(
            architecture_for_path(Path("kernel/mips64.rs")),
            ArchitectureFamily.MIPS64,
        )

    def test_cpu_matrix_is_isa_compatible(self):
        requested = [CPUS["znver3"], CPUS["neoverse-v1"], CPUS["power9"]]
        self.assertEqual(
            [cpu.name for cpu in compatible_cpus(requested, ArchitectureFamily.AARCH64)],
            ["neoverse-v1"],
        )

    def test_x86_defaults_exclude_arm_models(self):
        defaults = default_cpus_for_architecture(ArchitectureFamily.X86_64)
        self.assertTrue(defaults)
        self.assertTrue(all(cpu.family in ("amd", "intel") for cpu in defaults))

    def test_foreign_rust_target_mapping(self):
        self.assertEqual(
            rust_target_for_path(Path("aarch64.rs")),
            "aarch64-unknown-linux-gnu",
        )
        self.assertEqual(
            rust_target_for_path(Path("riscv32.rs")),
            "riscv32imac-unknown-none-elf",
        )

    def test_32_bit_architectures_have_compatible_models(self):
        expected = {
            ArchitectureFamily.LOONGARCH32: "loongarch32-generic",
            ArchitectureFamily.MIPS32: "mips32-generic-r2",
            ArchitectureFamily.RISCV32: "rocket-rv32",
            ArchitectureFamily.X86_32: "skylake-x86-32",
            ArchitectureFamily.ARM32: "cortex-a9-arm32",
            ArchitectureFamily.POWER32: "power9-32",
        }
        for architecture, cpu_name in expected.items():
            with self.subTest(architecture=architecture.value):
                defaults = default_cpus_for_architecture(architecture)
                self.assertIn(cpu_name, [cpu.name for cpu in defaults])

    def test_host_aliases_cover_endian_and_long_machine_names(self):
        with patch("asm_analyzer.targets.platform.machine", return_value="mips64el"):
            self.assertEqual(host_architecture(), ArchitectureFamily.MIPS64)
        with patch("asm_analyzer.targets.platform.machine", return_value="powerpc"):
            self.assertEqual(host_architecture(), ArchitectureFamily.POWER32)


if __name__ == "__main__":
    unittest.main()
