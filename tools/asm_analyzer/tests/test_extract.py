"""Regression tests for fail-closed Rust inline-assembly extraction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asm_analyzer.extract import (
    Operand,
    extract_named_asm_regions,
    real_asm_for_block,
    real_asm_for_source,
    render_snippet,
)
from asm_analyzer.kernel_source import extract_kernel_asm, extract_kernel_variants


class TestExtraction(unittest.TestCase):
    def test_explicit_register_outputs_use_assignable_variables(self):
        snippet = render_snippet(
            ['"divq {divisor}"'],
            [
                Operand(name=None, cls="rax", kind="inout", explicit=True),
                Operand(name=None, cls="rdx", kind="inout", explicit=True),
                Operand(name="divisor", cls="reg", kind="in"),
            ],
            None,
        )
        self.assertIn('inout("rax") __operand_0', snippet)
        self.assertIn('inout("rdx") __operand_1', snippet)
        self.assertNotIn('inout("rax") 0', snippet)
        self.assertIn("divisor = in(reg) divisor", snippet)

    def test_explicit_discard_output_stays_a_clobber(self):
        snippet = render_snippet(
            ['"add 3, 3, 4"'],
            [Operand(name=None, cls="xer", kind="out", explicit=True, discard=True)],
            None,
        )
        self.assertIn('out("xer") _', snippet)
        self.assertNotIn('out("xer") __operand_0', snippet)

    def test_generic_register_classes_keep_named_placeholders(self):
        snippet = render_snippet(
            ['"setc {carry}"'],
            [Operand(name="carry", cls="reg_byte", kind="lateout")],
            None,
        )
        self.assertIn("carry = lateout(reg_byte) carry", snippet)

    def test_generated_inputs_are_opaque_and_distinct(self):
        snippet = render_snippet(
            ['"movq ({src}), %rax"', '"movq %rax, ({dst})"'],
            [
                Operand(name="src", cls="reg", kind="in"),
                Operand(name="dst", cls="reg", kind="in"),
            ],
            None,
        )
        self.assertIn("black_box(4096usize)", snippet)
        self.assertIn("black_box(8192usize)", snippet)

    def test_pure_inout_block_is_emitted_as_resolved_assembly(self):
        assembly, error = real_asm_for_block(
            ['"divq {divisor}"'],
            [
                Operand(name="divisor", cls="reg", kind="in"),
                Operand(name=None, cls="rax", kind="inout", explicit=True),
                Operand(name=None, cls="rdx", kind="inout", explicit=True),
            ],
            "options(pure, nomem, nostack, att_syntax)",
            False,
        )
        self.assertEqual(error, "")
        self.assertIsNotNone(assembly)
        resolved = "\n".join(assembly or [])
        self.assertIn("divq", resolved)
        self.assertNotIn("{divisor}", resolved)

    def test_macro_generated_source_fallback_emits_real_assembly(self):
        source = """
use core::arch::asm;
use super::Limb;
macro_rules! define_kernel {
    ($name:ident) => {
        #[inline(always)]
        pub unsafe fn $name(value: Limb) -> Limb {
            let mut out = value;
            unsafe { asm!("incq {out}", out = inout(reg) out, options(att_syntax)); }
            out
        }
    };
}
define_kernel!(generated_kernel);
"""
        assembly, error = real_asm_for_source(source, False)
        self.assertEqual(error, "")
        self.assertIsNotNone(assembly)
        self.assertIn("incq", "\n".join(assembly or []))

    def test_named_regions_keep_enclosing_symbol(self):
        regions = extract_named_asm_regions(
            """
generated_kernel:
#APP
incq %rax
#NO_APP
retq
"""
        )
        self.assertEqual(regions, [("generated_kernel", ["incq %rax"])])

    def test_arm_comment_markers_are_extracted(self):
        regions = extract_named_asm_regions(
            "arm_kernel:\n@APP\nadds r0, r0, r1\n@NO_APP\nbx lr",
        )
        self.assertEqual(regions, [("arm_kernel", ["adds r0, r0, r1"])])

    def test_macro_source_exposes_every_generated_kernel(self):
        source_text = """
use core::arch::asm;
use super::Limb;
macro_rules! define_kernel {
    ($name:ident, $instruction:literal) => {
        #[inline(always)]
        pub unsafe fn $name(value: Limb) -> Limb {
            let mut out = value;
            unsafe { asm!($instruction, out = inout(reg) out, options(att_syntax)); }
            out
        }
    };
}
define_kernel!(first_generated, "incq {out}");
define_kernel!(second_generated, "decq {out}");
"""
        with tempfile.NamedTemporaryFile("w", suffix=".rs", delete=False) as source:
            source.write(source_text)
            path = Path(source.name)
        try:
            variants, error = extract_kernel_variants(path)
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(error, "")
        self.assertEqual(
            {variant.name.rsplit("::", 1)[-1] for variant in variants},
            {"first_generated", "second_generated"},
        )

    def test_fixed_macro_symbol_provides_logical_limb_width(self):
        path = (
            Path(__file__).resolve().parents[3]
            / "src/int/logic/unsigned/math/arch/mul_basecase_unchecked/x86_64_adx.rs"
        )
        variants, error = extract_kernel_variants(path)
        self.assertEqual(error, "")
        widths = {
            variant.name.rsplit("::", 1)[-1]: variant.limbs_per_iteration
            for variant in variants
        }
        self.assertEqual(widths["add_mul_10_limbs_unchecked"], 10)
        self.assertEqual(widths["mul_2x10_limbs_unchecked"], 10)

    @patch(
        "asm_analyzer.kernel_source.real_asm_for_block",
        return_value=(None, "synthetic rustc failure"),
    )
    def test_rustc_failure_never_returns_template_as_assembly(self, _extract):
        with tempfile.NamedTemporaryFile("w", suffix=".rs", delete=False) as source:
            source.write('unsafe { asm!("divq {d}", d = in(reg) 3usize); }')
            path = Path(source.name)
        try:
            assembly, error = extract_kernel_asm(path)
        finally:
            path.unlink(missing_ok=True)

        self.assertIsNone(assembly)
        self.assertIn("rustc inline-assembly extraction failed", error)
        self.assertNotIn("template fallback", error)


if __name__ == "__main__":
    unittest.main()
