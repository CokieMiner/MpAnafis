"""Tests for guarded confirmed-schedule source rewriting."""

from __future__ import annotations

import unittest

from asm_analyzer.search.source_apply import rewrite_confirmed_schedule


class TestSourceApply(unittest.TestCase):
    def test_reorders_exactly_mapped_instruction_lines(self):
        source = """unsafe {
    asm!(
        \"add {a}, {b}\", // add
        \"mul {c}, {d}\", // mul
        \"xor {e}, {f}\", // xor
        a = in(reg) a,
    );
}
"""
        rewritten = rewrite_confirmed_schedule(
            source,
            2,
            "add x0, x1\nmul x2, x3\nxor x4, x5",
            "add x0, x1\nmul x2, x3\nxor x4, x5",
            "mul x2, x3\nadd x0, x1\nxor x4, x5",
        )
        self.assertLess(rewritten.index("// mul"), rewritten.index("// add"))

    def test_rejects_changed_instruction(self):
        source = """asm!(
    \"add {a}, {b}\",
    \"mul {c}, {d}\",
);
"""
        with self.assertRaisesRegex(ValueError, "changed an instruction"):
            rewrite_confirmed_schedule(
                source,
                1,
                "add x0, x1\nmul x2, x3",
                "add x0, x1\nmul x2, x3",
                "sub x0, x1\nmul x2, x3",
            )

    def test_rejects_standalone_comment_inside_region(self):
        source = """asm!(
    \"add {a}, {b}\",
    // dependency boundary
    \"mul {c}, {d}\",
);
"""
        with self.assertRaisesRegex(ValueError, "manual review"):
            rewrite_confirmed_schedule(
                source,
                1,
                "add x0, x1\nmul x2, x3",
                "add x0, x1\nmul x2, x3",
                "mul x2, x3\nadd x0, x1",
            )


if __name__ == "__main__":
    unittest.main()
