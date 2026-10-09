"""Rust lexical boundaries and conservative source application."""

import tempfile
import unittest
from pathlib import Path

from asm_analyzer.extraction_harness import render_snippet
from asm_analyzer.extraction_parser import extract_asm_blocks, find_asm_blocks, parse_operand, split_args
from asm_analyzer.search.source_apply import rewrite_confirmed_schedule


class TestParserBoundaries(unittest.TestCase):
    def test_discovery_ignores_comments_strings_and_character_literals(self):
        source = '''// asm!("fake");
/* outer /* nested */ asm!("fake"); */
const TEXT: &str = r##"asm!("fake")"##;
const QUOTE: char = '"';
unsafe { asm! ("real"); }
'''
        blocks = find_asm_blocks(source)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(source[slice(*blocks[0])], 'asm! ("real")')

    def test_split_arguments_handles_nested_comments_and_terminal_raw_string(self):
        self.assertEqual(split_args('"a", /* outer /* nested */ end */ r#"b,c"#'), ['"a"', 'r#"b,c"#'])

    def test_whitespace_after_macro_name_preserves_first_template(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernel.rs"
            path.write_text('asm ! \n ("nop", options (nostack));', encoding="utf-8")
            self.assertEqual(extract_asm_blocks(path)[0].instructions, ['"nop"'])

    def test_unsupported_arguments_are_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernel.rs"
            path.write_text('asm!("nop", clobber_abi("C"));', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported asm! argument"):
                extract_asm_blocks(path)

    def test_inlateout_discard_keeps_the_input(self):
        operand = parse_operand('inlateout("rax") value => _')
        self.assertEqual(operand.kind, "inlateout")
        self.assertTrue(operand.discard)
        snippet = render_snippet(['r#"incq %rax"#'], [operand], "options(att_syntax)")
        self.assertIn('r#"incq %rax"#', snippet)
        self.assertIn('inlateout("rax") __operand_0 => _', snippet)
        self.assertFalse(parse_operand('out("rax") _named_variable').discard)

    def test_source_application_rejects_ambiguous_physical_lines(self):
        for templates in (
            '    "movq %rax, %rbx\\nmovq %rcx, %rdx",\n    "",',
            '    "movq %rax, %rbx", "movq %rcx, %rdx",',
            '    concat!("movq %rax", ", %rbx"),\n    "movq %rcx, %rdx",',
            '    "movq %rax, %rbx; movq %rcx, %rdx",\n    "",',
        ):
            source = "asm!(\n" + templates + "\n);"
            with self.subTest(templates=templates), self.assertRaises(ValueError):
                rewrite_confirmed_schedule(source, 1, "a\nb", "a\nb", "b\na")

    def test_source_application_supports_standalone_raw_strings(self):
        source = 'asm!(\n    r#"a"#,\n    r##"b"##,\n);'
        rewritten = rewrite_confirmed_schedule(source, 1, "a\nb", "a\nb", "b\na")
        self.assertLess(rewritten.index('r##"b"'), rewritten.index('r#"a"'))


if __name__ == "__main__":
    unittest.main()
