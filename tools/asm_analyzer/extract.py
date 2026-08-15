#!/usr/bin/env python3
"""Extract real assembly that rustc emits for an asm! block."""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

from .asm_util import run, wsl_path

_PLACEHOLDER = re.compile(r"\{(\w+)(?::(\w+))?\}")
_MEM = re.compile(r"\(\s*\{(\w+)\}(?:\s*,\s*\{(\w+)\})?\s*(?:,\s*[0-9a-zA-Z_]+)?\)")


@dataclass
class AsmOperand:
    name: Optional[str]
    cls: Optional[str]      # "reg" | concrete register string
    kind: str               # "in" | "out" | "inout" | "lateout"


@dataclass
class Operand:
    """One asm! operand clause: name = kind(cls) expr."""
    name: Optional[str]
    cls: Optional[str]
    kind: str


@dataclass
class AsmBlock:
    line: int
    instructions: List[str]
    operands: List[AsmOperand]
    options: Optional[str]


def _skip_string(text: str, i: int) -> int:
    n = len(text)
    if text[i] == "r":
        hashes = 0
        j = i + 1
        while j < n and text[j] == "#":
            hashes += 1
            j += 1
        if j < n and text[j] == '"':
            term = '"' + "#" * hashes
            end = text.find(term, j + 1)
            return end + len(term) if end != -1 else n
    if text[i] == '"':
        j = i + 1
        while j < n:
            if text[j] == "\\":
                j += 2
                continue
            if text[j] == '"':
                return j + 1
            j += 1
    return n


def _skip_comment(text: str, i: int) -> int:
    n = len(text)
    if text.startswith("//", i):
        j = text.find("\n", i)
        return j if j != -1 else n
    if text.startswith("/*", i):
        j = text.find("*/", i + 2)
        return j + 2 if j != -1 else n
    return i


def find_asm_blocks(text: str) -> List[Tuple[int, int]]:
    blocks: List[Tuple[int, int]] = []
    for m in re.finditer(r"\basm!\s*\(", text):
        open_pos = m.end() - 1
        depth = 0
        i = open_pos
        n = len(text)
        while i < n:
            c = text[i]
            if c == '"':
                i = _skip_string(text, i)
                continue
            if c == "/" and i + 1 < n and text[i + 1] in ("/", "*"):
                i = _skip_comment(text, i)
                continue
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
                if depth == 0:
                    blocks.append((m.start(), i + 1))
                    break
            i += 1
    return blocks


def split_args(body: str) -> List[str]:
    parts: List[str] = []
    cur: List[str] = []
    depth = 0
    i = 0
    n = len(body)
    while i < n:
        c = body[i]
        if c == '"':
            end = _skip_string(body, i)
            cur.append(body[i:end])
            i = end
            continue
        if c == "/" and i + 1 < n and body[i + 1] in ("/", "*"):
            i = _skip_comment(body, i)
            continue
        if c in "([{":
            depth += 1
            cur.append(c)
        elif c in ")]}":
            depth -= 1
            cur.append(c)
        elif c == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    if cur:
        parts.append("".join(cur).strip())
    return [p for p in parts if p]


def is_string_literal(arg: str) -> bool:
    return arg.startswith('"') or arg.startswith('r"')


def parse_operand(arg: str) -> Optional[AsmOperand]:
    arg = arg.strip()
    name: Optional[str] = None
    body = arg
    eq = arg.find("=")
    if eq != -1 and arg[:eq].strip().isidentifier():
        name = arg[:eq].strip()
        body = arg[eq + 1:].strip()
    m = re.match(r"(in|out|inout|lateout)\s*\((.*?)\)", body, re.S)
    if not m:
        return None
    kind, inner = m.group(1), m.group(2).strip()
    cls = None
    if inner == "reg":
        cls = "reg"
    elif inner.startswith('"') and inner.endswith('"'):
        cls = inner[1:-1]
    return AsmOperand(name=name, cls=cls, kind=kind)


def extract_asm_blocks(path: Path) -> List[AsmBlock]:
    """Parse every asm! block in path; return raw template args + operands."""
    text = path.read_text(encoding="utf-8")
    blocks: List[AsmBlock] = []
    for start, end in find_asm_blocks(text):
        line = text.count("\n", 0, start) + 1
        body = text[start + len("asm!("):end - 1]
        args = split_args(body)
        insts = [a for a in args if is_string_literal(a)]
        opts = next((a for a in args if a.startswith("options(")), None)
        ops = [parse_operand(a) for a in args if not is_string_literal(a) and not a.startswith("options(")]
        valid_ops = [op for op in ops if op is not None]
        blocks.append(AsmBlock(line=line, instructions=insts, operands=valid_ops, options=opts))
    return blocks


def _classify_pointers(template: str, operands: List[Operand]) -> Set[str]:
    ptr = set()
    for m in _MEM.finditer(template):
        if m.group(1):
            ptr.add(m.group(1))
    return {op.name for op in operands if op.name and op.name in ptr}


def _expand_macro_templates(lines: List[str]) -> List[str]:
    out = []
    for ln in lines:
        s = ln
        s = re.sub(r'stringify!\(\$offset\)', '"0"', s)
        s = re.sub(r'stringify!\(\$close1\)', '"8"', s)
        s = re.sub(r'stringify!\(\$close2\)', '"16"', s)
        s = re.sub(r'stringify!\(\$close3\)', '"24"', s)
        s = re.sub(r'stringify!\(\$close4\)', '"32"', s)
        s = re.sub(r'stringify!\(\$\w+\)', '"0"', s)
        s_strip = s.strip()
        if s_strip.startswith("concat!(") and s_strip.endswith(")"):
            inner = s_strip[7:-1].strip()
            parts = re.findall(r'"([^"]*)"', inner)
            if parts:
                joined = "".join(parts)
                s = f'"{joined}"'
        out.append(s)
    return out


def render_snippet(template_lines: List[str], operands: List[Operand],
                   options_text: Optional[str]) -> str:
    """Render a standalone Rust translation unit containing the inline assembly block."""
    template_lines = _expand_macro_templates(template_lines)
    template = "\n".join(template_lines)
    pointers = _classify_pointers(template, operands)
    names = [op.name for op in operands if op.name]

    lines: List[str] = [
        "#![no_std]",
        "#![allow(unused_mut, unused_unsafe, unused_variables, clippy::all)]",
        "use core::arch::asm;",
        "",
        "#[inline(never)]",
        "#[no_mangle]",
        "pub unsafe fn __ks_kernel() {",
    ]
    for n in names:
        if n in pointers:
            lines.append(f"    let mut {n}: *mut usize = core::ptr::null_mut();")
        else:
            lines.append(f"    let mut {n}: usize = 0;")
    lines.append("    unsafe {")
    lines.append("        asm!(")
    for t in template_lines:
        if t.startswith('"') or t.startswith('r"') or t.startswith("concat!("):
            lines.append(f"            {t},")
        else:
            esc = t.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'            "{esc}",')
    for op in operands:
        if op.cls == "reg":
            expr = op.name if op.name else "0"
            lines.append(f"            {op.name} = {op.kind}(reg) {expr},")
        else:
            reg = op.cls or "rax"
            if op.name:
                lines.append(f'            {op.name} = {op.kind}("{reg}") {op.name},')
            elif op.kind in ("in", "inout"):
                lines.append(f'            {op.kind}("{reg}") 0,')
            else:
                lines.append(f'            {op.kind}("{reg}") _,')
    if options_text:
        lines.append(f"            {options_text},")
    lines.append("        );")
    lines.append("    }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def compile_snippet(snippet: str, workdir: Path, use_wsl: bool) -> tuple[Optional[str], str]:
    """Compile a generated Rust snippet to target assembly via rustc."""
    src = workdir / "k.rs"
    out = workdir / "k.s"
    src.write_text(snippet, encoding="utf-8")
    src_arg = wsl_path(src) if use_wsl else str(src)
    out_arg = wsl_path(out) if use_wsl else str(out)
    cmd = ["rustc", "--edition=2021", "-C", "opt-level=3",
           "--crate-type=lib", "--emit=asm", "-o", out_arg, src_arg]
    r = run(cmd, use_wsl)
    if r.returncode != 0:
        return None, (r.stderr or r.stdout or "rustc failed")[:800]
    if not out.exists():
        return None, "rustc produced no assembly output"
    return out.read_text(encoding="utf-8", errors="replace"), ""


def extract_asm_region(asm_text: str) -> List[str]:
    """Extract compiler-emitted inline assembly statements delimited by #APP and #NO_APP."""
    out: List[str] = []
    inside = False
    for ln in asm_text.splitlines():
        s = ln.strip()
        if s == "#APP":
            inside = True
            continue
        if s == "#NO_APP":
            inside = False
            continue
        if inside and s:
            out.append(s)
    return out


def real_asm_for_block(template_lines: List[str], operands: List[Operand | AsmOperand],
                       options_text: Optional[str], use_wsl: bool) -> tuple[Optional[List[str]], str]:
    """Compile an inline assembly block through rustc and extract real compiler-lowered assembly."""
    ops = [Operand(name=op.name, cls=op.cls, kind=op.kind) for op in operands]
    snippet = render_snippet(template_lines, ops, options_text)
    with tempfile.TemporaryDirectory() as tmp:
        asm_text, err = compile_snippet(snippet, Path(tmp), use_wsl)
        if asm_text is None:
            return None, err
        body = extract_asm_region(asm_text)
        if not body:
            return None, "no #APP/#NO_APP inline-asm region emitted"
        return body, ""
