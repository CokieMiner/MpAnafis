"""Rust compilation harness for lowering parsed inline assembly."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import List, Optional, Set, Tuple

from .asm_util import run, wsl_path
from .emitted_asm import extract_asm_region, extract_named_asm_regions
from .extraction_parser import Operand, is_string_literal

_MEMORY_PLACEHOLDER_RE = re.compile(
    r"\(\s*\{(\w+)\}(?:\s*,\s*\{(\w+)\})?\s*(?:,\s*[0-9a-zA-Z_]+)?\)",
)
_BUILD_STD_TARGETS = frozenset(
    ("mips-unknown-linux-gnu", "mips64-unknown-linux-gnuabi64"),
)
_BUILD_STD_TARGET_DIR = (
    Path(__file__).resolve().parents[2] / "target" / "asm-analyzer-build-std"
)


def render_snippet(
    template_lines: List[str],
    operands: List[Operand],
    options_text: Optional[str],
) -> str:
    """Render a standalone Rust translation unit for one inline-asm block."""
    expanded_templates = template_lines
    template = "\n".join(expanded_templates)
    pointers = _classify_pointers(template, operands)
    variables = [
        operand.name or f"__operand_{index}"
        for index, operand in enumerate(operands)
    ]

    lines: List[str] = [
        "#![no_std]",
        "#![allow(unused_mut, unused_unsafe, unused_variables, clippy::all)]",
        "use core::arch::asm;",
        "",
        "#[inline(never)]",
        "#[no_mangle]",
        "pub unsafe fn __ks_kernel() {",
    ]
    for index, variable in enumerate(variables):
        operand = operands[index]
        if variable in pointers:
            address = 0x1000 * (index + 1)
            lines.append(
                f"    let mut {variable} = core::hint::black_box({address}usize) "
                "as *mut usize;",
            )
        elif operand.cls == "reg_byte" and not operand.explicit:
            lines.append(
                f"    let mut {variable}: u8 = core::hint::black_box({index + 1}u8);",
            )
        else:
            lines.append(
                f"    let mut {variable}: usize = "
                f"core::hint::black_box({index + 1}usize);",
            )
    lines.extend(("    unsafe {", "        asm!("))
    for template_line in expanded_templates:
        if is_string_literal(template_line) or template_line.startswith("concat!("):
            lines.append(f"            {template_line},")
        else:
            escaped = template_line.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'            "{escaped}",')
    for index, operand in enumerate(operands):
        variable = variables[index]
        if not operand.explicit:
            register_class = operand.cls or "reg"
            prefix = f"{operand.name} = " if operand.name else ""
            lines.append(
                f"            {prefix}{operand.kind}({register_class}) {variable},",
            )
        else:
            register = operand.cls or "rax"
            value = (
                f"{variable} => _" if operand.discard and operand.kind in ("inout", "inlateout")
                else "_" if operand.discard else variable
            )
            lines.append(f'            {operand.kind}("{register}") {value},')
    if options_text:
        lines.append(f"            {options_text},")
    lines.extend(("        );", "    }"))
    for variable in variables:
        lines.append(f"    core::hint::black_box({variable});")
    lines.append("}")
    return "\n".join(lines) + "\n"


def compile_snippet(
    snippet: str,
    workdir: Path,
    use_wsl: bool,
    target: Optional[str] = None,
) -> tuple[Optional[str], str]:
    """Compile a generated Rust snippet to target assembly through rustc."""
    source_path = workdir / "k.rs"
    output_path = workdir / "k.s"
    source_path.write_text(snippet, encoding="utf-8")
    source_argument = wsl_path(source_path) if use_wsl else str(source_path)
    output_argument = wsl_path(output_path) if use_wsl else str(output_path)
    command = [
        "rustc", "--edition=2021", "-C", "opt-level=3", "--crate-type=lib",
        "--emit=asm", "-o", output_argument, source_argument,
    ]
    if target:
        command[1:1] = ["--target", target]
    result = run(command, use_wsl)
    if result.returncode != 0:
        diagnostic = result.stderr or result.stdout or "rustc failed"
        if target in _BUILD_STD_TARGETS and "can't find crate for `core`" in diagnostic:
            return _compile_snippet_with_build_std(
                workdir,
                output_path,
                use_wsl,
                target,
            )
        return None, diagnostic[:800]
    if not output_path.exists():
        return None, "rustc produced no assembly output"
    return output_path.read_text(encoding="utf-8", errors="replace"), ""


def _compile_snippet_with_build_std(
    workdir: Path,
    output_path: Path,
    use_wsl: bool,
    target: str,
) -> tuple[Optional[str], str]:
    """Compile targets whose prebuilt ``core`` is not distributed by rustup."""
    source_path = workdir / "k.rs"
    source = source_path.read_text(encoding="utf-8")
    source_path.write_text(
        source.replace(
            "#![no_std]\n",
            "#![no_std]\n#![feature(asm_experimental_arch)]\n",
            1,
        ),
        encoding="utf-8",
    )
    manifest_path = workdir / "Cargo.toml"
    manifest_path.write_text(
        "[package]\n"
        'name = "asm-analyzer-kernel"\n'
        'version = "0.0.0"\n'
        'edition = "2021"\n'
        "\n"
        "[lib]\n"
        'path = "k.rs"\n'
        'crate-type = ["lib"]\n',
        encoding="utf-8",
    )
    manifest_argument = wsl_path(manifest_path) if use_wsl else str(manifest_path)
    output_argument = wsl_path(output_path) if use_wsl else str(output_path)
    target_dir_argument = (
        wsl_path(_BUILD_STD_TARGET_DIR) if use_wsl else str(_BUILD_STD_TARGET_DIR)
    )
    result = run(
        [
            "cargo",
            "rustc",
            "-Z",
            "build-std=core",
            "--manifest-path",
            manifest_argument,
            "--target",
            target,
            "--target-dir",
            target_dir_argument,
            "--release",
            "--",
            "--emit=asm",
            "-o",
            output_argument,
        ],
        use_wsl,
    )
    if result.returncode != 0:
        diagnostic = result.stderr or result.stdout or "cargo build-std failed"
        return None, diagnostic[:800]
    candidates = [output_path, *sorted(workdir.glob("*.s"))]
    emitted_path = next((path for path in candidates if path.exists()), None)
    if emitted_path is None:
        outputs = ", ".join(sorted(path.name for path in workdir.iterdir()))
        return None, f"cargo build-std produced no assembly output (files: {outputs})"
    return emitted_path.read_text(encoding="utf-8", errors="replace"), ""


def real_asm_for_block(
    template_lines: List[str],
    operands: List[Operand],
    options_text: Optional[str],
    use_wsl: bool,
    target: Optional[str] = None,
) -> tuple[Optional[List[str]], str]:
    """Compile one inline-assembly block and return compiler-resolved assembly."""
    snippet = render_snippet(template_lines, operands, options_text)
    with tempfile.TemporaryDirectory() as temporary_directory:
        asm_text, error = compile_snippet(
            snippet,
            Path(temporary_directory),
            use_wsl,
            target=target,
        )
        if asm_text is None:
            return None, error
        body = extract_asm_region(asm_text)
        if not body:
            return None, "no #APP/#NO_APP inline-asm region emitted"
        return body, ""


def real_asm_for_source(
    source: str,
    use_wsl: bool,
    target: Optional[str] = None,
) -> tuple[Optional[List[str]], str]:
    """Compile a self-contained architecture source and select its largest region."""
    regions, error = real_asm_regions_for_source(source, use_wsl, target=target)
    if regions is None:
        return None, error
    return max((body for _, body in regions), key=len), ""


def real_asm_regions_for_source(
    source: str,
    use_wsl: bool,
    target: Optional[str] = None,
    *,
    support_source: str = "",
) -> tuple[Optional[List[Tuple[str, List[str]]]], str]:
    """Compile architecture source with optional parent macros and return asm regions."""
    if "use super::Limb;" not in source:
        return None, "source expansion only supports the narrow parent Limb import"
    kernel_source = source.replace("use super::Limb;", "type Limb = usize;").replace(
        "#[inline(always)]",
        "#[inline(never)]\n#[no_mangle]",
    )
    # Preserve module-level documentation/attributes while giving the kernel
    # the same lexical access to shared cfg macros as its production module.
    if support_source:
        kernel_source = support_source + "\nmod kernel {\n" + kernel_source + "\n}\n"
    snippet = (
        "#![no_std]\n"
        "#![allow(unsafe_code, unused_imports, dead_code, clippy::all)]\n"
        + kernel_source
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        asm_text, error = compile_snippet(
            snippet,
            Path(temporary_directory),
            use_wsl,
            target=target,
        )
        if asm_text is None:
            return None, error
        regions = extract_named_asm_regions(asm_text)
        if not regions:
            return None, "no #APP/#NO_APP inline-asm region emitted"
        return regions, ""


def _classify_pointers(template: str, operands: List[Operand]) -> Set[str]:
    pointer_names = {
        match.group(1)
        for match in _MEMORY_PLACEHOLDER_RE.finditer(template)
        if match.group(1)
    }
    return {
        operand.name
        for operand in operands
        if operand.name and operand.name in pointer_names
    }


__all__ = [
    "compile_snippet",
    "real_asm_for_block",
    "real_asm_for_source",
    "real_asm_regions_for_source",
    "render_snippet",
]
