"""Repository kernel discovery and fail-closed Rust assembly extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .backends.mca_driver import REPO_ROOT
from .extract import (
    extract_asm_blocks,
    real_asm_for_block,
    real_asm_regions_for_source,
)
from .targets import rust_target_for_path

_ARCH_ROOT = REPO_ROOT / "src" / "int" / "logic" / "unsigned" / "math" / "arch"


@dataclass(frozen=True)
class KernelAssembly:
    """One independently emitted assembly block from a repository source."""

    name: str
    asm: str
    source_line: Optional[int] = None
    limbs_per_iteration: Optional[int] = None


def discover_kernels(arch_dir: Optional[Path] = None) -> List[Path]:
    """Find architecture implementation files containing inline assembly."""
    root = arch_dir or _ARCH_ROOT
    files: List[Path] = []
    for path in root.rglob("*.rs"):
        if path.name in ("mod.rs", "runtime_dispatch.rs", "kernels.rs") or "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if re.search(r"\basm\s*!", text) is None:
            continue
        if path.name.startswith(
            (
                "x86_64",
                "aarch64",
                "arm",
                "riscv",
                "powerpc",
                "s390x",
                "loongarch",
                "mips",
            )
        ) or path.name == "x86.rs":
            files.append(path)
    return sorted(files)


def extract_kernel_asm(path: Path, use_wsl: bool = False) -> tuple[Optional[str], str]:
    """Extract the largest compiler-emitted variant for single-kernel commands."""
    variants, error = extract_kernel_variants(path, use_wsl=use_wsl)
    if not variants:
        return None, error
    return max(variants, key=lambda variant: len(variant.asm.splitlines())).asm, ""


def extract_kernel_variants(
    path: Path,
    use_wsl: bool = False,
) -> tuple[List[KernelAssembly], str]:
    """Extract every real compiler-emitted assembly variant in a source.

    Macro-defined files compile as complete source so each generated function
    retains its actual repetition count and symbol name. Non-macro files
    compile every inline-assembly block independently. Rust templates are
    never submitted to simulator backends.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".s":
        return [KernelAssembly(name=path.stem, asm=text)], ""

    target = rust_target_for_path(path)
    base_name = kernel_name(path)
    if "macro_rules!" in text:
        support_source = ""
        if "select_arch_kernel!" in text:
            support_source = (_ARCH_ROOT / "kernel_selection.rs").read_text(encoding="utf-8")
        regions, error = real_asm_regions_for_source(
            text,
            use_wsl,
            target=target,
            support_source=support_source,
        )
        if regions is None:
            return [], f"rustc macro-source extraction failed: {error}"
        return [
            KernelAssembly(
                name=f"{base_name}::{name}",
                asm="\n".join(body),
                limbs_per_iteration=_fixed_limb_count(name),
            )
            for name, body in regions
        ], ""

    try:
        blocks = extract_asm_blocks(path)
    except ValueError as error:
        return [], str(error)
    if not blocks:
        return [], "no asm! blocks found"

    variants: List[KernelAssembly] = []
    failures: List[str] = []
    multiple = len(blocks) > 1
    for block in blocks:
        body_lines, error = real_asm_for_block(
            block.instructions,
            block.operands,
            block.options,
            use_wsl,
            target=target,
        )
        if body_lines is not None:
            name = f"{base_name}:asm@{block.line}" if multiple else base_name
            variants.append(
                KernelAssembly(
                    name=name,
                    asm="\n".join(body_lines),
                    source_line=block.line,
                )
            )
        else:
            failures.append(f"line {block.line}: {error}")

    if failures:
        return [], (
            "rustc inline-assembly extraction failed for one or more blocks: "
            + "; ".join(failures)
        )
    return variants, ""


def kernel_name(path: Path) -> str:
    """Return a stable repository-relative kernel name when possible."""
    try:
        return str(path.resolve().relative_to(_ARCH_ROOT.resolve())).replace(
            "\\",
            "/",
        ).removesuffix(".rs")
    except ValueError:
        return path.stem


def _fixed_limb_count(symbol: str) -> Optional[int]:
    match = re.fullmatch(
        r"(?:add_mul_|mul_2x)(\d+)_limbs_unchecked",
        symbol,
    )
    return int(match.group(1)) if match is not None else None


__all__ = [
    "KernelAssembly",
    "discover_kernels",
    "extract_kernel_asm",
    "extract_kernel_variants",
    "kernel_name",
]
