"""Automated repository-wide microarchitectural analysis sweep."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

from ..backends import make_backends
from ..backends.mca_driver import REPO_ROOT
from ..features import extract_kernel_report
from ..models import CpuSpec, DEFAULT_MATRIX, parse_cpus
from ..report.markdown import render_sweep_markdown
from ..report.json_export import export_reports_to_json
from ..types import ArchitectureFamily, KernelAnalysisReport
from ..extract import (
    AsmOperand,
    find_asm_blocks,
    is_string_literal,
    parse_operand,
    split_args,
    real_asm_for_block,
)


def discover_kernels(arch_dir: Optional[Path] = None) -> List[Path]:
    """Find all architecture kernel implementation files."""
    root = arch_dir or (REPO_ROOT / "src" / "int" / "logic" / "unsigned" / "math" / "arch")
    files: List[Path] = []
    for p in root.rglob("*.rs"):
        if p.name in ("mod.rs", "runtime_dispatch.rs", "kernels.rs") or "tests" in p.parts:
            continue
        if (
            p.name.startswith("x86_64")
            or p.name.startswith("aarch64")
            or p.name == "x86.rs"
            or p.name.startswith("arm")
            or p.name.startswith("riscv")
            or p.name.startswith("powerpc")
            or p.name.startswith("s390x")
            or p.name.startswith("loongarch")
            or p.name.startswith("mips")
        ):
            files.append(p)
    return sorted(files)


def extract_kernel_asm(path: Path, use_wsl: bool = False) -> tuple[Optional[str], str]:
    """Extract real AT&T assembly body for the primary asm! block in a kernel file (or read .s file)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".s":
        return text, ""

    blocks = find_asm_blocks(text)
    if not blocks:
        return None, "no asm! blocks found"

    best_start, best_end = max(blocks, key=lambda b: b[1] - b[0])
    block_text = text[best_start:best_end]

    open_paren = block_text.find("(")
    close_paren = block_text.rfind(")")
    args = split_args(block_text[open_paren + 1:close_paren].strip())

    template_lines: List[str] = []
    operands: List[AsmOperand] = []
    options_text: Optional[str] = None

    for a in args:
        a_strip = a.strip()
        if is_string_literal(a_strip):
            template_lines.append(a_strip)
        elif a_strip.startswith("options("):
            options_text = a_strip
        else:
            op = parse_operand(a_strip)
            if op:
                operands.append(op)

    body_lines, err = real_asm_for_block(template_lines, operands, options_text, use_wsl)
    if body_lines is None:
        stripped: List[str] = []
        for line in template_lines:
            s = line.strip().strip('"').strip()
            if s and not s.startswith((".", "#", "//")):
                stripped.append(s)
        if stripped:
            return "\n".join(stripped), f"template fallback: {err}"
        return None, err
    return "\n".join(body_lines), ""


def run_sweep(
    target_path: Optional[str] = None,
    cpus: Optional[List[CpuSpec]] = None,
    use_wsl: bool = False,
    markdown: bool = False,
    as_json: bool = False,
) -> int:
    """Execute kernel sweep across targets and print output."""
    cpu_specs = cpus or parse_cpus(",".join(DEFAULT_MATRIX))
    cpu_names = [c.name for c in cpu_specs]

    if target_path:
        p = Path(target_path)
        kernel_files = [p] if p.is_file() else discover_kernels(p)
    else:
        kernel_files = discover_kernels()

    backends = make_backends(["llvm-mca", "osaca", "uica"], wsl=use_wsl)
    reports: List[KernelAnalysisReport] = []

    for kpath in kernel_files:
        asm_code, err = extract_kernel_asm(kpath, use_wsl=use_wsl)
        if not asm_code:
            if not as_json and not markdown:
                print(f"Skipping {kpath}: {err}", file=sys.stderr)
            continue

        try:
            rel = kpath.relative_to(REPO_ROOT / "src" / "int" / "logic" / "unsigned" / "math" / "arch")
            kname = str(rel).replace("\\", "/").replace(".rs", "")
        except ValueError:
            kname = kpath.stem

        cpu_cycles: Dict[str, float] = {}
        for cpu in cpu_specs:
            best_cycle: Optional[float] = None
            for bname in ("llvm-mca", "osaca", "uica"):
                b = backends.get(bname)
                if b and b.supports(cpu):
                    cyc = b.analyze(asm_code, cpu)
                    if cyc is not None:
                        # Store per-backend result with qualified key
                        cpu_cycles[f"{cpu.name}/{bname}"] = cyc
                        if best_cycle is None or cyc < best_cycle:
                            best_cycle = cyc
            # Store the consensus best for the summary column
            if best_cycle is not None:
                cpu_cycles[cpu.name] = best_cycle

        if "aarch64" in kpath.name:
            arch_family = ArchitectureFamily.AARCH64
        elif "arm" in kpath.name:
            arch_family = ArchitectureFamily.ARM32
        elif "x86.rs" in kpath.name:
            arch_family = ArchitectureFamily.X86_32
        elif "riscv64" in kpath.name:
            arch_family = ArchitectureFamily.RISCV64
        elif "s390x" in kpath.name:
            arch_family = ArchitectureFamily.S390X
        elif "power" in kpath.name or "ppc" in kpath.name:
            arch_family = ArchitectureFamily.POWER64
        else:
            arch_family = ArchitectureFamily.X86_64

        rep = extract_kernel_report(
            asm=asm_code,
            kernel_name=kname,
            target_arch=arch_family,
            cpu_cycles=cpu_cycles,
        )
        reports.append(rep)

    if as_json:
        print(export_reports_to_json(reports))
    elif markdown:
        print(render_sweep_markdown(reports, cpu_names))
    else:
        print(render_sweep_markdown(reports, cpu_names))

    return 0
