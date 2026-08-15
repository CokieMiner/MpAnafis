"""Single assembly file microarchitectural analysis command."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from ..backends import make_backends
from ..features import extract_kernel_report
from ..models import CpuSpec, DEFAULT_MATRIX, parse_cpus
from ..report.markdown import render_sweep_markdown
from ..types import ArchitectureFamily


def run_analyze(
    asm_path: str,
    cpus: Optional[List[CpuSpec]] = None,
    use_wsl: bool = False,
    as_json: bool = False,
) -> int:
    """Analyze a single AT&T assembly file (.s) or Rust kernel file (.rs)."""
    p = Path(asm_path)
    if not p.exists():
        print(f"Error: File not found: {p}", file=sys.stderr)
        return 1

    if p.suffix == ".rs":
        from .sweep import extract_kernel_asm
        asm_code, err = extract_kernel_asm(p, use_wsl=use_wsl)
        if not asm_code:
            print(f"Error: Could not extract assembly from {p}: {err}", file=sys.stderr)
            return 1
        asm_text = asm_code
    else:
        asm_text = p.read_text(encoding="utf-8", errors="replace")

    path_str = str(p).lower()
    if "aarch64" in path_str:
        target_arch = ArchitectureFamily.AARCH64
    elif "arm" in path_str and "aarch64" not in path_str:
        target_arch = ArchitectureFamily.ARM32
    elif "riscv64" in path_str:
        target_arch = ArchitectureFamily.RISCV64
    elif "powerpc" in path_str or "ppc" in path_str:
        target_arch = ArchitectureFamily.POWER64
    elif "s390x" in path_str:
        target_arch = ArchitectureFamily.S390X
    elif "x86" in path_str and "x86_64" not in path_str:
        target_arch = ArchitectureFamily.X86_32
    else:
        target_arch = ArchitectureFamily.X86_64

    cpu_specs = cpus or parse_cpus(",".join(DEFAULT_MATRIX))
    backends = make_backends(["llvm-mca", "osaca", "uica"], wsl=use_wsl)

    cpu_cycles: Dict[str, float] = {}
    for cpu in cpu_specs:
        for bname in ("llvm-mca", "osaca", "uica"):
            b = backends.get(bname)
            if b and b.supports(cpu):
                cyc = b.analyze(asm_text, cpu)
                if cyc is not None:
                    cpu_cycles[cpu.name] = cyc
                    break

    report = extract_kernel_report(
        asm=asm_text,
        kernel_name=p.stem,
        target_arch=target_arch,
        cpu_cycles=cpu_cycles,
    )

    if as_json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(render_sweep_markdown([report], [c.name for c in cpu_specs]))

    return 0
