"""Side-by-side comparison engine for assembly kernel variants."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

from ..backends import make_backends
from ..features import extract_kernel_report
from ..models import CpuSpec, DEFAULT_MATRIX, parse_cpus
from ..report.markdown import render_diff_markdown
from ..report.json_export import export_diff_to_json
from ..types import KernelComparisonDiff
from .sweep import extract_kernel_asm


def compare_kernels(
    path_a: Path,
    path_b: Path,
    cpus: Optional[List[CpuSpec]] = None,
    use_wsl: bool = False,
) -> KernelComparisonDiff:
    """Analyze and generate side-by-side comparison diff between two kernels."""
    cpu_specs = cpus or parse_cpus(",".join(DEFAULT_MATRIX))
    backends = make_backends(["llvm-mca", "osaca", "uica"], wsl=use_wsl)

    # Extract ASM for kernel A
    if path_a.suffix == ".rs":
        asm_a, err_a = extract_kernel_asm(path_a, use_wsl=use_wsl)
        name_a = path_a.stem
    else:
        asm_a = path_a.read_text(encoding="utf-8", errors="replace")
        name_a = path_a.stem
        err_a = ""

    if not asm_a:
        raise ValueError(f"Could not load assembly for {path_a}: {err_a}")

    # Extract ASM for kernel B
    if path_b.suffix == ".rs":
        asm_b, err_b = extract_kernel_asm(path_b, use_wsl=use_wsl)
        name_b = path_b.stem
    else:
        asm_b = path_b.read_text(encoding="utf-8", errors="replace")
        name_b = path_b.stem
        err_b = ""

    if not asm_b:
        raise ValueError(f"Could not load assembly for {path_b}: {err_b}")

    # Simulate cycles across CPU targets
    cycles_a: Dict[str, float] = {}
    cycles_b: Dict[str, float] = {}
    cycle_deltas: Dict[str, float] = {}
    speedup_ratios: Dict[str, float] = {}

    for cpu in cpu_specs:
        for bname in ("llvm-mca", "osaca", "uica"):
            b = backends.get(bname)
            if b and b.supports(cpu):
                cyc_a = b.analyze(asm_a, cpu)
                cyc_b = b.analyze(asm_b, cpu)
                if cyc_a is not None:
                    cycles_a[cpu.name] = cyc_a
                if cyc_b is not None:
                    cycles_b[cpu.name] = cyc_b
                if cyc_a is not None and cyc_b is not None:
                    cycle_deltas[cpu.name] = cyc_b - cyc_a
                    speedup_ratios[cpu.name] = (cyc_a / cyc_b - 1.0) if cyc_b > 0 else 0.0
                break

    rep_a = extract_kernel_report(asm_a, kernel_name=name_a, cpu_cycles=cycles_a)
    rep_b = extract_kernel_report(asm_b, kernel_name=name_b, cpu_cycles=cycles_b)

    return KernelComparisonDiff(
        kernel_a=rep_a,
        kernel_b=rep_b,
        cycle_deltas=cycle_deltas,
        load_delta=rep_b.memory.loads - rep_a.memory.loads,
        store_delta=rep_b.memory.stores - rep_a.memory.stores,
        rmw_delta=rep_b.memory.read_modify_writes - rep_a.memory.read_modify_writes,
        gpr_delta=rep_b.registers.gprs_used - rep_a.registers.gprs_used,
        speedup_ratios=speedup_ratios,
    )


def run_diff(
    kernel_a_path: str,
    kernel_b_path: str,
    cpus: Optional[List[CpuSpec]] = None,
    use_wsl: bool = False,
    as_json: bool = False,
) -> int:
    """Execute kernel diff comparison and print results."""
    p_a = Path(kernel_a_path)
    p_b = Path(kernel_b_path)

    if not p_a.exists():
        print(f"Error: Path not found: {p_a}", file=sys.stderr)
        return 1
    if not p_b.exists():
        print(f"Error: Path not found: {p_b}", file=sys.stderr)
        return 1

    diff = compare_kernels(p_a, p_b, cpus=cpus, use_wsl=use_wsl)

    if as_json:
        print(export_diff_to_json(diff))
    else:
        print(render_diff_markdown(diff))

    return 0
