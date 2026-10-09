"""Side-by-side comparison engine for assembly kernel variants."""

from __future__ import annotations

import sys
import statistics
from pathlib import Path
from typing import Dict, List, Optional

from ..backends import make_backends
from ..features import extract_kernel_report
from ..kernel_source import extract_kernel_asm
from ..models import ANALYTICAL_BACKENDS, CpuSpec
from ..regions import select_analysis_region
from ..simulation import simulate_backends
from ..targets import architecture_for_path, compatible_cpus, default_cpus_for_architecture
from ..report.markdown import render_diff_markdown
from ..report.json_export import export_diff_to_json
from ..types import KernelComparisonDiff


def run_diff(
    kernel_a_path: str,
    kernel_b_path: str,
    cpus: Optional[List[CpuSpec]] = None,
    backend_names: Optional[List[str]] = None,
    use_wsl: bool = False,
    as_json: bool = False,
) -> int:
    """Execute kernel diff comparison and print results."""
    path_a = Path(kernel_a_path)
    path_b = Path(kernel_b_path)

    if not path_a.exists():
        print(f"Error: Path not found: {path_a}", file=sys.stderr)
        return 1
    if not path_b.exists():
        print(f"Error: Path not found: {path_b}", file=sys.stderr)
        return 1

    try:
        diff = compare_kernels(
            path_a,
            path_b,
            cpus=cpus,
            backend_names=backend_names,
            use_wsl=use_wsl,
        )
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if not diff.cycle_deltas:
        print("Error: no selected backend produced paired cycle estimates", file=sys.stderr)
        return 1

    if as_json:
        print(export_diff_to_json(diff))
    else:
        print(render_diff_markdown(diff))

    for failure in diff.backend_failures:
        print(f"Backend failure: {failure}", file=sys.stderr)
    return 1 if diff.backend_failures else 0


def compare_kernels(
    path_a: Path,
    path_b: Path,
    cpus: Optional[List[CpuSpec]] = None,
    backend_names: Optional[List[str]] = None,
    use_wsl: bool = False,
) -> KernelComparisonDiff:
    """Analyze and generate side-by-side comparison diff between two kernels."""
    selected_backends = backend_names or list(ANALYTICAL_BACKENDS)
    backends = make_backends(selected_backends, wsl=use_wsl)

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
    region_a = select_analysis_region(asm_a, architecture_for_path(path_a))
    analysis_asm_a = region_a.asm

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
    region_b = select_analysis_region(asm_b, architecture_for_path(path_b))
    analysis_asm_b = region_b.asm

    architecture_a = architecture_for_path(path_a)
    architecture_b = architecture_for_path(path_b)
    if architecture_a != architecture_b:
        raise ValueError(
            f"cannot compare {architecture_a.value} assembly with {architecture_b.value} assembly"
        )
    cpu_specs = (
        compatible_cpus(cpus, architecture_a)
        if cpus is not None
        else default_cpus_for_architecture(architecture_a)
    )
    if not cpu_specs:
        raise ValueError(f"no CPU models compatible with {architecture_a.value}")

    simulation_a = simulate_backends(analysis_asm_a, cpu_specs, selected_backends, backends)
    simulation_b = simulate_backends(analysis_asm_b, cpu_specs, selected_backends, backends)

    # Retain only paired backend cells so variant medians compare equal model
    # coverage rather than unrelated subsets.
    cycles_a: Dict[str, float] = {}
    cycles_b: Dict[str, float] = {}

    for cpu in cpu_specs:
        paired_a: List[float] = []
        paired_b: List[float] = []
        for bname in selected_backends:
            cyc_a = simulation_a.cycles.get(cpu.name, {}).get(bname)
            cyc_b = simulation_b.cycles.get(cpu.name, {}).get(bname)
            if cyc_a is not None and cyc_b is not None:
                paired_a.append(cyc_a)
                paired_b.append(cyc_b)
                key = f"{cpu.name}/{bname}"
                cycles_a[key] = cyc_a
                cycles_b[key] = cyc_b
        if paired_a:
            cycles_a[cpu.name] = statistics.median(paired_a)
            cycles_b[cpu.name] = statistics.median(paired_b)

    failures = tuple(
        f"A {message}" for message in simulation_a.failure_messages()
    ) + tuple(
        f"B {message}" for message in simulation_b.failure_messages()
    )

    rep_a = extract_kernel_report(
        analysis_asm_a,
        kernel_name=name_a,
        target_arch=architecture_a,
        cpu_cycles=cycles_a,
        analysis_scope=region_a.kind,
        analysis_label=region_a.label,
        backend_failures=simulation_a.failure_messages(),
        cfg_block_count=region_a.block_count,
        cfg_edge_count=region_a.cfg_edge_count,
    )
    rep_b = extract_kernel_report(
        analysis_asm_b,
        kernel_name=name_b,
        target_arch=architecture_b,
        cpu_cycles=cycles_b,
        analysis_scope=region_b.kind,
        analysis_label=region_b.label,
        backend_failures=simulation_b.failure_messages(),
        cfg_block_count=region_b.block_count,
        cfg_edge_count=region_b.cfg_edge_count,
    )

    normalized_a = rep_a.cpu_cycles_per_limb
    normalized_b = rep_b.cpu_cycles_per_limb
    paired_keys = normalized_a.keys() & normalized_b.keys()
    cycle_deltas = {
        key: normalized_b[key] - normalized_a[key]
        for key in paired_keys
    }
    speedup_ratios = {
        key: (normalized_a[key] / normalized_b[key] - 1.0)
        if normalized_b[key] > 0 else 0.0
        for key in paired_keys
    }

    return KernelComparisonDiff(
        kernel_a=rep_a,
        kernel_b=rep_b,
        cycle_deltas=cycle_deltas,
        load_delta=rep_b.memory.loads - rep_a.memory.loads,
        store_delta=rep_b.memory.stores - rep_a.memory.stores,
        rmw_delta=rep_b.memory.read_modify_writes - rep_a.memory.read_modify_writes,
        gpr_delta=rep_b.registers.gprs_used - rep_a.registers.gprs_used,
        speedup_ratios=speedup_ratios,
        backend_failures=failures,
    )
