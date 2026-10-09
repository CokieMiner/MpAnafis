"""Automated repository-wide microarchitectural analysis sweep."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from ..backends import make_backends
from ..features import extract_kernel_report
from ..kernel_source import discover_kernels, extract_kernel_variants
from ..models import ANALYTICAL_BACKENDS, CPUS, CpuSpec
from ..report.json_export import export_reports_to_json
from ..report.markdown import render_sweep_markdown
from ..regions import select_analysis_region
from ..simulation import simulate_backends
from ..targets import (
    architecture_for_path,
    compatible_cpus,
    default_cpus_for_architecture,
)
from ..types import ArchitectureFamily, KernelAnalysisReport


def run_sweep(
    target_path: Optional[str] = None,
    cpus: Optional[List[CpuSpec]] = None,
    backend_names: Optional[List[str]] = None,
    use_wsl: bool = False,
    markdown: bool = False,
    as_json: bool = False,
) -> int:
    """Execute kernel sweep across targets and print output."""
    if target_path:
        p = Path(target_path)
        kernel_files = [p] if p.is_file() else discover_kernels(p)
        explicit_file = p.is_file()
    else:
        kernel_files = [
            path for path in discover_kernels()
            if architecture_for_path(path) == ArchitectureFamily.X86_64
        ]
        explicit_file = False

    if cpus is not None and not explicit_file:
        kernel_files = [
            path
            for path in kernel_files
            if compatible_cpus(cpus, architecture_for_path(path))
        ]

    architectures = {architecture_for_path(path) for path in kernel_files}
    if cpus is not None:
        requested_cpus = cpus
    elif len(architectures) == 1:
        requested_cpus = default_cpus_for_architecture(next(iter(architectures)))
    else:
        requested_cpus = list(CPUS.values())
    cpu_names = [cpu.name for cpu in requested_cpus]

    selected_backends = backend_names or list(ANALYTICAL_BACKENDS)
    backends = make_backends(selected_backends, wsl=use_wsl)
    reports: List[KernelAnalysisReport] = []
    failures: List[tuple[Path, str]] = []
    backend_failures: List[tuple[str, str]] = []

    for kpath in kernel_files:
        variants, err = extract_kernel_variants(kpath, use_wsl=use_wsl)
        if not variants:
            failures.append((kpath, err))
            print(f"Skipping {kpath}: {err}", file=sys.stderr)
            continue
        arch_family = architecture_for_path(kpath)
        cpu_specs = compatible_cpus(requested_cpus, arch_family)
        if not cpu_specs:
            failures.append((kpath, f"no CPU models compatible with {arch_family.value}"))
            continue

        for variant in variants:
            region = select_analysis_region(variant.asm, arch_family)
            analysis_asm = region.asm
            simulation = simulate_backends(
                analysis_asm,
                cpu_specs,
                selected_backends,
                backends,
            )
            cpu_cycles = simulation.flattened_with_medians()
            if not cpu_cycles:
                backend_failures.append((variant.name, "no selected backend produced a cycle estimate"))
            backend_failures.extend(
                (variant.name, message) for message in simulation.failure_messages()
            )

            report = extract_kernel_report(
                asm=analysis_asm,
                kernel_name=variant.name,
                target_arch=arch_family,
                cpu_cycles=cpu_cycles,
                cpu_model_results=simulation.model_results(),
                analysis_scope=region.kind,
                analysis_label=region.label,
                backend_failures=simulation.failure_messages(),
                limbs_per_iteration=variant.limbs_per_iteration,
                cfg_block_count=region.block_count,
                cfg_edge_count=region.cfg_edge_count,
            )
            reports.append(report)

    if as_json:
        print(export_reports_to_json(reports))
    elif markdown:
        print(render_sweep_markdown(reports, cpu_names))
    else:
        print(render_sweep_markdown(reports, cpu_names))

    if failures:
        print(
            f"Error: sweep incomplete: {len(failures)} of {len(kernel_files)} kernels failed extraction",
            file=sys.stderr,
        )
        return 1
    if backend_failures:
        for variant_name, failure in backend_failures:
            print(f"Backend failure for {variant_name}: {failure}", file=sys.stderr)
        print(
            f"Error: sweep incomplete: {len(backend_failures)} backend cells failed",
            file=sys.stderr,
        )
        return 1
    if not reports:
        print("Error: sweep produced no kernel reports", file=sys.stderr)
        return 1
    return 0
