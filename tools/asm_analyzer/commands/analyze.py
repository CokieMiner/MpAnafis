"""Single assembly file microarchitectural analysis command."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

from ..backends import make_backends
from ..features import extract_kernel_report
from ..kernel_source import extract_kernel_asm
from ..models import ANALYTICAL_BACKENDS, CpuSpec
from ..regions import select_analysis_region
from ..simulation import simulate_backends
from ..targets import architecture_for_path, compatible_cpus, default_cpus_for_architecture
from ..report.markdown import render_sweep_markdown


def run_analyze(
    asm_path: str,
    cpus: Optional[List[CpuSpec]] = None,
    backend_names: Optional[List[str]] = None,
    use_wsl: bool = False,
    as_json: bool = False,
) -> int:
    """Analyze a single AT&T assembly file (.s) or Rust kernel file (.rs)."""
    p = Path(asm_path)
    if not p.exists():
        print(f"Error: File not found: {p}", file=sys.stderr)
        return 1

    if p.suffix == ".rs":
        asm_code, err = extract_kernel_asm(p, use_wsl=use_wsl)
        if not asm_code:
            print(f"Error: Could not extract assembly from {p}: {err}", file=sys.stderr)
            return 1
        asm_text = asm_code
    else:
        asm_text = p.read_text(encoding="utf-8", errors="replace")

    target_arch = architecture_for_path(p)
    region = select_analysis_region(asm_text, target_arch)
    analysis_asm = region.asm

    cpu_specs = compatible_cpus(cpus, target_arch) if cpus is not None else default_cpus_for_architecture(target_arch)
    if not cpu_specs:
        print(f"Error: no CPU models compatible with {target_arch.value}", file=sys.stderr)
        return 1
    selected_backends = backend_names or list(ANALYTICAL_BACKENDS)
    backends = make_backends(selected_backends, wsl=use_wsl)

    simulation = simulate_backends(analysis_asm, cpu_specs, selected_backends, backends)
    cpu_cycles = simulation.flattened_with_medians()

    report = extract_kernel_report(
        asm=analysis_asm,
        kernel_name=p.stem,
        target_arch=target_arch,
        cpu_cycles=cpu_cycles,
        cpu_model_results=simulation.model_results(),
        analysis_scope=region.kind,
        analysis_label=region.label,
        backend_failures=simulation.failure_messages(),
        cfg_block_count=region.block_count,
        cfg_edge_count=region.cfg_edge_count,
    )

    if as_json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(render_sweep_markdown([report], [c.name for c in cpu_specs]))

    for failure in simulation.failure_messages():
        print(f"Backend failure: {failure}", file=sys.stderr)
    if not cpu_cycles:
        print("Error: no selected backend produced a cycle estimate", file=sys.stderr)
        return 1
    if simulation.failures:
        return 1
    return 0
