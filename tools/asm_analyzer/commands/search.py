"""Randomized topological instruction scheduler and optimizer command."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

from ..kernel_source import extract_kernel_asm
from ..models import CpuSpec
from ..regions import select_analysis_region
from ..search.engine import search_kernel
from ..targets import architecture_for_path, compatible_cpus, default_cpus_for_architecture, host_architecture


def run_search(
    kernel_path: str,
    cpus: Optional[List[CpuSpec]] = None,
    backend_names: Optional[List[str]] = None,
    candidates: int = 50,
    seed: int = 42,
    use_wsl: bool = False,
    no_alias: bool = False,
    disjoint_pointers: Optional[str] = None,
    as_json: bool = False,
) -> int:
    """Execute randomized DAG scheduler search on a kernel file or assembly snippet."""
    p = Path(kernel_path)
    if not p.exists():
        print(f"Error: File not found: {p}", file=sys.stderr)
        return 1

    if p.suffix == ".rs":
        asm_body, err = extract_kernel_asm(p, use_wsl=use_wsl)
        if not asm_body:
            print(f"Error extracting assembly: {err}", file=sys.stderr)
            return 1
    else:
        asm_body = p.read_text(encoding="utf-8", errors="replace")

    architecture = architecture_for_path(p)
    region = select_analysis_region(asm_body, architecture)
    asm_body = region.schedulable_asm()

    cpu_specs = (
        compatible_cpus(cpus, architecture)
        if cpus is not None
        else default_cpus_for_architecture(architecture)
    )
    if not cpu_specs:
        print(
            f"Search failed: no CPU models compatible with {architecture.value}",
            file=sys.stderr,
        )
        return 1

    disjoint_bases = None
    if disjoint_pointers:
        disjoint_bases = {ptr.strip().lstrip("%") for ptr in disjoint_pointers.split(",") if ptr.strip()}
    elif no_alias:
        disjoint_bases = {"rdi", "rsi", "rdx", "rcx", "r8", "r9", "r10", "r11", "x0", "x1", "x2", "x3", "x4", "x5", "a0", "a1", "a2", "a3"}

    results, err = search_kernel(
        asm_body,
        cpu_specs,
        backend_names=backend_names,
        candidates_count=candidates,
        seed=seed,
        use_wsl=use_wsl,
        architecture=architecture,
        run_diff_test=architecture == host_architecture(),
        disjoint_bases=disjoint_bases,
    )

    if not results:
        print(f"Search failed: {err}", file=sys.stderr)
        return 1

    if as_json:
        data = [
            {
                "idx": r.idx,
                "is_valid": r.is_valid,
                "score": r.score,
                "coverage": r.coverage,
                "missing": r.missing,
                "cycles": r.cycles,
                "samples": r.samples,
                "comparisons": r.comparisons,
                "provenance": r.provenance,
                "body": r.body,
            }
            for r in results
        ]
        print(json.dumps(data, indent=2))
    else:
        print(f"# Kernel Schedule Search for `{p.name}`\n")
        print(f"Generated and evaluated {len(results)} valid schedule candidates.")
        print(
            f"\nTop Ranked Modeled Candidate (Original = 0, Candidate = {results[0].idx}, "
            f"Normalized regret = {results[0].score:.4f}):\n"
        )
        for line in results[0].body.splitlines():
            print(f"    {line}")
        print("")

    if err:
        print(f"Search incomplete: {err}", file=sys.stderr)
        return 1
    return 0
