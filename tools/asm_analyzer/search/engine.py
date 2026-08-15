"""Core randomized kernel rewrite search engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..backends import make_backends
from ..diff_test import diff_test_variants
from ..models import CpuSpec
from .ast import parse_line
from .dag import build_dag, topological_permutations


@dataclass
class CandidateResult:
    idx: int
    body: str
    is_valid: bool
    cycles: Dict[str, Dict[str, Optional[float]]]  # cpu -> (backend -> cost)


def search_kernel(
    asm_body: str,
    cpus: List[CpuSpec],
    candidates_count: int = 100,
    seed: int = 42,
    use_wsl: bool = False,
    run_diff_test: bool = True,
) -> Tuple[List[CandidateResult], str]:
    """Search for optimal topological instruction schedules of an assembly body."""
    raw_lines = [ln.strip() for ln in asm_body.splitlines() if ln.strip()]
    insts = [parse_line(ln) for ln in raw_lines]
    valid_insts = [inst for inst in insts if inst is not None]

    if not valid_insts:
        return [], "No valid instructions parsed from body"

    nodes = build_dag(valid_insts)
    perms = topological_permutations(nodes, count=candidates_count, seed=seed)

    if not perms:
        return [], "Could not generate topological permutations"

    bodies = ["\n".join(valid_insts[i].line for i in perm) for perm in perms]
    orig_body = "\n".join(valid_insts[i].line for i in range(len(valid_insts)))
    all_bodies = [orig_body] + bodies

    diff_error: str = ""
    if run_diff_test:
        try:
            diff_results = diff_test_variants(all_bodies, cases=50, use_wsl=use_wsl)
        except Exception as err:
            diff_error = f"Differential testing error: {err}"
            diff_results = [False] * len(all_bodies)
    else:
        diff_results = [True] * len(all_bodies)

    backends = make_backends(["llvm-mca", "osaca", "uica"], wsl=use_wsl)
    results: List[CandidateResult] = []

    for idx, (body, is_ok) in enumerate(zip(all_bodies, diff_results)):
        if not is_ok:
            continue

        cpu_map: Dict[str, Dict[str, Optional[float]]] = {}
        for cpu in cpus:
            b_map: Dict[str, Optional[float]] = {}
            for bname in ("llvm-mca", "osaca", "uica"):
                b = backends.get(bname)
                if b and b.supports(cpu):
                    cyc = b.analyze(body, cpu)
                    b_map[bname] = cyc
            cpu_map[cpu.name] = b_map

        results.append(CandidateResult(idx=idx, body=body, is_valid=is_ok, cycles=cpu_map))

    return results, diff_error
