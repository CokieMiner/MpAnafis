"""Core randomized kernel rewrite search engine."""

from __future__ import annotations

import math
import statistics
from typing import Callable, Dict, List, Optional, Set, Tuple

from ..backends import make_backends
from ..diff_test import diff_test_variants
from ..models import ANALYTICAL_BACKENDS, CpuSpec
from ..regions import is_branch_instruction
from ..simulation import simulate_backends
from ..types import ArchitectureFamily
from .adapters import adapter_for
from .ast import Instr, Spec, parse_line
from .dag import (
    build_dag,
    topological_permutations,
)
from .hardware import evaluate_hardware_candidates
from .results import CandidateResult


def _rank_candidates(
    results: List[CandidateResult],
    expected_cells: Optional[set[tuple[str, str]]] = None,
) -> None:
    """Rank candidates by normalized regret with explicit coverage priority."""
    observed_cells = {
        (cpu, backend)
        for result in results
        for cpu, backend_map in result.cycles.items()
        for backend, cost in backend_map.items()
        if cost is not None and math.isfinite(cost) and cost > 0
    }
    cells = expected_cells if expected_cells is not None else observed_cells
    best_by_cell = {
        cell: min(
            cost
            for result in results
            if (cost := result.cycles.get(cell[0], {}).get(cell[1])) is not None
            and math.isfinite(cost)
            and cost > 0
        )
        for cell in observed_cells
    }

    for result in results:
        regrets: List[float] = []
        for cpu, backend in observed_cells:
            cost = result.cycles.get(cpu, {}).get(backend)
            if cost is not None and math.isfinite(cost) and cost > 0:
                regrets.append(cost / best_by_cell[(cpu, backend)])
        result.coverage = len(regrets)
        result.missing = len(cells) - result.coverage
        result.score = statistics.fmean(regrets) if regrets else None

    results.sort(
        key=lambda result: (
            result.missing,
            result.score if result.score is not None else math.inf,
            result.idx,
        )
    )


def search_kernel(
    asm_body: str,
    cpus: List[CpuSpec],
    backend_names: Optional[List[str]] = None,
    candidates_count: int = 100,
    seed: int = 42,
    use_wsl: bool = False,
    run_diff_test: bool = True,
    architecture: ArchitectureFamily = ArchitectureFamily.X86_64,
    allow_unmodeled: bool = False,
    disjoint_bases: Optional[Set[str] | Set[Tuple[str, str]]] = None,
    relax_renaming: bool = False,
) -> Tuple[List[CandidateResult], str]:
    """Search for low-regret topological instruction schedules of an assembly body."""
    if candidates_count < 1:
        return [], "candidate count must be positive"
    if relax_renaming:
        return [], "schedule search cannot rename registers without whole-kernel live-out and scratch contracts"
    selected_backends = backend_names or list(ANALYTICAL_BACKENDS)
    if any(name not in ANALYTICAL_BACKENDS for name in selected_backends) and selected_backends != ["nanobench"]:
        return [], "schedule search accepts analytical backends or nanoBench alone"
    adapter = adapter_for(architecture)
    raw_lines = [
        line.strip()
        for line in asm_body.splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "//"))
    ]
    segments, parse_error = _schedulable_segments(
        raw_lines, adapter.parse,
        delay_slots=architecture in (ArchitectureFamily.MIPS32, ArchitectureFamily.MIPS64),
    )
    if parse_error:
        return [], parse_error
    all_bodies = _candidate_bodies(
        raw_lines,
        segments,
        candidates_count,
        seed,
        adapter.spec,
        architecture=architecture,
        disjoint_bases=disjoint_bases,
    )
    if len(all_bodies) < 2:
        all_bodies.append(all_bodies[0])

    diff_error: str = ""
    if run_diff_test:
        try:
            diff_results = diff_test_variants(
                all_bodies,
                cases=50,
                use_wsl=use_wsl,
                architecture=architecture,
            )
        except Exception as err:
            diff_error = f"Differential testing error: {err}"
            diff_results = [False] * len(all_bodies)
    else:
        diff_results = [True] * len(all_bodies)

    backends = make_backends(selected_backends, wsl=use_wsl)
    results: List[CandidateResult] = []
    expected_cells = {
        (cpu.name, backend_name)
        for cpu in cpus
        for backend_name in selected_backends
        if (backend := backends.get(backend_name)) is not None and backend.supports(cpu)
    }
    backend_failures: set[str] = set()

    if selected_backends == ["nanobench"]:
        backend = backends.get("nanobench")
        if backend is None:
            return [], "nanoBench backend was not constructed"
        results, hardware_failures = evaluate_hardware_candidates(
            all_bodies,
            diff_results,
            cpus,
            backend,
        )
        if results and results[0].score is None:
            diagnostic = "; ".join(hardware_failures)
            return [], diagnostic or "No hardware measurement produced a cycle estimate"
        diagnostics = [
            message
            for message in (diff_error, *sorted(hardware_failures))
            if message
        ]
        return results, "\n".join(diagnostics)

    for idx, (body, is_ok) in enumerate(zip(all_bodies, diff_results)):
        if not is_ok:
            continue

        simulation = simulate_backends(body, cpus, selected_backends, backends)
        cpu_map: Dict[str, Dict[str, Optional[float]]] = {
            cpu.name: {
                backend_name: simulation.scheduling_costs.get(cpu.name, {}).get(backend_name)
                for backend_name in selected_backends
                if (cpu.name, backend_name) in expected_cells
            }
            for cpu in cpus
        }
        backend_failures.update(simulation.failure_messages())

        results.append(
            CandidateResult(
                idx=idx,
                body=body,
                is_valid=is_ok,
                cycles=cpu_map,
                provenance={
                    "static_cost_metrics": simulation.scheduling_metrics,
                    "validation": "native_differential" if run_diff_test else "static_only",
                },
            ),
        )

    _rank_candidates(results, expected_cells)
    if results and results[0].score is None:
        message = "No selected backend produced a cycle estimate for any candidate"
        if allow_unmodeled:
            diagnostics = [message, diff_error, *sorted(backend_failures)]
            return results, "\n".join(item for item in diagnostics if item)
        return [], message
    diagnostics = [message for message in (diff_error, *sorted(backend_failures)) if message]
    return results, "\n".join(diagnostics)


def _schedulable_segments(
    raw_lines: List[str],
    parser: Callable[[str], Optional[Instr]] = parse_line,
    delay_slots: bool = False,
) -> Tuple[List[Tuple[List[int], List[Instr]]], str]:
    """Split a CFG region into movable basic-block interiors."""
    segments: List[Tuple[List[int], List[Instr]]] = []
    positions: List[int] = []
    instructions: List[Instr] = []
    fixed_delay_slot = False

    def finish_segment() -> None:
        if instructions:
            segments.append((positions.copy(), instructions.copy()))
            positions.clear()
            instructions.clear()

    for position, line in enumerate(raw_lines):
        if line.endswith(":"):
            finish_segment()
            continue
        if line.startswith((".abiversion", ".align", ".machine", ".option", ".p2align", ".set")):
            finish_segment()
            continue
        instruction = parser(line)
        if instruction is None:
            return [], (
                "Schedule search requires parsed instructions and labels; "
                f"unsupported source line: {line}"
            )
        mnemonic = instruction.mnemonic
        if mnemonic.startswith(("call", "ret")):
            return [], f"Schedule search cannot execute region terminator: {line}"
        if fixed_delay_slot:
            if is_branch_instruction(line):
                return [], "a branch in a delay slot cannot be scheduled"
            fixed_delay_slot = False
            continue
        if is_branch_instruction(line):
            finish_segment()
            fixed_delay_slot = delay_slots
            continue
        positions.append(position)
        instructions.append(instruction)
    finish_segment()
    if fixed_delay_slot:
        return [], "branch delay slot is missing from the assembly region"
    if not segments:
        return [], "No schedulable instructions parsed from body"
    return segments, ""


def _candidate_bodies(
    raw_lines: List[str],
    segments: List[Tuple[List[int], List[Instr]]],
    count: int,
    seed: int,
    spec_provider: Callable[[Instr], Spec],
    architecture: ArchitectureFamily = ArchitectureFamily.X86_64,
    disjoint_bases: Optional[Set[str] | Set[Tuple[str, str]]] = None,
) -> List[str]:
    segment_orders: List[Tuple[List[int], List[List[str]]]] = []
    for segment_index, (positions, instructions) in enumerate(segments):
        nodes = build_dag(
            instructions,
            spec_provider=spec_provider,
            disjoint_bases=disjoint_bases,
        )
        permutations = topological_permutations(
            nodes,
            count=count,
            seed=seed + segment_index,
        )
        rendered_permutations: List[List[str]] = [
            [instructions[idx].line for idx in perm]
            for perm in permutations
        ]

        segment_orders.append((positions, rendered_permutations))

    bodies = ["\n".join(raw_lines)]
    seen = {bodies[0]}
    for candidate_index in range(count):
        candidate = raw_lines.copy()
        for positions, permutations in segment_orders:
            if not permutations:
                continue
            order = permutations[candidate_index % len(permutations)]
            for position, line in zip(positions, order):
                candidate[position] = line
        body = "\n".join(candidate)
        if body not in seen:
            seen.add(body)
            bodies.append(body)
            if len(bodies) >= count + 1:
                break
    return bodies
