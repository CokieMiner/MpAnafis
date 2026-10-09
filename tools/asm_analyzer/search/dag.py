"""Dependency DAG construction, critical-path calculation, and topological schedulers."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from ..types import ArchitectureFamily
from .ast import (
    FLAG_FULL,
    REGS64,
    Instr,
    Op,
    Spec,
    clone_instruction_with_renaming,
    get_instruction_spec,
)
from .memory_dependencies import may_alias, memory_operand


@dataclass
class DagNode:
    idx: int
    instr: Instr
    spec: Spec
    preds: Set[int]
    succs: Set[int]
    latency: int = 1


def _estimate_latency(sp: Spec, inst: Instr) -> int:
    """Heuristic static instruction latency in execution cycles."""
    mnem = inst.mnemonic.lower()
    if (
        mnem.startswith(("mul", "imul"))
        or mnem in ("umulh", "smulh", "umaal", "umlal", "mulld", "mulhdu")
    ):
        return 3
    if sp.mem == "load":
        return 4
    if sp.mem == "store":
        return 1
    if mnem in ("shld", "shrd"):
        return 3
    return 1


def build_dag(
    instructions: List[Instr],
    spec_provider: Callable[[Instr], Spec] = get_instruction_spec,
    disjoint_bases: Optional[Set[str] | Set[Tuple[str, str]]] = None,
) -> List[DagNode]:
    """Construct dataflow dependency DAG over a list of instructions."""
    n = len(instructions)
    specs = [spec_provider(inst) for inst in instructions]
    preds: List[Set[int]] = [set() for _ in range(n)]
    succs: List[Set[int]] = [set() for _ in range(n)]

    last_def: Dict[str, int] = {}
    readers_since_def: Dict[str, Set[int]] = {}
    last_flag_def: Dict[str, int] = {}
    readers_since_flag_def: Dict[str, Set[int]] = {}
    prior_memory: List[Tuple[int, str, Optional[Op]]] = []
    last_barrier: int = -1

    def flags_overlap(left: str, right: str) -> bool:
        return left == FLAG_FULL or right == FLAG_FULL or left == right

    for i in range(n):
        sp = specs[i]
        is_barrier = sp.unknown or instructions[i].mnemonic.startswith(("j", "call", "ret"))

        if last_barrier != -1:
            preds[i].add(last_barrier)
        if is_barrier:
            preds[i].update(range(i))

        # Register RAW dependencies
        for r in sp.uses:
            if r in last_def:
                preds[i].add(last_def[r])

        # Register WAW and WAR dependencies
        for r in sp.defs:
            if r in last_def:
                preds[i].add(last_def[r])
            preds[i].update(readers_since_def.get(r, set()))

        # Flag dependencies
        for f in sp.flags_read:
            for defined_flag, defining_instruction in last_flag_def.items():
                if flags_overlap(f, defined_flag):
                    preds[i].add(defining_instruction)

        for f in sp.flags_write:
            for defined_flag, defining_instruction in last_flag_def.items():
                if flags_overlap(f, defined_flag):
                    preds[i].add(defining_instruction)
            for read_flag, reading_instructions in readers_since_flag_def.items():
                if flags_overlap(f, read_flag):
                    preds[i].update(reading_instructions)

        # Memory dependencies. Proved disjoint intervals or non-aliasing bases are independent.
        operand = memory_operand(instructions[i])
        if sp.mem is not None:
            for prior_index, prior_kind, prior_operand in prior_memory:
                if (
                    sp.mem == "store" or prior_kind == "store"
                ) and may_alias(prior_operand, operand, disjoint_bases=disjoint_bases):
                    preds[i].add(prior_index)
            prior_memory.append((i, sp.mem, operand))

        # Update last defs
        for r in sp.defs:
            last_def[r] = i
            readers_since_def[r] = set()
        for r in sp.uses - sp.defs:
            readers_since_def.setdefault(r, set()).add(i)
        for f in sp.flags_write:
            for read_flag in list(readers_since_flag_def):
                if f == FLAG_FULL or f == read_flag:
                    readers_since_flag_def[read_flag] = set()
            last_flag_def[f] = i
        for f in sp.flags_read:
            if not any(written_flag == FLAG_FULL or written_flag == f for written_flag in sp.flags_write):
                readers_since_flag_def.setdefault(f, set()).add(i)
        if is_barrier:
            last_barrier = i

    for i in range(n):
        for p in preds[i]:
            succs[p].add(i)

    nodes = []
    for i in range(n):
        lat = _estimate_latency(specs[i], instructions[i])
        nodes.append(DagNode(i, instructions[i], specs[i], preds[i], succs[i], latency=lat))
    return nodes


def compute_critical_paths(nodes: List[DagNode]) -> List[int]:
    """Compute bottom-up critical path depth for each node in the DAG."""
    n = len(nodes)
    depths = [0] * n
    in_degree = [len(node.succs) for node in nodes]
    queue = deque(i for i in range(n) if in_degree[i] == 0)

    while queue:
        curr = queue.popleft()
        max_succ_depth = max([depths[s] for s in nodes[curr].succs], default=0)
        depths[curr] = nodes[curr].latency + max_succ_depth

        for p in nodes[curr].preds:
            in_degree[p] -= 1
            if in_degree[p] == 0:
                queue.append(p)

    return depths


def compute_transitive_closure(nodes: Sequence[DagNode]) -> List[List[bool]]:
    """Compute all-pairs reachability matrix for the DAG."""
    n = len(nodes)
    reach = [[False] * n for _ in range(n)]
    for i in range(n):
        reach[i][i] = True
        for s in nodes[i].succs:
            reach[i][s] = True
    for k in range(n):
        for i in range(n):
            if reach[i][k]:
                for j in range(n):
                    if reach[k][j]:
                        reach[i][j] = True
    return reach


def evaluate_schedule_cost(schedule: Sequence[int], nodes: Sequence[DagNode]) -> float:
    """Heuristic pipeline latency and stall cost for a candidate schedule."""
    completion: Dict[int, int] = {}
    total_cycles = 0
    stall_penalty = 0

    for step, node_idx in enumerate(schedule):
        node = nodes[node_idx]
        ready_cycle = (
            max((completion[p] for p in node.preds), default=0)
            if node.preds
            else 0
        )
        issue_cycle = max(step, ready_cycle)
        if ready_cycle > step:
            stall_penalty += (ready_cycle - step) * 2
        finish_cycle = issue_cycle + node.latency
        completion[node_idx] = finish_cycle
        total_cycles = max(total_cycles, finish_cycle)

    return float(total_cycles + stall_penalty)


def heuristic_topological_schedule(nodes: List[DagNode]) -> List[int]:
    """Generate a deterministic critical-path list schedule."""
    n = len(nodes)
    if n == 0:
        return []
    depths = compute_critical_paths(nodes)
    in_degree = [len(node.preds) for node in nodes]
    ready = [i for i in range(n) if in_degree[i] == 0]
    schedule: List[int] = []

    while ready:
        ready.sort(key=lambda idx: (depths[idx], nodes[idx].latency, -idx), reverse=True)
        chosen = ready.pop(0)
        schedule.append(chosen)

        for s in nodes[chosen].succs:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                ready.append(s)

    return schedule


def exact_optimal_schedule(
    nodes: List[DagNode], max_states: int = 20_000,
) -> Optional[List[int]]:
    """Return a proved optimum, or None when the bounded search is exhausted."""
    n = len(nodes)
    if n == 0 or n > 20:
        return None

    best_schedule: Optional[List[int]] = None
    min_cost = math.inf
    in_degree = [len(node.preds) for node in nodes]
    states = 0
    exhausted = False

    def search(
        current_schedule: List[int],
        current_in_degree: List[int],
    ) -> None:
        nonlocal best_schedule, min_cost, states, exhausted
        if states >= max_states:
            exhausted = True
            return
        states += 1
        if len(current_schedule) == n:
            cost = evaluate_schedule_cost(current_schedule, nodes)
            if cost < min_cost:
                min_cost = cost
                best_schedule = current_schedule.copy()
            return

        ready = [i for i in range(n) if current_in_degree[i] == 0 and i not in current_schedule]
        for chosen in ready:
            next_in_degree = current_in_degree.copy()
            for s in nodes[chosen].succs:
                next_in_degree[s] -= 1
            current_schedule.append(chosen)
            search(current_schedule, next_in_degree)
            current_schedule.pop()
            if exhausted:
                return

    search([], in_degree.copy())
    return None if exhausted else best_schedule


def simulated_annealing_permutations(
    nodes: List[DagNode],
    count: int = 50,
    seed: int = 42,
    temp_start: float = 2.0,
    cooling_rate: float = 0.92,
) -> List[List[int]]:
    """Explore schedule neighborhood by swapping adjacent independent DAG nodes."""
    n = len(nodes)
    if count <= 0:
        return []
    if n < 2:
        return [list(range(n))]

    rng = random.Random(seed)
    orderings: List[Tuple[int, ...]] = []
    seen: Set[Tuple[int, ...]] = set()

    base_schedule = heuristic_topological_schedule(nodes)
    if len(base_schedule) == n:
        t_base = tuple(base_schedule)
        orderings.append(t_base)
        seen.add(t_base)

    current = base_schedule.copy()
    current_cost = evaluate_schedule_cost(current, nodes)
    temp = temp_start

    rounds = count * 20
    for _ in range(rounds):
        if len(orderings) >= count:
            break
        i = rng.randint(0, n - 2)
        u, v = current[i], current[i + 1]
        # In a topological order, a longer u -> v path needs an intervening
        # node. Adjacent nodes can therefore swap exactly when no edge joins them.
        if v not in nodes[u].succs and u not in nodes[v].succs:
            candidate = current.copy()
            candidate[i], candidate[i + 1] = candidate[i + 1], candidate[i]
            candidate_cost = evaluate_schedule_cost(candidate, nodes)
            delta = candidate_cost - current_cost

            if delta <= 0 or (temp > 1e-4 and rng.random() < math.exp(-delta / temp)):
                current = candidate
                current_cost = candidate_cost
                t_cand = tuple(current)
                if t_cand not in seen:
                    seen.add(t_cand)
                    orderings.append(t_cand)

        temp = max(1e-4, temp * cooling_rate)

    return [list(o) for o in orderings]


def topological_permutations(
    nodes: List[DagNode],
    count: int = 100,
    seed: int = 42,
) -> List[List[int]]:
    """Generate high-quality candidate schedules combining heuristics, annealing, and list search."""
    n = len(nodes)
    if n == 0 or count <= 0:
        return []

    depths = compute_critical_paths(nodes)
    rng = random.Random(seed)
    orderings: List[Tuple[int, ...]] = []
    seen: Set[Tuple[int, ...]] = set()

    # 1. Exact optimal schedule for small DAGs (N <= 12)
    if n <= 12:
        exact = exact_optimal_schedule(nodes)
        if exact is not None:
            t_exact = tuple(exact)
            orderings.append(t_exact)
            seen.add(t_exact)

    # 2. Pure critical-path heuristic schedule
    best_heuristic = tuple(heuristic_topological_schedule(nodes))
    if len(best_heuristic) == n and best_heuristic not in seen:
        orderings.append(best_heuristic)
        seen.add(best_heuristic)

    # 3. Simulated annealing local search
    annealed = simulated_annealing_permutations(nodes, count=count // 2, seed=seed)
    for sched in annealed:
        t_sched = tuple(sched)
        if t_sched not in seen:
            seen.add(t_sched)
            orderings.append(t_sched)

    # 4. Weighted randomized list scheduling exploration
    attempts = count * 10
    for _ in range(attempts):
        if len(orderings) >= count:
            break
        in_degree = [len(node.preds) for node in nodes]
        ready = [i for i in range(n) if in_degree[i] == 0]
        order: List[int] = []

        while ready:
            if len(ready) == 1 or rng.random() < 0.2:
                chosen = rng.choice(ready)
            else:
                weights = [max(1, depths[r]) for r in ready]
                chosen = rng.choices(ready, weights=weights, k=1)[0]

            ready.remove(chosen)
            order.append(chosen)

            for s in nodes[chosen].succs:
                in_degree[s] -= 1
                if in_degree[s] == 0:
                    ready.append(s)

        if len(order) == n:
            t_order = tuple(order)
            if t_order not in seen:
                seen.add(t_order)
                orderings.append(t_order)

    return [list(o) for o in orderings[:count]]


def compute_live_ranges(
    instructions: List[Instr],
    spec_provider: Callable[[Instr], Spec] = get_instruction_spec,
) -> Dict[str, List[Tuple[int, int]]]:
    """Compute active def-use live ranges for registers in an instruction block."""
    specs = [spec_provider(inst) for inst in instructions]
    live_ranges: Dict[str, List[Tuple[int, int]]] = {}
    current_defs: Dict[str, int] = {}
    last_uses: Dict[str, int] = {}

    for idx, sp in enumerate(specs):
        for r in sp.uses:
            last_uses[r] = idx
        for r in sp.defs:
            if r in current_defs:
                start = current_defs[r]
                end = last_uses.get(r, start)
                live_ranges.setdefault(r, []).append((start, end))
            current_defs[r] = idx
            last_uses[r] = idx

    for r, start in current_defs.items():
        end = last_uses.get(r, start)
        live_ranges.setdefault(r, []).append((start, end))

    return live_ranges


def relax_anti_dependencies_with_renaming(
    instructions: List[Instr],
    architecture: ArchitectureFamily,
    spec_provider: Callable[[Instr], Spec] = get_instruction_spec,
    available_spares: Optional[Sequence[str]] = None,
    live_out: Optional[Set[str]] = None,
) -> List[Instr]:
    """Rename full x86-64 definitions with explicit scratch and live-out contracts.

    Spares must be caller-authorized clobbers. Without complete live-out data,
    every register may carry state into another block or the enclosing function.
    """
    if not available_spares or live_out is None or architecture != ArchitectureFamily.X86_64:
        return instructions
    specs = [spec_provider(inst) for inst in instructions]
    if any(sp.unknown for sp in specs):
        return instructions
    live_ranges = compute_live_ranges(instructions, spec_provider)

    # Determine spare registers
    used_regs = set().union(
        *(spec_provider(inst).uses | spec_provider(inst).defs for inst in instructions),
    )
    spares = [
        r for r in available_spares
        if r in REGS64 and r not in {"rsp", "rbp"} and r not in used_regs and r not in live_out
    ]
    if not spares:
        return instructions

    renamed_instructions = instructions.copy()
    spare_idx = 0

    for reg, ranges in live_ranges.items():
        if len(ranges) >= 2 and spare_idx < len(spares) and reg not in live_out and reg not in {"rsp", "rbp"}:
            # The second live range is disjoint from the first: rename it to a spare register
            spare_reg = spares[spare_idx]
            second_range_start, second_range_end = ranges[1]
            if reg in specs[second_range_start].uses:
                continue  # Read-modify-write needs the preceding value.
            if len(ranges) > 2 and reg in specs[ranges[2][0]].uses:
                continue
            if any(
                reg in (specs[index].uses | specs[index].defs)
                and (not any(reg in op.regs for op in instructions[index].ops)
                     or (reg in {"rax", "rdx"} and instructions[index].mnemonic.startswith(("mul", "imul")))
                     or any(op.kind == "reg" and op.base == reg and op.width < 32
                            for op in instructions[index].ops))
                for index in range(second_range_start, second_range_end + 1)
            ):
                continue
            spare_idx += 1
            rename_map = {reg: spare_reg}

            for line_idx in range(second_range_start, second_range_end + 1):
                inst = renamed_instructions[line_idx]
                renamed_instructions[line_idx] = clone_instruction_with_renaming(inst, rename_map)

    return renamed_instructions


__all__ = [
    "DagNode",
    "build_dag",
    "compute_critical_paths",
    "compute_live_ranges",
    "compute_transitive_closure",
    "evaluate_schedule_cost",
    "exact_optimal_schedule",
    "heuristic_topological_schedule",
    "relax_anti_dependencies_with_renaming",
    "simulated_annealing_permutations",
    "topological_permutations",
]
