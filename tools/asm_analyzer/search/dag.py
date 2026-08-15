"""Dependency DAG construction, critical-path calculation, and topological schedulers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from .ast import (
    FLAG_FULL,
    Instr,
    Spec,
    get_instruction_spec,
)


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
    if mnem.startswith("mul") or mnem.startswith("imul") or mnem in ("umulh", "smulh", "umaal", "umlal"):
        return 3
    if sp.mem == "load":
        return 4
    if sp.mem == "store":
        return 1
    if mnem in ("shld", "shrd"):
        return 3
    return 1


def build_dag(instructions: List[Instr]) -> List[DagNode]:
    """Construct dataflow dependency DAG over a list of instructions."""
    n = len(instructions)
    specs = [get_instruction_spec(inst) for inst in instructions]
    preds: List[Set[int]] = [set() for _ in range(n)]
    succs: List[Set[int]] = [set() for _ in range(n)]

    last_def: Dict[str, int] = {}
    last_flag_def: Dict[str, int] = {}
    last_mem_write: int = -1
    last_mem_reads: List[int] = []

    for i in range(n):
        sp = specs[i]

        # Register RAW dependencies
        for r in sp.uses:
            if r in last_def:
                preds[i].add(last_def[r])

        # Register WAW and WAR dependencies
        for r in sp.defs:
            if r in last_def:
                preds[i].add(last_def[r])

        # Flag dependencies
        for f in sp.flags_read:
            if f in last_flag_def:
                preds[i].add(last_flag_def[f])
            if FLAG_FULL in last_flag_def:
                preds[i].add(last_flag_def[FLAG_FULL])

        for f in sp.flags_write:
            if f == FLAG_FULL:
                for v in last_flag_def.values():
                    preds[i].add(v)
            elif f in last_flag_def:
                preds[i].add(last_flag_def[f])

        # Memory dependencies
        if sp.mem == "store":
            if last_mem_write != -1:
                preds[i].add(last_mem_write)
            for r_idx in last_mem_reads:
                preds[i].add(r_idx)
            last_mem_write = i
            last_mem_reads = []
        elif sp.mem == "load":
            if last_mem_write != -1:
                preds[i].add(last_mem_write)
            last_mem_reads.append(i)

        # Update last defs
        for r in sp.defs:
            last_def[r] = i
        for f in sp.flags_write:
            last_flag_def[f] = i

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
    # Topological reverse pass
    in_degree = [len(node.succs) for node in nodes]
    queue = [i for i in range(n) if in_degree[i] == 0]

    while queue:
        curr = queue.pop(0)
        max_succ_depth = max([depths[s] for s in nodes[curr].succs], default=0)
        depths[curr] = nodes[curr].latency + max_succ_depth

        for p in nodes[curr].preds:
            in_degree[p] -= 1
            if in_degree[p] == 0:
                queue.append(p)

    return depths


def heuristic_topological_schedule(nodes: List[DagNode]) -> List[int]:
    """Generate optimal deterministic schedule using critical-path list scheduling."""
    n = len(nodes)
    if n == 0:
        return []
    depths = compute_critical_paths(nodes)
    in_degree = [len(node.preds) for node in nodes]
    ready = [i for i in range(n) if in_degree[i] == 0]
    schedule: List[int] = []

    while ready:
        # Prioritize node with longest critical path to unblock dependents earliest
        ready.sort(key=lambda idx: (depths[idx], nodes[idx].latency, -idx), reverse=True)
        chosen = ready.pop(0)
        schedule.append(chosen)

        for s in nodes[chosen].succs:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                ready.append(s)

    return schedule


def topological_permutations(nodes: List[DagNode], count: int = 100, seed: int = 42) -> List[List[int]]:
    """Generate high-quality candidate schedules mixing critical-path heuristics and randomized exploration."""
    n = len(nodes)
    if n == 0:
        return []

    depths = compute_critical_paths(nodes)
    rng = random.Random(seed)
    orderings: List[Tuple[int, ...]] = []
    seen: Set[Tuple[int, ...]] = set()

    # Always include the pure critical-path heuristic schedule first
    best_heuristic = tuple(heuristic_topological_schedule(nodes))
    if len(best_heuristic) == n:
        orderings.append(best_heuristic)
        seen.add(best_heuristic)

    # Weighted randomized list scheduling exploration
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
                # Weighted probability biased towards high critical path depth
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

    return [list(o) for o in orderings]
