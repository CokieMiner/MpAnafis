"""Assembly kernel feature extraction facade."""

from __future__ import annotations

from typing import Dict, List, Optional
from ..consensus.dataset import FeatureSet
from ..types import (
    ArchitectureFamily,
    BranchStats,
    KernelAnalysisReport,
    MemoryAccessStats,
    MemoryHierarchyStats,
    MultiplierStats,
    PortPressureStats,
    RegisterStats,
    UopCacheStats,
    VectorizationFeasibility,
)
from .branch_prediction import analyze_branch_patterns
from .memory import analyze_memory_accesses, estimate_unroll_factor
from .memory_hierarchy import analyze_memory_hierarchy
from .multiplier import analyze_multiplier
from .ports import analyze_port_pressure
from .registers import analyze_registers
from .uop_cache import analyze_uop_cache
from .vectorization import analyze_vectorization_feasibility


def extract_features(asm: str, reports: Optional[List[object]] = None) -> FeatureSet:
    """Extract standard FeatureSet for dataset and consensus error modeling."""
    mem_stats = analyze_memory_accesses(asm)
    reg_stats = analyze_registers(asm)
    mul_stats = analyze_multiplier(asm)
    unroll = estimate_unroll_factor(asm)
    instr_count = len(asm.splitlines())

    return FeatureSet(
        instruction_count=instr_count,
        gpr_count=reg_stats.gprs_used,
        mem_loads=mem_stats.loads,
        mem_stores=mem_stats.stores,
        rmw_count=mem_stats.read_modify_writes,
        unroll_factor=unroll,
        mul_latency_slack=mul_stats.min_slack,
        cache_straddles=mem_stats.cache_line_straddles,
    )


def extract_kernel_report(
    asm: str,
    kernel_name: str = "kernel",
    target_arch: ArchitectureFamily = ArchitectureFamily.X86_64,
    cpu_cycles: Optional[Dict[str, float]] = None,
) -> KernelAnalysisReport:
    """Extract complete microarchitectural feature report for an assembly block.

    x86-specific analyzers (port pressure, µOp cache, multiplier slack) are
    skipped for non-x86 targets where they would produce meaningless results.
    """
    mem_stats = analyze_memory_accesses(asm)
    reg_stats = analyze_registers(asm)
    branch_stats = analyze_branch_patterns(asm)
    unroll = estimate_unroll_factor(asm)

    mul_stats = analyze_multiplier(asm)
    is_x86 = target_arch in (ArchitectureFamily.X86_64, ArchitectureFamily.X86_32)

    if is_x86:
        port_stats = analyze_port_pressure(asm)
        uop_stats = analyze_uop_cache(asm)
    else:
        port_stats = PortPressureStats()
        uop_stats = UopCacheStats()

    return KernelAnalysisReport(
        kernel_name=kernel_name,
        target_arch=target_arch,
        unroll_factor=unroll,
        memory=mem_stats,
        registers=reg_stats,
        multiplier=mul_stats,
        port_pressure=port_stats,
        uop_cache=uop_stats,
        branch=branch_stats,
        cpu_cycles=cpu_cycles or {},
        raw_asm=asm,
    )


__all__ = [
    "analyze_branch_patterns",
    "analyze_memory_accesses",
    "analyze_memory_hierarchy",
    "analyze_multiplier",
    "analyze_port_pressure",
    "analyze_registers",
    "analyze_uop_cache",
    "analyze_vectorization_feasibility",
    "estimate_unroll_factor",
    "extract_features",
    "extract_kernel_report",
]
