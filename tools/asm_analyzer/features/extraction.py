"""Assemble static kernel reports and identify each model's applicability."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from ..asm_util import instr_lines
from ..semantics import semantics_for
from ..search.memory_dependencies import analyze_loop_memory_dependencies
from ..types import (
    AnalysisAssessment,
    AnalysisConfidence,
    ArchitectureFamily,
    FeatureSet,
    KernelAnalysisReport,
    MemoryDependencyStats,
    UopCacheStats,
)
from .aarch64 import analyze_aarch64_instructions
from .branch_prediction import analyze_branch_patterns
from .instruction_width import analyze_instruction_widths
from .memory import analyze_memory_accesses, estimate_unroll_factor
from .memory_hierarchy import analyze_memory_hierarchy
from .multiplier import analyze_multiplier
from .ports import analyze_port_pressure
from .registers import analyze_registers
from .short_loop import analyze_short_loops
from .stlf import analyze_stlf_hazards
from .uop_cache import analyze_uop_cache
from .vectorization import analyze_vectorization_feasibility
from .x86_32_loop import analyze_x86_32_loop_control


def extract_features(
    asm: str,
    target_arch: ArchitectureFamily = ArchitectureFamily.X86_64,
) -> FeatureSet:
    """Extract the standard static FeatureSet for one kernel variant."""
    semantics = semantics_for(target_arch)
    mem_stats = analyze_memory_accesses(
        asm,
        semantics.limb_bytes,
        semantics.cache_line_bytes,
        target_arch,
    )
    reg_stats = analyze_registers(asm, semantics)
    mul_stats = analyze_multiplier(asm)
    unroll = estimate_unroll_factor(asm, semantics.limb_bytes)
    instr_count = len(instr_lines(asm))

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
    cpu_model_results: Optional[Dict[str, Dict[str, object]]] = None,
    analysis_scope: str = "block",
    analysis_label: Optional[str] = None,
    backend_failures: Tuple[str, ...] = (),
    limbs_per_iteration: Optional[int] = None,
    operand_limb_count: Optional[int] = None,
    cfg_block_count: int = 0,
    cfg_edge_count: int = 0,
) -> KernelAnalysisReport:
    """Extract complete microarchitectural feature report for an assembly block.

    The port-pressure and multiplier models understand multiple ISAs.  The
    µOp-cache model remains x86-specific.
    """
    semantics = semantics_for(target_arch)
    mem_stats = analyze_memory_accesses(
        asm,
        semantics.limb_bytes,
        semantics.cache_line_bytes,
        target_arch,
    )
    reg_stats = analyze_registers(asm, semantics)
    instruction_width_stats = analyze_instruction_widths(asm, target_arch)
    branch_stats = analyze_branch_patterns(
        asm,
        encoded_bytes=(
            instruction_width_stats.total_bytes
            if instruction_width_stats.exact
            else None
        ),
    )
    is_x86 = target_arch in (ArchitectureFamily.X86_64, ArchitectureFamily.X86_32)
    memory_dependency_stats = (
        analyze_loop_memory_dependencies(asm) if is_x86 else MemoryDependencyStats()
    )
    unroll = limbs_per_iteration or estimate_unroll_factor(
        asm,
        semantics.limb_bytes,
    )

    mul_stats = analyze_multiplier(asm)
    port_stats = analyze_port_pressure(asm, target_arch)
    if semantics.has_x86_uop_cache_model:
        uop_stats = analyze_uop_cache(asm)
    else:
        uop_stats = UopCacheStats()

    stlf_stats = analyze_stlf_hazards(
        asm,
        semantics.limb_bytes,
        semantics.cache_line_bytes,
    )
    hierarchy_limb_count = operand_limb_count or unroll
    hierarchy_stats = analyze_memory_hierarchy(
        hierarchy_limb_count,
        pointer_width_bytes=semantics.limb_bytes,
    )
    short_loop_stats = (
        analyze_short_loops(asm, unroll_factor=unroll)
        if analysis_scope == "loop"
        else None
    )
    vectorization_stats = analyze_vectorization_feasibility(asm) if is_x86 else None
    aarch64_stats = (
        analyze_aarch64_instructions(asm)
        if target_arch is ArchitectureFamily.AARCH64
        else None
    )
    x86_32_stats = (
        analyze_x86_32_loop_control(asm)
        if target_arch is ArchitectureFamily.X86_32
        else None
    )
    assessments = _analysis_assessments(
        target_arch,
        analysis_scope,
        operand_limb_count is not None,
        port_stats.model_name is not None,
    )

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
        instruction_width=instruction_width_stats,
        memory_dependencies=memory_dependency_stats,
        cpu_cycles=cpu_cycles or {},
        cpu_model_results=cpu_model_results or {},
        raw_asm=asm,
        analysis_scope=analysis_scope,
        analysis_label=analysis_label,
        backend_failures=backend_failures,
        limb_bytes=semantics.limb_bytes,
        cache_line_bytes=semantics.cache_line_bytes,
        cfg_block_count=cfg_block_count,
        cfg_edge_count=cfg_edge_count,
        stlf=stlf_stats,
        memory_hierarchy=hierarchy_stats,
        short_loop=short_loop_stats,
        vectorization=vectorization_stats,
        aarch64=aarch64_stats,
        x86_32_loop=x86_32_stats,
        assessments=assessments,
    )


def _analysis_assessments(
    target_arch: ArchitectureFamily,
    analysis_scope: str,
    has_operand_size: bool,
    has_port_model: bool,
) -> Dict[str, AnalysisAssessment]:
    is_x86 = target_arch in (ArchitectureFamily.X86_64, ArchitectureFamily.X86_32)
    is_aarch64 = target_arch is ArchitectureFamily.AARCH64
    is_riscv = target_arch in (ArchitectureFamily.RISCV64, ArchitectureFamily.RISCV32)
    is_power = target_arch in (ArchitectureFamily.POWER64, ArchitectureFamily.POWER32)
    stlf_supported = is_x86 or is_aarch64
    return {
        "memory": AnalysisAssessment(
            True,
            AnalysisConfidence.STRUCTURAL
            if is_x86 or is_aarch64 or is_riscv
            else AnalysisConfidence.HEURISTIC,
            "Target load/store instruction and operand classification.",
        ),
        "memory_dependencies": AnalysisAssessment(
            is_x86,
            AnalysisConfidence.STRUCTURAL if is_x86 else AnalysisConfidence.UNAVAILABLE,
            "Byte-range alias proof and pointer-stride loop-carried analysis."
            if is_x86 else "The reviewed dependency parser currently supports x86 operands.",
        ),
        "registers": AnalysisAssessment(
            True,
            AnalysisConfidence.HEURISTIC if is_power else AnalysisConfidence.STRUCTURAL,
            (
                "POWER bare-register syntax can be ambiguous with small immediates."
                if is_power
                else "Target-specific architectural register scan."
            ),
        ),
        "multiplier": AnalysisAssessment(
            True,
            AnalysisConfidence.HEURISTIC,
            "Product-consumer instruction distance; it is not a latency model.",
        ),
        "branch": AnalysisAssessment(
            True,
            AnalysisConfidence.STRUCTURAL,
            "Branch count is structural; byte density uses exact encoding when available.",
        ),
        "port_pressure": AnalysisAssessment(
            has_port_model,
            AnalysisConfidence.HEURISTIC if has_port_model else AnalysisConfidence.UNAVAILABLE,
            "Generic ISA execution-unit model." if has_port_model
            else "No reviewed port model exists for this ISA.",
        ),
        "uop_cache": AnalysisAssessment(
            is_x86,
            AnalysisConfidence.HEURISTIC if is_x86 else AnalysisConfidence.UNAVAILABLE,
            "Generic x86 decode and uop-cache estimate." if is_x86
            else "The implemented uop-cache model is x86-specific.",
        ),
        "stlf": AnalysisAssessment(
            stlf_supported,
            AnalysisConfidence.HEURISTIC if stlf_supported else AnalysisConfidence.UNAVAILABLE,
            "Address overlap is structural; penalty magnitude is CPU-dependent."
            if stlf_supported else "The operand parser is not validated for this ISA.",
        ),
        "memory_hierarchy": AnalysisAssessment(
            True,
            AnalysisConfidence.HEURISTIC,
            (
                "Uses caller-provided operand length."
                if has_operand_size
                else "Only the emitted iteration footprint is known; full operands were not supplied."
            ),
        ),
        "short_loop": AnalysisAssessment(
            analysis_scope == "loop",
            AnalysisConfidence.HEURISTIC if analysis_scope == "loop"
            else AnalysisConfidence.UNAVAILABLE,
            "Finite-trip estimate over the selected loop body." if analysis_scope == "loop"
            else "No repeated loop body was selected.",
        ),
        "vectorization": AnalysisAssessment(
            is_x86,
            AnalysisConfidence.HEURISTIC if is_x86 else AnalysisConfidence.UNAVAILABLE,
            "x86 AVX2 and AVX-512 instruction-mix screen." if is_x86
            else "The implemented vectorization screen is x86-specific.",
        ),
        "aarch64": AnalysisAssessment(
            is_aarch64,
            AnalysisConfidence.STRUCTURAL if is_aarch64 else AnalysisConfidence.UNAVAILABLE,
            "AArch64 pair-memory, multiply, carry, and register scan." if is_aarch64
            else "AArch64-specific analysis does not apply.",
        ),
        "x86_32_loop": AnalysisAssessment(
            target_arch is ArchitectureFamily.X86_32,
            AnalysisConfidence.STRUCTURAL if target_arch is ArchitectureFamily.X86_32
            else AnalysisConfidence.UNAVAILABLE,
            "32-bit x86 stack and carry invariant scan."
            if target_arch is ArchitectureFamily.X86_32
            else "32-bit x86 stack-loop analysis does not apply.",
        ),
    }


__all__ = [
    "extract_features",
    "extract_kernel_report",
]
