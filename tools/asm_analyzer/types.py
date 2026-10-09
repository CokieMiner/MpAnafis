"""Data-driven type definitions and immutable dataclasses for asm_analyzer.

Defines all domain models, metrics, and structured reports used across
the assembly analyzer suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class ArchitectureFamily(str, Enum):
    """Supported CPU architecture families."""
    X86_64 = "x86_64"
    X86_32 = "x86"
    AARCH64 = "aarch64"
    ARM32 = "arm32"
    RISCV64 = "riscv64"
    RISCV32 = "riscv32"
    POWER64 = "power64"
    POWER32 = "power32"
    S390X = "s390x"
    MIPS64 = "mips64"
    MIPS32 = "mips32"
    LOONGARCH64 = "loongarch64"
    LOONGARCH32 = "loongarch32"


class AnalysisConfidence(str, Enum):
    """Strength of evidence behind one static analysis result."""

    EXACT = "exact"
    STRUCTURAL = "structural"
    HEURISTIC = "heuristic"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class AnalysisAssessment:
    """Applicability and confidence metadata for one report section."""

    applicable: bool
    confidence: AnalysisConfidence
    rationale: str


@dataclass(frozen=True)
class MemoryAccessStats:
    """Statistics for memory interactions within an assembly kernel block."""
    loads: int = 0
    stores: int = 0
    read_modify_writes: int = 0
    cache_line_straddles: int = 0

    @property
    def total_accesses(self) -> int:
        """Total memory operations."""
        return self.loads + self.stores + self.read_modify_writes

    @property
    def has_rmw(self) -> bool:
        """Whether the block contains any read-modify-write instruction."""
        return self.read_modify_writes > 0


@dataclass(frozen=True)
class RegisterStats:
    """Register usage and pressure analysis."""
    gprs_used: int = 0
    gpr_names: Tuple[str, ...] = ()
    simds_used: int = 0
    simd_names: Tuple[str, ...] = ()
    flags_read: Tuple[str, ...] = ()
    flags_written: Tuple[str, ...] = ()
    allocatable_gprs: int = 14

    @property
    def is_gpr_pressure_high(self) -> bool:
        """Whether observed GPR use exceeds the target's allocatable budget."""
        return self.gprs_used > self.allocatable_gprs


@dataclass(frozen=True)
class MultiplierStats:
    """Multiplier latency, slack, and pipelining characteristics."""
    mul_count: int = 0
    min_slack: Optional[int] = None
    is_paired_pipeline: bool = False

    @property
    def has_multiplier_stall(self) -> bool:
        """Whether multiplier has 0-instruction consumption slack."""
        return self.mul_count > 0 and self.min_slack is not None and self.min_slack == 0


@dataclass(frozen=True)
class PortPressureStats:
    """Breakdown of execution port / ALU binding counts per block."""
    intel_ports: Dict[str, float] = field(default_factory=dict)
    amd_alus: Dict[str, float] = field(default_factory=dict)
    arm_units: Dict[str, float] = field(default_factory=dict)
    target_units: Dict[str, float] = field(default_factory=dict)
    model_name: Optional[str] = None
    bottleneck_port: Optional[str] = None
    bottleneck_cycles: float = 0.0


@dataclass(frozen=True)
class UopCacheStats:
    """Decode width and µOp cache (DSB / Op-Cache) saturation characteristics."""
    instruction_count: int = 0
    estimated_uops: int = 0
    estimated_bytes: int = 0
    fits_intel_dsb: bool = True
    fits_amd_op_cache: bool = True
    recommended_max_unroll: int = 4


@dataclass(frozen=True)
class MemoryHierarchyStats:
    """Working set footprint and cache hierarchy tier mapping."""
    limb_count: int = 0
    working_set_bytes: int = 0
    cache_tier: str = "L1D"
    spills_l1d: bool = False
    spills_l2: bool = False
    suggest_cache_blocking: bool = False
    arithmetic_intensity: float = 0.0
    estimated_memory_bound: bool = False


@dataclass(frozen=True)
class BranchStats:
    """Branch target buffer (BTB) density and loop entry alignment."""
    branch_count: int = 0
    branches_per_64_bytes: float = 0.0
    has_btb_density_hazard: bool = False
    has_unaligned_loop_head: bool = False


@dataclass(frozen=True)
class InstructionWidthStats:
    """Exact encoded instruction sizes when a target assembler is available."""

    exact: bool = False
    total_bytes: int = 0
    instruction_bytes: Tuple[int, ...] = ()
    source: Optional[str] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class MemoryDependencyStats:
    """Conservative alias and loop-carried memory dependency results."""

    memory_operations: int = 0
    proved_disjoint_pairs: int = 0
    may_alias_pairs: int = 0
    pointer_strides: Dict[str, int] = field(default_factory=dict)
    loop_carried_dependencies: Tuple[str, ...] = ()
    unknown_cross_iteration_pairs: int = 0


@dataclass(frozen=True)
class VectorizationFeasibility:
    """SIMD (AVX2 / AVX-512 IFMA) vectorization feasibility assessment."""
    is_avx2_candidate: bool = False
    is_avx512_ifma_candidate: bool = False
    lane_count_256: int = 4
    lane_count_512: int = 8
    rationale: str = ""


@dataclass(frozen=True)
class StlfHazard:
    """One Store-to-Load Forwarding hazard detected in a memory access sequence."""
    hazard_type: str
    store_line: str
    load_line: str
    distance_instructions: int
    penalty_cycles: float
    description: str


@dataclass(frozen=True)
class StlfAnalysis:
    """Aggregated STLF hazard analysis results for an assembly block."""
    has_stlf_hazard: bool = False
    hazard_count: int = 0
    max_penalty_cycles: float = 0.0
    hazards: Tuple[StlfHazard, ...] = ()


@dataclass(frozen=True)
class ShortLoopStats:
    """Heuristic finite-loop cost estimates for small operand lengths."""

    prologue_instructions: int = 0
    loop_body_instructions: int = 0
    epilogue_instructions: int = 0
    estimated_cycles_by_limbs: Dict[int, float] = field(default_factory=dict)
    branch_mispredict_overhead_cycles: float = 0.0


@dataclass(frozen=True)
class Aarch64Stats:
    """AArch64-specific instruction mix and register statistics."""

    gprs_used: int = 0
    loads: int = 0
    stores: int = 0
    pair_loads_ldp: int = 0
    pair_stores_stp: int = 0
    mul_count: int = 0
    umulh_count: int = 0
    carry_instructions: int = 0


@dataclass(frozen=True)
class X86_32LoopStats:
    """32-bit x86 stack-loop and control-flow invariant results."""

    is_32bit_stack_loop: bool = False
    net_stack_delta: int = 0
    has_stack_imbalance: bool = False
    has_flag_clobber_hazard: bool = False
    has_stride_mismatch: bool = False
    unroll_stride_bytes: int = 0
    counter_step_limbs: int = 0
    diagnostics: Tuple[str, ...] = ()


@dataclass(frozen=True)
class KernelAnalysisReport:
    """Unified analysis report for an individual assembly kernel."""
    kernel_name: str
    target_arch: ArchitectureFamily
    unroll_factor: int
    memory: MemoryAccessStats
    registers: RegisterStats
    multiplier: MultiplierStats
    port_pressure: PortPressureStats
    uop_cache: UopCacheStats = field(default_factory=UopCacheStats)
    branch: BranchStats = field(default_factory=BranchStats)
    instruction_width: InstructionWidthStats = field(
        default_factory=InstructionWidthStats,
    )
    memory_dependencies: MemoryDependencyStats = field(
        default_factory=MemoryDependencyStats,
    )
    cpu_cycles: Dict[str, float] = field(default_factory=dict)
    cpu_model_results: Dict[str, Dict[str, object]] = field(default_factory=dict)
    raw_asm: str = ""
    analysis_scope: str = "block"
    analysis_label: Optional[str] = None
    backend_failures: Tuple[str, ...] = ()
    limb_bytes: int = 8
    cache_line_bytes: int = 64
    cfg_block_count: int = 0
    cfg_edge_count: int = 0
    stlf: StlfAnalysis = field(default_factory=StlfAnalysis)
    memory_hierarchy: MemoryHierarchyStats = field(default_factory=MemoryHierarchyStats)
    short_loop: Optional[ShortLoopStats] = None
    vectorization: Optional[VectorizationFeasibility] = None
    aarch64: Optional[Aarch64Stats] = None
    x86_32_loop: Optional[X86_32LoopStats] = None
    assessments: Dict[str, AnalysisAssessment] = field(default_factory=dict)

    @property
    def cpu_cycles_per_limb(self) -> Dict[str, float]:
        """Normalize loop costs by the estimated limb unroll factor."""
        divisor = max(1, self.unroll_factor)
        return {name: cycles / divisor for name, cycles in self.cpu_cycles.items()}

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return {
            "kernel_name": self.kernel_name,
            "target_arch": self.target_arch.value,
            "analysis_scope": self.analysis_scope,
            "analysis_label": self.analysis_label,
            "backend_failures": list(self.backend_failures),
            "limb_bytes": self.limb_bytes,
            "cache_line_bytes": self.cache_line_bytes,
            "cfg_block_count": self.cfg_block_count,
            "cfg_edge_count": self.cfg_edge_count,
            "unroll_factor": self.unroll_factor,
            "memory": {
                "loads": self.memory.loads,
                "stores": self.memory.stores,
                "read_modify_writes": self.memory.read_modify_writes,
                "cache_line_straddles": self.memory.cache_line_straddles,
            },
            "registers": {
                "gprs_used": self.registers.gprs_used,
                "gpr_names": list(self.registers.gpr_names),
                "simds_used": self.registers.simds_used,
                "simd_names": list(self.registers.simd_names),
                "flags_read": list(self.registers.flags_read),
                "flags_written": list(self.registers.flags_written),
                "allocatable_gprs": self.registers.allocatable_gprs,
            },
            "multiplier": {
                "mul_count": self.multiplier.mul_count,
                "min_slack": self.multiplier.min_slack,
                "is_paired_pipeline": self.multiplier.is_paired_pipeline,
            },
            "port_pressure": {
                "intel_ports": self.port_pressure.intel_ports,
                "amd_alus": self.port_pressure.amd_alus,
                "arm_units": self.port_pressure.arm_units,
                "target_units": self.port_pressure.target_units,
                "model_name": self.port_pressure.model_name,
                "bottleneck_port": self.port_pressure.bottleneck_port,
                "bottleneck_cycles": self.port_pressure.bottleneck_cycles,
            },
            "uop_cache": {
                "instruction_count": self.uop_cache.instruction_count,
                "estimated_uops": self.uop_cache.estimated_uops,
                "estimated_bytes": self.uop_cache.estimated_bytes,
                "fits_intel_dsb": self.uop_cache.fits_intel_dsb,
                "fits_amd_op_cache": self.uop_cache.fits_amd_op_cache,
                "recommended_max_unroll": self.uop_cache.recommended_max_unroll,
            },
            "branch": {
                "branch_count": self.branch.branch_count,
                "branches_per_64_bytes": self.branch.branches_per_64_bytes,
                "has_btb_density_hazard": self.branch.has_btb_density_hazard,
                "has_unaligned_loop_head": self.branch.has_unaligned_loop_head,
            },
            "instruction_width": {
                "exact": self.instruction_width.exact,
                "total_bytes": self.instruction_width.total_bytes,
                "instruction_bytes": list(self.instruction_width.instruction_bytes),
                "source": self.instruction_width.source,
                "error": self.instruction_width.error,
            },
            "memory_dependencies": {
                "memory_operations": self.memory_dependencies.memory_operations,
                "proved_disjoint_pairs": self.memory_dependencies.proved_disjoint_pairs,
                "may_alias_pairs": self.memory_dependencies.may_alias_pairs,
                "pointer_strides": self.memory_dependencies.pointer_strides,
                "loop_carried_dependencies": list(
                    self.memory_dependencies.loop_carried_dependencies,
                ),
                "unknown_cross_iteration_pairs": (
                    self.memory_dependencies.unknown_cross_iteration_pairs
                ),
            },
            "cpu_model_results": self.cpu_model_results,
            "stlf": {
                "has_stlf_hazard": self.stlf.has_stlf_hazard,
                "hazard_count": self.stlf.hazard_count,
                "max_penalty_cycles": self.stlf.max_penalty_cycles,
                "hazards": [
                    {
                        "hazard_type": hazard.hazard_type,
                        "store_line": hazard.store_line,
                        "load_line": hazard.load_line,
                        "distance_instructions": hazard.distance_instructions,
                        "penalty_cycles": hazard.penalty_cycles,
                        "description": hazard.description,
                    }
                    for hazard in self.stlf.hazards
                ],
            },
            "memory_hierarchy": {
                "limb_count": self.memory_hierarchy.limb_count,
                "working_set_bytes": self.memory_hierarchy.working_set_bytes,
                "cache_tier": self.memory_hierarchy.cache_tier,
                "spills_l1d": self.memory_hierarchy.spills_l1d,
                "spills_l2": self.memory_hierarchy.spills_l2,
                "suggest_cache_blocking": self.memory_hierarchy.suggest_cache_blocking,
                "arithmetic_intensity": self.memory_hierarchy.arithmetic_intensity,
                "estimated_memory_bound": self.memory_hierarchy.estimated_memory_bound,
            },
            "short_loop": _short_loop_to_dict(self.short_loop),
            "vectorization": _vectorization_to_dict(self.vectorization),
            "aarch64": _aarch64_to_dict(self.aarch64),
            "x86_32_loop": _x86_32_loop_to_dict(self.x86_32_loop),
            "assessments": {
                name: {
                    "applicable": assessment.applicable,
                    "confidence": assessment.confidence.value,
                    "rationale": assessment.rationale,
                }
                for name, assessment in self.assessments.items()
            },
            "cpu_cycles": self.cpu_cycles,
            "cpu_cycles_per_limb": self.cpu_cycles_per_limb,
        }


@dataclass(frozen=True)
class KernelComparisonDiff:
    """Side-by-side comparison between two kernel variants."""
    kernel_a: KernelAnalysisReport
    kernel_b: KernelAnalysisReport
    cycle_deltas: Dict[str, float] = field(default_factory=dict)
    load_delta: int = 0
    store_delta: int = 0
    rmw_delta: int = 0
    gpr_delta: int = 0
    speedup_ratios: Dict[str, float] = field(default_factory=dict)
    backend_failures: Tuple[str, ...] = ()


def _short_loop_to_dict(stats: Optional[ShortLoopStats]) -> Optional[Dict[str, Any]]:
    if stats is None:
        return None
    return {
        "prologue_instructions": stats.prologue_instructions,
        "loop_body_instructions": stats.loop_body_instructions,
        "epilogue_instructions": stats.epilogue_instructions,
        "estimated_cycles_by_limbs": stats.estimated_cycles_by_limbs,
        "branch_mispredict_overhead_cycles": stats.branch_mispredict_overhead_cycles,
    }


def _vectorization_to_dict(
    stats: Optional[VectorizationFeasibility],
) -> Optional[Dict[str, Any]]:
    if stats is None:
        return None
    return {
        "is_avx2_candidate": stats.is_avx2_candidate,
        "is_avx512_ifma_candidate": stats.is_avx512_ifma_candidate,
        "lane_count_256": stats.lane_count_256,
        "lane_count_512": stats.lane_count_512,
        "rationale": stats.rationale,
    }


def _aarch64_to_dict(stats: Optional[Aarch64Stats]) -> Optional[Dict[str, Any]]:
    if stats is None:
        return None
    return {
        "gprs_used": stats.gprs_used,
        "loads": stats.loads,
        "stores": stats.stores,
        "pair_loads_ldp": stats.pair_loads_ldp,
        "pair_stores_stp": stats.pair_stores_stp,
        "mul_count": stats.mul_count,
        "umulh_count": stats.umulh_count,
        "carry_instructions": stats.carry_instructions,
    }


def _x86_32_loop_to_dict(
    stats: Optional[X86_32LoopStats],
) -> Optional[Dict[str, Any]]:
    if stats is None:
        return None
    return {
        "is_32bit_stack_loop": stats.is_32bit_stack_loop,
        "net_stack_delta": stats.net_stack_delta,
        "has_stack_imbalance": stats.has_stack_imbalance,
        "has_flag_clobber_hazard": stats.has_flag_clobber_hazard,
        "has_stride_mismatch": stats.has_stride_mismatch,
        "unroll_stride_bytes": stats.unroll_stride_bytes,
        "counter_step_limbs": stats.counter_step_limbs,
        "diagnostics": list(stats.diagnostics),
    }


@dataclass
class FeatureSet:
    """Static features of one kernel variant on one CPU."""
    instruction_count: int
    uops: Optional[float] = None
    dependency_depth: Optional[float] = None
    flag_chain_length: Optional[int] = None
    code_size: Optional[int] = None
    gpr_count: Optional[int] = None
    mem_loads: Optional[int] = None
    mem_stores: Optional[int] = None
    rmw_count: Optional[int] = None
    unroll_factor: Optional[int] = None
    mul_latency_slack: Optional[int] = None
    cache_straddles: Optional[int] = None
