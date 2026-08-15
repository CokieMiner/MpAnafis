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
    def has_rmw_hazard(self) -> bool:
        """Whether any store-forwarding RMW hazard exists."""
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

    @property
    def is_gpr_pressure_high(self) -> bool:
        """Whether GPR count exceeds x86-64 safe allocatable limit (14)."""
        return self.gprs_used > 14


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


@dataclass(frozen=True)
class BranchStats:
    """Branch target buffer (BTB) density and loop entry alignment."""
    branch_count: int = 0
    branches_per_64_bytes: float = 0.0
    has_btb_density_hazard: bool = False
    has_unaligned_loop_head: bool = False


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
    cpu_cycles: Dict[str, float] = field(default_factory=dict)
    raw_asm: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return {
            "kernel_name": self.kernel_name,
            "target_arch": self.target_arch.value,
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
                "flags_read": list(self.registers.flags_read),
                "flags_written": list(self.registers.flags_written),
            },
            "multiplier": {
                "mul_count": self.multiplier.mul_count,
                "min_slack": self.multiplier.min_slack,
                "is_paired_pipeline": self.multiplier.is_paired_pipeline,
            },
            "port_pressure": {
                "intel_ports": self.port_pressure.intel_ports,
                "amd_alus": self.port_pressure.amd_alus,
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
            "cpu_cycles": self.cpu_cycles,
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
