"""Working-set footprint, cache hierarchy, and Roofline model analyzer.

Models multi-precision operand memory requirements against hardware cache tiers
(L1 Data Cache, L2 Cache, L3 Cache, and Main Memory DRAM) to determine when
in-register processing is optimal and when cache-blocked algorithms or prefetching
should be activated.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from ..types import MemoryHierarchyStats

L1D_CAPACITY_BYTES = 32 * 1024        # 32 KB (Baseline x86-64 / ARM L1D)
L2_CAPACITY_BYTES = 512 * 1024        # 512 KB (Baseline L2 per core)
L3_CAPACITY_BYTES = 32 * 1024 * 1024  # 32 MB (Baseline shared L3)

# Typical peak memory bandwidths (GB/s) on modern desktop / server CPUs
BANDWIDTH_L1D_GBPS = 150.0
BANDWIDTH_L2_GBPS = 80.0
BANDWIDTH_L3_GBPS = 45.0
BANDWIDTH_DRAM_GBPS = 35.0


def analyze_memory_hierarchy(
    limb_count: int,
    num_operands: int = 3,
    pointer_width_bytes: int = 8,
    ops_per_limb: float = 2.0,
) -> MemoryHierarchyStats:
    """Analyze memory working set size, cache tier, and Roofline constraints."""
    working_set = limb_count * pointer_width_bytes * num_operands

    if working_set <= L1D_CAPACITY_BYTES:
        tier = "L1D"
        spills_l1 = False
        spills_l2 = False
        suggest_blocking = False
        effective_bandwidth = BANDWIDTH_L1D_GBPS
    elif working_set <= L2_CAPACITY_BYTES:
        tier = "L2"
        spills_l1 = True
        spills_l2 = False
        suggest_blocking = False
        effective_bandwidth = BANDWIDTH_L2_GBPS
    elif working_set <= L3_CAPACITY_BYTES:
        tier = "L3"
        spills_l1 = True
        spills_l2 = True
        suggest_blocking = True
        effective_bandwidth = BANDWIDTH_L3_GBPS
    else:
        tier = "DRAM"
        spills_l1 = True
        spills_l2 = True
        suggest_blocking = True
        effective_bandwidth = BANDWIDTH_DRAM_GBPS

    # Arithmetic Intensity (Operations per Byte transferred)
    bytes_per_limb = pointer_width_bytes * num_operands
    arithmetic_intensity = ops_per_limb / max(bytes_per_limb, 1)

    # Compute vs Memory Bound classification:
    # A typical core can execute ~4 ALU ops/cycle at 4 GHz = 16 GOps/s.
    # If Arithmetic Intensity * Memory Bandwidth < Peak ALU, the loop is Memory-Bound.
    peak_alu_gops = 16.0
    attainable_gops = min(peak_alu_gops, arithmetic_intensity * effective_bandwidth)
    is_memory_bound = (arithmetic_intensity * effective_bandwidth) < peak_alu_gops

    return MemoryHierarchyStats(
        limb_count=limb_count,
        working_set_bytes=working_set,
        cache_tier=tier,
        spills_l1d=spills_l1,
        spills_l2=spills_l2,
        suggest_cache_blocking=suggest_blocking,
    )
