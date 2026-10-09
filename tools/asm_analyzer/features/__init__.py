"""Static assembly feature facade."""

from .aarch64 import analyze_aarch64_instructions
from .branch_prediction import analyze_branch_patterns
from .extraction import extract_features, extract_kernel_report
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

__all__ = [
    "analyze_aarch64_instructions",
    "analyze_branch_patterns",
    "analyze_memory_accesses",
    "analyze_memory_hierarchy",
    "analyze_multiplier",
    "analyze_port_pressure",
    "analyze_registers",
    "analyze_short_loops",
    "analyze_stlf_hazards",
    "analyze_uop_cache",
    "analyze_vectorization_feasibility",
    "analyze_x86_32_loop_control",
    "estimate_unroll_factor",
    "extract_features",
    "extract_kernel_report",
]
