"""Automated microarchitectural optimization suggestion engine.

Scans assembly kernel bodies for performance anti-patterns and generates
actionable, concrete optimization recommendations with code snippets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from ..asm_util import extract_mnemonic
from .branch_prediction import analyze_branch_patterns
from .multiplier import analyze_multiplier
from .registers import analyze_registers
from .uop_cache import analyze_uop_cache
from .vectorization import analyze_vectorization_feasibility


class SuggestionSeverity(str, Enum):
    """Severity level of an optimization suggestion."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class OptimizationSuggestion:
    """Actionable optimization suggestion for an assembly kernel."""
    severity: SuggestionSeverity
    rule_id: str
    title: str
    description: str
    suggested_fix: str
    line_number: Optional[int] = None
    problematic_code: Optional[str] = None


_RMW_OP_RE = re.compile(r"(-?\d*)\(([^)]*)\)")


def generate_suggestions(asm: str, kernel_name: str = "kernel") -> List[OptimizationSuggestion]:
    """Analyze assembly block and generate a list of concrete optimization suggestions."""
    suggestions: List[OptimizationSuggestion] = []
    lines = asm.splitlines()

    # Rule 1: In-Memory Read-Modify-Write (RMW) Detection (x86 CISC-specific)
    # Load-store architectures (AArch64, ARM32, RISC-V, PowerPC, s390x) do not have in-memory RMW.
    _X86_RMW_MNEMS = {
        "add", "addq", "addl", "adc", "adcq", "adcl",
        "sub", "subq", "subl", "sbb", "sbbq", "sbbl",
        "inc", "incq", "incl", "dec", "decq", "decl",
        "and", "andq", "andl", "or", "orq", "orl", "xor", "xorq", "xorl",
        "not", "notq", "notl", "neg", "negq", "negl",
        "shl", "shlq", "shll", "shr", "shrq", "shrl", "sar", "sarq", "sarl",
        "rol", "rolq", "roll", "ror", "rorq", "rorl",
    }
    for idx, raw_line in enumerate(lines, start=1):
        line = raw_line.split("#")[0].split("//")[0].strip()
        if not line or line.startswith(".") or line.endswith(":"):
            continue

        mnem = extract_mnemonic(line)
        if mnem in ("jmp", "jz", "jnz", "js", "jns", "jc", "jnc", "call", "ret", "nop"):
            continue

        # In-memory RMW only occurs on x86 when an ALU instruction targets a memory destination
        if mnem in _X86_RMW_MNEMS:
            parts = line.split(None, 1)
            if len(parts) >= 2:
                operands = [op.strip() for op in parts[1].split(",")]
                if operands and _RMW_OP_RE.search(operands[-1]) and ("%" in operands[-1] or "(" in operands[-1]):
                    dest_mem = operands[-1]
                    # 32-bit x86 stack loop control counters (subl $N, (%esp) / decl (%esp)) are intentional
                    # and required by the 6-GPR hardware budget. Do not flag loop counters as RMW hazards.
                    if "%esp" in dest_mem or "esp" in dest_mem:
                        continue
                    else:
                        # In-place limb destination updates (e.g. adcl/adcq %eax, 0({dst})) do not stall
                        # when indices advance and write through store buffers. Flag non-streaming RMWs.
                        is_streaming_inplace = "{dst}" in dest_mem or "{dst_ptr}" in dest_mem
                        if not is_streaming_inplace:
                            suggestions.append(OptimizationSuggestion(
                                severity=SuggestionSeverity.CRITICAL,
                                rule_id="OPT001-RMW-HAZARD",
                                title="In-Memory Read-Modify-Write Instruction",
                                description=f"Instruction `{line}` performs an in-memory RMW on `{dest_mem}`.",
                                suggested_fix="Load into a register, perform ALU arithmetic, and write back.",
                                line_number=idx,
                                problematic_code=line,
                            ))

    # Rule 2: Multiplier Latency & Zero-Slack Consumption
    mul_stats = analyze_multiplier(asm)
    if mul_stats.has_multiplier_stall and not mul_stats.is_paired_pipeline:
        # Check if this is explicitly a baseline/legacy fallback kernel (superseded by BMI2/ADX on modern CPUs)
        stem = kernel_name.lower().replace(".rs", "")
        is_legacy_fallback = stem in ("x86_64", "x86", "fallback") or stem.endswith("_x86_64") or stem.endswith("_x86") or "fallback" in stem
        if is_legacy_fallback:
            suggestions.append(OptimizationSuggestion(
                severity=SuggestionSeverity.INFO,
                rule_id="OPT002-LEGACY-MUL-SLACK",
                title="Baseline Pre-BMI2 Multiplier Slack (Expected Fallback)",
                description="Legacy 1-operand `mulq` serializes %rax:%rdx outputs on pre-BMI2 CPUs. "
                            "Modern CPUs dynamically dispatch to `x86_64_adx` with zero-stall parallel multiplier streams.",
                suggested_fix="Expected architectural behavior for baseline fallback; modern targets use BMI2/ADX runtime dispatch.",
            ))
        else:
            suggestions.append(OptimizationSuggestion(
                severity=SuggestionSeverity.WARNING,
                rule_id="OPT002-MUL-SLACK-STALL",
                title="Zero Multiplier Slack (Latency Stall)",
                description=f"Multiplier product is consumed immediately (instruction distance = 0). "
                            f"On x86 and ARM, 64-bit integer multiplication has a 3–4 cycle latency. "
                            f"When the very next instruction reads the product register, the pipeline "
                            f"stalls until the result is ready.",
                suggested_fix="Hoist independent memory loads or non-dependent arithmetic between the multiply "
                              "and its product consumption to hide execution latency.",
            ))

    # Rule 3: Fall-Through .p2align Decode Hazard
    for idx, raw_line in enumerate(lines, start=1):
        line = raw_line.split("#")[0].split("//")[0].strip()
        if line == ".p2align 4" or line.startswith(".p2align"):
            if idx > 1:
                prev_line = lines[idx - 2].split("#")[0].split("//")[0].strip()
                prev_mnem = extract_mnemonic(prev_line)
                if prev_mnem in ("js", "jns", "jz", "jnz", "jc", "jnc", "ja", "jb", "jae", "jbe"):
                    suggestions.append(OptimizationSuggestion(
                        severity=SuggestionSeverity.WARNING,
                        rule_id="OPT003-ALIGN-FALLTHROUGH",
                        title="Fall-Through .p2align Execution Hazard",
                        description=f"`.p2align` directive at line {idx} is placed on a straight fall-through path "
                                    f"directly following conditional branch `{prev_line}`, injecting multi-byte NOPs into the Op-Cache.",
                        suggested_fix="Remove `.p2align` from straight fall-through code paths. Align only non-fall-through loop targets.",
                        line_number=idx,
                        problematic_code=raw_line.strip(),
                    ))

    # Rule 4: High Register Pressure
    reg_stats = analyze_registers(asm)
    if reg_stats.is_gpr_pressure_high:
        suggestions.append(OptimizationSuggestion(
            severity=SuggestionSeverity.WARNING,
            rule_id="OPT004-HIGH-GPR-PRESSURE",
            title="High General-Purpose Register Pressure",
            description=f"Kernel block uses {reg_stats.gprs_used} distinct GPRs (exceeds the 14 allocatable limit). "
                        f"Excessive register usage forces stack spills during register allocation.",
            suggested_fix="Reuse scratch registers between independent limb calculations.",
        ))

    # Rule 5: µOp Cache Saturation & Sizing
    uop_stats = analyze_uop_cache(asm)
    if not uop_stats.fits_intel_dsb:
        suggestions.append(OptimizationSuggestion(
            severity=SuggestionSeverity.WARNING,
            rule_id="OPT005-UOP-CACHE-OVERFLOW",
            title="µOp Cache Capacity Overflow",
            description=f"Unrolled body contains ~{uop_stats.estimated_uops} µOps, exceeding Intel DSB cache limit "
                        f"({uop_stats.recommended_max_unroll}x unrolling recommended). Exceeding capacity forces decoder to fall back to legacy MITE.",
            suggested_fix=f"Reduce loop unrolling factor to {uop_stats.recommended_max_unroll}x to ensure the inner loop fits entirely within the µOp cache.",
        ))

    # Rule 6: Branch Target Buffer (BTB) Density
    branch_stats = analyze_branch_patterns(asm)
    if branch_stats.has_btb_density_hazard:
        suggestions.append(OptimizationSuggestion(
            severity=SuggestionSeverity.WARNING,
            rule_id="OPT006-BTB-DENSITY-HIGH",
            title="High Branch Density in Code Window",
            description=f"Code window contains {branch_stats.branches_per_64_bytes} branches per 64 bytes (exceeds BTB limit of 3).",
            suggested_fix="Unroll loop or fuse sequential conditional jumps to reduce branch instruction density.",
        ))

    # Rule 7: SIMD / AVX-512 IFMA Vectorization Opportunity
    vec_stats = analyze_vectorization_feasibility(asm)
    if vec_stats.is_avx512_ifma_candidate:
        suggestions.append(OptimizationSuggestion(
            severity=SuggestionSeverity.INFO,
            rule_id="OPT007-AVX512-IFMA-OPPORTUNITY",
            title="Candidate for AVX-512 IFMA 52-bit Vectorization",
            description=f"Kernel contains high multiplication density ({vec_stats.rationale}).",
            suggested_fix="Consider adding a 52-bit radix AVX-512 IFMA implementation (`_mm512_madd52lo_epu64`) for wide inputs.",
        ))

    # Rule 8: Store-to-Load Forwarding (STLF) Hazard Detection
    from .stlf import analyze_stlf_hazards
    stlf_stats = analyze_stlf_hazards(asm)
    if stlf_stats.has_stlf_hazard:
        for h in stlf_stats.hazards:
            suggestions.append(OptimizationSuggestion(
                severity=SuggestionSeverity.WARNING,
                rule_id="OPT008-STLF-FORWARDING-HAZARD",
                title="Store-to-Load Forwarding (STLF) Pipeline Stall",
                description=f"{h.description} (+{h.penalty_cycles} cycles penalty).",
                suggested_fix="Use the register carry-forward pattern to retain written limb values across loop iterations rather than reloading from memory.",
                problematic_code=f"Store: {h.store_line} -> Load: {h.load_line}",
            ))

    # Rule 9: Redundant-Radix (Radix-2^52) IFMA Parallelization Advisor
    if vec_stats.is_avx512_ifma_candidate:
        # Check if the kernel has dense carry chains where deferred carry accumulation provides 8x speedup
        has_dense_carries = any(m in asm for m in ("adcx", "adox", "adc", "adcq", "adcs"))
        if has_dense_carries:
            suggestions.append(OptimizationSuggestion(
                severity=SuggestionSeverity.INFO,
                rule_id="OPT009-IFMA-REDUNDANT-RADIX",
                title="Redundant Radix-2^52 IFMA Acceleration Feasibility",
                description="Dense carry propagation and multiplier streams detected. "
                            "In Radix-2^52 representation, carries accumulate into the 12 spare top bits "
                            "without scalar flag serialization, enabling 8 concurrent 52x52->104 bit "
                            "fused multiply-accumulates via AVX-512 `vpmadd52luq`/`vpmadd52huq`.",
                suggested_fix="For wide multi-limb operations (>= 8 limbs), evaluate redundant Radix-2^52 representation.",
            ))

    # Rule 10, 11, 12: 32-bit x86 Stack Loop Control & Frame Balance Verification
    from .x86_32_loop import analyze_x86_32_loop_control
    loop_stats = analyze_x86_32_loop_control(asm)
    if loop_stats.is_32bit_stack_loop:
        if loop_stats.has_stack_imbalance:
            suggestions.append(OptimizationSuggestion(
                severity=SuggestionSeverity.CRITICAL,
                rule_id="OPT010-STACK-IMBALANCE",
                title="Stack Frame Imbalance on Exit Path",
                description=f"Net stack pointer delta is {loop_stats.net_stack_delta:+d} bytes at exit. "
                            f"Stack pushes/allocations are not strictly balanced with deallocations.",
                suggested_fix="Ensure all `pushl` and `subl $N, %esp` allocations are balanced with `addl $N, %esp` before return.",
            ))
        if loop_stats.has_flag_clobber_hazard:
            suggestions.append(OptimizationSuggestion(
                severity=SuggestionSeverity.CRITICAL,
                rule_id="OPT011-FLAG-CLOBBER-LOOP-CONTROL",
                title="Live Condition Flag Clobbered by Loop Counter Arithmetic",
                description="Loop counter modification (`subl`/`decl` on stack) occurs while the Carry/Borrow flag is live without prior register mask capture.",
                suggested_fix="Capture the live flag into a register mask (e.g. `sbbl {reg}, {reg}`) before executing stack counter arithmetic, and restore it at loop entry (`addl $1, {reg}`).",
            ))
        if loop_stats.has_stride_mismatch:
            suggestions.append(OptimizationSuggestion(
                severity=SuggestionSeverity.WARNING,
                rule_id="OPT012-LOOP-STRIDE-MISMATCH",
                title="Loop Pointer Displacement vs Counter Step Mismatch",
                description=f"Pointer increment ({loop_stats.unroll_stride_bytes} bytes) does not match stack counter step ({loop_stats.counter_step_limbs} limbs).",
                suggested_fix="Ensure pointer displacements advance by exactly `counter_step * 4` bytes per unrolled iteration.",
            ))

    return suggestions
