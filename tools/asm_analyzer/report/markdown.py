"""Markdown table rendering for assembly analysis reports and diffs."""

from __future__ import annotations

from typing import List
from ..consensus.score import Cell, ConsensusResult
from ..types import KernelAnalysisReport, KernelComparisonDiff


def render_cell_table(cells: List[Cell], variants: List[str],
                      result: ConsensusResult, top: int = 8) -> str:
    """Render a per-CPU cell table across variants."""
    valid = [c for c in cells if c.best is not None]
    if not valid:
        return f"  {result.cpu}: no usable measurements\n"
    col_labels = [f"{c.backend}:{c.cpu}" for c in valid]
    widths = [max(len(l), 8) for l in col_labels]

    def _fmt_cost(cell: Cell, v: str) -> str:
        cost = cell.costs.get(v)
        if cost is None:
            return "-"
        s = f"{cost:.2f}"
        if v in cell.winners:
            s = f"*{s}"
        return s

    ordered = [v for v in result.ranking if v in variants][:top]
    lines: List[str] = []
    cols = [l.ljust(w) for l, w in zip(col_labels, widths)]
    lines.append(f"  {'variant'.ljust(14)} " + " ".join(cols))
    lines.append("  " + "-" * (14 + sum(widths) + len(widths)))

    for v in ordered:
        row_costs = [_fmt_cost(c, v).ljust(w) for c, w in zip(valid, widths)]
        lines.append(f"  {v.ljust(14)} " + " ".join(row_costs))

    return "\n".join(lines) + "\n"


def render_sweep_markdown(reports: List[KernelAnalysisReport], cpu_matrix: List[str]) -> str:
    """Render full kernel sweep markdown table."""
    headers = [
        "Kernel", "Unroll", "GPRs (15)", "Mem (L/S/RMW)", "Mul Slack", "Straddles"
    ] + [f"{cpu} (cyc)" for cpu in cpu_matrix]

    lines = [
        "# Microarchitectural Kernel Sweep & Analysis",
        "",
        "| " + " | ".join(headers) + " |",
        "|:" + "|:".join(["---:"] * len(headers)) + "|",
    ]

    for r in reports:
        gpr_str = f"⚠️ {r.registers.gprs_used}" if r.registers.is_gpr_pressure_high else str(r.registers.gprs_used)
        mem_str = f"{r.memory.loads}/{r.memory.stores}/{r.memory.read_modify_writes}"
        if r.memory.read_modify_writes > 0:
            mem_str = f"⚠️ {mem_str}"

        if r.multiplier.mul_count == 0:
            mul_str = "N/A"
        elif r.multiplier.is_paired_pipeline:
            mul_str = f"⚡ paired ({r.multiplier.min_slack or 0})"
        elif r.multiplier.has_multiplier_stall:
            mul_str = "⚠️ 0 (stall)"
        else:
            mul_str = str(r.multiplier.min_slack)

        straddles_str = f"⚠️ {r.memory.cache_line_straddles}" if r.memory.cache_line_straddles > 0 else "0"

        cpu_cols = []
        for cpu in cpu_matrix:
            cyc = r.cpu_cycles.get(cpu)
            if cyc is not None:
                cpu_cols.append(f"{cyc:.2f}")
            else:
                cpu_cols.append("-")

        row = [
            f"`{r.kernel_name}`",
            f"{r.unroll_factor}x",
            gpr_str,
            mem_str,
            mul_str,
            straddles_str,
        ] + cpu_cols
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("> **Key / Metrics**:")
    lines.append("> - **Unroll**: Estimated unrolling factor across 64-bit limb accesses.")
    lines.append("> - **GPRs**: Number of distinct 64-bit GPRs used (15 max without rsp). ⚠️ indicates spill risk.")
    lines.append("> - **Mem (L/S/RMW)**: Memory loads / pure stores / read-modify-write instructions.")
    lines.append("> - **Mul Slack**: Distance between `mulx`/`mul` and first consumption (⚠️ 0 means multiplier latency stall).")
    lines.append("> - **Straddles**: Number of memory accesses crossing 64-byte boundaries unaligned.")
    lines.append("")
    return "\n".join(lines)


def render_diff_markdown(diff: KernelComparisonDiff) -> str:
    """Render side-by-side comparison between two kernel variants."""
    lines = [
        f"# Microarchitectural Comparison: `{diff.kernel_a.kernel_name}` vs `{diff.kernel_b.kernel_name}`",
        "",
        "| Metric | Variant A (`" + diff.kernel_a.kernel_name + "`) | Variant B (`" + diff.kernel_b.kernel_name + "`) | Delta (B - A) | Impact |",
        "|:---|:---:|:---:|:---:|:---|",
        f"| **Unroll Factor** | {diff.kernel_a.unroll_factor}x | {diff.kernel_b.unroll_factor}x | {diff.kernel_b.unroll_factor - diff.kernel_a.unroll_factor:+d} | Loop overhead |",
        f"| **GPRs Used** | {diff.kernel_a.registers.gprs_used} | {diff.kernel_b.registers.gprs_used} | {diff.gpr_delta:+d} | Register pressure |",
        f"| **Memory Loads** | {diff.kernel_a.memory.loads} | {diff.kernel_b.memory.loads} | {diff.load_delta:+d} | L1D Bandwidth |",
        f"| **Memory Stores** | {diff.kernel_a.memory.stores} | {diff.kernel_b.memory.stores} | {diff.store_delta:+d} | Store Buffer |",
        f"| **Read-Modify-Writes** | {diff.kernel_a.memory.read_modify_writes} | {diff.kernel_b.memory.read_modify_writes} | {diff.rmw_delta:+d} | Store-to-Load Stalls |",
        f"| **Multiplier Slack** | {diff.kernel_a.multiplier.min_slack} | {diff.kernel_b.multiplier.min_slack} | - | Latency Hiding |",
        "",
        "### Cycle Throughput Across CPU Targets",
        "",
        "| CPU Target | Variant A (cyc) | Variant B (cyc) | Speedup / Delta |",
        "|:---|:---:|:---:|:---:|",
    ]

    all_cpus = sorted(set(diff.kernel_a.cpu_cycles.keys()) | set(diff.kernel_b.cpu_cycles.keys()))
    for cpu in all_cpus:
        cyc_a = diff.kernel_a.cpu_cycles.get(cpu)
        cyc_b = diff.kernel_b.cpu_cycles.get(cpu)
        delta_str = "-"
        if cyc_a is not None and cyc_b is not None:
            delta = cyc_b - cyc_a
            speedup = (cyc_a / cyc_b - 1.0) * 100.0 if cyc_b > 0 else 0.0
            if delta < 0:
                delta_str = f"🚀 **{abs(speedup):.1f}% faster** ({delta:+.2f} cyc)"
            elif delta > 0:
                delta_str = f"⚠️ {speedup:.1f}% slower ({delta:+.2f} cyc)"
            else:
                delta_str = "Identical (0.00 cyc)"

        col_a = f"{cyc_a:.2f}" if cyc_a is not None else "-"
        col_b = f"{cyc_b:.2f}" if cyc_b is not None else "-"
        lines.append(f"| `{cpu}` | {col_a} | {col_b} | {delta_str} |")

    lines.append("")
    return "\n".join(lines)
