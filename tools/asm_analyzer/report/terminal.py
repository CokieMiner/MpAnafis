"""Rich terminal visualization, ANSI colors, and Unicode box-drawing formatters."""

from __future__ import annotations

from typing import List

from ..features.suggestions import OptimizationSuggestion, SuggestionSeverity
from ..types import KernelComparisonDiff

# ANSI Color Codes
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
WHITE = "\033[37m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"


def colorize(text: str, color: str, enable_color: bool = True) -> str:
    """Wrap text in ANSI color escape codes when enabled."""
    if not enable_color:
        return text
    return f"{color}{text}{RESET}"


def render_terminal_diff(diff: KernelComparisonDiff, enable_color: bool = True) -> str:
    """Render a beautifully formatted side-by-side terminal diff."""
    title = f" Microarchitectural Diff: {diff.kernel_a.kernel_name} vs {diff.kernel_b.kernel_name} "
    width = 82
    border = "=" * width

    lines = [
        colorize(border, CYAN, enable_color),
        colorize(title.center(width, " "), BOLD + WHITE, enable_color),
        colorize(border, CYAN, enable_color),
        "",
        f" {'Metric':<26} | {'Variant A':<14} | {'Variant B':<14} | {'Delta':<18}",
        "-" * width,
    ]

    def _format_row(metric: str, val_a: str, val_b: str, delta_val: int, inverse: bool = False) -> str:
        delta_str = f"{delta_val:+d}" if delta_val != 0 else "0"
        if delta_val < 0:
            color = RED if inverse else GREEN
        elif delta_val > 0:
            color = GREEN if inverse else RED
        else:
            color = WHITE
        styled_delta = colorize(f"{delta_str:<18}", color, enable_color)
        return f" {metric:<26} | {val_a:<14} | {val_b:<14} | {styled_delta}"

    # General instructions & unroll
    inst_delta = diff.kernel_b.uop_cache.instruction_count - diff.kernel_a.uop_cache.instruction_count
    lines.append(_format_row("Instruction Count", str(diff.kernel_a.uop_cache.instruction_count), str(diff.kernel_b.uop_cache.instruction_count), inst_delta))
    lines.append(_format_row("Unroll Factor", f"{diff.kernel_a.unroll_factor}x", f"{diff.kernel_b.unroll_factor}x",
                            diff.kernel_b.unroll_factor - diff.kernel_a.unroll_factor, inverse=True))

    # Registers & Multipliers
    lines.append(_format_row("GPRs Used", str(diff.kernel_a.registers.gprs_used), str(diff.kernel_b.registers.gprs_used), diff.gpr_delta))
    lines.append(_format_row("Multipliers (mul/mulx)", str(diff.kernel_a.multiplier.mul_count), str(diff.kernel_b.multiplier.mul_count),
                            diff.kernel_b.multiplier.mul_count - diff.kernel_a.multiplier.mul_count))

    slack_a = f"{diff.kernel_a.multiplier.min_slack} cyc" if diff.kernel_a.multiplier.min_slack is not None else "N/A"
    slack_b = f"{diff.kernel_b.multiplier.min_slack} cyc" if diff.kernel_b.multiplier.min_slack is not None else "N/A"
    lines.append(f" {'Multiplier Min Slack':<26} | {slack_a:<14} | {slack_b:<14} | {'--':<18}")

    # Memory Accesses
    lines.append(_format_row("Memory Loads", str(diff.kernel_a.memory.loads), str(diff.kernel_b.memory.loads), diff.load_delta))
    lines.append(_format_row("Memory Stores", str(diff.kernel_a.memory.stores), str(diff.kernel_b.memory.stores), diff.store_delta))
    lines.append(_format_row("Read-Modify-Writes", str(diff.kernel_a.memory.read_modify_writes),
                            str(diff.kernel_b.memory.read_modify_writes), diff.rmw_delta))

    # Execution units & bottlenecks
    bot_a = diff.kernel_a.port_pressure.bottleneck_port or "None"
    bot_b = diff.kernel_b.port_pressure.bottleneck_port or "None"
    lines.append(f" {'Port Bottleneck':<26} | {bot_a[:14]:<14} | {bot_b[:14]:<14} | {'--':<18}")

    lines.append("-" * width)
    lines.append(colorize(" Cycle Throughput Across CPU Targets", BOLD + WHITE, enable_color))
    lines.append("-" * width)

    all_cpus = sorted(set(diff.kernel_a.cpu_cycles.keys()) | set(diff.kernel_b.cpu_cycles.keys()))
    for cpu in all_cpus:
        cyc_a = diff.kernel_a.cpu_cycles.get(cpu)
        cyc_b = diff.kernel_b.cpu_cycles.get(cpu)
        if cyc_a is not None and cyc_b is not None:
            delta = cyc_b - cyc_a
            speedup = (cyc_a / cyc_b - 1.0) * 100.0 if cyc_b > 0 else 0.0
            if delta < 0:
                delta_str = colorize(f"🚀 {abs(speedup):.1f}% faster ({delta:+.2f} cyc)", GREEN, enable_color)
            elif delta > 0:
                delta_str = colorize(f"⚠️ {speedup:.1f}% slower ({delta:+.2f} cyc)", RED, enable_color)
            else:
                delta_str = "Identical"
            lines.append(f" {cpu:<26} | {cyc_a:<14.2f} | {cyc_b:<14.2f} | {delta_str}")

    lines.append(colorize(border, CYAN, enable_color))
    return "\n".join(lines)


def render_terminal_suggestions(suggestions: List[OptimizationSuggestion], enable_color: bool = True) -> str:
    """Render optimization suggestions as a rich terminal report."""
    if not suggestions:
        return colorize("✅ No microarchitectural hazards found! Assembly kernel is clean.", GREEN, enable_color)

    lines = [
        colorize(f"Found {len(suggestions)} Optimization Suggestions:", BOLD + WHITE, enable_color),
        "",
    ]

    for s in suggestions:
        if s.severity == SuggestionSeverity.CRITICAL:
            sev_tag = colorize("[CRITICAL]", BOLD + RED, enable_color)
        elif s.severity == SuggestionSeverity.WARNING:
            sev_tag = colorize("[WARNING]", BOLD + YELLOW, enable_color)
        else:
            sev_tag = colorize("[INFO]", BOLD + BLUE, enable_color)

        lines.append(f" {sev_tag} {colorize(s.rule_id, BOLD, enable_color)}: {s.title}")
        lines.append(f"          {s.description}")
        lines.append(f"          Action: {colorize(s.suggested_fix, CYAN, enable_color)}")
        if hasattr(s, "estimated_speedup") and getattr(s, "estimated_speedup", None):
            lines.append(f"          Est. Gain: {colorize(s.estimated_speedup, GREEN, enable_color)}")
        lines.append("")

    return "\n".join(lines)
