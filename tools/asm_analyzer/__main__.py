#!/usr/bin/env python3
"""Unified CLI entrypoint for the Assembly Analyzer (`asm_analyzer`).

Usage:
    # Scan a kernel and generate actionable optimization suggestions
    python3 -m asm_analyzer suggest src/int/logic/unsigned/math/arch/add_mul_limbs_unchecked/x86_64_adx.rs

    # Search for optimal topological instruction schedules
    python3 -m asm_analyzer search src/int/logic/unsigned/math/arch/add_mul_limbs_unchecked/x86_64_adx.rs

    # Side-by-side comparison between two kernel variants
    python3 -m asm_analyzer diff path/to/kernel_a.rs path/to/kernel_b.rs

    # Sweep and analyze all x86 kernels
    python3 -m asm_analyzer sweep --markdown

    # Single-file analysis
    python3 -m asm_analyzer analyze --asm path/to/kernel.s

    # Hardware PMU profiling
    python3 -m asm_analyzer pmu -- cargo test --lib -- fused_multiply

    # Backend and CPU model probe
    python3 -m asm_analyzer check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure tools/ root is in sys.path
_TOOLS_DIR = Path(__file__).resolve().parent.parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from asm_analyzer.commands import (
    run_analyze,
    run_calibrate,
    run_check,
    run_diff,
    run_pmu,
    run_search,
    run_suggest,
    run_sweep,
)
from asm_analyzer.models import DEFAULT_BACKENDS, DEFAULT_MATRIX, parse_backends, parse_cpus


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--wsl", action="store_true", help="run toolchain binaries through WSL")
    common.add_argument("--color", action="store_true", default=sys.stdout.isatty(),
                        help="enable rich ANSI color output in terminal")
    common.add_argument("--no-color", action="store_false", dest="color",
                        help="disable ANSI color output")
    common.add_argument("--backend", default=",".join(DEFAULT_BACKENDS),
                        help="comma-separated backends (default: llvm-mca,osaca,uica)")
    common.add_argument("--cpu", default=",".join(DEFAULT_MATRIX),
                        help="comma-separated CPU models")

    p = argparse.ArgumentParser(
        prog="asm_analyzer",
        description="Microarchitectural assembly analysis, pipeline simulation, "
                    "optimization suggestions, DAG scheduler search, side-by-side diffing, and PMU hardware profiling suite.",
        parents=[common],
    )

    sub = p.add_subparsers(dest="command", required=True)

    # suggest / audit
    sug = sub.add_parser("suggest", parents=[common], help="analyze kernel and generate actionable optimization advice")
    sug.add_argument("kernel", help="path to kernel file (.rs or .s)")
    sug.add_argument("--json", action="store_true", help="output JSON")

    aud = sub.add_parser("audit", parents=[common], help="audit kernel and generate actionable optimization advice (alias for suggest)")
    aud.add_argument("kernel", help="path to kernel file (.rs or .s)")
    aud.add_argument("--json", action="store_true", help="output JSON")

    # search
    src = sub.add_parser("search", parents=[common], help="search for optimal topological instruction schedules")
    src.add_argument("kernel", help="path to kernel file (.rs or .s)")
    src.add_argument("--candidates", type=int, default=50, help="number of topological candidates to evaluate")
    src.add_argument("--seed", type=int, default=42, help="random seed")
    src.add_argument("--json", action="store_true", help="output JSON")

    # diff
    df = sub.add_parser("diff", parents=[common], help="side-by-side microarchitectural diff between two kernels")
    df.add_argument("kernel_a", help="path to first kernel file (.rs or .s)")
    df.add_argument("kernel_b", help="path to second kernel file (.rs or .s)")
    df.add_argument("--json", action="store_true", help="output JSON")

    # sweep
    sw = sub.add_parser("sweep", parents=[common], help="sweep and analyze all x86_64 kernels across CPUs")
    sw.add_argument("path", nargs="?", default=None, help="optional path to directory or specific kernel")
    sw.add_argument("--markdown", action="store_true", help="render Markdown table")
    sw.add_argument("--json", action="store_true", help="output JSON")

    # analyze
    an = sub.add_parser("analyze", parents=[common], help="analyze a single assembly file (.s) or Rust kernel (.rs)")
    an.add_argument("kernel", nargs="?", default=None, help="path to kernel file (.rs or .s)")
    an.add_argument("--asm", default=None, help="optional path to assembly file (.s)")
    an.add_argument("--json", action="store_true", help="output JSON")

    # pmu
    pmu = sub.add_parser("pmu", parents=[common], help="run command under hardware PMU counters (Linux perf)")
    pmu.add_argument("cmd", nargs=argparse.REMAINDER, help="command to execute and profile")
    pmu.add_argument("--json", action="store_true", help="output JSON")

    # calibrate
    cal = sub.add_parser("calibrate", parents=[common], help="measure kernels on host and update empirical dataset and error models")
    cal.add_argument("kernel", nargs="?", default=None, help="optional path to specific kernel file")
    cal.add_argument("--runs", type=int, default=5, help="number of benchmark repetitions (default: 5)")
    cal.add_argument("--json", action="store_true", help="output JSON")

    # check
    chk = sub.add_parser("check", parents=[common], help="probe available simulator backends and CPU support")
    chk.add_argument("--json", action="store_true", help="output JSON")

    return p


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    cpus = parse_cpus(args.cpu)
    backends = parse_backends(args.backend)

    if args.command in ("suggest", "audit"):
        return run_suggest(args.kernel, use_wsl=args.wsl, enable_color=args.color, as_json=args.json)
    elif args.command == "search":
        return run_search(args.kernel, cpus=cpus, candidates=args.candidates,
                          seed=args.seed, use_wsl=args.wsl, as_json=args.json)
    elif args.command == "diff":
        return run_diff(args.kernel_a, args.kernel_b, cpus=cpus,
                        use_wsl=args.wsl, as_json=args.json)
    elif args.command == "sweep":
        return run_sweep(target_path=args.path, cpus=cpus, use_wsl=args.wsl,
                          markdown=args.markdown, as_json=args.json)
    elif args.command == "analyze":
        target = args.kernel or args.asm
        if not target:
            parser.error("analyze requires a kernel path or --asm")
        return run_analyze(target, cpus=cpus, use_wsl=args.wsl, as_json=args.json)
    elif args.command == "pmu":
        return run_pmu(args.cmd, as_json=args.json)
    elif args.command == "calibrate":
        return run_calibrate(args.kernel, backend_name=args.backend, cpu_name=args.cpu,
                             use_wsl=args.wsl, runs=args.runs, as_json=args.json)
    elif args.command == "check":
        return run_check(backends, cpus, use_wsl=args.wsl, as_json=args.json)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
