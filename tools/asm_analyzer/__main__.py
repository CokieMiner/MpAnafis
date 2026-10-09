#!/usr/bin/env python3
"""Unified CLI entrypoint for the Assembly Analyzer (`asm_analyzer`).

Usage:
    # Scan a kernel and generate actionable optimization suggestions
    python3 -m asm_analyzer suggest src/int/logic/unsigned/math/arch/add_mul_limbs_unchecked/x86_64_adx.rs

    # Search for low-regret topological instruction schedules
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
import subprocess
import sys

from .commands import (
    run_analyze,
    run_check,
    run_diff,
    run_optimize,
    run_pmu,
    run_search,
    run_suggest,
    run_sweep,
)
from .models import (
    DEFAULT_BACKENDS,
    DEFAULT_MATRIX,
    parse_backends,
    parse_cpus,
)


def main(argv: list[str] | None = None) -> int:
    """Parse and dispatch one analyzer command."""
    try:
        return _dispatch(argv)
    except (OSError, UnicodeError, ValueError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


def _dispatch(argv: list[str] | None) -> int:
    """Validate arguments and invoke the selected command."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        cpus = parse_cpus(args.cpu) if args.cpu is not None else None
        backends = parse_backends(args.backend)
        if cpus == [] or not backends:
            parser.error("CPU and backend lists must not be empty")
    except KeyError as error:
        parser.error(str(error.args[0]))
    for option in ("candidates", "hardware_shortlist"):
        if hasattr(args, option) and getattr(args, option) < 1:
            parser.error(f"--{option.replace('_', '-')} must be positive")

    if args.command in ("suggest", "audit"):
        return run_suggest(
            args.kernel,
            use_wsl=args.wsl,
            enable_color=args.color,
            as_json=args.json,
        )
    if args.command == "search":
        return run_search(
            args.kernel,
            cpus=cpus,
            backend_names=backends,
            candidates=args.candidates,
            seed=args.seed,
            use_wsl=args.wsl,
            no_alias=args.no_alias,
            disjoint_pointers=args.disjoint_pointers,
            as_json=args.json,
        )
    if args.command == "diff":
        return run_diff(
            args.kernel_a,
            args.kernel_b,
            cpus=cpus,
            backend_names=backends,
            use_wsl=args.wsl,
            as_json=args.json,
        )
    if args.command == "optimize":
        return run_optimize(
            target_path=args.path,
            cpus=cpus,
            backend_names=backends,
            candidates=args.candidates,
            hardware_shortlist=args.hardware_shortlist,
            seed=args.seed,
            use_wsl=args.wsl,
            hardware=args.hardware,
            no_alias=args.no_alias,
            disjoint_pointers=args.disjoint_pointers,
            apply_confirmed=args.apply_confirmed,
            as_json=args.json,
        )
    if args.command == "sweep":
        return run_sweep(
            target_path=args.path,
            cpus=cpus,
            backend_names=backends,
            use_wsl=args.wsl,
            markdown=args.markdown,
            as_json=args.json,
        )
    if args.command == "analyze":
        target = args.kernel or args.asm
        if not target:
            parser.error("analyze requires a kernel path or --asm")
        return run_analyze(
            target,
            cpus=cpus,
            backend_names=backends,
            use_wsl=args.wsl,
            as_json=args.json,
        )
    if args.command == "pmu":
        return run_pmu(args.cmd, as_json=args.json)
    if args.command == "check":
        return run_check(backends, cpus, use_wsl=args.wsl, as_json=args.json)

    parser.print_help()
    return 1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="asm_analyzer",
        description="Microarchitectural assembly analysis, pipeline simulation, "
                    "optimization suggestions, DAG scheduler search, side-by-side diffing, and PMU hardware profiling suite.",
    )

    sub = p.add_subparsers(dest="command", required=True)

    # suggest / audit
    sug = sub.add_parser("suggest", parents=[_common_parser()], help="analyze kernel and generate actionable optimization advice")
    sug.add_argument("kernel", help="path to kernel file (.rs or .s)")
    sug.add_argument("--json", action="store_true", help="output JSON")

    aud = sub.add_parser("audit", parents=[_common_parser()], help="audit kernel and generate actionable optimization advice (alias for suggest)")
    aud.add_argument("kernel", help="path to kernel file (.rs or .s)")
    aud.add_argument("--json", action="store_true", help="output JSON")

    # search
    src = sub.add_parser(
        "search",
        parents=[_common_parser()],
        help="search for low-regret topological instruction schedules",
    )
    src.add_argument("kernel", help="path to kernel file (.rs or .s)")
    src.add_argument("--candidates", type=int, default=50, help="number of topological candidates to evaluate")
    src.add_argument("--seed", type=int, default=42, help="random seed")
    src.add_argument("--no-alias", action="store_true", help="assume distinct pointer base registers do not alias")
    src.add_argument("--disjoint-pointers", default=None, help="comma-separated list of non-aliasing pointer registers (e.g. 'rsi,rdi')")
    src.add_argument("--json", action="store_true", help="output JSON")

    # diff
    df = sub.add_parser("diff", parents=[_common_parser()], help="side-by-side microarchitectural diff between two kernels")
    df.add_argument("kernel_a", help="path to first kernel file (.rs or .s)")
    df.add_argument("kernel_b", help="path to second kernel file (.rs or .s)")
    df.add_argument("--json", action="store_true", help="output JSON")

    # optimize
    opt = sub.add_parser(
        "optimize",
        parents=[_common_parser()],
        help="batch-search eligible kernels with optional native holdout confirmation",
    )
    opt.add_argument("path", nargs="?", default=None, help="optional kernel file or architecture directory")
    opt.add_argument("--candidates", type=int, default=50, help="schedule candidates generated per kernel")
    opt.add_argument("--hardware-shortlist", type=int, default=4, help="statically ranked alternatives sent to hardware screening")
    opt.add_argument("--seed", type=int, default=42, help="random schedule seed")
    opt.add_argument("--no-alias", action="store_true", help="assume distinct pointer base registers do not alias")
    opt.add_argument("--disjoint-pointers", default=None, help="comma-separated list of non-aliasing pointer registers (e.g. 'rsi,rdi')")
    opt.add_argument("--hardware", action="store_true", help="run corrected screening and independent holdout on the exact local host")
    opt.add_argument(
        "--apply-confirmed",
        action="store_true",
        help="atomically apply only independently confirmed schedules that map exactly to Rust asm! lines",
    )
    opt.add_argument("--json", action="store_true", help="output JSON")

    # sweep
    sw = sub.add_parser("sweep", parents=[_common_parser()], help="sweep and analyze all x86_64 kernels across CPUs")
    sw.add_argument("path", nargs="?", default=None, help="optional path to directory or specific kernel")
    sw.add_argument("--markdown", action="store_true", help="render Markdown table")
    sw.add_argument("--json", action="store_true", help="output JSON")

    # analyze
    an = sub.add_parser("analyze", parents=[_common_parser()], help="analyze a single assembly file (.s) or Rust kernel (.rs)")
    an.add_argument("kernel", nargs="?", default=None, help="path to kernel file (.rs or .s)")
    an.add_argument("--asm", default=None, help="optional path to assembly file (.s)")
    an.add_argument("--json", action="store_true", help="output JSON")

    # pmu
    pmu = sub.add_parser("pmu", parents=[_common_parser()], help="run command under hardware PMU counters (Linux perf)")
    pmu.add_argument("cmd", nargs=argparse.REMAINDER, help="command to execute and profile")
    pmu.add_argument("--json", action="store_true", help="output JSON")

    # check
    chk = sub.add_parser("check", parents=[_common_parser(",".join(DEFAULT_MATRIX))], help="probe available simulator backends and CPU support")
    chk.add_argument("--json", action="store_true", help="output JSON")

    return p


def _common_parser(cpu_default: str | None = None) -> argparse.ArgumentParser:
    """Build an independent common-options parent for one subcommand.

    ``argparse`` reuses action objects from parent parsers.  Each subcommand
    needs its own parent so per-command CPU defaults stay independent.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--wsl",
        action="store_true",
        help="run toolchain binaries through WSL",
    )
    common.add_argument(
        "--color",
        action="store_true",
        default=sys.stdout.isatty(),
        help="enable rich ANSI color output in terminal",
    )
    common.add_argument(
        "--no-color",
        action="store_false",
        dest="color",
        help="disable ANSI color output",
    )
    common.add_argument(
        "--backend",
        default=",".join(DEFAULT_BACKENDS),
        help="comma-separated backends (default: llvm-mca,osaca,uica)",
    )
    common.add_argument(
        "--cpu",
        default=cpu_default,
        help="comma-separated CPU models (default: target-compatible matrix)",
    )
    return common


if __name__ == "__main__":
    sys.exit(main())
