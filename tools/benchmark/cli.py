"""One CLI for benchmark discovery, plans, execution, and reports."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import asdict
from pathlib import Path

from .catalog import execution_plan, select, validate_catalog
from .divan import parse_measurements
from .models import SUITES, BenchmarkError, Measurement
from .paths import validate_output_path
from .plan import validate_plan
from .publish import publish_run
from .report import write_report
from .runner import ROOT, build_binary, discover, run_plan
from .source_audit import check_tree


def main(argv: list[str] | None = None) -> int:
    parser = argument_parser()
    args = parser.parse_args(argv)
    try:
        if args.command in {"plan", "run", "report", "publish"}:
            args.output = validate_output_path(args.output, documentation=args.command == "publish")
        if args.command == "plan" and args.script:
            args.script = validate_output_path(args.script)
            args.results = validate_output_path(args.results)
        if args.command == "publish":
            destination = publish_run(args.input, args.output, args.name, args.description)
            print(f"Wrote documentation record to {destination}")
            return 0
        if args.command == "check":
            findings = check_tree(ROOT / "benches/public_api")
            if findings:
                raise BenchmarkError("benchmark source findings:\n" + "\n".join(f"{f['path']}:{f['line']} [{f['kind']}] {f['detail']}" for f in findings))
            if args.source_only:
                print("Benchmark source findings: 0")
                return 0
        if args.command == "report":
            rows = []
            for index, path in enumerate(args.input):
                if path.suffix == ".json":
                    data = json.loads(path.read_text())
                    for entry in data:
                        entry["run"] = f"{index}:{entry['run']}"
                        rows.append(Measurement(**entry))
                else:
                    rows.extend(parse_measurements(path.read_text(), run=str(index), suite=args.suite))
            write_report(rows, args.output, plots=args.plot)
            return 0

        plan = json.loads(args.plan.read_text()) if getattr(args, "plan", None) else None
        if plan is not None and not isinstance(plan, dict):
            raise BenchmarkError("plan must be a JSON object")
        features = plan["features"] if plan is not None else args.features
        if not isinstance(features, str):
            raise BenchmarkError("features must be a Cargo feature string")
        binary = args.binary.resolve() if args.binary else build_binary(features, timeout=args.build_timeout)
        catalog = discover(binary, timeout=60)
        selected = select(catalog, args.case)
        validate_catalog(selected, require_comparison=args.require_comparison)
        if args.command in {"list", "check"}:
            if args.json:
                print(json.dumps([asdict(item) for item in selected], indent=2))
            else:
                for item in selected:
                    print(f"{item.path}\t{','.join(item.engines)}")
                print(f"{len(selected)} function/scenario groups", file=sys.stderr)
            return 0
        if plan is None:
            available = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
            cpus = [int(value) for value in args.cpus.split(",")] if args.cpus else available[:args.threads]
            plan = {
                "schema_version": 1, "features": features, "arguments": args.arg,
                "settings": {"cpus": cpus, "threads": args.threads, "samples": args.samples,
                             "sample_size": args.sample_size, "timeout": args.timeout},
                "runs": execution_plan(selected, args.arg, rounds=args.rounds, compare=not args.mp_only),
            }
        validate_plan(plan, catalog)
        if args.command == "plan":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(plan, indent=2) + "\n")
            if args.script:
                invocation = [sys.executable, str(ROOT / "tools/bench.py"), "run", "--plan",
                              str(args.output.resolve()), "--output", str(args.results.resolve())]
                if args.binary:
                    invocation.extend(["--binary", str(binary)])
                args.script.parent.mkdir(parents=True, exist_ok=True)
                args.script.write_text("#!/bin/sh\nset -eu\nexec " + shlex.join(invocation) + ' "$@"\n')
                args.script.chmod(args.script.stat().st_mode | 0o100)
            print(f"Wrote {len(plan['runs'])} isolated runs to {args.output}")
            return 0
        run_plan(binary, plan, args.output, smoke=args.smoke)
        if not args.smoke:
            rows = [Measurement(**row) for row in json.loads((args.output / "measurements.json").read_text())]
            write_report(rows, args.output, plots=args.plot)
        return 0
    except (BenchmarkError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"bench: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("bench: interrupted; partial results are retained", file=sys.stderr)
        return 130


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("list", "check", "plan", "run"):
        command = subcommands.add_parser(name)
        command.add_argument("--binary", type=Path, help="reuse an explicitly supplied benchmark executable")
        command.add_argument("--features", default="std,rayon", help="Cargo features; add _internal-tune for FLINT comparisons")
        command.add_argument("--build-timeout", type=float, default=1800)
        command.add_argument("--case", action="append", default=[], help="full-path glob; repeat to select multiple functions")
        command.add_argument("--require-comparison", action="store_true")
        if name in {"list", "check"}:
            command.add_argument("--json", action="store_true")
            if name == "check":
                command.add_argument("--source-only", action="store_true", help="audit declarations without building or running Rust")
            continue
        command.add_argument("--arg", action="append", default=[], help="exact numeric benchmark argument; repeat for multiple sizes")
        command.add_argument("--rounds", type=int, default=1, help="number of A/B/B/A rounds")
        command.add_argument("--cpus", help="comma-separated allowed CPU IDs; defaults to the first available CPUs")
        command.add_argument("--threads", type=int, default=1, help="Rayon worker count; distinct from Divan contention threads")
        command.add_argument("--samples", type=int, default=30)
        command.add_argument("--sample-size", type=int, default=32)
        command.add_argument("--timeout", type=float, default=120, help="wall-clock limit per isolated run")
        command.add_argument("--mp-only", action="store_true")
        command.add_argument("--output", type=Path, required=True)
        if name == "plan":
            command.add_argument("--script", type=Path, help="also generate a shell entry point for this plan")
            command.add_argument("--results", type=Path, default=Path("target/bench-results"))
        else:
            command.add_argument("--plan", type=Path, help="execute an existing generated JSON plan")
            command.add_argument("--smoke", action="store_true", help="execute once without reporting timings")
            command.add_argument("--plot", action="store_true")
    report = subcommands.add_parser("report")
    report.add_argument("input", nargs="+", type=Path)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--plot", action="store_true")
    report.add_argument("--suite", choices=sorted(SUITES), default="public_api", help="suite of raw Divan text inputs; JSON records carry their own suite")
    publish = subcommands.add_parser("publish", help="export time-versus-size figures and measurement details into docs")
    publish.add_argument("input", type=Path, help="completed runner output directory")
    publish.add_argument("--name", required=True, help="unique documentation record name")
    publish.add_argument("--description", required=True, help="scope and interpretation of these measurements")
    publish.add_argument("--output", type=Path, default=ROOT / "docs/int/benchmarks", help="documentation benchmark root")
    return parser
