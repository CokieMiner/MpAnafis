"""Build once, discover the actual Cargo artifact, and execute isolated benchmark runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

from .divan import parse_catalog, parse_measurements, tree_rows
from .models import BenchmarkError
from .paths import ROOT, validate_output_path


def command(args: list[str], *, timeout: float, env: dict | None = None) -> subprocess.CompletedProcess:
    try:
        result = execute(args, timeout=timeout, env=env)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BenchmarkError(f"command failed: {args!r}: {error}") from error
    if result.returncode:
        raise BenchmarkError(f"command exited {result.returncode}: {args!r}\n{result.stderr[-8000:]}\n{result.stdout[-2000:]}")
    return result


def execute(args: list[str], *, timeout: float, env: dict | None = None) -> subprocess.CompletedProcess:
    """Stop the complete Cargo/benchmark process group on timeout or interruption."""
    with subprocess.Popen(args, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, start_new_session=os.name == "posix") as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            if isinstance(error, subprocess.TimeoutExpired):
                error.stdout, error.stderr = stdout, stderr
            raise
    return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)


def build_binary(features: str, *, timeout: float) -> Path:
    print(f"Building public_api with features {features!r}...", file=sys.stderr, flush=True)
    result = command(["cargo", "bench", "--bench", "public_api", "--features", features,
                      "--no-run", "--message-format=json"], timeout=timeout)
    artifacts = []
    for line in result.stdout.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (record.get("reason") == "compiler-artifact"
                and record.get("target", {}).get("name") == "public_api"
                and record.get("executable")):
            artifacts.append(Path(record["executable"]))
    if len(artifacts) != 1 or not artifacts[0].is_file():
        raise BenchmarkError("Cargo did not report exactly one public_api executable")
    return artifacts[0].resolve()


def discover(binary: Path, *, timeout: float):
    return parse_catalog(command([str(binary), "--list", "--color", "never"],
                                 timeout=timeout, env=measurement_environment(1)).stdout)


def measurement_environment(threads: int) -> dict[str, str]:
    # Explicit flags control Divan; inherited overrides must not silently turn
    # runs into tests or change the sample count, filters, timer, or counters.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("DIVAN_") and key != "NEXTEST"}
    env["RAYON_NUM_THREADS"] = str(threads)
    return env


def run_plan(binary: Path, plan: dict, output: Path, *, smoke: bool = False) -> dict:
    """Write raw output and metadata for each run before parsing any timings."""
    output = validate_output_path(output)
    settings = plan["settings"]
    cpus = settings["cpus"]
    if cpus:
        if not hasattr(os, "sched_getaffinity") or not set(cpus) <= os.sched_getaffinity(0):
            raise BenchmarkError(f"CPUs {cpus} are outside the current affinity set")
    if output.exists() and any(output.iterdir()):
        raise BenchmarkError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1, "plan": plan, "smoke": smoke,
        "platform": platform.platform(), "machine": platform.machine(),
        "host": platform.node(), "processor": processor_name(),
        "binary": str(binary), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "rustc": command(["rustc", "--version"], timeout=30).stdout.strip(),
        "git_head": command(["git", "rev-parse", "HEAD"], timeout=30).stdout.strip(),
        "git_status": command(["git", "status", "--porcelain"], timeout=30).stdout,
        "started_unix": time.time(), "status": "running", "commands": [],
    }
    metadata_path = output / "run.json"
    configuration = {key: metadata[key] for key in ("binary_sha256", "platform", "machine", "host", "processor", "rustc")}
    configuration.update(settings=settings, features=plan.get("features"),
                         environment={key: value for key, value in os.environ.items()
                                      if key.startswith(("MP_ANAFIS_", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"))})
    metadata["configuration"] = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:16]
    metadata["configuration_details"] = configuration
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    measurements = []
    env = measurement_environment(settings["threads"])
    try:
        for index, entry in enumerate(plan["runs"]):
            run_id = f"{index:04d}"
            args = [str(binary), "--test" if smoke else "--bench", entry["filter"],
                    "--color", "never", "--sample-count", str(settings["samples"]),
                    "--sample-size", str(settings["sample_size"])]
            if cpus:
                args = ["taskset", "--cpu-list", ",".join(map(str, cpus)), *args]
            metadata["commands"].append(args)
            print(f"[{index + 1}/{len(plan['runs'])}] {entry['path']} {entry['engine']}", flush=True)
            try:
                result = execute(args, env=env, timeout=settings["timeout"])
            except subprocess.TimeoutExpired as error:
                (output / f"{run_id}.stdout.txt").write_bytes(_bytes(error.stdout))
                (output / f"{run_id}.stderr.txt").write_bytes(_bytes(error.stderr))
                raise BenchmarkError(f"benchmark timed out: {entry['path']} {entry['engine']}") from error
            (output / f"{run_id}.stdout.txt").write_text(result.stdout)
            (output / f"{run_id}.stderr.txt").write_text(result.stderr)
            if result.returncode:
                raise BenchmarkError(f"benchmark exited {result.returncode}; see {output / (run_id + '.stderr.txt')}")
            if smoke:
                found = discover_from_smoke(result.stdout)
                if (entry["path"], entry["engine"]) not in found:
                    raise BenchmarkError(f"smoke filter ran no matching case: {entry['filter']}")
                expected = set(entry.get("arguments", plan["arguments"]))
                actual = {path[-1] for path, _, _ in tree_rows(result.stdout)
                          if len(path) >= 2 and path[-2] == entry["engine"]}
                if expected and actual != expected:
                    raise BenchmarkError(f"requested arguments are not all available: {entry['path']}")
            else:
                rows = parse_measurements(result.stdout, run=run_id)
                if any(row.path != entry["path"] or row.engine != entry["engine"] for row in rows):
                    raise BenchmarkError("benchmark filter included an unexpected function or engine")
                expected = set(entry.get("arguments", plan["arguments"]))
                if expected and {row.argument for row in rows} != expected:
                    raise BenchmarkError(f"requested arguments are not all available: {entry['path']}")
                measurements.extend(asdict(replace(row, configuration=metadata["configuration"])) for row in rows)
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        metadata["status"] = "complete"
    except (BenchmarkError, OSError, KeyboardInterrupt) as error:
        metadata["status"] = "failed"
        metadata["error"] = str(error)
        raise
    finally:
        metadata["finished_unix"] = time.time()
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        (output / "measurements.json").write_text(json.dumps(measurements, indent=2) + "\n")
    return metadata


def discover_from_smoke(text: str) -> set[tuple[str, str]]:
    return {(item.path, engine) for item in parse_catalog(text) for engine in item.engines}


def processor_name() -> str:
    """Record a CPU model when Linux exposes it; retain a portable fallback."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() in {"model name", "Processor", "cpu model"}:
                return value.strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _bytes(value: str | bytes | None) -> bytes:
    return value.encode() if isinstance(value, str) else value or b""
