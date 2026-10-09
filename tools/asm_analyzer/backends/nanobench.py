#!/usr/bin/env python3
"""nanoBench hardware PMU cycle measurement backend.

Measures real CPU cycles for isolated assembly blocks on the host machine
via the nanoBench kernel module / user driver.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Set

from ..analyzer import Analyzer, KernelReport
from ..asm_util import GPR_ALIAS_MAP, classify_regs, host_cpu_name, named_registers
from ..models import CpuSpec
from ..search.ast import get_instruction_spec, parse_line
from .nanobench_flow import (
    LOOP_MNEMONICS as _LOOP_MNEMONICS,
    asm_flow as _asm_flow,
    canonical_reg as _canonical_reg,
    validate_measurable as _validate_measurable,
)

IS_WINDOWS = os.name == "nt"
REPO_ROOT = Path(__file__).resolve().parents[3]

#: Deterministic seed for live-in scalar registers in nanoBench init code.
#: Zeroing lets ``dec``/``js``-style internal loops exit immediately (the
#: harness then times its own scaffolding instead of the kernel), while an
#: unseeded counter inherited from a nanoBench buffer address makes ``loop``
#: iterate astronomically. A small positive constant bounds every
#: scalar-fed counted back-edge to a handful of deterministic iterations.
MEASURE_SEED = 2


class NanobenchAnalyzer(Analyzer):
    """Empirical hardware cycle counter backend via nanoBench."""

    name = "nanobench"

    def __init__(self) -> None:
        self._bin = _discover_nanobench_binary()
        self._host = _host_cpu()

    def available(self) -> bool:
        """Return True if nanoBench executable is present."""
        return self._bin is not None

    def supports(self, cpu: CpuSpec) -> bool:
        """True if the target CPU matches the physical host machine."""
        return self.available() and self._host != "unknown" and cpu.name == self._host

    def analyze(self, asm_code: str, cpu: CpuSpec, iterations: int = 200) -> Optional[float]:
        """Measure real CPU cycles for an assembly block."""
        report = self.analyze_report(asm_code, cpu, iterations=iterations)
        return report.cycles

    def analyze_report(self, asm_code: str, cpu: CpuSpec, iterations: int = 200) -> KernelReport:
        """Measure assembly block and return rich empirical report."""
        if not self.available():
            return KernelReport(
                backend=self.name,
                cpu=cpu.name,
                ok=False,
                note=(
                    "nanoBench executable not found. Install from "
                    "https://github.com/andreas-abel/nanoBench"
                ),
            )
        if not self.supports(cpu):
            return KernelReport(
                backend=self.name,
                cpu=cpu.name,
                ok=False,
                note=f"nanoBench measures the host ({self._host}) only, not foreign target '{cpu.name}'",
            )

        if not all(shutil.which(tool) for tool in ("as", "objcopy", "objdump")):
            return KernelReport(
                backend=self.name,
                cpu=cpu.name,
                ok=False,
                note="binutils (as/objcopy/objdump) not found in PATH",
            )

        logical_cpu = _single_cpu_affinity()
        if logical_cpu is None:
            return KernelReport(
                backend=self.name,
                cpu=cpu.name,
                ok=False,
                note="nanoBench requires process affinity to exactly one logical CPU",
            )
        msr_device = Path(f"/dev/cpu/{logical_cpu}/msr")
        if not os.access(msr_device, os.R_OK | os.W_OK):
            return KernelReport(
                backend=self.name,
                cpu=cpu.name,
                ok=False,
                note=(
                    f"nanoBench requires read/write access to {msr_device}; "
                    "run the pinned analyzer command with sudo"
                ),
            )

        pointers, scalars = classify_regs(asm_code)
        refusal = _validate_measurable(asm_code, pointers, scalars)
        if refusal is not None:
            return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note=refusal)

        unroll, loop_count = _select_shape(asm_code, iterations, scalars)
        temporary = tempfile.TemporaryDirectory(prefix="asm-nanobench-")
        work = Path(temporary.name)
        asm_file = work / "kernel.s"
        obj_file = work / "kernel.o"
        bin_file = work / "kernel.bin"
        init_asm_file = work / "init.s"
        init_obj_file = work / "init.o"
        init_bin_file = work / "init.bin"

        try:
            asm_file.write_text(f".text\n{asm_code.rstrip()}\n", encoding="utf-8")
            init_asm_file.write_text(_initialization_asm(asm_code), encoding="utf-8")
            error = _assemble_binary(asm_file, obj_file, bin_file)
            if error:
                return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note=error)
            error = _assemble_binary(init_asm_file, init_obj_file, init_bin_file)
            if error:
                return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note=error)

            cmd = [
                str(self._bin),
                "-code", str(bin_file),
                "-code_init", str(init_bin_file),
                "-unroll_count", str(unroll),
                "-loop_count", str(loop_count),
                "-n_measurements", "11",
                "-warm_up_count", "5",
                "-initial_warm_up_count", "5",
                "-cpu", str(logical_cpu),
                "-median",
                "-basic_mode",
                "-fixed_counters",
            ]

            r = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30)
            text = r.stdout + "\n" + r.stderr
            if r.returncode == 0:
                measurement = _parse_cycle_measurement(text)
                if measurement is not None:
                    counter, cycles = measurement
                    note = (
                        f"counter={counter}; nanoBench normalized over "
                        f"unroll={unroll}, loop_count={loop_count}"
                    )
                    if not math.isfinite(cycles) or cycles <= 0:
                        return KernelReport(
                            backend=self.name,
                            cpu=cpu.name,
                            ok=False,
                            note=f"{note}; rejected non-positive measurement",
                            cycles=cycles,
                            raw_output=text,
                        )
                    return KernelReport(
                        backend=self.name,
                        cpu=cpu.name,
                        ok=True,
                        note=note,
                        cycles=cycles,
                        raw_output=text,
                    )
                return KernelReport(
                    backend=self.name,
                    cpu=cpu.name,
                    ok=False,
                    note=f"Could not parse CORE_CYCLES or RDTSC from output:\n{text[:300]}",
                )

            return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note=f"nanoBench execution error:\n{text[:300]}")
        except subprocess.TimeoutExpired:
            return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note="nanoBench timed out after 30 seconds")
        except Exception as err:
            return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note=f"nanoBench exception: {err}")
        finally:
            temporary.cleanup()


def _assemble_binary(source: Path, obj: Path, binary: Path) -> str:
    assembled = subprocess.run(
        ["as", str(source), "-o", str(obj)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if assembled.returncode != 0:
        return f"as failed for {source.name}: {assembled.stderr[:200]}"
    relocations = subprocess.run(
        ["objdump", "-r", str(obj)], capture_output=True, text=True,
        check=False, timeout=30,
    )
    if relocations.returncode != 0:
        return f"could not inspect relocations for {source.name}"
    if re.search(r"(?m)^\s*[0-9a-fA-F]+\s+R_", relocations.stdout):
        return "nanoBench snippets must not contain unresolved relocations"
    extracted = subprocess.run(
        ["objcopy", "-O", "binary", "-j", ".text", str(obj), str(binary)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if extracted.returncode != 0 or not binary.exists():
        return f"objcopy failed for {source.name}: {extracted.stderr[:200]}"
    return ""


def _initialization_asm(asm_code: str) -> str:
    """Build untimed deterministic register and memory initialization.

    Pointer registers resolve to nanoBench-provided 1 MiB buffer centers
    (``r14``/``rdi``/``rsi``/``rbp`` natively, the rest derived from ``r14``).
    Scalar live-ins receive :data:`MEASURE_SEED` instead of zero so internal
    counted loops execute a small deterministic trip count rather than
    skipping (zero) or running away (buffer-address residue).
    """
    pointers, scalars = classify_regs(asm_code)
    # nanoBench already initializes these registers to the centers of distinct
    # 1 MiB allocations. Preserve them so common source/destination operands do
    # not accidentally benchmark an aliasing case.
    memory_bases = {"r14", "rdi", "rsi", "rbp"}
    lines = [".text"]
    offset = 4096
    for pointer in sorted(pointers - memory_bases):
        lines.append(f"    leaq {offset}(%r14), %{pointer}")
        offset += 4096

    # Seed implicit integer inputs as well as textual register operands.
    for line in asm_code.splitlines():
        if instruction := parse_line(line):
            scalars.update(get_instruction_spec(instruction).uses & set(GPR_ALIAS_MAP.values()))
    scalars.discard("rsp")
    for scalar in sorted(scalars - pointers):
        lines.append(f"    movq ${MEASURE_SEED}, %{scalar}")
    # Establish deterministic CF and OF even for kernels with no scalar inputs.
    lines.append("    testq %rsp, %rsp")
    return "\n".join(lines) + "\n"


#: Registers nanoBench's counter reads clobber between copies (RAX/RCX/RDX
#: always; R8-R13 on AMD, R8-R11 on Intel — the union is used). A reseed
#: source in this set replays init state only on the first copy, so it can
#: never establish per-execution idempotency.
_CLOBBERED_BETWEEN_COPIES = frozenset({
    "rax", "rcx", "rdx",
    "r8", "r9", "r10", "r11", "r12", "r13",
})


def _written_regs(parsed) -> Set[str]:
    """Collect explicit and implicit definitions from the dependency model."""
    written: Set[str] = set()
    for mnemonic, operands in parsed:
        if not mnemonic:
            continue
        instruction = parse_line(mnemonic + " " + ", ".join(operands))
        written.update(get_instruction_spec(instruction).defs)
        if mnemonic in _LOOP_MNEMONICS:
            written.add("rcx")
    return written


def _reseeded_before(parsed, reg: str, upto: int, scalars: Set[str], written: Set[str]) -> bool:
    """True when a ``mov`` re-derives ``reg`` from bounded state before ``upto``.

    Accepted sources are immediates and scalar registers the blob itself never
    writes (init-seeded live-ins), excluding registers nanoBench's counter
    reads clobber between copies: only the survivors replay identically on
    every execution.
    """
    for mnemonic, operands in parsed[:upto]:
        if not mnemonic.startswith("mov") or len(operands) != 2:
            continue
        if _canonical_reg(operands[-1]) != reg:
            continue
        source = operands[0].strip()
        if re.fullmatch(r"\$-?\d+", source):
            return True
        origin = _canonical_reg(source)
        if (
            origin is not None
            and origin in scalars
            and origin not in written
            and origin not in _CLOBBERED_BETWEEN_COPIES
        ):
            return True
    return False


def _blob_is_idempotent(asm_code: str, scalars: Set[str]) -> bool:
    """True when back-to-back executions each perform identical work.

    nanoBench averages over dozens of byte-copy executions sharing register
    state (init runs once). A blob is idempotent when every counted back-edge
    re-derives its counter per execution: straight-line code always qualifies,
    while a counter merely decremented from its init seed (never reseeded
    in-blob) is consumed by the first pass, leaving later copies timing
    skipped-loop skeletons. Such one-shot blobs need a single-execution
    shape instead of copy averaging.
    """
    parsed, resolve_target = _asm_flow(asm_code)
    written = _written_regs(parsed)
    pointers, _ = classify_regs(asm_code)
    if pointers & written:
        return False  # Repeated copies otherwise accumulate pointer displacement.
    for index, (mnemonic, operands) in enumerate(parsed):
        if not operands:
            continue
        if mnemonic in _LOOP_MNEMONICS:
            if _reseeded_before(parsed, "rcx", index, scalars, written):
                continue
            return False
        if not mnemonic.startswith("j") or mnemonic == "jmp":
            continue
        target = resolve_target(operands[-1], index)
        if target is None or target >= index:
            continue
        for candidate, ops in parsed:
            if candidate in ("dec", "decq", "decl", "decw", "decb",
                             "inc", "incq", "incl", "incw", "incb") and ops:
                reg = _canonical_reg(ops[-1])
                if reg is not None and not _reseeded_before(
                    parsed, reg, target, scalars, written
                ):
                    return False
    return True


def _select_shape(asm_code: str, iterations: int, scalars: Set[str]) -> tuple[int, int]:
    """Choose the nanoBench repetition shape for one blob.

    Idempotent blobs use copy averaging for precision; one-shot blobs measure
    a single execution (``loop_count=0`` runs the copies once with no R15
    outer loop) so the median covers full executions instead of skeletons.
    """
    if not _blob_is_idempotent(asm_code, scalars):
        return 1, 0
    return _measurement_shape(asm_code, iterations)


def _measurement_shape(asm_code: str, iterations: int) -> tuple[int, int]:
    """Choose a precise measurement shape without overflowing nanoBench memory."""
    if "r15" in named_registers(asm_code):
        # nanoBench reserves R15 for its generated loop counter.
        return max(4, min(int(iterations), 200)), 0
    unroll = max(4, min(int(iterations), 64))
    return unroll, max(1, 1024 // unroll)


def _parse_cycle_measurement(output: str) -> Optional[tuple[str, float]]:
    """Read nanoBench's already-normalized cycle metric.

    Intel exposes ``CORE_CYCLES`` through fixed counters. The user-space AMD
    path intentionally exposes ``RDTSC`` instead, which is suitable for pinned
    A/B/B/A comparisons but represents reference-clock ticks rather than core
    clock cycles.
    """
    parsed: dict[str, float] = {}
    pattern = re.compile(
        r"^\s*(CORE_CYCLES|RDTSC)\s*:\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)\s*$",
    )
    for line in output.splitlines():
        if match := pattern.match(line):
            parsed[match.group(1)] = float(match.group(2))
    for counter in ("CORE_CYCLES", "RDTSC"):
        if counter in parsed:
            return counter, parsed[counter]
    return None


def _host_cpu() -> str:
    """Detect host CPU architecture name.

    Delegates to ``asm_util.host_cpu_name()`` where ``lscpu`` is available.
    Windows remains unknown because vendor strings do not identify an exact
    microarchitecture safely.
    """
    if IS_WINDOWS:
        return "unknown"
    return host_cpu_name()


def _discover_nanobench_binary() -> Optional[Path]:
    """Locate nanoBench executable in standard locations."""
    which_path = shutil.which("nanoBench")
    if which_path:
        return Path(which_path).resolve()
    candidates = [
        Path.home() / ".local" / "bin" / "nanoBench",
        Path.home() / "nanoBench" / "user" / "nanoBench",
        Path.home() / "nanoBench" / "nanoBench",
        Path.home() / "nanobench" / "nanoBench",
        Path("/usr/local/bin/nanoBench"),
        Path("/opt/nanoBench/nanoBench"),
        Path("/opt/nanoBench/user/nanoBench"),
    ]
    sudo_user = os.environ.get("SUDO_USER")
    if sudo_user:
        user_home = Path(f"/home/{sudo_user}")
        candidates.extend([
            user_home / ".local" / "bin" / "nanoBench",
            user_home / "nanoBench" / "user" / "nanoBench",
            user_home / "nanoBench" / "nanoBench",
            user_home / "nanobench" / "nanoBench",
        ])
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    return None


def _single_cpu_affinity() -> Optional[int]:
    if not hasattr(os, "sched_getaffinity"):
        return None
    try:
        allowed = os.sched_getaffinity(0)
    except OSError:
        return None
    return next(iter(allowed)) if len(allowed) == 1 else None
