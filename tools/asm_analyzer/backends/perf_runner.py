"""Experimental whole-process perf timing for Linux assembly workloads.

Counts include launcher and loop overhead. These measurements cannot serve as
isolated kernel costs or confirm automatic schedule application.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from ..analyzer import Analyzer, KernelReport
from ..asm_util import host_cpu_name
from ..models import CpuSpec
from ..targets import host_architecture


class PerfAnalyzer(Analyzer):
    """Empirical hardware cycle counter backend via Linux perf."""

    name = "perf"

    def __init__(self) -> None:
        self._perf = shutil.which("perf")
        self._host_arch = host_architecture()
        self._host_name = host_cpu_name()

    def available(self) -> bool:
        """Return True if Linux perf executable is present."""
        return self._perf is not None

    def supports(self, cpu: CpuSpec) -> bool:
        """True if the target CPU matches the physical host machine."""
        if not self.available():
            return False
        if cpu.name == self._host_name and self._host_name != "unknown":
            return True
        return False

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
                note="Linux perf executable not found in PATH",
            )
        if not self.supports(cpu):
            return KernelReport(
                backend=self.name,
                cpu=cpu.name,
                ok=False,
                note=f"perf measures the native host only, not foreign target '{cpu.name}'",
            )

        affinity = _single_cpu_affinity()
        if affinity is None:
            return KernelReport(
                backend=self.name,
                cpu=cpu.name,
                ok=False,
                note="hardware measurement requires process affinity to exactly one logical CPU",
            )

        loops = max(100, iterations)
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            src_file = work / "bench.s"
            obj_file = work / "bench.o"
            exe_file = work / "bench"

            # Create standalone benchmark loop
            bench_code = _create_bench_assembly(asm_code, loops, self._host_arch)
            src_file.write_text(bench_code, encoding="utf-8")

            assembled = subprocess.run(
                ["as", str(src_file), "-o", str(obj_file)],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            if assembled.returncode != 0:
                return KernelReport(
                    backend=self.name,
                    cpu=cpu.name,
                    ok=False,
                    note=f"as failed: {assembled.stderr[:200]}",
                )

            linked = subprocess.run(
                ["gcc", "-nostdlib", "-no-pie", str(obj_file), "-o", str(exe_file)],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            if linked.returncode != 0:
                linked = subprocess.run(
                    ["ld", str(obj_file), "-o", str(exe_file)],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
                if linked.returncode != 0:
                    return KernelReport(
                        backend=self.name,
                        cpu=cpu.name,
                        ok=False,
                        note=f"link failed: {linked.stderr[:200]}",
                    )

            perf_cmd = [
                "perf", "stat", "-x,", "-e", "cycles",
                "--", str(exe_file),
            ]
            run_res = subprocess.run(
                perf_cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            if run_res.returncode != 0:
                return KernelReport(
                    backend=self.name,
                    cpu=cpu.name,
                    ok=False,
                    note=f"perf stat failed: {run_res.stderr[:200]}",
                )

            cycles = _parse_perf_cycles(run_res.stderr)
            if cycles is not None and cycles > 0 and math.isfinite(cycles):
                cycles_per_iter = cycles / loops
                return KernelReport(
                    backend=self.name,
                    cpu=cpu.name,
                    ok=True,
                    cycles=cycles_per_iter,
                    note=f"whole-process perf cycles over {loops} loops; includes harness overhead; {cycles_per_iter:.2f} cycles/iter",
                    raw_output=run_res.stderr,
                )

            return KernelReport(
                backend=self.name,
                cpu=cpu.name,
                ok=False,
                note=f"could not parse cycles from perf output:\n{run_res.stderr[:300]}",
            )


def _single_cpu_affinity() -> Optional[int]:
    if not hasattr(os, "sched_getaffinity"):
        return None
    try:
        allowed = os.sched_getaffinity(0)
    except OSError:
        return None
    return next(iter(allowed)) if len(allowed) == 1 else None


def _parse_perf_cycles(stderr_output: str) -> Optional[float]:
    for line in stderr_output.splitlines():
        parts = line.split(",")
        if len(parts) >= 3 and parts[2].strip() == "cycles":
            try:
                return float(parts[0].strip())
            except ValueError:
                pass
    return None


def _create_bench_assembly(kernel_asm: str, loops: int, arch: Optional[object]) -> str:
    """Create minimal standalone assembly program running kernel_asm in a loop."""
    clean_body = "\n".join(
        f"    {line.strip()}"
        for line in kernel_asm.splitlines()
        if line.strip()
    )

    from ..types import ArchitectureFamily
    if arch not in (ArchitectureFamily.X86_64, ArchitectureFamily.AARCH64,
                    ArchitectureFamily.RISCV64, ArchitectureFamily.RISCV32):
        raise ValueError("perf harness is unavailable for this architecture")
    if arch == ArchitectureFamily.AARCH64:
        return f"""\
.bss
.p2align 6
scratch_buf:
    .zero 65536

.text
.globl _start
_start:
    adrp x20, scratch_buf
    add x20, x20, :lo12:scratch_buf
    add x0, x20, #4096
    add x1, x20, #8192
    add x2, x20, #12288
    add x3, x20, #16384
    add x4, x20, #20480
    add x5, x20, #24576
    add x6, x20, #28672
    add x7, x20, #32768
    mov x19, #{loops}
1:
{clean_body}
    subs x19, x19, #1
    b.ne 1b

    mov x8, #93
    mov x0, #0
    svc #0
"""
    elif arch in (ArchitectureFamily.RISCV64, ArchitectureFamily.RISCV32):
        return f"""\
.bss
.p2align 6
scratch_buf:
    .zero 65536

.text
.globl _start
_start:
    la t0, scratch_buf
    li a0, 4096
    add a0, t0, a0
    li a1, 8192
    add a1, t0, a1
    li a2, 12288
    add a2, t0, a2
    li a3, 16384
    add a3, t0, a3
    li a4, 20480
    add a4, t0, a4
    li a5, 24576
    add a5, t0, a5
    li s0, {loops}
1:
{clean_body}
    addi s0, s0, -1
    bnez s0, 1b

    li a7, 93
    li a0, 0
    ecall
"""
    else:  # x86-64 default
        return f"""\
.bss
.p2align 6
scratch_buf:
    .zero 65536

.text
.globl _start
_start:
    lea scratch_buf(%rip), %r14
    lea 4096(%r14), %rdi
    lea 8192(%r14), %rsi
    lea 12288(%r14), %rdx
    lea 16384(%r14), %rbx
    lea 20480(%r14), %r8
    lea 24576(%r14), %r9
    lea 28672(%r14), %r10
    lea 32768(%r14), %r11
    lea 36864(%r14), %r12
    lea 40960(%r14), %r13
    push %rbp
    lea 45056(%r14), %rbp

    mov ${loops}, %rcx
1:
    push %rcx
{clean_body}
    pop %rcx
    dec %rcx
    jnz 1b

    pop %rbp
    mov $60, %rax
    xor %rdi, %rdi
    syscall
"""


__all__ = ["PerfAnalyzer"]
