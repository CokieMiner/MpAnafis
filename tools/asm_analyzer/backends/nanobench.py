#!/usr/bin/env python3
"""nanoBench hardware PMU cycle measurement backend.

Measures real CPU cycles for isolated assembly blocks on the host machine
via the nanoBench kernel module / user driver.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Set, Tuple

from ..analyzer import Analyzer, KernelReport
from ..asm_util import classify_regs
from ..models import CpuSpec

IS_WINDOWS = os.name == "nt"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _host_cpu() -> str:
    """Detect host CPU architecture name.

    Delegates to ``asm_util.host_cpu_name()`` for Linux, with a Windows
    environment-variable fallback.
    """
    if IS_WINDOWS:
        arch = os.environ.get("PROCESSOR_ARCHITECTURE", "").lower()
        ident = os.environ.get("PROCESSOR_IDENTIFIER", "").lower()
        if "amd" in ident or "authenticamd" in ident:
            return "znver3"
        if "intel" in ident:
            return "skylake"
        return "x86_64"
    from ..asm_util import host_cpu_name
    return host_cpu_name()


def _discover_nanobench_binary() -> Optional[Path]:
    """Locate nanoBench executable in standard locations."""
    which_path = shutil.which("nanoBench")
    if which_path:
        return Path(which_path)
    candidates = [
        Path.home() / "nanoBench" / "nanoBench",
        Path.home() / "nanobench" / "nanoBench",
        Path("/usr/local/bin/nanoBench"),
        Path("/opt/nanoBench/nanoBench"),
    ]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    return None


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
        return self.available() and (cpu.name == self._host or cpu.family in ("amd", "intel", "x86_64"))

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

        if not shutil.which("as") or not shutil.which("objcopy"):
            return KernelReport(
                backend=self.name,
                cpu=cpu.name,
                ok=False,
                note="binutils (as/objcopy) not found in PATH",
            )

        unroll = max(4, min(int(iterations), 200))
        tag = hashlib.sha1(f"{asm_code}\n{cpu.name}".encode("utf-8")).hexdigest()[:16]
        work = REPO_ROOT / "tools" / "_mca" / "nanobench" / tag
        work.mkdir(parents=True, exist_ok=True)
        asm_file = work / "kernel.s"
        obj_file = work / "kernel.o"
        bin_file = work / "kernel.bin"
        out_file = work / "nanobench.out"

        pointers, scalars = classify_regs(asm_code)
        body = [".text", ".globl main", "main:"]
        for p in pointers:
            body.append(f"    leaq 1024(%rsp), %{p}")
        for s in scalars:
            body.append(f"    xorq %{s}, %{s}")
        body.append(asm_code.rstrip("\n"))
        body.append("    ret\n")

        asm_file.write_text("\n".join(body), encoding="utf-8")
        try:
            ar = subprocess.run(
                ["as", str(asm_file), "-o", str(obj_file)],
                capture_output=True, text=True, check=False, timeout=30,
            )
            if ar.returncode != 0:
                return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note=f"as failed: {ar.stderr[:200]}")

            or_ = subprocess.run(
                ["objcopy", "-O", "binary", "-j", ".text", str(obj_file), str(bin_file)],
                capture_output=True, text=True, check=False, timeout=30,
            )
            if or_.returncode != 0 or not bin_file.exists():
                return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note=f"objcopy failed: {or_.stderr[:200]}")

            cmd = [
                str(self._bin),
                "-code", str(bin_file),
                "-unroll_count", str(unroll),
                "-loop_count", "1",
                "-n_measurements", "5",
                "-avg",
            ]

            r = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30)
            text = r.stdout + "\n" + r.stderr
            if r.returncode == 0:
                cycles = None
                for line in text.splitlines():
                    if "CORE_CYCLES:" in line:
                        parts = line.split(":")
                        if len(parts) >= 2:
                            try:
                                cycles = float(parts[1].strip()) / unroll
                            except ValueError:
                                pass
                if cycles is not None:
                    return KernelReport(backend=self.name, cpu=cpu.name, ok=True, cycles=cycles, raw_output=text)
                return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note=f"Could not parse CORE_CYCLES from output:\n{text[:300]}")

            return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note=f"nanoBench execution error:\n{text[:300]}")
        except subprocess.TimeoutExpired:
            return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note="nanoBench timed out after 30 seconds")
        except Exception as err:
            return KernelReport(backend=self.name, cpu=cpu.name, ok=False, note=f"nanoBench exception: {err}")
        finally:
            for f in (asm_file, obj_file, bin_file, out_file):
                f.unlink(missing_ok=True)
