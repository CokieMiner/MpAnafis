#!/usr/bin/env python3
"""llvm-mca driver for the assembly analyzer backends.

Discovers llvm-mca binaries and executes analytical target scheduling models
natively or through WSL with bounded execution timeouts.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..analyzer import InstrMetrics

IS_WINDOWS = os.name == "nt"
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class McaResult:
    """Parsed llvm-mca result with throughput and order-sensitive simulation cost."""

    throughput_cycles: Optional[float]
    simulated_cycles: Optional[float]
    uops: Optional[float]
    dispatch_width: Optional[int]
    port_pressure: Dict[str, float]
    instruction_metrics: List[InstrMetrics]
    raw_output: str


def discover_mca_binary(wsl: bool) -> Optional[str]:
    """Find an installed llvm-mca binary, probing WSL when requested."""
    if wsl and not IS_WINDOWS:
        wsl = False
    candidates = ["llvm-mca", "llvm-mca-20", "llvm-mca-19", "llvm-mca-18", "llvm-mca-17"]
    if wsl:
        probe = "for c in " + " ".join(candidates) + "; do command -v $c && break; done"
        try:
            r = subprocess.run(
                ["wsl", "-e", "bash", "-lc", probe],
                capture_output=True, text=True, check=False, timeout=10,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return None
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().splitlines()[0].strip()
        return None
    for c in candidates:
        if shutil.which(c):
            return c
    return None


class McaDriver:
    """Runs llvm-mca natively or through WSL, capturing stdout with timeouts."""

    def __init__(self, mca: str, wsl: bool) -> None:
        self.mca = mca
        self.wsl = wsl
        self._avail: Optional[bool] = None

    def _mca_available(self) -> bool:
        if self._avail is None:
            self._avail = shutil.which(self.mca) is not None or self.wsl
        return self._avail

    def _to_wsl_path(self, path: Path) -> str:
        text = str(path)
        drive = text[0].lower()
        rest = text[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"

    def _run(self, args: List[str], cwd: Path, timeout: int = 30) -> subprocess.CompletedProcess[str]:
        try:
            if self.wsl:
                return subprocess.run(
                    ["wsl", "-e", *args],
                    cwd=cwd, capture_output=True, text=True, check=False,
                    stdin=subprocess.DEVNULL, timeout=timeout,
                )
            return subprocess.run(
                args, cwd=cwd, capture_output=True, text=True, check=False,
                stdin=subprocess.DEVNULL, timeout=timeout,
            )
        except subprocess.TimeoutExpired as err:
            return subprocess.CompletedProcess(
                args=args,
                returncode=-1,
                stdout=err.stdout or "" if isinstance(err.stdout, str) else "",
                stderr=f"TimeoutExpired: process timed out after {timeout}s",
            )
        except Exception as err:
            return subprocess.CompletedProcess(
                args=args,
                returncode=-1,
                stdout="",
                stderr=f"Execution error: {err}",
            )

    def list_cpus(self, triple: Optional[str] = None) -> List[str]:
        """Return CPU models llvm-mca supports."""
        if not self._mca_available():
            return []
        cmd = [self.mca]
        if triple:
            cmd.append(f"-mtriple={triple}")
        cmd.append("-mcpu=help")
        r = self._run(cmd, REPO_ROOT, timeout=15)
        text = (r.stderr or "") + "\n" + (r.stdout or "")
        cpus: List[str] = []
        for line in text.splitlines():
            m = re.match(r"\s*\*?\s*([a-z0-9][a-z0-9\-]*)", line)
            if m and m.group(1) not in ("Available", "targets", "for"):
                cpus.append(m.group(1))
        return cpus

    def run_on_asm_detailed(
        self,
        asm_code: str,
        cpu: str,
        iterations: int = 200,
        triple: Optional[str] = None,
        features: Tuple[str, ...] = (),
        timeout: int = 30,
    ) -> McaResult:
        """Run llvm-mca and retain throughput, resources, and instruction timings."""
        if not self._mca_available():
            return McaResult(None, None, None, None, {}, [], "llvm-mca is not available")
        with tempfile.NamedTemporaryFile("w", suffix=".s", delete=False) as f:
            f.write(asm_code)
            tmp = Path(f.name)
        try:
            target_path = self._to_wsl_path(tmp) if self.wsl else str(tmp)
            cmd = [self.mca]
            if triple:
                cmd.append(f"-mtriple={triple}")
            if features:
                cmd.append(f"-mattr={','.join(features)}")
            cmd.extend([f"-mcpu={cpu}", f"-iterations={iterations}", target_path])
            r = self._run(cmd, REPO_ROOT, timeout=timeout)
            if r.returncode != 0:
                err_msg = r.stderr.strip() if r.stderr else f"llvm-mca exited with code {r.returncode}"
                return McaResult(None, None, None, None, {}, [], err_msg)

            cycles: Optional[float] = None
            total_cycles: Optional[float] = None
            uops: Optional[float] = None
            dispatch_width: Optional[int] = None

            lines = r.stdout.splitlines()
            for line in lines:
                if "Block RThroughput:" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        try:
                            cycles = float(parts[1].strip())
                        except ValueError:
                            pass
                elif "Total Cycles:" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        try:
                            total_cycles = float(parts[1].strip())
                        except ValueError:
                            pass
                elif "Total uOps:" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        try:
                            uops = float(parts[1].strip()) / max(iterations, 1)
                        except ValueError:
                            pass

                elif "Dispatch Width:" in line:
                    try:
                        dispatch_width = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass

            port_pressure = _parse_resource_pressure(lines)
            instruction_metrics = _parse_instruction_metrics(lines)

            return McaResult(
                throughput_cycles=cycles,
                simulated_cycles=(
                    total_cycles / max(iterations, 1)
                    if total_cycles is not None
                    else None
                ),
                uops=uops,
                dispatch_width=dispatch_width,
                port_pressure=port_pressure,
                instruction_metrics=instruction_metrics,
                raw_output=r.stdout,
            )
        finally:
            tmp.unlink(missing_ok=True)

    def run_on_asm(
        self,
        asm_code: str,
        cpu: str,
        iterations: int = 200,
        triple: Optional[str] = None,
        features: Tuple[str, ...] = (),
        timeout: int = 30,
    ) -> Optional[float]:
        """Run llvm-mca on an assembly string, returning block RThroughput."""
        result = self.run_on_asm_detailed(
            asm_code=asm_code,
            cpu=cpu,
            iterations=iterations,
            triple=triple,
            features=features,
            timeout=timeout,
        )
        return result.throughput_cycles


def _parse_resource_pressure(lines: List[str]) -> Dict[str, float]:
    resources: Dict[str, str] = {}
    pressure: Dict[str, float] = {}
    for line in lines:
        match = re.match(r"^\[([0-9]+(?:\.[0-9]+)?)\]\s+-\s+(\S.*)$", line)
        if match is not None:
            resources[match.group(1)] = match.group(2).strip()

    for index, line in enumerate(lines):
        if line.strip() != "Resource pressure per iteration:":
            continue
        header_index = index + 1
        values_index = index + 2
        if values_index >= len(lines):
            break
        keys = re.findall(r"\[([0-9]+(?:\.[0-9]+)?)\]", lines[header_index])
        values = lines[values_index].split()
        for key, value in zip(keys, values):
            if value != "-":
                try:
                    pressure[resources.get(key, key)] = float(value)
                except ValueError:
                    continue
        break
    return pressure


def _parse_instruction_metrics(lines: List[str]) -> List[InstrMetrics]:
    metrics: List[InstrMetrics] = []
    in_table = False
    for line in lines:
        if line.rstrip().endswith("Instructions:") and "[1]" in line:
            in_table = True
            continue
        if not in_table:
            continue
        if not line.strip():
            if metrics:
                break
            continue
        match = re.match(
            r"^\s*(\d+)\s+(\d+)\s+([0-9]+(?:\.[0-9]+)?)\s+(.*)$",
            line,
        )
        if match is None:
            continue
        remainder = match.group(4).strip()
        instruction = re.sub(r"^(?:\*\s*){1,3}", "", remainder).strip()
        metrics.append(
            InstrMetrics(
                line=instruction,
                latency=float(match.group(2)),
                throughput=float(match.group(3)),
                uops=float(match.group(1)),
            )
        )
    return metrics
