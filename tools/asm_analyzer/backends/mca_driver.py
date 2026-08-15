#!/usr/bin/env python3
"""llvm-mca driver for the assembly analyzer backends.

Discovers llvm-mca binaries and executes cycle-accurate instruction pipeline
simulations natively or through WSL with bounded execution timeouts.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IS_WINDOWS = os.name == "nt"
REPO_ROOT = Path(__file__).resolve().parents[3]


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
                inner = " ".join(args)
                return subprocess.run(
                    ["wsl", "-e", "bash", "-lc", inner],
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
        timeout: int = 30,
    ) -> Tuple[Optional[float], Optional[float], Dict[str, float], str]:
        """Run llvm-mca on an assembly string, returning (cycles, uops, port_pressure, raw_output)."""
        if not self._mca_available():
            return None, None, {}, "llvm-mca is not available"
        with tempfile.NamedTemporaryFile("w", suffix=".s", delete=False) as f:
            f.write(asm_code)
            tmp = Path(f.name)
        try:
            target_path = self._to_wsl_path(tmp) if self.wsl else str(tmp)
            cmd = [self.mca]
            if triple:
                cmd.append(f"-mtriple={triple}")
            cmd.extend([f"-mcpu={cpu}", f"-iterations={iterations}", target_path])
            r = self._run(cmd, REPO_ROOT, timeout=timeout)
            if r.returncode != 0:
                err_msg = r.stderr.strip() if r.stderr else f"llvm-mca exited with code {r.returncode}"
                return None, None, {}, err_msg

            cycles: Optional[float] = None
            uops: Optional[float] = None
            port_pressure: Dict[str, float] = {}

            lines = r.stdout.splitlines()
            for line in lines:
                if "Block RThroughput:" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        try:
                            cycles = float(parts[1].strip())
                        except ValueError:
                            pass
                elif "Total uOps:" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        try:
                            uops = float(parts[1].strip())
                        except ValueError:
                            pass

            return cycles, uops, port_pressure, r.stdout
        finally:
            tmp.unlink(missing_ok=True)

    def run_on_asm(
        self,
        asm_code: str,
        cpu: str,
        iterations: int = 200,
        triple: Optional[str] = None,
        timeout: int = 30,
    ) -> Optional[float]:
        """Run llvm-mca on an assembly string, returning block RThroughput."""
        cycles, _, _, _ = self.run_on_asm_detailed(
            asm_code=asm_code,
            cpu=cpu,
            iterations=iterations,
            triple=triple,
            timeout=timeout,
        )
        return cycles
