"""Hardware PMU (Performance Monitoring Unit) profiling command.

Captures bare-metal hardware event counters (cycles, instructions, IPC,
backend stalls, branch mispredicts, and cache traffic) via Linux `perf`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class PmuProfile:
    """Hardware counter statistics captured from a benchmark run."""
    cycles: Optional[int] = None
    instructions: Optional[int] = None
    ipc: Optional[float] = None
    branches: Optional[int] = None
    branch_misses: Optional[int] = None
    branch_miss_rate: Optional[float] = None
    cache_references: Optional[int] = None
    cache_misses: Optional[int] = None
    cache_miss_rate: Optional[float] = None
    stalled_cycles_frontend: Optional[int] = None
    stalled_cycles_backend: Optional[int] = None

    def to_dict(self) -> Dict[str, Optional[float | int]]:
        """Convert PMU profile to dictionary."""
        return {
            "cycles": self.cycles,
            "instructions": self.instructions,
            "ipc": self.ipc,
            "branches": self.branches,
            "branch_misses": self.branch_misses,
            "branch_miss_rate": self.branch_miss_rate,
            "cache_references": self.cache_references,
            "cache_misses": self.cache_misses,
            "cache_miss_rate": self.cache_miss_rate,
            "stalled_cycles_frontend": self.stalled_cycles_frontend,
            "stalled_cycles_backend": self.stalled_cycles_backend,
        }

    def to_markdown(self) -> str:
        """Format the profile as a readable Markdown table."""
        rows = [
            ("Cycles", f"{self.cycles:,}" if self.cycles is not None else "N/A"),
            ("Instructions", f"{self.instructions:,}" if self.instructions is not None else "N/A"),
            ("IPC (Instructions/Cycle)", f"{self.ipc:.2f}" if self.ipc is not None else "N/A"),
            ("Branches", f"{self.branches:,}" if self.branches is not None else "N/A"),
            (
                "Branch Mispredict Rate",
                f"{self.branch_miss_rate * 100:.2f}%" if self.branch_miss_rate is not None else "N/A",
            ),
            ("Cache References", f"{self.cache_references:,}" if self.cache_references is not None else "N/A"),
            (
                "Cache Miss Rate",
                f"{self.cache_miss_rate * 100:.2f}%" if self.cache_miss_rate is not None else "N/A",
            ),
            (
                "Frontend Stalled Cycles",
                f"{self.stalled_cycles_frontend:,}" if self.stalled_cycles_frontend is not None else "N/A",
            ),
            (
                "Backend Stalled Cycles",
                f"{self.stalled_cycles_backend:,}" if self.stalled_cycles_backend is not None else "N/A",
            ),
        ]
        lines = ["# Hardware PMU Performance Profile", "", "| Metric | Value |", "|:---|:---|"]
        for metric, val in rows:
            lines.append(f"| {metric} | {val} |")
        lines.append("")
        return "\n".join(lines)


def is_perf_available() -> bool:
    """Return True if the Linux `perf` tool is installed and executable."""
    return shutil.which("perf") is not None


def profile_command(cmd: Sequence[str], events: Optional[Sequence[str]] = None) -> Optional[PmuProfile]:
    """Execute `cmd` under `perf stat` and parse hardware counters."""
    if not is_perf_available():
        return None

    if events is None:
        events = [
            "cycles",
            "instructions",
            "branches",
            "branch-misses",
            "cache-references",
            "cache-misses",
            "stalled-cycles-frontend",
            "stalled-cycles-backend",
        ]

    perf_cmd = ["perf", "stat", "-x,", "-e", ",".join(events), "--", *cmd]

    try:
        proc = subprocess.run(
            perf_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception:
        return None

    raw_counts: Dict[str, int] = {}
    for line in proc.stderr.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) >= 3:
            count_str = parts[0].strip()
            event_name = parts[2].strip()
            if count_str.isdigit():
                raw_counts[event_name] = int(count_str)

    cycles = raw_counts.get("cycles")
    instructions = raw_counts.get("instructions")
    ipc = (instructions / cycles) if (cycles and instructions and cycles > 0) else None

    branches = raw_counts.get("branches")
    branch_misses = raw_counts.get("branch-misses")
    branch_miss_rate = (
        (branch_misses / branches)
        if (branches and branch_misses is not None and branches > 0)
        else None
    )

    cache_refs = raw_counts.get("cache-references")
    cache_misses = raw_counts.get("cache-misses")
    cache_miss_rate = (
        (cache_misses / cache_refs)
        if (cache_refs and cache_misses is not None and cache_refs > 0)
        else None
    )

    return PmuProfile(
        cycles=cycles,
        instructions=instructions,
        ipc=ipc,
        branches=branches,
        branch_misses=branch_misses,
        branch_miss_rate=branch_miss_rate,
        cache_references=cache_refs,
        cache_misses=cache_misses,
        cache_miss_rate=cache_miss_rate,
        stalled_cycles_frontend=raw_counts.get("stalled-cycles-frontend"),
        stalled_cycles_backend=raw_counts.get("stalled-cycles-backend"),
    )


def run_pmu(cmd: List[str], as_json: bool = False) -> int:
    """Run command under PMU profiling and display results."""
    if not cmd:
        print("Error: No command specified for PMU profiling.", file=sys.stderr)
        return 1

    profile = profile_command(cmd)
    if profile is None:
        print("Error: Linux `perf` is not available or failed to execute.", file=sys.stderr)
        return 1

    if as_json:
        print(json.dumps(profile.to_dict(), indent=2))
    else:
        print(profile.to_markdown())

    return 0
