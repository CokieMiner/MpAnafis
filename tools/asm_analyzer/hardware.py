"""Read-only host topology and benchmark-environment capture."""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Dict, Optional


def capture_hardware_context(logical_cpu: int) -> Dict[str, object]:
    """Capture CPU, cache, affinity, frequency, and kernel provenance."""
    cpu_root = Path(f"/sys/devices/system/cpu/cpu{logical_cpu}")
    return {
        "model_name": _cpu_model_name(),
        "logical_cpu": logical_cpu,
        "affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else [],
        "thread_siblings": _read(cpu_root / "topology/thread_siblings_list"),
        "caches": _cache_topology(cpu_root / "cache"),
        "scaling_governor": _read(cpu_root / "cpufreq/scaling_governor"),
        "scaling_min_khz": _read_int(cpu_root / "cpufreq/scaling_min_freq"),
        "scaling_max_khz": _read_int(cpu_root / "cpufreq/scaling_max_freq"),
        "boost_enabled": _read(Path("/sys/devices/system/cpu/cpufreq/boost")),
        "kernel": platform.release(),
    }


def _cache_topology(cache_root: Path) -> Dict[str, Dict[str, object]]:
    caches: Dict[str, Dict[str, object]] = {}
    for index in sorted(cache_root.glob("index*")):
        level = _read(index / "level")
        cache_type = _read(index / "type")
        if level is None or cache_type is None:
            continue
        key = f"L{level}{cache_type.lower()}"
        caches[key] = {
            "size": _read(index / "size"),
            "ways": _read_int(index / "ways_of_associativity"),
            "line_bytes": _read_int(index / "coherency_line_size"),
            "shared_cpu_list": _read(index / "shared_cpu_list"),
        }
    return caches


def _cpu_model_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    try:
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _read_int(path: Path) -> Optional[int]:
    value = _read(path)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


__all__ = ["capture_hardware_context"]
