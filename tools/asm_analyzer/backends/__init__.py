#!/usr/bin/env python3
"""Backend factory and registry for the assembly analyzer suite."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional
from ..analyzer import Analyzer
from ..models import CpuSpec
from .llvm_mca import LlvmMcaAnalyzer
from .nanobench import NanobenchAnalyzer
from .osaca import OsacaAnalyzer
from .uica import UicaAnalyzer

_BACKEND_CLASSES: Dict[str, Callable[..., Analyzer]] = {
    "llvm-mca": LlvmMcaAnalyzer,
    "osaca": OsacaAnalyzer,
    "uica": UicaAnalyzer,
    "nanobench": NanobenchAnalyzer,
}


def make_backends(backends: Optional[List[str]] = None,
                  wsl: Optional[bool] = None) -> Dict[str, Analyzer]:
    """Instantiate requested backend analyzers."""
    if backends is None:
        backends = list(_BACKEND_CLASSES)
    out: Dict[str, Analyzer] = {}
    for name in backends:
        cls = _BACKEND_CLASSES.get(name)
        if cls is None:
            continue
        if name in ("llvm-mca", "osaca", "uica"):
            out[name] = cls(wsl=wsl)
        else:
            out[name] = cls()
    return out


def supported_backends(cpu: CpuSpec, backends: Dict[str, Analyzer]) -> List[str]:
    """Backends that both know this CPU and are available."""
    return [name for name in _BACKEND_CLASSES
            if name in backends and backends[name].available()
            and backends[name].supports(cpu)]


__all__ = ["make_backends", "supported_backends", "_BACKEND_CLASSES"]
