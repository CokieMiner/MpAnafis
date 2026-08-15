"""Backend and CPU capability probe command."""

from __future__ import annotations

import json
from typing import Dict, List

from ..backends import make_backends
from ..models import CpuSpec


def run_check(
    backends_list: List[str],
    cpus_list: List[CpuSpec],
    use_wsl: bool = False,
    as_json: bool = False,
) -> int:
    """Probe simulator backends and logical CPU support."""
    backends = make_backends(backends_list, wsl=use_wsl)
    rows: List[Dict[str, object]] = []

    for name in backends_list:
        a = backends.get(name)
        avail = a.available() if a else False
        rows.append({"backend": name, "available": avail})

    cpu_rows: List[Dict[str, object]] = []
    for cpu in cpus_list:
        supported = [b for b in backends_list if backends.get(b) and backends[b].supports(cpu)]
        cpu_rows.append({
            "cpu": cpu.name,
            "family": cpu.family,
            "supported_backends": supported,
        })

    if as_json:
        print(json.dumps({"backends": rows, "cpus": cpu_rows}, indent=2))
    else:
        print("# Assembly Analyzer Capability Probe\n")
        print("| Backend | Status |")
        print("|:---|:---|")
        for r in rows:
            status = "✅ Available" if r["available"] else "❌ Not found"
            print(f"| `{r['backend']}` | {status} |")

        print("\n### Supported CPU Architecture Models\n")
        print("| CPU Model | Architecture | Active Backends |")
        print("|:---|:---|:---|")
        for c in cpu_rows:
            b_str = ", ".join(c["supported_backends"]) if c["supported_backends"] else "None"
            print(f"| `{c['cpu']}` | `{c['family']}` | {b_str} |")
        print("")

    return 0
