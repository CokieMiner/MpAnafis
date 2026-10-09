"""Backend and CPU capability probe command."""

from __future__ import annotations

import json
from typing import Dict, List

from ..backends import make_backends
from ..models import CpuSpec

_SMOKE_ASSEMBLY = {
    "amd": "addq %rax, %rbx",
    "arm": "add x0, x0, x1",
    "arm32": "add r0, r0, r1",
    "intel": "addq %rax, %rbx",
    "loongarch32": "add.w $a0, $a0, $a1",
    "loongarch64": "add.d $a0, $a0, $a1",
    "mips32": "addu $4, $4, $5",
    "mips64": "daddu $4, $4, $5",
    "ppc": "add 3, 3, 4",
    "ppc32": "add 3, 3, 4",
    "riscv": "add a0, a0, a1",
    "riscv32": "add a0, a0, a1",
    "s390x": "agr %r2, %r3",
    "x86_32": "addl %eax, %ebx",
}


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
    probe_failed = False
    for cpu in cpus_list:
        supported: List[str] = []
        failed: Dict[str, str] = {}
        smoke_asm = _SMOKE_ASSEMBLY.get(cpu.family, "")
        for backend_name in backends_list:
            backend = backends.get(backend_name)
            if backend is None or not backend.supports(cpu):
                continue
            report = backend.analyze_report(smoke_asm, cpu)
            if report.ok and report.cycles is not None:
                supported.append(backend_name)
            else:
                probe_failed = True
                reason = report.note or report.raw_output or "no cycle estimate"
                failed[backend_name] = reason.strip().splitlines()[0]
        cpu_rows.append({
            "cpu": cpu.name,
            "family": cpu.family,
            "supported_backends": supported,
            "failed_backends": failed,
        })

    if as_json:
        print(json.dumps({"backends": rows, "cpus": cpu_rows}, indent=2))
    else:
        print("# Assembly Analyzer Capability Probe\n")
        print("| Backend | Status |")
        print("|:---|:---|")
        for r in rows:
            status = "Available" if r["available"] else "Not found"
            print(f"| `{r['backend']}` | {status} |")

        print("\n### Supported CPU Architecture Models\n")
        print("| CPU Model | Architecture | Active Backends | Failed Probes |")
        print("|:---|:---|:---|:---|")
        for c in cpu_rows:
            b_str = ", ".join(c["supported_backends"]) if c["supported_backends"] else "None"
            failed_str = ", ".join(
                f"{name}: {reason}" for name, reason in c["failed_backends"].items()
            ) or "None"
            print(f"| `{c['cpu']}` | `{c['family']}` | {b_str} | {failed_str} |")
        print("")

    return 1 if probe_failed else 0
