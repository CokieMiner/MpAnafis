#!/usr/bin/env python3
"""Shared AT&T assembly helpers for the assembly analyzer suite.

Holds the WSL subprocess shim, the 64-bit GPR slot order, index-aware
pointer/scalar register classifiers, host CPU auto-detection, and
instruction parsing primitives.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Slot order shared by the wrappers and drivers: the 15 GPRs (rsp excluded).
GPR64: List[str] = [
    "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11",
    "r12", "r13", "r14", "r15", "rbp",
]

# 64-bit -> 32-bit register names for zero-extension.
R32_MAP = {
    "rax": "eax", "rbx": "ebx", "rcx": "ecx", "rdx": "edx",
    "rsi": "esi", "rdi": "edi", "rbp": "ebp",
    "r8": "r8d", "r9": "r9d", "r10": "r10d", "r11": "r11d",
    "r12": "r12d", "r13": "r13d", "r14": "r14d", "r15": "r15d",
}

# Canonical mapping: every x86-64 GPR alias -> its 64-bit base name.
# Includes all width variants (64/32/16/8-bit) for all 16 registers.
# This is the single source of truth for register alias normalization;
# features/registers.py, features/multiplier.py, and search/ast.py all
# import from here instead of maintaining separate copies.
GPR_ALIAS_MAP: Dict[str, str] = {
    "rax": "rax", "eax": "rax", "ax": "rax", "al": "rax", "ah": "rax",
    "rbx": "rbx", "ebx": "rbx", "bx": "rbx", "bl": "rbx", "bh": "rbx",
    "rcx": "rcx", "ecx": "rcx", "cx": "rcx", "cl": "rcx", "ch": "rcx",
    "rdx": "rdx", "edx": "rdx", "dx": "rdx", "dl": "rdx", "dh": "rdx",
    "rsi": "rsi", "esi": "rsi", "si": "rsi", "sil": "rsi",
    "rdi": "rdi", "edi": "rdi", "di": "rdi", "dil": "rdi",
    "rbp": "rbp", "ebp": "rbp", "bp": "rbp", "bpl": "rbp",
    "rsp": "rsp", "esp": "rsp", "sp": "rsp", "spl": "rsp",
    "r8": "r8", "r8d": "r8", "r8w": "r8", "r8b": "r8",
    "r9": "r9", "r9d": "r9", "r9w": "r9", "r9b": "r9",
    "r10": "r10", "r10d": "r10", "r10w": "r10", "r10b": "r10",
    "r11": "r11", "r11d": "r11", "r11w": "r11", "r11b": "r11",
    "r12": "r12", "r12d": "r12", "r12w": "r12", "r12b": "r12",
    "r13": "r13", "r13d": "r13", "r13w": "r13", "r13b": "r13",
    "r14": "r14", "r14d": "r14", "r14w": "r14", "r14b": "r14",
    "r15": "r15", "r15d": "r15", "r15w": "r15", "r15b": "r15",
}

# AT&T memory operand `(base, index, scale)`.
_MEM_TOK = re.compile(r"\(([^,)]*)(?:,\s*([^,)]*))?(?:,\s*[^,)]*)?\)")
_REG_TOK = re.compile(r"%([a-z][a-z0-9]*)")
_MEM_OPERAND_RE = re.compile(r"(-?\d*)\(([^)]*)\)")


def wsl_path(p: Path) -> str:
    """Convert a Windows path to a WSL path (/mnt/c/...)."""
    s = str(p).replace("\\", "/")
    m = re.match(r"^([A-Za-z]):(/.*)$", s)
    if m:
        return f"/mnt/{m.group(1).lower()}{m.group(2)}"
    return s


def run(cmd: List[str], use_wsl: bool) -> subprocess.CompletedProcess:
    """Run a toolchain command, wrapping it through WSL when requested."""
    if use_wsl:
        quoted = " ".join(shlex.quote(c) for c in cmd)
        cmd = ["wsl.exe", "-e", "bash", "-lc", quoted]
    env = os.environ.copy()
    cargo_bin = str(Path.home() / ".cargo" / "bin")
    if cargo_bin not in env.get("PATH", ""):
        env["PATH"] = f"{cargo_bin}:{env.get('PATH', '')}"
    return subprocess.run(
        cmd, capture_output=True, text=True, check=False,
        stdin=subprocess.DEVNULL, timeout=180, env=env,
    )


def instr_lines(asm: str) -> List[str]:
    """Strip comments, empty lines, and directives from an assembly string."""
    out: List[str] = []
    for raw in asm.splitlines():
        line = raw.split("#")[0].split("//")[0].strip()
        if not line or line.startswith(".") or line.endswith(":"):
            continue
        out.append(line)
    return out


def extract_mnemonic(line: str) -> str:
    """Extract lower-case instruction mnemonic without suffixes."""
    parts = line.split(None, 1)
    if not parts:
        return ""
    raw = parts[0].lower().split(".")[0]
    for base in (
        "mov", "add", "sub", "mul", "imul", "div", "idiv", "cmp", "test",
        "and", "or", "xor", "shl", "shr", "sar", "rol", "ror", "push", "pop",
        "adc", "sbb", "mulx", "adcx", "adox"
    ):
        if raw == base or (len(raw) == len(base) + 1 and raw.startswith(base) and raw[-1] in "bwlq"):
            return base
    return raw


def named_registers(body: str) -> Set[str]:
    """Concrete registers named in an AT&T body (base names, no duplicates)."""
    return set(_REG_TOK.findall(body))


def classify_regs(body: str) -> Tuple[Set[str], Set[str]]:
    """Pointers vs scalars among the concrete registers of an AT&T body."""
    pointers: Set[str] = set()
    indexes: Set[str] = set()
    for ln in body.splitlines():
        s = ln.split("#", 1)[0].strip()
        if not s or s.startswith(".") or s.endswith(":"):
            continue
        for m in _MEM_TOK.finditer(s):
            base, idx = m.group(1), m.group(2)
            if base and base.startswith("%"):
                pointers.add(base[1:])
            if idx and idx.startswith("%"):
                indexes.add(idx[1:])
    pointers -= indexes
    pointers.discard("rsp")
    scalars = {r for r in named_registers(body) if r not in pointers and r != "rsp"}
    return pointers, scalars


def host_cpu_name() -> str:
    """Return a short normalized host CPU identifier (e.g. 'znver4', 'skylake')."""
    try:
        r = subprocess.run(["lscpu"], capture_output=True, text=True, check=False,
                           stdin=subprocess.DEVNULL, timeout=20)
        text = r.stdout or ""
        for line in text.splitlines():
            if line.lower().startswith("model name"):
                name = line.split(":", 1)[1].strip().lower()
                # Check newest AMD first to avoid overlap with Intel model numbers
                if "zen 5" in name or "ryzen 9 9" in name or "ryzen 7 9" in name:
                    return "znver5"
                if "zen 4" in name or "ryzen 9 7" in name or "ryzen 7 7" in name or "ryzen 5 7" in name:
                    return "znver4"
                if "zen 3" in name or "ryzen 9 5" in name or "ryzen 7 5" in name or "ryzen 5 5" in name:
                    return "znver3"
                if "zen 2" in name or "ryzen 9 3" in name or "ryzen 7 3" in name or "ryzen 5 3" in name or "ryzen 7 4" in name:
                    return "znver2"
                # Intel checks — these come after AMD to avoid cross-matching
                if "alder lake" in name or "raptor lake" in name or "i9-12" in name or "i7-12" in name or "i9-13" in name or "i9-14" in name:
                    return "alderlake"
                if "ice lake" in name or "i7-10" in name:
                    return "icelake-server"
                if "skylake" in name or "i7-6" in name or "i7-8" in name:
                    return "skylake"
    except (OSError, UnicodeDecodeError):
        pass
    return "znver2"
