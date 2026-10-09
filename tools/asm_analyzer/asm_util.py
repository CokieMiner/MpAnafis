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
        without_slashes = raw.split("//", 1)[0]
        line = re.split(
            r"\s+#(?![-+]?(?:0x[0-9a-fA-F]+|\d))",
            without_slashes,
            maxsplit=1,
        )[0].strip()
        if (
            not line
            or line.startswith((".", "#"))
            or line.endswith(":")
        ):
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
        "adc", "sbb", "mulx", "adcx", "adox", "lea"
    ):
        if raw == base or (len(raw) == len(base) + 1 and raw.startswith(base) and raw[-1] in "bwlq"):
            return base
    return raw


def split_asm_operands(text: str) -> List[str]:
    """Split comma-delimited operands while preserving address tuples."""
    operands: List[str] = []
    current: List[str] = []
    depth = 0
    for character in text:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        if character == "," and depth == 0:
            operands.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    if current:
        operands.append("".join(current).strip())
    return [operand for operand in operands if operand]


def named_registers(body: str) -> Set[str]:
    """Concrete x86 GPRs named in an AT&T body, normalized to 64-bit names."""
    return {
        base
        for register in _REG_TOK.findall(body)
        if (base := GPR_ALIAS_MAP.get(register)) is not None
    }


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
                if canonical := GPR_ALIAS_MAP.get(base[1:]):
                    pointers.add(canonical)
            if idx and idx.startswith("%"):
                if canonical := GPR_ALIAS_MAP.get(idx[1:]):
                    indexes.add(canonical)
    pointers -= indexes
    pointers.discard("rsp")
    scalars = {r for r in named_registers(body) if r not in pointers and r != "rsp"}
    return pointers, scalars


def host_cpu_name() -> str:
    """Return the normalized host CPU identifier, or ``unknown``.

    Hardware measurement must never guess a microarchitecture: a guessed name
    could attach empirical data to a foreign CPU model.
    """
    model_name = ""
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        model_name = _model_name_from_cpuinfo(cpuinfo)
    except (OSError, UnicodeDecodeError):
        pass
    if not model_name:
        try:
            env = os.environ.copy()
            env["LC_ALL"] = "C"
            result = subprocess.run(
                ["lscpu"],
                capture_output=True,
                text=True,
                check=False,
                stdin=subprocess.DEVNULL,
                timeout=20,
                env=env,
            )
            model_name = _model_name_from_lscpu(result.stdout or "")
        except (OSError, UnicodeDecodeError, subprocess.TimeoutExpired):
            pass
    return _cpu_name_from_model(model_name)


def _model_name_from_cpuinfo(cpuinfo: str) -> str:
    for line in cpuinfo.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() == "model name":
            return value.strip()
    return ""


def _model_name_from_lscpu(output: str) -> str:
    for line in output.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() == "model name":
            return value.strip()
    return ""


def _cpu_name_from_model(model_name: str) -> str:
    name = model_name.lower()
    # Check newest AMD first to avoid overlap with Intel model numbers.
    if (
        re.search(r"\bzen\s*5\b", name)
        or "ryzen ai 7 350" in name
        or "ryzen ai 5 340" in name
    ):
        return "znver5"
    if re.search(r"\bzen\s*4\b", name):
        return "znver4"
    if re.search(r"\bzen\s*3\b", name):
        return "znver3"
    if re.search(r"\bzen\s*2\b", name):
        return "znver2"
    # Intel checks come after AMD to avoid cross-matching model numbers.
    if "alder lake" in name:
        return "alderlake"
    if "ice lake" in name:
        return "icelake-server" if "xeon" in name else "ice-lake"
    if "skylake" in name:
        return "skylake"
    return "unknown"
