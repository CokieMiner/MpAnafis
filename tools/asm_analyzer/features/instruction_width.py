"""Exact machine-code width extraction through LLVM's target assemblers."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict

from ..types import ArchitectureFamily, InstructionWidthStats

_TRIPLES: Dict[ArchitectureFamily, str] = {
    ArchitectureFamily.X86_64: "x86_64",
    ArchitectureFamily.X86_32: "i386",
    ArchitectureFamily.AARCH64: "aarch64",
    ArchitectureFamily.ARM32: "arm",
    ArchitectureFamily.RISCV64: "riscv64",
    ArchitectureFamily.RISCV32: "riscv32",
    ArchitectureFamily.POWER64: "powerpc64le",
    ArchitectureFamily.POWER32: "powerpc",
    ArchitectureFamily.S390X: "s390x",
    ArchitectureFamily.MIPS64: "mips64",
    ArchitectureFamily.MIPS32: "mips",
    ArchitectureFamily.LOONGARCH64: "loongarch64",
    ArchitectureFamily.LOONGARCH32: "loongarch32",
}
_FEATURES: Dict[ArchitectureFamily, str] = {
    ArchitectureFamily.ARM32: "+v7",
    ArchitectureFamily.RISCV64: "+m,+a,+c",
    ArchitectureFamily.RISCV32: "+m,+a,+c",
}
_ENCODING_RE = re.compile(r"encoding:\s*\[([^]]*)\]")


def analyze_instruction_widths(
    asm: str,
    target_arch: ArchitectureFamily,
) -> InstructionWidthStats:
    """Assemble a block and return exact encoded widths when LLVM supports it."""
    llvm_mc = shutil.which("llvm-mc")
    if llvm_mc is None:
        return InstructionWidthStats(error="llvm-mc is not installed")

    with tempfile.NamedTemporaryFile("w", suffix=".s", delete=False) as source:
        source.write(asm)
        source_path = Path(source.name)
    command = [
        llvm_mc,
        f"-triple={_TRIPLES[target_arch]}",
        "--show-encoding",
    ]
    if features := _FEATURES.get(target_arch):
        command.append(f"-mattr={features}")
    command.append(str(source_path))
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return InstructionWidthStats(error=str(error))
    finally:
        source_path.unlink(missing_ok=True)

    if result.returncode != 0:
        diagnostic = result.stderr.strip().splitlines()
        return InstructionWidthStats(
            error=diagnostic[0] if diagnostic else "llvm-mc failed",
        )

    widths = tuple(
        len([byte for byte in match.group(1).split(",") if byte.strip()])
        for line in result.stdout.splitlines()
        if (match := _ENCODING_RE.search(line)) is not None
    )
    if not widths:
        return InstructionWidthStats(error="llvm-mc emitted no instruction encodings")
    return InstructionWidthStats(
        exact=True,
        total_bytes=sum(widths),
        instruction_bytes=widths,
        source="llvm-mc --show-encoding",
    )


__all__ = ["analyze_instruction_widths"]
