"""Randomized topological instruction scheduler and optimizer command."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

from ..extract import extract_asm_blocks, real_asm_for_block
from ..models import CpuSpec, DEFAULT_MATRIX, parse_cpus
from ..search.engine import search_kernel


def run_search(
    kernel_path: str,
    cpus: Optional[List[CpuSpec]] = None,
    candidates: int = 50,
    seed: int = 42,
    use_wsl: bool = False,
    as_json: bool = False,
) -> int:
    """Execute randomized DAG scheduler search on a kernel file or assembly snippet."""
    p = Path(kernel_path)
    if not p.exists():
        print(f"Error: File not found: {p}", file=sys.stderr)
        return 1

    if p.suffix == ".rs":
        blocks = extract_asm_blocks(p)
        if not blocks:
            print(f"Error: No asm! blocks found in {p}", file=sys.stderr)
            return 1
        best_block = max(blocks, key=lambda b: len(b.instructions))
        asm_lines, err = real_asm_for_block(best_block.instructions, best_block.operands, best_block.options, use_wsl)
        if not asm_lines:
            print(f"Error extracting assembly: {err}", file=sys.stderr)
            return 1
        asm_body = "\n".join(asm_lines)
    else:
        asm_body = p.read_text(encoding="utf-8", errors="replace")

    cpu_specs = cpus or parse_cpus(",".join(DEFAULT_MATRIX))
    results, err = search_kernel(asm_body, cpu_specs, candidates_count=candidates, seed=seed, use_wsl=use_wsl)

    if not results:
        print(f"Search failed: {err}", file=sys.stderr)
        return 1

    if as_json:
        data = [
            {
                "idx": r.idx,
                "is_valid": r.is_valid,
                "cycles": r.cycles,
                "body": r.body,
            }
            for r in results
        ]
        print(json.dumps(data, indent=2))
    else:
        print(f"# Kernel Schedule Search for `{p.name}`\n")
        print(f"Generated and evaluated {len(results)} valid topological schedule candidates.")
        print(f"\nTop Candidate (Original = 0, Best = {results[0].idx}):\n")
        for line in results[0].body.splitlines():
            print(f"    {line}")
        print("")

    return 0
