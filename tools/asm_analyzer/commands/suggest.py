"""Optimization suggestion command for assembly kernel files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..features.suggestions import generate_suggestions
from ..report.terminal import render_terminal_suggestions
from .sweep import extract_kernel_asm


def run_suggest(
    kernel_path: str,
    use_wsl: bool = False,
    enable_color: bool = True,
    as_json: bool = False,
) -> int:
    """Scan assembly file and display optimization recommendations."""
    p = Path(kernel_path)
    if not p.exists():
        print(f"Error: File not found: {p}", file=sys.stderr)
        return 1

    if p.suffix == ".rs":
        asm_code, err = extract_kernel_asm(p, use_wsl=use_wsl)
        if not asm_code:
            print(f"Error extracting assembly from {p}: {err}", file=sys.stderr)
            return 1
    else:
        asm_code = p.read_text(encoding="utf-8", errors="replace")

    suggestions = generate_suggestions(asm_code, kernel_name=p.stem)

    if as_json:
        data = [
            {
                "severity": s.severity.value,
                "rule_id": s.rule_id,
                "title": s.title,
                "description": s.description,
                "suggested_fix": s.suggested_fix,
                "line_number": s.line_number,
                "problematic_code": s.problematic_code,
            }
            for s in suggestions
        ]
        print(json.dumps(data, indent=2))
    else:
        print(f"# Assembly Optimization Report for `{p.name}`\n")
        print(render_terminal_suggestions(suggestions, enable_color=enable_color))

    return 0
