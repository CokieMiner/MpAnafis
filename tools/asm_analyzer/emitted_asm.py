"""Extraction of compiler-emitted inline-assembly marker regions."""

from __future__ import annotations

import re
from typing import List, Tuple

_APP_MARKERS = ("#APP", "//APP", "@APP")
_NO_APP_MARKERS = ("#NO_APP", "//NO_APP", "@NO_APP")
_SYMBOL_RE = re.compile(r"([A-Za-z_][\w.$@]*):")


def extract_asm_region(asm_text: str) -> List[str]:
    """Flatten every emitted inline-assembly marker region."""
    return [line for region in extract_asm_regions(asm_text) for line in region]


def extract_asm_regions(asm_text: str) -> List[List[str]]:
    """Extract each compiler-emitted inline-assembly region independently."""
    return [body for _, body in extract_named_asm_regions(asm_text)]


def extract_named_asm_regions(asm_text: str) -> List[Tuple[str, List[str]]]:
    """Extract emitted inline-assembly regions with their enclosing symbols."""
    regions: List[Tuple[str, List[str]]] = []
    current_symbol = ""
    active_symbol = ""
    body: List[str] = []
    inside = False
    for line in asm_text.splitlines():
        stripped = line.strip()
        symbol = _SYMBOL_RE.fullmatch(stripped)
        if symbol is not None:
            current_symbol = symbol.group(1)
        if stripped in _APP_MARKERS:
            inside = True
            active_symbol = current_symbol
            body = []
            continue
        if stripped in _NO_APP_MARKERS:
            if inside and body:
                regions.append((active_symbol or f"asm_{len(regions) + 1}", body))
            inside = False
            continue
        if inside and stripped:
            body.append(stripped)
    if inside and body:
        regions.append((active_symbol or f"asm_{len(regions) + 1}", body))
    return regions


__all__ = [
    "extract_asm_region",
    "extract_asm_regions",
    "extract_named_asm_regions",
]
