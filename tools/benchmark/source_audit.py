"""Enforce symmetric declarations in handwritten benchmark pairs and shared templates."""

from __future__ import annotations

import re
from pathlib import Path

try:
    from audit.items import top_level_items
    from audit.rust_source import clean_rust_code, split_top_level
except ModuleNotFoundError:
    from tools.audit.items import top_level_items
    from tools.audit.rust_source import clean_rust_code, split_top_level

from .models import ENGINES


def check_source(text: str, path: str = "fixture.rs") -> list[dict]:
    findings = []
    def add(kind, offset, detail):
        findings.append({"kind": kind, "path": path, "line": text.count("\n", 0, offset) + 1, "detail": detail})

    def inspect(source: str, base: int):
        cleaned = clean_rust_code(source, scrub_attributes=False)
        try:
            items = top_level_items(source)
        except ValueError as error:
            add("unparsed_benchmark_source", base, str(error))
            return
        benches = {}
        for item in items:
            if item.kind == "mod" and cleaned[item.start:item.end].rstrip().endswith("}"):
                opening = cleaned.index("{", item.start, item.end)
                inspect(source[opening + 1:item.end - 1], base + opening + 1)
            annotation = next((a for a in item.attributes if re.match(r"divan\s*::\s*bench\s*\(", a.code)), None)
            if annotation is None:
                continue
            if item.name not in ENGINES:
                add("nonstandard_benchmark_engine", base + item.start, f"use mp/rug/gmp/flint engine names, not {item.name!r}")
                continue
            config = {}
            contents = annotation.code[annotation.code.index("(") + 1:annotation.code.rfind(")")]
            for field in split_top_level(contents):
                key, separator, value = field.partition("=")
                config[key.strip()] = re.sub(r"\s+", "", value) if separator else True
            if item.name in benches:
                add("duplicate_benchmark_engine", base + item.start, "each function/scenario declares an engine once")
            body = cleaned[item.start:item.end]
            timing = tuple(re.findall(r"\.\s*(bench_local(?:_values|_refs)?|bench(?:_values|_refs)?)\s*\(", body))
            benches[item.name] = (item, config, timing)
        if benches:
            if "mp" not in benches:
                add("benchmark_without_mp", base, "comparison group needs an mp entry")
                return
            mp, expected, expected_timing = benches["mp"]
            for engine, (item, config, timing) in benches.items():
                if config != expected:
                    add("asymmetric_benchmark_configuration", base + item.start, f"{engine} and mp must use identical argument ladders, sampling, and counters")
                if timing != expected_timing:
                    add("asymmetric_benchmark_timing", base + item.start, f"{engine} and mp must use the same timing and input-reset strategy")
    inspect(text, 0)
    return findings


def check_tree(root: Path) -> list[dict]:
    findings = []
    for path in sorted(root.rglob("*.rs")):
        findings.extend(check_source(path.read_text(), str(path.relative_to(root))))
    return findings
