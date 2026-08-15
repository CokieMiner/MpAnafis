"""Audit mechanically enforceable Rust source-structure rules."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .common import Finding, ROOT, SRC
from .import_parser import is_test_path
from .rust_source import clean_rust_code

COHESION_LINE_LIMIT = 600
SMALL_FILE_LINE_LIMIT = 200
ARCHITECTURE_ROOT = "src/int/logic/unsigned/math/arch/"
MATH_ROOT = "src/int/logic/unsigned/math/"

EXPECTED_NAMESPACE_CONSUMERS: Dict[str, Tuple[str, ...]] = {
    "Addition": (f"{MATH_ROOT}div/", f"{MATH_ROOT}mul/"),
    "Division": (
        f"{MATH_ROOT}barrett.rs",
        f"{MATH_ROOT}gcd/",
        f"{MATH_ROOT}modular.rs",
        f"{MATH_ROOT}montgomery.rs",
        f"{MATH_ROOT}pow.rs",
        f"{MATH_ROOT}roots/",
        f"{MATH_ROOT}theory/",
    ),
    "Gcd": (f"{MATH_ROOT}div/", f"{MATH_ROOT}theory/"),
    "LowProduct": (f"{MATH_ROOT}barrett.rs", f"{MATH_ROOT}div/"),
    "Multiplication": (f"{MATH_ROOT}div/",),
}

FORBIDDEN_VISIBILITY_RE = re.compile(r"\bpub\s*\(\s*(?:super\s*\)|in\b[^)]*\))")
INLINE_TEST_RE = re.compile(
    r"#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]\s*(?:#\s*\[[^]]*\]\s*)*mod\s+[A-Za-z_][A-Za-z0-9_]*\s*\{",
    re.MULTILINE,
)
PLACEHOLDER_RE = re.compile(r"\b(?:todo|unimplemented)\s*!\s*\(")
PLUMBING_ITEM_RE = re.compile(
    r"^\s*(?:pub(?:\s*\([^)]*\))?\s+)?(?:unsafe\s+)?(?:"
    r"fn\b|impl\b|struct\b|enum\b|union\b|trait\b|type\b|"
    r"const\b|static\b|macro_rules\s*!|macro\b)"
)
INLINE_MODULE_RE = re.compile(r"^\s*(?:pub(?:\s*\([^)]*\))?\s+)?mod\s+[A-Za-z_][A-Za-z0-9_]*\s*\{")
PRIVATE_LIB_USE_RE = re.compile(r"^\s*use\s+")


@dataclass(frozen=True)
class SizeReview:
    path: str
    lines: int


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def plumbing_findings(path: Path, cleaned: str) -> List[Finding]:
    if path.name not in {"mod.rs", "lib.rs"}:
        return []

    findings: List[Finding] = []
    rel = str(path.relative_to(ROOT)).replace("\\", "/")

    for match in INLINE_TEST_RE.finditer(cleaned):
        findings.append(Finding("inline_test_module_in_plumbing_file", rel, line_number(cleaned, match.start()), match.group(0)))

    brace_depth = 0
    for line_number_value, line in enumerate(cleaned.splitlines(), start=1):
        if brace_depth == 0:
            if path.name == "lib.rs" and PRIVATE_LIB_USE_RE.match(line):
                findings.append(Finding("private_import_in_library_facade", rel, line_number_value, line.strip()))
            elif PLUMBING_ITEM_RE.match(line):
                findings.append(Finding("implementation_in_plumbing_file", rel, line_number_value, line.strip()))
            elif INLINE_MODULE_RE.match(line):
                findings.append(Finding("inline_module_in_plumbing_file", rel, line_number_value, line.strip()))
        brace_depth += line.count("{") - line.count("}")
    return findings


def run_structure_audit(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit mechanically enforceable Rust source-structure rules.")
    parser.add_argument("--json", action="store_true", help="Print findings as JSON.")
    args = parser.parse_args(argv)

    findings: List[Finding] = []
    oversized_files: List[SizeReview] = []

    for path in sorted(SRC.rglob("*.rs")):
        if is_test_path(path):
            continue
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        text = path.read_text(encoding="utf-8")
        cleaned = clean_rust_code(text, scrub_attributes=False)

        for match in FORBIDDEN_VISIBILITY_RE.finditer(cleaned):
            findings.append(Finding("forbidden_visibility", rel, line_number(cleaned, match.start()), match.group(0)))

        for match in PLACEHOLDER_RE.finditer(cleaned):
            findings.append(Finding("placeholder_in_production", rel, line_number(cleaned, match.start()), match.group(0)))

        findings.extend(plumbing_findings(path, cleaned))

        line_count = len(text.splitlines())
        if line_count > COHESION_LINE_LIMIT:
            oversized_files.append(SizeReview(path=rel, lines=line_count))

    if args.json:
        payload = {
            "findings": [f.__dict__ for f in findings],
            "oversized_files": [o.__dict__ for o in oversized_files],
        }
        print(json.dumps(payload, indent=2))
        return 0 if not findings else 1

    print(f"Structure findings: {len(findings)}")
    print(f"Files > 600 lines: {len(oversized_files)}")

    if findings:
        print("\nFindings:")
        for f in findings:
            print(f"  {f.path}:{f.line} [{f.kind}] {f.detail}")
        return 1

    return 0
