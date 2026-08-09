#!/usr/bin/env python3
"""Inventory Rust ``allow`` attributes in production and test-oriented code."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rust_source import clean_rust_code, matching_delimiter, split_top_level


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORIES = (
    "src",
    "benches",
    "examples",
    "tests",
    "fuzz/fuzz_targets",
    "tools/tune",
    "build_support",
)
TEST_PATH_PARTS = {"benches", "tests", "fuzz", "fuzz_targets"}
CFG_TEST_RE = re.compile(
    r"#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]",
    re.MULTILINE,
)
TEST_MODULE_RE = re.compile(
    r"\bmod\s+(?:tests|[A-Za-z_][A-Za-z0-9_]*_tests)\s*\{",
    re.MULTILINE,
)
ALLOW_RE = re.compile(r"(?P<prefix>#!|#)\s*\[\s*allow\s*\(")


@dataclass(frozen=True)
class AllowEntry:
    kind: str
    path: str
    line: int
    lints: tuple[str, ...]
    in_test: bool


def rust_source_paths(
    root: Path = ROOT,
    *,
    production_only: bool = False,
) -> list[Path]:
    paths: set[Path] = set()
    directories = ("src",) if production_only else SOURCE_DIRECTORIES
    for relative in directories:
        directory = root / relative
        if directory.is_dir():
            paths.update(directory.rglob("*.rs"))
    build_script = root / "build.rs"
    if not production_only and build_script.is_file():
        paths.add(build_script)
    return sorted(paths)


def is_whole_file_test(rel_path: str) -> bool:
    parts = rel_path.replace("\\", "/").split("/")
    stem = Path(parts[-1]).stem
    return (
        any(part in TEST_PATH_PARTS for part in parts)
        or stem == "tests"
        or stem.endswith("_tests")
    )


def skip_attributes_and_whitespace(text: str, offset: int) -> int:
    while offset < len(text):
        if text[offset].isspace():
            offset += 1
            continue
        if text.startswith("#![", offset):
            opening = offset + 2
        elif text.startswith("#[", offset):
            opening = offset + 1
        else:
            break
        closing = matching_delimiter(text, opening, "[", "]")
        if closing is None:
            break
        offset = closing + 1
    return offset


def find_item_end(text: str, offset: int) -> int | None:
    depths = {"(": 0, "[": 0, "<": 0}
    closing = {")": "(", "]": "[", ">": "<"}
    in_initializer = False
    while offset < len(text):
        char = text[offset]
        if char in {"(", "["}:
            depths[char] += 1
        elif char == "<" and not in_initializer:
            depths[char] += 1
        elif char in closing and depths[closing[char]] > 0:
            depths[closing[char]] -= 1
        elif char == "=" and not any(depths.values()):
            in_initializer = True
        elif char in "{;" and not any(depths.values()):
            if char == ";":
                return offset + 1
            body_end = matching_delimiter(text, offset, "{", "}")
            return len(text) if body_end is None else body_end + 1
        offset += 1
    return None


def find_test_ranges(text: str) -> list[tuple[int, int]]:
    """Return sorted, merged byte ranges occupied by test-only items."""

    cleaned = clean_rust_code(text, scrub_attributes=False)
    ranges: list[tuple[int, int]] = []

    for match in TEST_MODULE_RE.finditer(cleaned):
        opening = cleaned.find("{", match.start(), match.end())
        closing = matching_delimiter(cleaned, opening, "{", "}")
        if closing is not None:
            ranges.append((match.start(), closing + 1))

    for match in CFG_TEST_RE.finditer(cleaned):
        item_start = skip_attributes_and_whitespace(cleaned, match.end())
        item_end = find_item_end(cleaned, item_start)
        if item_end is not None:
            ranges.append((match.start(), item_end))

    if not ranges:
        return []
    ranges.sort()
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def extract_allows(
    text: str,
    rel_path: str,
    test_ranges: list[tuple[int, int]],
    whole_file_test: bool,
) -> list[AllowEntry]:
    cleaned = clean_rust_code(text, scrub_attributes=False)
    entries: list[AllowEntry] = []
    for match in ALLOW_RE.finditer(cleaned):
        opening = cleaned.rfind("(", match.start(), match.end())
        closing = matching_delimiter(cleaned, opening, "(", ")")
        if closing is None:
            continue
        parts = split_top_level(cleaned[opening + 1 : closing])
        lints = tuple(
            part.strip()
            for part in parts
            if part.strip() and part.split("=", maxsplit=1)[0].strip() != "reason"
        )
        if not lints:
            continue
        start = match.start()
        entries.append(
            AllowEntry(
                kind=(
                    "inner #![allow]"
                    if match.group("prefix") == "#!"
                    else "outer #[allow]"
                ),
                path=rel_path,
                line=cleaned.count("\n", 0, start) + 1,
                lints=lints,
                in_test=(
                    whole_file_test
                    or any(range_start <= start < range_end for range_start, range_end in test_ranges)
                ),
            )
        )
    return entries


def select_lints(
    entries: list[AllowEntry],
    *,
    lint: str | None,
    mode: str,
    production_only: bool,
    test_only: bool,
) -> list[AllowEntry]:
    wanted = None
    if lint is not None:
        wanted = {lint}
        if "::" not in lint:
            wanted.add(f"clippy::{lint}")

    selected: list[AllowEntry] = []
    for entry in entries:
        if production_only and entry.in_test:
            continue
        if test_only and not entry.in_test:
            continue
        lints = entry.lints
        if mode == "clippy":
            lints = tuple(name for name in lints if name.startswith("clippy::"))
        elif mode == "non_clippy":
            lints = tuple(name for name in lints if not name.startswith("clippy::"))
        if wanted is not None:
            lints = tuple(name for name in lints if name in wanted)
        if lints:
            selected.append(
                AllowEntry(entry.kind, entry.path, entry.line, lints, entry.in_test)
            )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--lint")
    location = parser.add_mutually_exclusive_group()
    location.add_argument("--prod-only", action="store_true")
    location.add_argument("--test-only", action="store_true")
    lint_mode = parser.add_mutually_exclusive_group()
    lint_mode.add_argument(
        "--non-clippy",
        action="store_true",
        help="Report only non-Clippy lint allowances",
    )
    lint_mode.add_argument(
        "--all-lints",
        action="store_true",
        help="Report both Clippy and non-Clippy lint allowances",
    )
    args = parser.parse_args()

    entries: list[AllowEntry] = []
    for path in rust_source_paths(production_only=args.prod_only):
        text = path.read_text(encoding="utf-8")
        relative = str(path.relative_to(ROOT))
        entries.extend(
            extract_allows(
                text,
                relative,
                find_test_ranges(text),
                is_whole_file_test(relative),
            )
        )

    mode = "all" if args.all_lints else "non_clippy" if args.non_clippy else "clippy"
    entries = select_lints(
        entries,
        lint=args.lint,
        mode=mode,
        production_only=args.prod_only,
        test_only=args.test_only,
    )

    prod_counts: Counter[str] = Counter()
    test_counts: Counter[str] = Counter()
    prod_files: dict[str, set[str]] = {}
    test_files: dict[str, set[str]] = {}
    for entry in entries:
        counts = test_counts if entry.in_test else prod_counts
        files = test_files if entry.in_test else prod_files
        for lint in entry.lints:
            counts[lint] += 1
            files.setdefault(lint, set()).add(entry.path)

    all_lints = sorted(set(prod_counts) | set(test_counts))
    if args.json:
        output = {
            "mode": mode,
            "total_allows": len(entries),
            "prod_total": sum(prod_counts.values()),
            "test_total": sum(test_counts.values()),
            "by_lint": {
                lint: {
                    "prod_count": prod_counts.get(lint, 0),
                    "test_count": test_counts.get(lint, 0),
                    "prod_files": sorted(prod_files.get(lint, set())),
                    "test_files": sorted(test_files.get(lint, set())),
                }
                for lint in all_lints
            },
        }
        print(json.dumps(output, indent=2))
        return 0

    print(f"{'Lint':52s} {'Prod':>6s} {'Test':>6s}  Prod files")
    print("-" * 110)
    for lint in all_lints:
        paths = sorted(prod_files.get(lint, set()))
        location_text = ", ".join(paths[:3])
        if len(paths) > 3:
            location_text += f" … (+{len(paths) - 3})"
        print(
            f"{lint:52s} {prod_counts.get(lint, 0):6d} "
            f"{test_counts.get(lint, 0):6d}  {location_text}"
        )
    print("-" * 110)
    print(f"{'TOTAL':52s} {sum(prod_counts.values()):6d} {sum(test_counts.values()):6d}")
    print(f"\nEntries scanned: {len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
