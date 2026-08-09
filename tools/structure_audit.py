#!/usr/bin/env python3
"""Audit mechanically enforceable Rust source-structure rules.

Checks production sources for:

- implementation items in ``mod.rs`` or ``lib.rs`` plumbing files;
- private imports in ``lib.rs`` that relay non-public implementation names;
- ``pub(super)`` and ``pub(in ...)`` visibility;
- inline ``#[cfg(test)] mod ... { ... }`` bodies;
- ``todo!`` and ``unimplemented!`` placeholders.

It also reports conservative cohesion advisories for namespace methods that
have no visible cross-file caller and at most one call in their defining file.
Those methods are candidates for private placement or call-site inlining.

File length is deliberately informational: cohesion cannot be decided by a
line counter. Files above 600 lines are listed for review. The complete
inventory of implementation files below 200 lines is available on request but
is not printed by default. Architecture backends are excluded because their
target-specific implementation and dispatch boundaries are intentionally
small; the remaining inventory still requires a cohesion review rather than a
mechanical merge.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from import_audit import ImportEdge, collect_findings, is_test_path, module_name, parse_alias
from rust_source import clean_rust_code


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
COHESION_LINE_LIMIT = 600
SMALL_FILE_LINE_LIMIT = 200
ARCHITECTURE_ROOT = "src/int/logic/unsigned/math/arch/"
MATH_ROOT = "src/int/logic/unsigned/math/"

# Cross-family dependencies that are part of the arithmetic design. Keep the
# consumer paths narrow: a new consumer outside these roots remains a cohesion
# review instead of disappearing behind a blanket namespace exception.
EXPECTED_NAMESPACE_CONSUMERS: dict[str, tuple[str, ...]] = {
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
    r"#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]\s*"
    r"(?:#\s*\[[^]]*\]\s*)*mod\s+[A-Za-z_][A-Za-z0-9_]*\s*\{",
    re.MULTILINE,
)
PLACEHOLDER_RE = re.compile(r"\b(?:todo|unimplemented)\s*!\s*\(")
PLUMBING_ITEM_RE = re.compile(
    r"^\s*(?:pub(?:\s*\([^)]*\))?\s+)?(?:unsafe\s+)?(?:"
    r"fn\b|impl\b|struct\b|enum\b|union\b|trait\b|type\b|"
    r"const\b|static\b|macro_rules\s*!|macro\b)"
)
INLINE_MODULE_RE = re.compile(
    r"^\s*(?:pub(?:\s*\([^)]*\))?\s+)?mod\s+"
    r"[A-Za-z_][A-Za-z0-9_]*\s*\{"
)
PRIVATE_LIB_USE_RE = re.compile(r"^\s*use\s+")


@dataclass(frozen=True)
class Finding:
    kind: str
    path: str
    line: int
    text: str


@dataclass(frozen=True)
class SizeReview:
    path: str
    lines: int


@dataclass(frozen=True)
class NamespaceReview:
    path: str
    line: int
    function: str
    importers: tuple[str, ...]


@dataclass(frozen=True)
class NamespaceSpreadReview:
    path: str
    line: int
    namespace: str
    importer_folders: tuple[str, ...]


@dataclass(frozen=True)
class ImplMethodReview:
    path: str
    line: int
    namespace: str
    method: str
    local_calls: int


@dataclass(frozen=True)
class NamespaceArityReview:
    path: str
    line: int
    namespace: str
    method: str


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def plumbing_findings(path: Path, cleaned: str) -> list[Finding]:
    if path.name not in {"mod.rs", "lib.rs"}:
        return []

    findings: list[Finding] = []
    brace_depth = 0
    for line_number_value, line in enumerate(cleaned.splitlines(), start=1):
        if brace_depth == 0:
            if path.name == "lib.rs" and PRIVATE_LIB_USE_RE.match(line):
                findings.append(
                    Finding(
                        kind="private_import_in_library_facade",
                        path=str(path.relative_to(ROOT)),
                        line=line_number_value,
                        text=line.strip(),
                    )
                )
            elif PLUMBING_ITEM_RE.match(line):
                findings.append(
                    Finding(
                        kind="implementation_in_plumbing_file",
                        path=str(path.relative_to(ROOT)),
                        line=line_number_value,
                        text=line.strip(),
                    )
                )
            elif INLINE_MODULE_RE.match(line):
                findings.append(
                    Finding(
                        kind="inline_module_in_plumbing_file",
                        path=str(path.relative_to(ROOT)),
                        line=line_number_value,
                        text=line.strip(),
                    )
                )
        brace_depth += line.count("{") - line.count("}")
    return findings


def public_free_functions(path: Path, cleaned: str) -> list[tuple[str, int]]:
    functions: list[tuple[str, int]] = []
    brace_depth = 0
    pattern = re.compile(r"^\s*pub\s+(?:unsafe\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)")
    for line_number_value, line in enumerate(cleaned.splitlines(), start=1):
        if brace_depth == 0:
            match = pattern.match(line)
            if match is not None:
                functions.append((match.group(1), line_number_value))
        brace_depth += line.count("{") - line.count("}")
    return functions


def namespace_types(cleaned: str) -> list[tuple[str, int]]:
    namespaces: list[tuple[str, int]] = []
    brace_depth = 0
    pattern = re.compile(r"^\s*pub\s+struct\s+([A-Za-z_][A-Za-z0-9_]*)\s*;")
    for line_number_value, line in enumerate(cleaned.splitlines(), start=1):
        if brace_depth == 0:
            match = pattern.match(line)
            if match is not None:
                namespaces.append((match.group(1), line_number_value))
        brace_depth += line.count("{") - line.count("}")
    return namespaces


def namespace_impl_methods(cleaned: str) -> list[tuple[str, str, int]]:
    """Collect public methods from simple inherent impl blocks.

    The project namespace pattern uses ``impl Namespace {``. Trait impls and
    generic impl headers are deliberately excluded rather than guessed at.
    """

    methods: list[tuple[str, str, int]] = []
    impl_pattern = re.compile(r"^\s*impl\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{")
    method_pattern = re.compile(
        r"^\s*pub(?:\s*\([^)]*\))?\s+"
        r"(?:(?:const|unsafe|async)\s+)*fn\s+([A-Za-z_][A-Za-z0-9_]*)"
    )
    brace_depth = 0
    owner: str | None = None
    body_depth = 0

    for line_number_value, line in enumerate(cleaned.splitlines(), start=1):
        if owner is None:
            impl_match = impl_pattern.match(line)
            if impl_match is not None:
                owner = impl_match.group(1)
                body_depth = brace_depth + 1
        elif brace_depth == body_depth:
            method_match = method_pattern.match(line)
            if method_match is not None:
                methods.append((owner, method_match.group(1), line_number_value))

        brace_depth += line.count("{") - line.count("}")
        if owner is not None and brace_depth < body_depth:
            owner = None
            body_depth = 0

    return methods


def importers_by_symbol(edges: list[ImportEdge]) -> dict[str, set[str]]:
    """Resolve parent re-export bindings before indexing symbol importers."""

    bindings: dict[str, set[str]] = {}
    for edge in edges:
        target, alias = parse_alias(edge.raw)
        name = alias or target.rsplit("::", maxsplit=1)[-1]
        bindings.setdefault(f"{edge.source}::{name}", set()).add(edge.target)

    def resolve(symbol: str) -> str | None:
        visited: set[str] = set()
        while symbol in bindings:
            if symbol in visited or len(bindings[symbol]) != 1:
                return None
            visited.add(symbol)
            symbol = next(iter(bindings[symbol]))
        return symbol

    importers: dict[str, set[str]] = {}
    for edge in edges:
        resolved = resolve(edge.target)
        if resolved is not None:
            importers.setdefault(resolved, set()).add(edge.path)
    return importers


def namespace_family_root(path: Path) -> Path:
    """Return the algorithm family that owns a namespace.

    The multiplication tower is one family: its dispatcher orchestrates every
    tier and higher tiers deliberately call lower tiers. Treating each tier
    directory as unrelated would report the intended dependency graph as a
    cohesion failure.
    """

    relative = path.relative_to(ROOT)
    parts = relative.parts
    math_prefix = ("src", "int", "logic", "unsigned", "math")
    if parts[: len(math_prefix)] == math_prefix:
        remainder = parts[len(math_prefix) :]
        if remainder and remainder[0] == "mul":
            return ROOT.joinpath(*math_prefix, "mul")
        if remainder:
            return ROOT.joinpath(*math_prefix, remainder[0])
    return path.parent


def namespace_method_call_count(
    source: str,
    owner: str,
    method: str,
    *,
    include_self_calls: bool,
) -> int:
    owner_call = re.compile(rf"\b{re.escape(owner)}\s*::\s*{re.escape(method)}\b")
    calls = len(owner_call.findall(source))
    if include_self_calls:
        self_call = re.compile(rf"\bSelf\s*::\s*{re.escape(method)}\b")
        calls += len(self_call.findall(source))
    return calls


def is_expected_namespace_consumer(namespace: str, importer: str) -> bool:
    """Return whether `importer` is an explicit cross-family dependency."""

    for allowed in EXPECTED_NAMESPACE_CONSUMERS.get(namespace, ()):
        if allowed.endswith("/"):
            if importer.startswith(allowed):
                return True
        elif importer == allowed:
            return True
    return False


def collect() -> tuple[
    list[Finding],
    list[SizeReview],
    list[SizeReview],
    list[NamespaceReview],
    list[NamespaceSpreadReview],
    list[ImplMethodReview],
    list[NamespaceArityReview],
]:
    findings: list[Finding] = []
    large_files: list[SizeReview] = []
    small_files: list[SizeReview] = []
    public_functions: list[tuple[Path, str, int]] = []
    namespaces: list[tuple[Path, str, int]] = []
    impl_methods: list[tuple[Path, str, str, int]] = []
    sources: dict[Path, str] = {}

    for path in sorted(SRC.rglob("*.rs")):
        if is_test_path(path):
            continue

        raw = path.read_text(encoding="utf-8")
        cleaned = clean_rust_code(raw)
        sources[path] = cleaned
        rel_path = str(path.relative_to(ROOT))

        for match in FORBIDDEN_VISIBILITY_RE.finditer(cleaned):
            findings.append(
                Finding(
                    kind="forbidden_visibility",
                    path=rel_path,
                    line=line_number(cleaned, match.start()),
                    text=match.group(0).strip(),
                )
            )

        for match in INLINE_TEST_RE.finditer(cleaned):
            findings.append(
                Finding(
                    kind="inline_test_module",
                    path=rel_path,
                    line=line_number(cleaned, match.start()),
                    text="move the test body to a dedicated test file",
                )
            )

        for match in PLACEHOLDER_RE.finditer(cleaned):
            findings.append(
                Finding(
                    kind="production_placeholder",
                    path=rel_path,
                    line=line_number(cleaned, match.start()),
                    text=match.group(0).strip(),
                )
            )

        findings.extend(plumbing_findings(path, cleaned))

        if path.name not in {"mod.rs", "lib.rs"}:
            if not rel_path.startswith(ARCHITECTURE_ROOT):
                public_functions.extend(
                    (path, name, line) for name, line in public_free_functions(path, cleaned)
                )
            for name, line in namespace_types(cleaned):
                namespaces.append((path, name, line))
                if name.endswith("Tuner"):
                    findings.append(
                        Finding(
                            kind="stateless_tuner_namespace",
                            path=rel_path,
                            line=line,
                            text=f"{name} must own reusable tuning state",
                        )
                    )
            impl_methods.extend(
                (path, owner, method, line)
                for owner, method, line in namespace_impl_methods(cleaned)
            )

        source_lines = len(raw.splitlines())
        if source_lines > COHESION_LINE_LIMIT:
            large_files.append(SizeReview(path=rel_path, lines=source_lines))
        elif (
            path.name not in {"mod.rs", "lib.rs"}
            and not rel_path.startswith(ARCHITECTURE_ROOT)
            and source_lines < SMALL_FILE_LINE_LIMIT
        ):
            small_files.append(SizeReview(path=rel_path, lines=source_lines))

    _, edges, _, _ = collect_findings()
    symbol_importers = importers_by_symbol(edges)

    namespace_reviews: list[NamespaceReview] = []
    public_function_counts = Counter(name for _path, name, _line in public_functions)
    for path, name, line in public_functions:
        if public_function_counts[name] != 1:
            continue
        canonical = f"{module_name(path)}::{name}"
        source_dir = path.parent.relative_to(ROOT)
        cross_folder_importers = tuple(
            sorted(
                importer
                for importer in symbol_importers.get(canonical, set())
                if Path(importer).parent != source_dir
            )
        )
        if cross_folder_importers:
            namespace_reviews.append(
                NamespaceReview(
                    path=str(path.relative_to(ROOT)),
                    line=line,
                    function=name,
                    importers=cross_folder_importers,
                )
            )

    namespace_spreads: list[NamespaceSpreadReview] = []
    spread_exceptions = {"ArchKernels", "InternalMpUint", "InternalPrecisionContext", "Tuner"}
    namespace_counts = Counter(name for _path, name, _line in namespaces)
    for path, name, line in namespaces:
        if name in spread_exceptions or namespace_counts[name] != 1:
            continue
        canonical = f"{module_name(path)}::{name}"
        declaration_folder = path.parent.relative_to(ROOT)
        cohesion_root = namespace_family_root(path)
        importer_folders = tuple(
            sorted(
                {
                    str(Path(importer).parent)
                    for importer in symbol_importers.get(canonical, set())
                    if Path(importer).name not in {"mod.rs", "lib.rs"}
                    if Path(importer).parent != declaration_folder
                    if not importer.startswith("src/int/tune_api/")
                    if not (ROOT / importer).is_relative_to(cohesion_root)
                    if not is_expected_namespace_consumer(name, importer)
                }
            )
        )
        if importer_folders:
            namespace_spreads.append(
                NamespaceSpreadReview(
                    path=str(path.relative_to(ROOT)),
                    line=line,
                    namespace=name,
                    importer_folders=importer_folders,
                )
            )

    namespace_names = {name for name, count in namespace_counts.items() if count == 1}
    impl_method_reviews: list[ImplMethodReview] = []
    for path, owner, method, line in impl_methods:
        relative_parts = path.relative_to(ROOT).parts
        if owner not in namespace_names or relative_parts[:3] == ("src", "int", "api"):
            continue

        owner_impl = re.compile(rf"^\s*impl\s+{re.escape(owner)}\s*\{{", re.MULTILINE)

        local_calls = 0
        external_callers: set[Path] = set()
        for source_path, source in sources.items():
            calls = namespace_method_call_count(
                source,
                owner,
                method,
                include_self_calls=owner_impl.search(source) is not None,
            )
            if calls == 0:
                continue
            if source_path == path:
                local_calls += calls
            else:
                external_callers.add(source_path)

        if not external_callers and local_calls <= 1:
            impl_method_reviews.append(
                ImplMethodReview(
                    path=str(path.relative_to(ROOT)),
                    line=line,
                    namespace=owner,
                    method=method,
                    local_calls=local_calls,
                )
            )

    methods_by_namespace: dict[str, set[str]] = {}
    for _, owner, method, _ in impl_methods:
        if owner in namespace_names:
            methods_by_namespace.setdefault(owner, set()).add(method)
    namespace_arity_reviews: list[NamespaceArityReview] = []
    for path, name, line in namespaces:
        methods = methods_by_namespace.get(name, set())
        if len(methods) == 1:
            namespace_arity_reviews.append(
                NamespaceArityReview(
                    path=str(path.relative_to(ROOT)),
                    line=line,
                    namespace=name,
                    method=next(iter(methods)),
                )
            )

    findings.sort(key=lambda item: (item.kind, item.path, item.line, item.text))
    large_files.sort(key=lambda item: (-item.lines, item.path))
    small_files.sort(key=lambda item: (item.lines, item.path))
    namespace_reviews.sort(key=lambda item: (item.path, item.line, item.function))
    namespace_spreads.sort(key=lambda item: (item.path, item.line, item.namespace))
    impl_method_reviews.sort(
        key=lambda item: (item.path, item.line, item.namespace, item.method)
    )
    namespace_arity_reviews.sort(
        key=lambda item: (item.path, item.line, item.namespace, item.method)
    )
    return (
        findings,
        large_files,
        small_files,
        namespace_reviews,
        namespace_spreads,
        impl_method_reviews,
        namespace_arity_reviews,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    parser.add_argument("--limit", type=int, default=200, help="Maximum findings to print")
    parser.add_argument(
        "--advisory-limit",
        type=int,
        default=50,
        help="Maximum entries printed for each advisory review class",
    )
    parser.add_argument(
        "--size-inventory",
        action="store_true",
        help="Print the informational inventory of non-architecture implementation files below 200 lines",
    )
    parser.add_argument(
        "--path-prefix",
        help="Only report files below this repository-relative path",
    )
    args = parser.parse_args()

    (
        findings,
        large_files,
        small_files,
        namespace_reviews,
        namespace_spreads,
        impl_method_reviews,
        namespace_arity_reviews,
    ) = collect()
    if args.path_prefix is not None:
        path_prefix = args.path_prefix.removeprefix("./").rstrip("/")

        def selected(item: object) -> bool:
            path = getattr(item, "path")
            return path == path_prefix or path.startswith(f"{path_prefix}/")

        findings = [item for item in findings if selected(item)]
        large_files = [item for item in large_files if selected(item)]
        small_files = [item for item in small_files if selected(item)]
        namespace_reviews = [item for item in namespace_reviews if selected(item)]
        namespace_spreads = [item for item in namespace_spreads if selected(item)]
        impl_method_reviews = [item for item in impl_method_reviews if selected(item)]
        namespace_arity_reviews = [item for item in namespace_arity_reviews if selected(item)]
    counts = Counter(finding.kind for finding in findings)

    if args.json:
        print(
            json.dumps(
                {
                    "root": str(ROOT),
                    "path_prefix": args.path_prefix,
                    "counts": dict(counts),
                    "findings": [asdict(finding) for finding in findings],
                    "large_file_reviews": [asdict(size) for size in large_files],
                    "small_file_reviews": [asdict(size) for size in small_files],
                    "namespace_reviews": [asdict(review) for review in namespace_reviews],
                    "namespace_spread_reviews": [asdict(review) for review in namespace_spreads],
                    "impl_method_reviews": [asdict(review) for review in impl_method_reviews],
                    "namespace_arity_reviews": [
                        asdict(review) for review in namespace_arity_reviews
                    ],
                },
                indent=2,
            )
        )
    else:
        print(f"Structural findings: {len(findings)}")
        for kind, count in sorted(counts.items()):
            print(f"  {kind}: {count}")
        for finding in findings[: args.limit]:
            print(f"{finding.kind}: {finding.path}:{finding.line}: {finding.text}")
        print(
            f"Cohesion reviews (>{COHESION_LINE_LIMIT} lines, advisory): "
            f"{len(large_files)}"
        )
        for size in large_files[: args.advisory_limit]:
            print(f"  {size.lines:4d}  {size.path}")
        if args.size_inventory:
            print(
                "Small implementation inventory "
                f"(<{SMALL_FILE_LINE_LIMIT} lines; arch excluded; informational): "
                f"{len(small_files)}"
            )
            for size in small_files[: args.advisory_limit]:
                print(f"  {size.lines:4d}  {size.path}")
        print(
            "Cross-folder public free-function reviews "
            f"(namespace/inherent candidate, advisory): {len(namespace_reviews)}"
        )
        for review in namespace_reviews[: args.advisory_limit]:
            print(
                f"  {review.path}:{review.line}: {review.function} "
                f"({len(review.importers)} importer(s))"
            )
        print(
            "Cross-family namespace cohesion reviews "
            f"(advisory): {len(namespace_spreads)}"
        )
        for review in namespace_spreads[: args.advisory_limit]:
            print(
                f"  {review.path}:{review.line}: {review.namespace} "
                f"({len(review.importer_folders)} folder(s))"
            )
        print(
            "File-local namespace method reviews "
            f"(zero or one local call, advisory): {len(impl_method_reviews)}"
        )
        for review in impl_method_reviews[: args.advisory_limit]:
            print(
                f"  {review.path}:{review.line}: "
                f"{review.namespace}::{review.method} "
                f"({review.local_calls} local call(s), no cross-file caller)"
            )
        print(
            "Single-method namespace reviews "
            f"(prefer a real owner or private function, advisory): {len(namespace_arity_reviews)}"
        )
        for review in namespace_arity_reviews[: args.advisory_limit]:
            print(
                f"  {review.path}:{review.line}: "
                f"{review.namespace}::{review.method}"
            )

    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
