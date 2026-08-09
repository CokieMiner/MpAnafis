#!/usr/bin/env python3
"""Audit Rust imports and module-boundary usage.

Checks:
- deterministic top-level private import ordering and grouping
- inline qualified paths used in code bodies outside `use` statements
- deep relative imports such as `use super::super::...`
- same-subsystem imports that bypass immediate-parent wiring
- cross-subsystem imports that bypass a top-level crate facade
- cross-boundary imports into `logic/`
- surface bypasses that jump past a boundary's top-level `logic` re-export
- optionally emits Graphviz DOT/SVG graphs with staircase-boundary clusters

Usage:
    python3 tools/import_audit.py
    python3 tools/import_audit.py --json
    python3 tools/import_audit.py --dot tools/import_graph.dot
"""

from __future__ import annotations

import argparse
import json
import keyword
import re
import subprocess
import tempfile
import tomllib
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from rust_source import clean_rust_code, split_top_level

# ============================================================================
# CONFIGURATION
# ============================================================================
# Project-specific exceptions can be tuned here without digging into the logic.

CONFIG = {
    "whitelisted_crate_root_imports": set(),
    
    # Specific type aliases that are allowed without warnings.
    "whitelisted_alias_imports": {
        ("FmtResult", "core::fmt::Result"),
    },
    
    "whitelisted_alias_import_prefixes": {},

    # Primitive types that act as namespaces but are not PascalCase.
    "rust_primitive_types": {
        "f32", "f64", "f64x4", "i8", "i16", "i32", "i64", "i128", "isize",
        "u8", "u16", "u32", "u64", "u128", "usize", "str", "bool", "char"
    },

    # Specific inline qualified paths that are allowed (e.g. array::from_fn)
    "whitelisted_inline_qualified_paths": {
        "alloc::vec",
        "array::from_fn",
    },

    # Architecture dispatch owns direct wiring between selectors and concrete
    # backends. Routing these imports through every parent mod.rs would merely
    # duplicate cfg-sensitive re-export plumbing.
    "architecture_wiring_basenames": {
        "runtime_dispatch.rs",
    },
    "architecture_wiring_paths": {
        "src/int/logic/unsigned/math/arch/backend_providers.rs",
        "src/int/logic/unsigned/math/arch/kernel_selection.rs",
        "src/int/logic/unsigned/math/arch/kernels.rs",
        "src/int/logic/unsigned/math/arch/x86_selectors.rs",
    },
}

# ============================================================================

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RUST_KEYWORDS = {
    "Self", "async", "await", "break", "const", "continue", "crate", "dyn",
    "else", "enum", "extern", "false", "fn", "for", "if", "impl", "in", "let",
    "loop", "match", "mod", "move", "mut", "pub", "ref", "return", "self",
    "static", "struct", "super", "trait", "true", "type", "unsafe", "use",
    "where", "while",
}

INLINE_MOD_RE = re.compile(r"^\s*(?:pub(?:\([^)]+\))?\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{")
MODULE_DECL_RE = re.compile(
    r"^\s*(?:pub(?:\([^)]+\))?\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\{|;)"
)
QUALIFIED_PATH_RE = re.compile(
    r"((?:\$crate|[A-Za-z_][A-Za-z0-9_]*)(?:::[A-Za-z_][A-Za-z0-9_]*)+)"
)
MACRO_INVOCATION_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*!\s*[(\{\[]")
TEST_PATH_PARTS = {"tests", "benches"}
IMPORT_GROUPS = ("standard", "alloc", "external", "crate", "super", "local")


def cargo_dependency_roots() -> set[str]:
    """Return production dependency names exactly as Rust imports them."""

    manifest = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    names = set(manifest.get("dependencies", {}))
    for target in manifest.get("target", {}).values():
        names.update(target.get("dependencies", {}))
    return {name.replace("-", "_") for name in names}


EXTERNAL_CRATE_ROOTS = {"alloc", "core", "std"} | cargo_dependency_roots()


@dataclass
class Finding:
    kind: str
    path: str
    line: int
    text: str

@dataclass
class ImportEdge:
    source: str
    target: str
    raw: str
    path: str
    line: int
    internal: bool

@dataclass
class AliasImport:
    path: str
    line: int
    source_module: str
    target: str
    alias: str
    usage_count: int


def top_level_module_names(lines: list[str]) -> set[str]:
    names: set[str] = set()
    brace_depth = 0
    for line in lines:
        if brace_depth == 0:
            match = MODULE_DECL_RE.match(line)
            if match is not None:
                names.add(match.group(1))
        brace_depth += line.count("{") - line.count("}")
    return names


class FileAnalyzer:
    """Encapsulates the state for analyzing a single file."""
    
    def __init__(self, path: Path, boundaries: set[str]):
        self.path = path
        self.raw_text = path.read_text(encoding="utf-8")
        self.scrubbed_text = clean_rust_code(self.raw_text)
        self.lines = self.scrubbed_text.splitlines()
        self.raw_lines = self.raw_text.splitlines()
        self.boundaries = boundaries
        self.rel_path = str(path.relative_to(ROOT))
        self.base_module = module_name(path)
        self.local_modules = top_level_module_names(self.lines)
        
        self.nested_modules: list[tuple[str, int]] = []
        self.brace_depths = {'{': 0, '(': 0, '[': 0}
        self.macro_body_depth: int | None = None
        self.current_macro_name: str | None = None
        self.macro_open_char: str | None = None
        
        self.in_use_stmt = False
        self.use_stmt_lines = []
        self.use_start_lineno = 0
        self.seen_code = False
        
        self.finding_keys: set[tuple[str, str, int, str]] = set()
        self.edges: list[ImportEdge] = []
        self.alias_imports: list[AliasImport] = []
        self.macro_invocations: list[tuple[str, str, int]] = []
        self.macro_definitions: dict[str, set[str]] = {}
        self.file_imports: set[str] = set()
        self.private_imports: list[tuple[int, int, str, int]] = []
        self.public_imports: list[tuple[int, int, str]] = []
        self.module_declaration_lines: list[int] = []

    def analyze(self) -> None:
        for lineno, line in enumerate(self.lines, start=1):
            self._analyze_line(lineno, line)
        self._check_import_order()
        self._check_reexport_order()

    def _analyze_line(self, lineno: int, line: str) -> None:
        is_use_line = False
        if not self.in_use_stmt:
            m = re.match(r"^\s*(?:pub(?:\([^)]+\))?\s+)?use\s+(.*)", line)
            if m:
                self.in_use_stmt = True
                is_use_line = True
                self.use_start_lineno = lineno
                self.use_stmt_lines = [m.group(1)]
                if ";" in line:
                    self.in_use_stmt = False
                    self._process_collected_use_stmt(lineno)
        else:
            is_use_line = True
            self.use_stmt_lines.append(line)
            if ";" in line:
                self.in_use_stmt = False
                self._process_collected_use_stmt(lineno)

        # Check for modular structure
        mod_match = INLINE_MOD_RE.match(line)
        if mod_match:
            next_depth = self.brace_depths['{'] + line.count("{") - line.count("}")
            if next_depth > self.brace_depths['{']:
                self.nested_modules.append((mod_match.group(1), next_depth))

        mr_match = re.search(r"\bmacro_rules!\s+([A-Za-z_][A-Za-z0-9_]*)\s*([{\[(])", line)
        if self.macro_body_depth is None and mr_match:
            self.current_macro_name = mr_match.group(1)
            self.macro_open_char = mr_match.group(2)
            self.macro_body_depth = self.brace_depths[self.macro_open_char]

        inside_macro = (self.macro_body_depth is not None and self.macro_open_char and self.brace_depths[self.macro_open_char] > self.macro_body_depth)
        inside_test_module = any(is_test_module_name(name) for name, _depth in self.nested_modules)
        is_mod_line = bool(re.match(r"^\s*(pub(\s*\([^)]*\))?\s+)?mod\s+", line))
        if is_mod_line and self.brace_depths["{"] == 0:
            self.module_declaration_lines.append(lineno)

        if is_use_line and self.seen_code and not inside_test_module and not inside_macro and not self.nested_modules:
            self.finding_keys.add(("use_not_at_top", self.rel_path, lineno, "use statement found after other code"))

        if not is_use_line and not inside_test_module and not is_mod_line and not mr_match:
            stripped = line.strip()
            is_ignored_prefix = stripped.startswith("pub(in ") or stripped.startswith("macro_rules!")
            if stripped and stripped != "}" and stripped != "];":
                if not stripped.startswith("extern crate ") and not stripped.startswith("compile_error!") and not is_ignored_prefix and not inside_macro:
                    self.seen_code = True
            if not is_ignored_prefix:
                self._handle_code_line(line, lineno, inside_macro)

        self._update_braces(line)

    def _check_import_order(self) -> None:
        previous: tuple[int, int, str, int] | None = None
        for current in self.private_imports:
            line, end_line, source, group = current
            if previous is not None:
                _previous_line, previous_end, previous_source, previous_group = previous
                if group < previous_group:
                    self.finding_keys.add(
                        (
                            "import_group_order",
                            self.rel_path,
                            line,
                            f"{source} must precede {previous_source}; expected "
                            + " -> ".join(IMPORT_GROUPS),
                        )
                    )
                elif (
                    group == previous_group
                    and "{" not in source
                    and "{" not in previous_source
                    and import_sort_key(source) < import_sort_key(previous_source)
                ):
                    self.finding_keys.add(
                        (
                            "import_path_order",
                            self.rel_path,
                            line,
                            f"{source} must sort before {previous_source}",
                        )
                    )

                between = self.raw_lines[previous_end: line - 1]
                blank_lines = sum(not item.strip() for item in between)
                expected_blanks = 1 if group != previous_group else 0
                if blank_lines != expected_blanks:
                    self.finding_keys.add(
                        (
                            "import_group_spacing",
                            self.rel_path,
                            line,
                            f"{IMPORT_GROUPS[group]} import follows {IMPORT_GROUPS[previous_group]} "
                            f"with {blank_lines} blank line(s); expected {expected_blanks}",
                        )
                    )
            previous = current

    def _check_reexport_order(self) -> None:
        if self.path.name not in {"mod.rs", "lib.rs"} or not self.public_imports:
            return

        last_module_line = max(self.module_declaration_lines, default=0)
        for line, _end_line, source in self.public_imports:
            if line < last_module_line:
                self.finding_keys.add(
                    (
                        "reexport_before_module_declarations",
                        self.rel_path,
                        line,
                        f"{source} must follow all module declarations",
                    )
                )

        previous_source: str | None = None
        for line, _end_line, source in self.public_imports:
            if (
                previous_source is not None
                and "{" not in source
                and "{" not in previous_source
                and import_sort_key(source.split("::", 1)[0])
                < import_sort_key(previous_source.split("::", 1)[0])
            ):
                self.finding_keys.add(
                    (
                        "reexport_path_order",
                        self.rel_path,
                        line,
                        f"{source} must sort before {previous_source}",
                    )
                )
            previous_source = source

        first_reexport_line = self.public_imports[0][0]
        for line, _end_line, source, _group in self.private_imports:
            if line > first_reexport_line:
                self.finding_keys.add(
                    (
                        "private_import_after_reexport",
                        self.rel_path,
                        line,
                        f"{source} must precede the facade re-export block",
                    )
                )

    def _process_collected_use_stmt(self, end_lineno: int) -> None:
        full_use = " ".join(self.use_stmt_lines).split(";", 1)[0].strip()
        self.use_stmt_lines = []
        
        source_parts = self.base_module.split("::") + [name for name, _depth in self.nested_modules]
        if source_parts == [""]: source_parts = []
        source_mod = "::".join(source_parts)
        source_boundary = boundary_for_module(source_mod, self.boundaries)
        inside_test_module = any(is_test_module_name(name) for name, _depth in self.nested_modules)

        is_pub_use = bool(re.match(r"^\s*pub(?:\([^)]+\))?\s+use\b", self.lines[self.use_start_lineno - 1]))
        if not self.nested_modules and self.brace_depths["{"] == 0 and self.macro_body_depth is None:
            if is_pub_use:
                self.public_imports.append((self.use_start_lineno, end_lineno, full_use))
            else:
                groups = import_groups(full_use)
                if len(groups) > 1:
                    names = ", ".join(IMPORT_GROUPS[group] for group in sorted(groups))
                    self.finding_keys.add(
                        (
                            "mixed_import_groups",
                            self.rel_path,
                            self.use_start_lineno,
                            f"one use tree mixes import groups: {names}",
                        )
                    )
                self.private_imports.append(
                    (
                        self.use_start_lineno,
                        end_lineno,
                        full_use,
                        min(groups),
                    )
                )
        
        if inside_test_module:
            return
            
        self._handle_use_statement(
            full_use,
            source_mod,
            source_boundary,
            self.use_start_lineno,
            end_lineno,
        )

    def _handle_use_statement(
        self,
        raw_use: str,
        source_mod: str,
        source_boundary: str | None,
        lineno: int,
        end_lineno: int,
    ) -> None:
        is_pub_use = bool(re.match(r"^\s*pub(?:\([^)]+\))?\s+use\b", self.lines[lineno - 1]))
        for raw_target, alias in expand_use_tree(raw_use):
            normalized = resolve_relative_path(source_mod, raw_target)
            self.file_imports.add(normalized)
            self.edges.append(
                ImportEdge(
                    source=source_mod,
                    target=normalized,
                    raw=raw_target,
                    path=self.rel_path,
                    line=lineno,
                    internal=is_internal_target(normalized),
                )
            )

            if alias and alias != "_" and not is_whitelisted_alias_import_for_path(self.rel_path, alias, normalized):
                usage_count = count_alias_usage(self.lines, alias, end_lineno)
                if not (usage_count == 0 and is_pub_use):
                    self.alias_imports.append(
                        AliasImport(
                            path=self.rel_path,
                            line=lineno,
                            source_module=source_mod,
                            target=normalized,
                            alias=alias,
                            usage_count=usage_count,
                        )
                    )
                if usage_count == 0 and not is_pub_use:
                    self.finding_keys.add(("unused_alias_import", self.rel_path, lineno, f"{alias} -> {normalized}"))

            if raw_target.startswith("super::super"):
                self.finding_keys.add(("deep_relative_import", self.rel_path, lineno, raw_target))

            import_violation = project_import_violation(
                raw_target,
                source_path=self.rel_path,
                source_mod=source_mod,
                plumbing_file=Path(self.rel_path).name in {"mod.rs", "lib.rs"},
                local_modules=self.local_modules,
            )
            if import_violation is not None:
                self.finding_keys.add((import_violation, self.rel_path, lineno, raw_target))

            if raw_target.startswith("crate::"):
                if source_boundary is not None:
                    target_boundary = boundary_for_module(normalized, self.boundaries)
                    if target_boundary == source_boundary:
                        self.finding_keys.add(("self_referential_crate_import", self.rel_path, lineno, f"{raw_target} (should be relative)"))

            if is_shallow_crate_import(raw_target) and not is_whitelisted_crate_root_import(raw_target):
                self.finding_keys.add(("crate_root_import", self.rel_path, lineno, raw_target))

            if "::logic::" in normalized and normalized.startswith("src::"):
                logic_boundary = normalized.split("::logic::", 1)[0]
                if source_boundary != logic_boundary:
                    self.finding_keys.add(("cross_boundary_logic_import", self.rel_path, lineno, f"{source_mod} -> {normalized}"))
                elif not (source_mod == logic_boundary or "::api" in source_mod or "::tune_api" in source_mod or source_mod.startswith(f"{logic_boundary}::logic")):
                    after_logic = normalized.split("::logic::", 1)[1]
                    if "::" in after_logic:
                        self.finding_keys.add(("deep_logic_surface_bypass", self.rel_path, lineno, f"{source_mod} -> {normalized}"))
                    else:
                        self.finding_keys.add(("logic_surface_import", self.rel_path, lineno, f"{source_mod} -> {normalized}"))

    def _handle_code_line(self, line: str, lineno: int, inside_macro: bool) -> None:
        for inv_match in MACRO_INVOCATION_RE.findall(line):
            if inv_match not in {"macro_rules"}:
                self.macro_invocations.append((self.rel_path, inv_match, lineno))

        for match in qualified_path_matches(line):
            if inside_macro and self.current_macro_name:
                resolved_path = match
                if resolved_path.startswith("$crate::"):
                    resolved_path = "src::" + resolved_path[8:]
                elif resolved_path.startswith("crate::"):
                    resolved_path = "src::" + resolved_path[7:]
                self.macro_definitions.setdefault(self.current_macro_name, set()).add(resolved_path)
            else:
                self.finding_keys.add(("inline_qualified_path", self.rel_path, lineno, match))

    def _update_braces(self, line: str) -> None:
        self.brace_depths['{'] += line.count("{") - line.count("}")
        self.brace_depths['('] += line.count("(") - line.count(")")
        self.brace_depths['['] += line.count("[") - line.count("]")
        
        if self.macro_body_depth is not None and self.macro_open_char and self.brace_depths[self.macro_open_char] <= self.macro_body_depth:
            self.macro_body_depth = None
            self.current_macro_name = None
            self.macro_open_char = None
            
        while self.nested_modules and self.brace_depths['{'] < self.nested_modules[-1][1]:
            self.nested_modules.pop()


def module_name(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "mod":
        parts.pop()
    return "::".join(parts)


def staircase_boundaries() -> set[str]:
    boundaries: set[str] = set()
    for directory in SRC.rglob("*"):
        if not directory.is_dir():
            continue
        has_api = (directory / "api.rs").exists() or (directory / "api" / "mod.rs").exists()
        if has_api and (directory / "logic").is_dir():
            rel = directory.relative_to(ROOT)
            boundaries.add("::".join(rel.parts))
    return boundaries


def is_test_path(path: Path) -> bool:
    if any(part in TEST_PATH_PARTS for part in path.parts):
        return True
    stem = path.stem
    return stem == "tests" or stem.endswith("_tests")


def is_test_module_name(name: str) -> bool:
    return name == "tests" or name.endswith("_tests")


def split_alias(path: str) -> str:
    parts = re.split(r"\s+as\s+", path, maxsplit=1)
    return parts[0].strip()


def import_target_group(raw_target: str) -> int:
    target = raw_target.lstrip(":").strip()
    root = target.split("::", 1)[0].split("{", 1)[0].strip()
    if root in {"core", "std"}:
        return 0
    if root == "alloc":
        return 1
    if root in EXTERNAL_CRATE_ROOTS:
        return 2
    if root == "crate":
        return 3
    if root == "super":
        return 4
    return 5


def import_groups(raw_use: str) -> set[int]:
    """Return every prescribed group represented by one use tree."""

    return {import_target_group(target) for target, _alias in expand_use_tree(raw_use)}


def import_sort_key(source: str) -> tuple[tuple[int, str | int], ...]:
    """Approximate rustfmt's numeric-aware ordering for import paths."""

    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", source)
        if part
    )


def parse_alias(path: str) -> tuple[str, str | None]:
    parts = re.split(r"\s+as\s+", path, maxsplit=1)
    target = parts[0].strip()
    alias = parts[1].strip() if len(parts) == 2 else None
    return target, alias


def expand_use_tree(tree: str) -> list[tuple[str, str | None]]:
    tree = tree.strip()
    if tree.startswith("{") and tree.endswith("}"):
        paths: list[tuple[str, str | None]] = []
        for item in split_top_level(tree[1:-1]):
            paths.extend(expand_use_tree(item))
        return paths

    depth = 0
    group_start = -1
    prefix_end = -1
    for idx, char in enumerate(tree):
        if char == "{":
            if depth == 0:
                group_start = idx
                prefix_end = idx - 2
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and group_start != -1:
                prefix = tree[:prefix_end].strip()
                group = tree[group_start + 1 : idx]
                suffix = tree[idx + 1 :].strip()
                if suffix.startswith("::"):
                    suffix = suffix[2:]
                paths: list[tuple[str, str | None]] = []
                for item in split_top_level(group):
                    expanded = expand_use_tree(item)
                    for part, alias in expanded:
                        combined = prefix if part == "self" else f"{prefix}::{part}"
                        if suffix:
                            combined = f"{combined}::{suffix}"
                        paths.append((combined, alias))
                return paths

    target, alias = parse_alias(tree)
    return [(target, alias)]


def resolve_relative_path(source: str, target: str) -> str:
    target = split_alias(target)
    if target == "self":
        return source
    if target == "super":
        return "::".join(source.split("::")[:-1])
    if target.startswith("crate::"):
        return f"src::{target[7:]}"
    if target.startswith("self::"):
        return f"{source}::{target[6:]}"
    if target.startswith("super::"):
        parts = source.split("::")
        rest = target
        while rest.startswith("super::"):
            rest = rest[7:]
            if len(parts) > 1:
                parts.pop()
        if rest:
            parts.extend(rest.split("::"))
        return "::".join(parts)
    root = target.lstrip(":").split("::", maxsplit=1)[0]
    if root not in EXTERNAL_CRATE_ROOTS:
        return f"{source}::{target}"
    return target


def boundary_for_module(module: str, boundaries: set[str]) -> str | None:
    candidates = [
        boundary
        for boundary in boundaries
        if module == boundary or module.startswith(f"{boundary}::")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.count("::"))


def is_internal_target(target: str) -> bool:
    return target.startswith("src::")


def is_shallow_crate_import(raw_target: str) -> bool:
    if not raw_target.startswith("crate::"):
        return False
    return "::" not in raw_target[7:]


def is_whitelisted_crate_root_import(raw_target: str) -> bool:
    return raw_target in CONFIG["whitelisted_crate_root_imports"]


def top_level_subsystem(module: str) -> str | None:
    parts = module.split("::")
    if len(parts) >= 2 and parts[0] == "src":
        return parts[1]
    return None


def is_architecture_wiring_file(path: str) -> bool:
    architecture_root = "src/int/logic/unsigned/math/arch/"
    return path.startswith(architecture_root) and (
        Path(path).name in CONFIG["architecture_wiring_basenames"]
        or path in CONFIG["architecture_wiring_paths"]
    )


def project_import_violation(
    raw_target: str,
    *,
    source_path: str,
    source_mod: str,
    plumbing_file: bool,
    local_modules: set[str] | None = None,
) -> str | None:
    """Classify a project-local import that bypasses an architectural facade.

    Within a top-level subsystem, implementation files receive names only as
    ``super::Thing``. Cross-subsystem dependencies use exactly
    ``crate::subsystem::Thing``. Plumbing may additionally bind a direct child
    as ``child::Thing``. Standard-library and dependency roots keep their
    native module depth.
    """
    target = split_alias(raw_target)
    parts = [part for part in target.split("::") if part]
    if not parts or parts[0] in EXTERNAL_CRATE_ROOTS:
        return None

    if parts[0] == "crate":
        if len(parts) != 3:
            return "deep_project_import"
        source_subsystem = top_level_subsystem(source_mod)
        target_subsystem = parts[1]
        if source_subsystem == target_subsystem:
            return "non_parent_plumbing_import" if plumbing_file else "non_parent_implementation_import"
        return None

    if is_architecture_wiring_file(source_path):
        return None

    if len(parts) == 2 and parts[0] in (local_modules or set()):
        return None

    if plumbing_file:
        if parts[0] == "super":
            return None if len(parts) == 2 else "non_parent_plumbing_import"
        if parts[0] == "self":
            return None if len(parts) <= 3 else "deep_project_import"
        return None if len(parts) <= 2 else "deep_project_import"

    if parts[0] != "super" or len(parts) != 2:
        return "non_parent_implementation_import"
    return None


def is_whitelisted_alias_import(alias: str, target: str) -> bool:
    return (alias, target) in CONFIG["whitelisted_alias_imports"]


def is_whitelisted_alias_import_for_path(path: str, alias: str, target: str) -> bool:
    if is_whitelisted_alias_import(alias, target):
        return True
    for prefix, allowed in CONFIG["whitelisted_alias_import_prefixes"].items():
        if path.startswith(prefix) and (alias, target) in allowed:
            return True
    return False


def qualified_path_matches(line: str) -> list[str]:
    matches: list[str] = []
    for qp in QUALIFIED_PATH_RE.finditer(line):
        candidate = qp.group(1).strip()
        parts = candidate.split("::")
        
        # Associated paths on types remain idiomatic; module-qualified calls
        # should instead bind their owning namespace through a shallow import.
        if len(parts) == 2:
            head = parts[0]
            if head not in ("crate", "super", "self"):
                is_pascal_case = head and head[0].isupper()
                is_primitive = head in CONFIG["rust_primitive_types"]
                if is_pascal_case or is_primitive:
                    continue
            if candidate in CONFIG.get("whitelisted_inline_qualified_paths", set()):
                continue

        # Skip double-underscore items
        if any(seg.startswith("__") for seg in parts):
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*::<", candidate):
            continue
        if "::{" in candidate:
            continue
            
        head = parts[0]
        if head in RUST_KEYWORDS or keyword.iskeyword(head):
            matches.append(candidate)
            continue
            
        matches.append(candidate)
    return matches


def count_alias_usage(lines: list[str], alias: str, end_line: int) -> int:
    pattern = re.compile(rf"\b{re.escape(alias)}\b")
    return sum(len(pattern.findall(line)) for line in lines[end_line:])


def collect_findings() -> tuple[list[Finding], list[ImportEdge], set[str], list[AliasImport]]:
    boundaries = staircase_boundaries()
    finding_keys: set[tuple[str, str, int, str]] = set()
    edges: list[ImportEdge] = []
    alias_imports: list[AliasImport] = []
    
    macro_definitions: dict[str, set[str]] = {}
    macro_invocations: list[tuple[str, str, int]] = []
    file_imports: dict[str, set[str]] = {}

    for path in sorted(SRC.rglob("*.rs")):
        if is_test_path(path):
            continue
            
        analyzer = FileAnalyzer(path, boundaries)
        analyzer.analyze()
        
        finding_keys.update(analyzer.finding_keys)
        edges.extend(analyzer.edges)
        alias_imports.extend(analyzer.alias_imports)
        
        # Merge macro info
        macro_invocations.extend(analyzer.macro_invocations)
        for macro, reqs in analyzer.macro_definitions.items():
            macro_definitions.setdefault(macro, set()).update(reqs)
        
        file_imports[analyzer.rel_path] = analyzer.file_imports

    # Process macros globally
    macro_import_missing_callers: dict[tuple[str, str], list[str]] = {}
    for file_path, inv_macro, lineno in macro_invocations:
        if inv_macro in macro_definitions:
            required_imports = macro_definitions[inv_macro]
            actual_imports = file_imports.get(file_path, set())
            
            for req_import in required_imports:
                caller_mod = file_path.replace('.rs', '').replace('/', '::')
                if caller_mod.endswith('::mod'):
                    caller_mod = caller_mod[:-5]
                req_mod = "::".join(req_import.split("::")[:-1])
                
                is_available = (req_import in actual_imports) or (caller_mod == req_mod)
                if not is_available:
                    macro_import_missing_callers.setdefault((inv_macro, req_import), []).append(f"{file_path}:{lineno}")

    for macro, req_imports in macro_definitions.items():
        for req_import in req_imports:
            missing_callers = macro_import_missing_callers.get((macro, req_import), [])
            if not missing_callers:
                finding_keys.add(
                    ("macro_inline_path_safe_to_remove", f"macro: {macro}", 1,
                     f"Macro uses inline '{req_import}' - SAFE to simplify because all callers import it")
                )

    findings = [Finding(kind=kind, path=path, line=line, text=text) for kind, path, line, text in sorted(finding_keys)]
    alias_imports.sort(key=lambda item: (item.path, item.line, item.alias, item.target))
    return findings, edges, boundaries, alias_imports


def write_dot(edges: list[ImportEdge], boundaries: set[str], dot_path: Path) -> None:
    dot_path.parent.mkdir(parents=True, exist_ok=True)
    nodes = {edge.source for edge in edges}
    nodes.update(edge.target for edge in edges if edge.internal)

    boundary_map: dict[str, list[str]] = {boundary: [] for boundary in sorted(boundaries)}
    loose_nodes: list[str] = []
    
    for node in sorted(nodes):
        boundary = boundary_for_module(node, boundaries)
        if boundary:
            boundary_map.setdefault(boundary, []).append(node)
        else:
            loose_nodes.append(node)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=dot_path.parent,
            prefix=f".{dot_path.name}.",
            delete=False,
        ) as fh:
            temporary_path = Path(fh.name)
            fh.write("digraph imports {\n  rankdir=\"LR\";\n  graph [fontname=\"monospace\"];\n  node [shape=\"box\", fontname=\"monospace\"];\n  edge [fontname=\"monospace\"];\n")
            cluster_index = 0
            for boundary, boundary_nodes in boundary_map.items():
                if not boundary_nodes:
                    continue
                fh.write(f"  subgraph cluster_{cluster_index} {{\n    label=\"{boundary}\";\n    color=\"lightgrey\";\n")
                for node in sorted(boundary_nodes):
                    fill = "white"
                    if node == boundary:
                        fill = "lightblue"
                    elif "::logic" in node:
                        fill = "lightyellow"
                    elif node.endswith("::api"):
                        fill = "honeydew"
                    fh.write(f'    "{node}" [style="filled", fillcolor="{fill}"];\n')
                fh.write("  }\n")
                cluster_index += 1

            for node in loose_nodes:
                fh.write(f'  "{node}";\n')

            for edge in edges:
                target = edge.target
                if not edge.internal:
                    if "::" in target:
                        target = target.split("::", 1)[0]
                    fh.write(f'  "{target}" [shape="ellipse", style="dashed"];\n')
                safe_source = edge.source.replace('"', '\\"')
                safe_target = target.replace('"', '\\"')
                fh.write(f'  "{safe_source}" -> "{safe_target}";\n')
            fh.write("}\n")
        temporary_path.replace(dot_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def svg_path_for(dot_path: Path) -> Path:
    """Return a distinct SVG sibling for any DOT filename."""

    candidate = dot_path.with_suffix(".svg")
    if candidate == dot_path:
        return dot_path.with_name(f"{dot_path.name}.svg")
    return candidate


def write_svg(dot_path: Path, svg_path: Path) -> None:
    """Render Graphviz output atomically, preserving any prior SVG on failure."""

    svg_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=svg_path.parent,
            prefix=f".{svg_path.name}.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        subprocess.run(
            ["dot", "-Tsvg", str(dot_path), "-o", str(temporary_path)],
            check=True,
        )
        temporary_path.replace(svg_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def structure_findings(boundaries: set[str]) -> list[Finding]:
    findings: list[Finding] = []
    for boundary in sorted(boundaries):
        rel = ROOT.joinpath(*boundary.split("::"))
        if not (rel / "mod.rs").exists():
            findings.append(Finding(kind="missing_boundary_mod", path=str(rel.relative_to(ROOT)), line=1, text=f"{boundary} has api.rs and logic/ but no mod.rs"))
        if not (rel / "logic" / "mod.rs").exists():
            findings.append(Finding(kind="missing_logic_mod", path=str((rel / "logic").relative_to(ROOT)), line=1, text=f"{boundary}::logic is missing mod.rs"))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Emit JSON findings")
    parser.add_argument(
        "--dot",
        help="Opt in to Graphviz DOT and SVG generation at the given path",
    )
    parser.add_argument("--limit", type=int, default=200, help="Maximum number of findings to print in text mode")
    parser.add_argument(
        "--path-prefix",
        help="Only report files below this repository-relative path",
    )
    args = parser.parse_args()

    findings, edges, boundaries, alias_imports = collect_findings()
    findings.extend(structure_findings(boundaries))
    findings.sort(key=lambda item: (item.kind, item.path, item.line, item.text))
    if args.path_prefix is not None:
        path_prefix = args.path_prefix.removeprefix("./").rstrip("/")
        findings = [
            finding
            for finding in findings
            if finding.path == path_prefix or finding.path.startswith(f"{path_prefix}/")
        ]
        alias_imports = [
            item
            for item in alias_imports
            if item.path == path_prefix or item.path.startswith(f"{path_prefix}/")
        ]
    graph: dict[str, str | None] | None = None
    if args.dot is not None:
        dot_path = Path(args.dot)
        svg_path = svg_path_for(dot_path)
        write_dot(edges, boundaries, dot_path)
        graph = {"dot": str(dot_path), "svg": str(svg_path), "error": None}
        try:
            write_svg(dot_path, svg_path)
        except (FileNotFoundError, OSError, subprocess.CalledProcessError) as error:
            graph["error"] = str(error)
    
    counts = Counter(finding.kind for finding in findings)
    unused_alias_count = sum(1 for item in alias_imports if item.usage_count == 0)

    if args.json:
        print(json.dumps({
            "root": str(ROOT),
            "path_prefix": args.path_prefix,
            "graph": graph,
            "boundary_count": len(boundaries),
            "boundaries": sorted(boundaries),
            "counts": dict(counts),
            "alias_import_count": len(alias_imports),
            "unused_alias_import_count": unused_alias_count,
            "alias_imports": [asdict(alias_import) for alias_import in alias_imports],
            "findings": [asdict(finding) for finding in findings],
        }, indent=2))
        return int(bool(findings))

    if graph is not None:
        print(f"DOT graph written to: {graph['dot']}")
        if graph["error"] is None:
            print(f"SVG graph written to: {graph['svg']}")
        else:
            print(f"Warning: SVG generation failed: {graph['error']}")
    print(f"Boundaries detected: {len(boundaries)}")
    print(f"Findings: {len(findings)}")
    print(f"Aliased imports: {len(alias_imports)}")
    print(f"Unused aliased imports: {unused_alias_count}")
    for kind, count in sorted(counts.items()):
        print(f"  {kind}: {count}")
    for finding in findings[: args.limit]:
        print(f"{finding.kind}: {finding.path}:{finding.line}: {finding.text}")

    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
