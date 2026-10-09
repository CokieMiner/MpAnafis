"""Repository source discovery shared by Rust audits."""

from __future__ import annotations

from pathlib import Path

from .common import ROOT

SOURCE_DIRECTORIES = (
    "src", "benches", "examples", "tests", "fuzz/src", "fuzz/fuzz_targets",
    "tools/tune", "build_support",
)


def rust_source_paths(root: Path = ROOT, *, production_only: bool = False) -> list[Path]:
    """Find maintained Rust sources without generated fuzz or Cargo artifacts."""
    directories = ("src",) if production_only else SOURCE_DIRECTORIES
    paths = {path for relative in directories for path in (root / relative).rglob("*.rs")}
    build_script = root / "build.rs"
    if not production_only and build_script.is_file():
        paths.add(build_script)
    return sorted(paths)
