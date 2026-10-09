"""Keep working benchmark artifacts separate from retained documentation records."""

from pathlib import Path

from .models import BenchmarkError

ROOT = Path(__file__).resolve().parents[2]


def validate_output_path(path: Path, *, documentation: bool = False, root: Path = ROOT) -> Path:
    """Resolve symlinks before enforcing repository output placement."""
    resolved = path.resolve()
    root = root.resolve()
    allowed = root / ("docs/int/benchmarks" if documentation else "target/bench-results")
    if resolved.is_relative_to(root) and not resolved.is_relative_to(allowed):
        raise BenchmarkError(f"repository benchmark output must be under {allowed}: {resolved}")
    return resolved
