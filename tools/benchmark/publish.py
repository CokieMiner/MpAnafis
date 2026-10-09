"""Export size-sweep figures with measurement details; keep raw evidence local."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .evidence import validate_record
from .models import BenchmarkError
from .paths import ROOT, validate_output_path
from .report import plot_summary, size_sweeps, summarize


def publish_run(source: Path, root: Path, name: str, description: str) -> Path:
    root = validate_output_path(root, documentation=True)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise BenchmarkError("record name must contain lowercase letters, digits, underscores, or hyphens")
    if not description.strip():
        raise BenchmarkError("a record needs a description of its scope and interpretation")
    metadata, rows, files = validate_record(source)
    summary = summarize(rows)
    if not size_sweeps(summary):
        raise BenchmarkError("documentation export requires a size sweep; keep single-size results in local tables")
    destination = validate_output_path(root / "public_api" / name, documentation=True)
    if destination.exists():
        raise BenchmarkError(f"documentation record already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A failed plot/export never leaves a partially published record.
    with tempfile.TemporaryDirectory(prefix=".bench-export-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "record"
        staged.mkdir()
        figures = plot_summary(summary, staged)
        dirty = bool(metadata.get("git_status", "").strip())
        record_input = source.resolve() / "measurements.json"
        try:
            record_input = record_input.relative_to(ROOT)
        except ValueError:
            pass
        rebuild = shlex.join(["python3", "tools/bench.py", "report", str(record_input),
                              "--output", f"target/bench-results/{name}-report", "--plot"])
        readme = [
            f"# {name}", "", description.strip(), "",
            "- Suite: public_api",
            f"- Configuration: `{metadata['configuration']}`",
            f"- Revision: `{metadata.get('git_head', 'unknown')}`",
            f"- Worktree at measurement: {'dirty; the revision alone does not reproduce these sources' if dirty else 'clean'}.",
            f"- CPU: {metadata.get('processor', 'unknown')}",
            f"- Measured subprocesses: {len(metadata['plan']['runs'])}",
            f"- Function/scenario groups: {len({row.path for row in rows})}",
            "", "## Time versus size", "",
            *[f"- [{figure.with_suffix('').as_posix()}]({figure.as_posix()})" for figure in figures],
            "", "Times are per benchmark iteration, including the case's documented operand batch.",
            "Points are medians of run medians; error bars show observed run ranges, not confidence intervals.",
            "Only engines with at least two numeric size arguments are plotted; other results remain in local tables.",
            "", "The [measurement details and figure checksums](manifest.json) include the host, compiler, plan, and source hashes.",
            "Raw captures and numeric reports remain in the ignored working run directory.",
            "They are validated before export but are not included in this Git record.",
            "Retain that run separately when sharing reproducible measurement evidence.",
            "", "From the repository root, regenerate summaries and plots into a working directory:", "",
            "```sh",
            rebuild,
            "```",
        ]
        (staged / "README.md").write_text("\n".join(readme) + "\n")
        checksums = {str(path.relative_to(staged)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(staged.rglob("*")) if path.is_file()}
        (staged / "manifest.json").write_text(json.dumps({
            "schema_version": 2, "suite": "public_api", "name": name,
            "exported_utc": datetime.now(timezone.utc).isoformat(),
            "measurement_details": {key: metadata[key] for key in (
                "configuration", "binary_sha256", "git_head", "rustc", "platform", "plan",
                "processor", "configuration_details", "started_unix", "finished_unix")
                           if key in metadata} | {"dirty_worktree": dirty},
            "source_files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in [source / "run.json", source / "measurements.json", *files]},
            "files": checksums,
        }, indent=2) + "\n")
        staged.rename(destination)
    return destination
