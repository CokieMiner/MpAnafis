"""TSV output rendering, file diff verification, and inventory execution."""

from __future__ import annotations

import argparse
import csv
import difflib
import io
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from .inventory import Inventory
from .models import COLUMNS, DEFAULT_JSON, DEFAULT_OUTPUT, InventoryError, Row


def load_inventory(path: Path) -> Inventory:
    """Load and validate a rustdoc JSON file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise InventoryError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise InventoryError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(data, dict):
        raise InventoryError("rustdoc JSON top level must be an object")
    return Inventory(data)


def render_tsv(rows: List[Row]) -> str:
    """Render deterministic UTF-8 TSV with LF line endings on every platform."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(asdict(row))
    return buffer.getvalue()


def check_output(output: Path, expected: str) -> bool:
    """Compare an existing inventory byte-for-byte and print a useful diff."""
    try:
        with output.open("r", encoding="utf-8", newline="") as source:
            actual = source.read()
    except OSError as error:
        print(f"api inventory check failed: cannot read {output}: {error}", file=sys.stderr)
        return False
    if actual == expected:
        print(f"api inventory is up to date: {output}")
        return True
    diff = difflib.unified_diff(
        actual.splitlines(keepends=True),
        expected.splitlines(keepends=True),
        fromfile=str(output),
        tofile=f"{output} (generated)",
    )
    sys.stderr.writelines(diff)
    print(f"api inventory is stale: {output}", file=sys.stderr)
    return False


def run_api_inventory(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint to generate or verify API inventory."""
    parser = argparse.ArgumentParser(
        description="Generate the externally reachable integer API inventory from rustdoc JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--rustdoc-json",
        type=Path,
        default=DEFAULT_JSON,
        help=f"rustdoc JSON input (default: {DEFAULT_JSON})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"TSV output (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail without writing if the checked-in TSV differs",
    )
    args = parser.parse_args(argv)

    try:
        contents = render_tsv(load_inventory(args.rustdoc_json).rows())
    except InventoryError as error:
        print(f"api inventory failed: {error}", file=sys.stderr)
        return 2

    if args.check:
        return 0 if check_output(args.output, contents) else 1

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as output:
            output.write(contents)
    except OSError as error:
        print(f"api inventory failed: cannot write {args.output}: {error}", file=sys.stderr)
        return 2

    print(f"wrote {len(contents.splitlines()) - 1} API rows to {args.output}")
    return 0
