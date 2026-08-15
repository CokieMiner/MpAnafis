"""Data models and constants for the API inventory generator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = ROOT / "target" / "doc" / "mp_anafis.json"
DEFAULT_OUTPUT = ROOT / "tools" / "api_inventory.tsv"
EXPECTED_CRATE = "mp_anafis"
SUPPORTED_FORMAT_VERSIONS = {60, 61}
COLUMNS = (
    "root_path",
    "receiver",
    "impl_kind",
    "trait_path",
    "item_kind",
    "item_name",
    "source_file",
    "source_line",
    "cfg",
    "attrs",
    "signature",
)


class InventoryError(ValueError):
    """The input is not the rustdoc JSON schema/configuration we require."""


@dataclass(frozen=True)
class Route:
    """One public path from the crate root to a rustdoc item."""
    path: str
    attrs: Tuple[Any, ...]


@dataclass(frozen=True)
class Row:
    """One deterministic inventory record."""
    root_path: str
    receiver: str
    impl_kind: str
    trait_path: str
    item_kind: str
    item_name: str
    source_file: str
    source_line: int | str
    cfg: str
    attrs: str
    signature: str


def compact_json(value: Any) -> str:
    """Serialize structured fields deterministically and without TSV newlines."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def is_public(item: dict[str, Any]) -> bool:
    """Return whether an item is directly public at its containing boundary."""
    return item.get("visibility") == "public"


def is_hidden(item: dict[str, Any]) -> bool:
    """Recognize rustdoc's current encodings of #[doc(hidden)]."""
    return any(
        "DocHidden" in compact_json(attr) or "doc(hidden)" in compact_json(attr)
        for attr in item.get("attrs", [])
    )


def is_cfg_attr(attr: Any) -> bool:
    """Recognize rustdoc's evaluated cfg/cfg_attr trace attributes."""
    encoded = compact_json(attr)
    return "CfgTrace(" in encoded or "CfgAttrTrace(" in encoded
