#!/usr/bin/env python3
"""Generate the externally reachable integer API inventory from rustdoc JSON.

Rustdoc JSON is intentionally unstable, so generation is a separate, explicit
step and this tool accepts only the schema version it has been reviewed against::

    cargo rustdoc --lib --all-features -- \\
        -Z unstable-options --output-format json --document-private-items \\
        -A rustdoc::broken-intra-doc-links
    python3 tools/api_inventory.py \\
        --rustdoc-json target/doc/arbi_anafis.json \\
        --output tools/api_inventory.tsv
    python3 tools/api_inventory.py \\
        --rustdoc-json target/doc/arbi_anafis.json \\
        --output tools/api_inventory.tsv --check

Private-item JSON is required because rustdoc records an enclosing module's
``cfg`` trace on that module, not necessarily on every expanded impl below it.
Reachability is nevertheless computed only through public paths from the crate
root. Signatures are normalized structured JSON: unlike source text, it keeps
resolved types and generic constraints without depending on rustdoc item IDs.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import io
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "target" / "doc" / "arbi_anafis.json"
DEFAULT_OUTPUT = ROOT / "tools" / "api_inventory.tsv"
EXPECTED_CRATE = "arbi_anafis"
SUPPORTED_FORMAT_VERSIONS = {60}
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
    attrs: tuple[Any, ...]


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
    """Recognize rustdoc's current encodings of ``#[doc(hidden)]``."""

    return any(
        "DocHidden" in compact_json(attr) or "doc(hidden)" in compact_json(attr)
        for attr in item.get("attrs", [])
    )


def is_cfg_attr(attr: Any) -> bool:
    """Recognize rustdoc's evaluated cfg/cfg_attr trace attributes."""

    encoded = compact_json(attr)
    return "CfgTrace(" in encoded or "CfgAttrTrace(" in encoded


class Inventory:
    """Validated rustdoc JSON plus public-reachability and rendering helpers."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self.index = data.get("index")
        self.paths = data.get("paths")
        if not isinstance(self.index, dict) or not isinstance(self.paths, dict):
            raise InventoryError("rustdoc JSON must contain object-valued index and paths")

        version = data.get("format_version")
        if version not in SUPPORTED_FORMAT_VERSIONS:
            expected = ", ".join(str(value) for value in sorted(SUPPORTED_FORMAT_VERSIONS))
            raise InventoryError(
                f"unsupported rustdoc JSON format_version {version!r}; expected {expected}"
            )
        if data.get("includes_private") is not True:
            raise InventoryError(
                "rustdoc JSON must be generated with --document-private-items so "
                "enclosing-module cfg traces are available"
            )

        self.root_id = data.get("root")
        root = self.item(self.root_id)
        module = root.get("inner", {}).get("module")
        if root.get("name") != EXPECTED_CRATE or not isinstance(module, dict):
            raise InventoryError(
                f"rustdoc root must be the {EXPECTED_CRATE!r} crate module"
            )
        if module.get("is_crate") is not True or root.get("crate_id") != 0:
            raise InventoryError("rustdoc root has inconsistent crate-module metadata")

        self.crate_name = root["name"]
        self.module_items = self._module_items_by_filename()
        self.routes = self._public_routes()
        if not self.routes:
            raise InventoryError("crate root exposes no public, non-hidden items")

    def item(self, item_id: Any) -> dict[str, Any]:
        """Resolve an item ID while accepting JSON's stringified map keys."""

        item = self.index.get(str(item_id)) if isinstance(self.index, dict) else None
        if not isinstance(item, dict):
            raise InventoryError(f"rustdoc index is missing item {item_id!r}")
        if item.get("id") != item_id:
            raise InventoryError(f"rustdoc item {item_id!r} has a mismatched id field")
        if not isinstance(item.get("inner"), dict) or len(item["inner"]) != 1:
            raise InventoryError(f"rustdoc item {item_id!r} has an invalid inner value")
        return item

    @staticmethod
    def kind(item: dict[str, Any]) -> str:
        """Return the single rustdoc item-kind discriminator."""

        return next(iter(item["inner"]))

    def _module_items_by_filename(self) -> dict[str, list[dict[str, Any]]]:
        modules: dict[str, list[dict[str, Any]]] = {}
        for item in self.index.values():
            if not isinstance(item, dict) or self.kind(item) != "module":
                continue
            span = item.get("span")
            if isinstance(span, dict) and isinstance(span.get("filename"), str):
                modules.setdefault(span["filename"], []).append(item)
        return modules

    def _public_routes(self) -> dict[int, list[Route]]:
        routes: dict[int, list[Route]] = {}
        visited: set[tuple[int, tuple[str, ...]]] = set()

        def add(item_id: int, parts: tuple[str, ...], attrs: tuple[Any, ...]) -> None:
            route = Route("::".join(parts), attrs)
            existing = routes.setdefault(item_id, [])
            if route not in existing:
                existing.append(route)

        def visit_module(
            module_id: int, parts: tuple[str, ...], inherited: tuple[Any, ...]
        ) -> None:
            marker = (module_id, parts)
            if marker in visited:
                return
            visited.add(marker)
            module_item = self.item(module_id)
            module = module_item["inner"].get("module")
            if not isinstance(module, dict) or not isinstance(module.get("items"), list):
                raise InventoryError(f"item {module_id!r} is not a valid module")
            for child_id in module["items"]:
                child = self.item(child_id)
                if not is_public(child) or is_hidden(child):
                    continue
                child_attrs = inherited + tuple(child.get("attrs", []))
                kind = self.kind(child)
                if kind == "use":
                    use = child["inner"]["use"]
                    target_id = use.get("id")
                    name = use.get("name")
                    if target_id is None or not isinstance(name, str):
                        raise InventoryError(f"public use item {child_id!r} has no target/name")
                    target = self.item(target_id)
                    if is_hidden(target):
                        continue
                    target_attrs = child_attrs + tuple(target.get("attrs", []))
                    if use.get("is_glob"):
                        visit_module(target_id, parts, target_attrs)
                    else:
                        target_parts = parts + (name,)
                        add(target_id, target_parts, child_attrs)
                        if self.kind(target) == "module":
                            visit_module(target_id, target_parts, target_attrs)
                    continue
                name = child.get("name")
                if not isinstance(name, str):
                    raise InventoryError(f"public item {child_id!r} has no name")
                child_parts = parts + (name,)
                add(child_id, child_parts, child_attrs)
                if kind == "module":
                    visit_module(child_id, child_parts, child_attrs)

        visit_module(self.root_id, (self.crate_name,), ())
        for item_routes in routes.values():
            item_routes.sort(key=lambda route: route.path)
        return routes

    def canonical_path(self, item_id: Any, fallback: str) -> str:
        """Prefer a public crate-root alias, then rustdoc's canonical path."""

        if item_id in self.routes:
            return self.routes[item_id][0].path
        path = self.paths.get(str(item_id))
        if isinstance(path, dict) and isinstance(path.get("path"), list):
            return "::".join(str(part) for part in path["path"])
        return fallback

    def is_local_id(self, item_id: int) -> bool:
        """Return whether a resolved type/trait ID belongs to this crate."""

        path = self.paths.get(str(item_id))
        if isinstance(path, dict):
            return path.get("crate_id") == 0
        item = self.index.get(str(item_id))
        return isinstance(item, dict) and item.get("crate_id") == 0

    def canonicalize(self, value: Any) -> Any:
        """Replace rustdoc IDs in a signature with stable, resolved paths."""

        if isinstance(value, list):
            return [self.canonicalize(entry) for entry in value]
        if not isinstance(value, dict):
            return value
        if "path" in value and "id" in value:
            result = {
                "path": self.canonical_path(value["id"], str(value.get("path", "")))
            }
            for key, entry in value.items():
                if key not in {"id", "path"}:
                    result[key] = self.canonicalize(entry)
            return result
        result: dict[str, Any] = {}
        for key, entry in value.items():
            if key == "id":
                raise InventoryError("unexpected bare rustdoc item id in signature data")
            result[key] = self.canonicalize(entry)
        return result

    def render_path(self, path: dict[str, Any]) -> str:
        """Render a rustdoc Path, including its generic arguments."""

        base = self.canonical_path(path.get("id"), str(path.get("path", "")))
        args = path.get("args")
        if args is None:
            return base
        if "angle_bracketed" in args:
            angle = args["angle_bracketed"]
            rendered = [self.render_generic_arg(arg) for arg in angle.get("args", [])]
            rendered.extend(self.render_constraint(value) for value in angle.get("constraints", []))
            return f"{base}<{', '.join(rendered)}>"
        if "parenthesized" in args:
            parenthesized = args["parenthesized"]
            inputs = ", ".join(
                self.render_type(value) for value in parenthesized.get("inputs", [])
            )
            output = parenthesized.get("output")
            suffix = "" if output is None else f" -> {self.render_type(output)}"
            return f"{base}({inputs}){suffix}"
        raise InventoryError(f"unknown rustdoc generic-args shape: {compact_json(args)}")

    def render_generic_arg(self, arg: dict[str, Any]) -> str:
        """Render one rustdoc GenericArg."""

        if "lifetime" in arg:
            return str(arg["lifetime"])
        if "type" in arg:
            return self.render_type(arg["type"])
        if "const" in arg:
            return compact_json(self.canonicalize(arg["const"]))
        if "infer" in arg:
            return "_"
        raise InventoryError(f"unknown rustdoc generic argument: {compact_json(arg)}")

    def render_constraint(self, constraint: dict[str, Any]) -> str:
        """Render an associated-type constraint in a trait path."""

        name = str(constraint.get("name", ""))
        args = constraint.get("args")
        if args is not None:
            name = self.render_path({"path": name, "id": None, "args": args})
        binding = constraint.get("binding", {})
        if "equality" in binding:
            equality = binding["equality"]
            if "type" in equality:
                return f"{name} = {self.render_type(equality['type'])}"
            return f"{name} = {compact_json(self.canonicalize(equality))}"
        if "constraint" in binding:
            return f"{name}: {compact_json(self.canonicalize(binding['constraint']))}"
        raise InventoryError(f"unknown associated constraint: {compact_json(constraint)}")

    def render_type(self, value: dict[str, Any]) -> str:
        """Render the rustdoc Type variants used by receivers and trait paths."""

        if "resolved_path" in value:
            return self.render_path(value["resolved_path"])
        if "primitive" in value:
            return str(value["primitive"])
        if "generic" in value:
            return str(value["generic"])
        if "borrowed_ref" in value:
            ref = value["borrowed_ref"]
            lifetime = f"{ref['lifetime']} " if ref.get("lifetime") else ""
            mutable = "mut " if ref.get("is_mutable") else ""
            return f"&{lifetime}{mutable}{self.render_type(ref['type'])}"
        if "raw_pointer" in value:
            pointer = value["raw_pointer"]
            mutability = "mut" if pointer.get("is_mutable") else "const"
            return f"*{mutability} {self.render_type(pointer['type'])}"
        if "tuple" in value:
            entries = [self.render_type(entry) for entry in value["tuple"]]
            suffix = "," if len(entries) == 1 else ""
            return f"({', '.join(entries)}{suffix})"
        if "slice" in value:
            return f"[{self.render_type(value['slice'])}]"
        if "array" in value:
            array = value["array"]
            return f"[{self.render_type(array['type'])}; {array['len']}]"
        if "qualified_path" in value:
            qualified = value["qualified_path"]
            trait = self.render_path(qualified["trait"])
            own = self.render_type(qualified["self_type"])
            args = qualified.get("args")
            suffix = "" if args is None else self.render_path(
                {"path": "", "id": None, "args": args}
            )
            return f"<{own} as {trait}>::{qualified['name']}{suffix}"
        if "infer" in value:
            return "_"
        # These uncommon variants remain faithful and deterministic rather than
        # guessing at Rust surface syntax that rustdoc may have changed.
        for variant in ("dyn_trait", "function_pointer", "impl_trait", "pat"):
            if variant in value:
                return compact_json(self.canonicalize(value))
        raise InventoryError(f"unknown rustdoc type shape: {compact_json(value)}")

    @staticmethod
    def referenced_ids(value: Any) -> set[int]:
        """Collect resolved item IDs from rustdoc type/path structures."""

        found: set[int] = set()
        if isinstance(value, list):
            for entry in value:
                found.update(Inventory.referenced_ids(entry))
        elif isinstance(value, dict):
            if isinstance(value.get("id"), int):
                found.add(value["id"])
            for entry in value.values():
                found.update(Inventory.referenced_ids(entry))
        return found

    def inherited_attrs(self, span: Any) -> tuple[Any, ...]:
        """Return attrs of enclosing modules in the item's source file."""

        if not isinstance(span, dict) or not isinstance(span.get("filename"), str):
            return ()
        begin = tuple(span.get("begin", ()))
        end = tuple(span.get("end", ()))
        if len(begin) != 2 or len(end) != 2:
            return ()
        enclosing: list[tuple[tuple[int, int], tuple[int, int], dict[str, Any]]] = []
        for module in self.module_items.get(span["filename"], []):
            module_span = module.get("span") or {}
            module_begin = tuple(module_span.get("begin", ()))
            module_end = tuple(module_span.get("end", ()))
            if len(module_begin) == 2 and module_begin <= begin <= end <= module_end:
                enclosing.append((module_begin, module_end, module))
        enclosing.sort(key=lambda entry: (entry[0], entry[1]))
        return tuple(
            attr
            for _, _, module in enclosing
            for attr in module.get("attrs", [])
            if is_cfg_attr(attr)
        )

    @staticmethod
    def attrs_columns(attrs: list[Any] | tuple[Any, ...]) -> tuple[str, str]:
        """Split cfg traces from other rustdoc attributes and deduplicate both."""

        cfg: dict[str, Any] = {}
        other: dict[str, Any] = {}
        for attr in attrs:
            encoded = compact_json(attr)
            target = cfg if is_cfg_attr(attr) else other
            target.setdefault(encoded, attr)
        return compact_json(list(cfg.values())), compact_json(list(other.values()))

    @staticmethod
    def location(
        item: dict[str, Any], fallback: dict[str, Any] | None = None
    ) -> tuple[str, int | str]:
        """Return source file and one-based line, falling back to the impl span."""

        span = item.get("span") or (fallback or {}).get("span")
        if not isinstance(span, dict):
            return "", ""
        begin = span.get("begin")
        line = begin[0] if isinstance(begin, list) and begin else ""
        return str(span.get("filename", "")), line

    def route_data(self, ids: set[int]) -> tuple[str, tuple[Any, ...]]:
        """Combine all relevant public root paths and their route attributes."""

        selected = [route for item_id in sorted(ids) for route in self.routes[item_id]]
        paths = ";".join(sorted({route.path for route in selected}))
        attrs = tuple(attr for route in selected for attr in route.attrs)
        return paths, attrs

    def signature(self, item: dict[str, Any]) -> str:
        """Build stable structured signature data for an exposed item."""

        kind = self.kind(item)
        inner = item["inner"][kind]
        if kind == "function":
            value = {
                "generics": inner["generics"],
                "header": inner["header"],
                "sig": inner["sig"],
            }
        elif kind == "struct":
            value = {
                "generics": inner["generics"],
                "shape": self.public_struct_shape(inner["kind"]),
            }
        elif kind == "enum":
            value = {
                "generics": inner["generics"],
                "has_stripped_variants": inner["has_stripped_variants"],
            }
        elif kind == "variant":
            value = self.variant_signature(inner)
        elif kind == "struct_field":
            value = inner
        else:
            value = inner
        return compact_json(self.canonicalize(value))

    def public_struct_shape(self, shape: Any) -> Any:
        """Describe a struct without leaking private-field types from private JSON."""

        if shape == "unit":
            return "unit"
        if not isinstance(shape, dict):
            raise InventoryError(f"unknown rustdoc struct shape: {compact_json(shape)}")
        key = "tuple" if "tuple" in shape else "plain"
        field_ids = shape["tuple"] if key == "tuple" else shape["plain"]["fields"]
        public_fields = []
        has_private_fields = False
        for field_id in field_ids:
            field = self.item(field_id)
            if is_public(field):
                public_fields.append(
                    {"name": field.get("name"), "type": field["inner"]["struct_field"]}
                )
            else:
                has_private_fields = True
        return {
            "kind": key,
            "has_private_fields": has_private_fields,
            "public_fields": public_fields,
        }

    def variant_signature(self, variant: dict[str, Any]) -> dict[str, Any]:
        """Dereference variant field IDs so the signature contains stable types."""

        kind = variant["kind"]
        if kind == "plain":
            shape: Any = "plain"
        elif "tuple" in kind:
            shape = {
                "tuple": [
                    self.item(field_id)["inner"]["struct_field"]
                    for field_id in kind["tuple"]
                ]
            }
        elif "struct" in kind:
            shape = {
                "struct": [
                    {
                        "name": self.item(field_id).get("name"),
                        "type": self.item(field_id)["inner"]["struct_field"],
                    }
                    for field_id in kind["struct"]
                ]
            }
        else:
            raise InventoryError(f"unknown rustdoc variant shape: {compact_json(kind)}")
        return {"discriminant": variant.get("discriminant"), "shape": shape}

    def make_row(
        self,
        *,
        roots: set[int],
        receiver: str,
        impl_kind: str,
        trait_path: str,
        item_kind: str,
        item_name: str,
        item: dict[str, Any],
        signature: str,
        extra_attrs: tuple[Any, ...] = (),
        fallback: dict[str, Any] | None = None,
    ) -> Row:
        """Combine reachability, source, cfg, attrs, and signature into a row."""

        root_path, route_attrs = self.route_data(roots)
        span = item.get("span") or (fallback or {}).get("span")
        attrs = route_attrs + self.inherited_attrs(span) + extra_attrs + tuple(
            item.get("attrs", [])
        )
        cfg, ordinary_attrs = self.attrs_columns(attrs)
        source_file, source_line = self.location(item, fallback)
        return Row(
            root_path=root_path,
            receiver=receiver,
            impl_kind=impl_kind,
            trait_path=trait_path,
            item_kind=item_kind,
            item_name=item_name,
            source_file=source_file,
            source_line=source_line,
            cfg=cfg,
            attrs=ordinary_attrs,
            signature=signature,
        )

    def rows(self) -> list[Row]:
        """Inventory root-exposed items, variants/fields, and relevant impls."""

        rows: list[Row] = []
        owner_kinds = {"struct", "enum", "union", "type_alias", "trait"}
        owner_ids = {
            item_id
            for item_id in self.routes
            if self.kind(self.item(item_id)) in owner_kinds
        }

        for item_id in sorted(self.routes):
            item = self.item(item_id)
            kind = self.kind(item)
            if kind == "module":
                continue
            roots = {item_id}
            public_path, _ = self.route_data(roots)
            rows.append(
                self.make_row(
                    roots=roots,
                    receiver=public_path if kind in owner_kinds else "",
                    impl_kind="type" if kind in owner_kinds else "root",
                    trait_path="",
                    item_kind=kind,
                    item_name=str(item.get("name") or public_path.rsplit("::", 1)[-1]),
                    item=item,
                    signature=self.signature(item),
                )
            )
            if kind == "enum":
                for variant_id in item["inner"]["enum"]["variants"]:
                    variant = self.item(variant_id)
                    rows.append(
                        self.make_row(
                            roots=roots,
                            receiver=public_path,
                            impl_kind="type",
                            trait_path="",
                            item_kind="variant",
                            item_name=str(variant.get("name", "")),
                            item=variant,
                            signature=self.signature(variant),
                        )
                    )
            if kind == "struct":
                shape = item["inner"]["struct"]["kind"]
                field_ids: list[int] = []
                if isinstance(shape, dict) and "tuple" in shape:
                    field_ids = shape["tuple"]
                elif isinstance(shape, dict) and "plain" in shape:
                    field_ids = shape["plain"]["fields"]
                for position, field_id in enumerate(field_ids):
                    field = self.item(field_id)
                    if not is_public(field):
                        continue
                    rows.append(
                        self.make_row(
                            roots=roots,
                            receiver=public_path,
                            impl_kind="type",
                            trait_path="",
                            item_kind="field",
                            item_name=str(field.get("name") or position),
                            item=field,
                            signature=self.signature(field),
                        )
                    )

        for impl_item in self.index.values():
            if (
                not isinstance(impl_item, dict)
                or impl_item.get("crate_id") != 0
                or self.kind(impl_item) != "impl"
                or is_hidden(impl_item)
            ):
                continue
            impl = impl_item["inner"]["impl"]
            if impl.get("is_synthetic") or impl.get("blanket_impl") is not None:
                continue
            receiver_references = self.referenced_ids(impl.get("for"))
            references = set(receiver_references)
            references.update(self.referenced_ids(impl.get("trait")))
            if any(
                self.is_local_id(item_id) and item_id not in self.routes
                for item_id in references
            ):
                continue
            roots = references & owner_ids
            if not roots:
                continue

            trait = impl.get("trait")
            if trait is not None:
                trait_id = trait.get("id")
                path = self.paths.get(str(trait_id), {})
                if path.get("crate_id") == 0 and trait_id not in self.routes:
                    continue
                trait_path = self.render_path(trait)
                if impl.get("is_negative"):
                    trait_path = f"!{trait_path}"
                impl_kind = "trait"
            else:
                trait_path = ""
                impl_kind = "inherent"

            receiver = self.render_type(impl["for"])
            impl_attrs = tuple(impl_item.get("attrs", []))
            if trait is not None:
                trait_name = trait_path.split("<", 1)[0].rsplit("::", 1)[-1]
                impl_signature = compact_json(
                    self.canonicalize(
                        {
                            "generics": impl["generics"],
                            "is_negative": impl["is_negative"],
                            "is_unsafe": impl["is_unsafe"],
                            "provided_trait_methods": sorted(
                                impl.get("provided_trait_methods", [])
                            ),
                        }
                    )
                )
                rows.append(
                    self.make_row(
                        roots=roots,
                        receiver=receiver,
                        impl_kind=impl_kind,
                        trait_path=trait_path,
                        item_kind="trait_impl",
                        item_name=trait_name,
                        item=impl_item,
                        signature=impl_signature,
                    )
                )

            for associated_id in impl.get("items", []):
                associated = self.item(associated_id)
                if is_hidden(associated):
                    continue
                if trait is None and not is_public(associated):
                    continue
                associated_kind = self.kind(associated)
                if associated_kind == "function":
                    inputs = associated["inner"]["function"]["sig"].get("inputs", [])
                    item_kind = (
                        "method" if inputs and inputs[0][0] == "self" else "associated_function"
                    )
                else:
                    item_kind = associated_kind
                rows.append(
                    self.make_row(
                        roots=roots,
                        receiver=receiver,
                        impl_kind=impl_kind,
                        trait_path=trait_path,
                        item_kind=item_kind,
                        item_name=str(associated.get("name", "")),
                        item=associated,
                        signature=self.signature(associated),
                        extra_attrs=impl_attrs,
                        fallback=impl_item,
                    )
                )

        unique = {tuple(asdict(row).values()): row for row in rows}
        return sorted(
            unique.values(),
            key=lambda row: tuple(str(value) for value in asdict(row).values()),
        )


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


def render_tsv(rows: list[Row]) -> str:
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
        # Disable universal-newline conversion: a checked-in CRLF file must not
        # compare equal to the generator's canonical LF output.
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the generator/checker command line."""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Generate or verify the API inventory."""

    args = parse_args(argv)
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


if __name__ == "__main__":
    raise SystemExit(main())
