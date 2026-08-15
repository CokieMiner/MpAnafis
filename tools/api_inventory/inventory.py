"""Validated rustdoc JSON inventory loader and row extractor."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Set, Tuple

from .models import (
    EXPECTED_CRATE,
    InventoryError,
    Route,
    Row,
    SUPPORTED_FORMAT_VERSIONS,
    compact_json,
    is_cfg_attr,
    is_hidden,
    is_public,
)
from .normalizer import Normalizer


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

        self.normalizer = Normalizer(self.canonical_path, self.item)

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
        visited: Set[Tuple[int, Tuple[str, ...]]] = set()

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
    def location(item: dict[str, Any], fallback: dict[str, Any] | None = None) -> tuple[str, int | str]:
        span = item.get("span") or (fallback or {}).get("span")
        if not isinstance(span, dict):
            return "", ""
        begin = span.get("begin")
        line = begin[0] if isinstance(begin, list) and begin else ""
        return str(span.get("filename", "")), line

    def route_data(self, ids: set[int]) -> tuple[str, tuple[Any, ...]]:
        selected = [route for item_id in sorted(ids) for route in self.routes[item_id]]
        paths = ";".join(sorted({route.path for route in selected}))
        attrs = tuple(attr for route in selected for attr in route.attrs)
        return paths, attrs

    def signature(self, item: dict[str, Any]) -> str:
        kind = self.kind(item)
        inner = item["inner"][kind]
        if kind == "function":
            value = {"generics": inner["generics"], "header": inner["header"], "sig": inner["sig"]}
        elif kind == "struct":
            value = {"generics": inner["generics"], "shape": self.normalizer.public_struct_shape(inner["kind"])}
        elif kind == "enum":
            value = {"generics": inner["generics"], "has_stripped_variants": inner["has_stripped_variants"]}
        elif kind == "variant":
            value = self.normalizer.variant_signature(inner)
        elif kind == "struct_field":
            value = inner
        else:
            value = inner
        return compact_json(self.normalizer.canonicalize(value))

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
        root_path, route_attrs = self.route_data(roots)
        span = item.get("span") or (fallback or {}).get("span")
        attrs = route_attrs + self.inherited_attrs(span) + extra_attrs + tuple(item.get("attrs", []))
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
        rows: list[Row] = []
        owner_kinds = {"struct", "enum", "union", "type_alias", "trait"}
        owner_ids = {
            item_id for item_id in self.routes if self.kind(self.item(item_id)) in owner_kinds
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
                trait_path = self.normalizer.render_path(trait)
                if impl.get("is_negative"):
                    trait_path = f"!{trait_path}"
                impl_kind = "trait"
            else:
                trait_path = ""
                impl_kind = "inherent"

            receiver = self.normalizer.render_type(impl["for"])
            impl_attrs = tuple(impl_item.get("attrs", []))
            if trait is not None:
                trait_name = trait_path.split("<", 1)[0].rsplit("::", 1)[-1]
                impl_signature = compact_json(
                    self.normalizer.canonicalize(
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
