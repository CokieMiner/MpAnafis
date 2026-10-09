"""Type and signature normalization for the API inventory generator."""

from __future__ import annotations

from typing import Any, Callable, Dict
from .models import InventoryError, compact_json, is_public


class Normalizer:
    """Type, generic argument, constraint, and struct/variant normalizer."""

    def __init__(self, canonical_path_fn: Callable[[Any, str], str], item_fn: Callable[[Any], Dict[str, Any]]) -> None:
        self.canonical_path = canonical_path_fn
        self.item = item_fn

    def canonicalize(self, value: Any) -> Any:
        """Replace rustdoc IDs in a signature with stable, resolved paths."""
        if isinstance(value, list):
            return [self.canonicalize(entry) for entry in value]
        if not isinstance(value, dict):
            return value
        if "path" in value and "id" in value:
            result = {"path": self.canonical_path(value["id"], str(value.get("path", "")))}
            for key, entry in value.items():
                if key not in {"id", "path"}:
                    result[key] = self.canonicalize(entry)
            return result
        result: Dict[str, Any] = {}
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
            inputs = ", ".join(self.render_type(value) for value in parenthesized.get("inputs", []))
            output = parenthesized.get("output")
            suffix = "" if output is None else f" -> {self.render_type(output)}"
            return f"{base}({inputs}){suffix}"
        raise InventoryError(f"unknown rustdoc generic-args shape: {compact_json(args)}")

    def render_generic_arg(self, arg: dict[str, Any]) -> str:
        """Render one rustdoc GenericArg."""
        if "lifetime" in arg: return str(arg["lifetime"])
        if "type" in arg: return self.render_type(arg["type"])
        if "const" in arg: return compact_json(self.canonicalize(arg["const"]))
        if "infer" in arg: return "_"
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
        if "resolved_path" in value: return self.render_path(value["resolved_path"])
        if "primitive" in value: return str(value["primitive"])
        if "generic" in value: return str(value["generic"])
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
        if "slice" in value: return f"[{self.render_type(value['slice'])}]"
        if "array" in value:
            array = value["array"]
            return f"[{self.render_type(array['type'])}; {array['len']}]"
        if "qualified_path" in value:
            qualified = value["qualified_path"]
            trait = self.render_path(qualified["trait"])
            own = self.render_type(qualified["self_type"])
            args = qualified.get("args")
            suffix = "" if args is None else self.render_path({"path": "", "id": None, "args": args})
            return f"<{own} as {trait}>::{qualified['name']}{suffix}"
        if "infer" in value: return "_"
        for variant in ("dyn_trait", "function_pointer", "impl_trait", "pat"):
            if variant in value: return compact_json(self.canonicalize(value))
        raise InventoryError(f"unknown rustdoc type shape: {compact_json(value)}")

    def public_struct_shape(self, shape: Any) -> Any:
        """Describe a struct without leaking private-field types from private JSON."""
        if shape == "unit": return "unit"
        if not isinstance(shape, dict): raise InventoryError(f"unknown rustdoc struct shape: {compact_json(shape)}")
        key = "tuple" if "tuple" in shape else "plain"
        field_ids = shape["tuple"] if key == "tuple" else shape["plain"]["fields"]
        public_fields = []
        has_private_fields = False
        for field_id in field_ids:
            field = self.item(field_id)
            if is_public(field):
                public_fields.append({"name": field.get("name"), "type": field["inner"]["struct_field"]})
            else:
                has_private_fields = True
        return {"kind": key, "has_private_fields": has_private_fields, "public_fields": public_fields}

    def variant_signature(self, variant: dict[str, Any]) -> dict[str, Any]:
        """Dereference variant field IDs so the signature contains stable types."""
        kind = variant["kind"]
        if kind == "plain": shape: Any = "plain"
        elif "tuple" in kind:
            shape = {"tuple": [self.item(field_id)["inner"]["struct_field"] for field_id in kind["tuple"]]}
        elif "struct" in kind:
            shape = {
                "struct": [
                    {"name": self.item(field_id).get("name"), "type": self.item(field_id)["inner"]["struct_field"]}
                    for field_id in kind["struct"]
                ]
            }
        else:
            raise InventoryError(f"unknown rustdoc variant shape: {compact_json(kind)}")
        return {"discriminant": variant.get("discriminant"), "shape": shape}
