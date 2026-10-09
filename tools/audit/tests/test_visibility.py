"""Crate-visible methods follow public reexports rather than pub declarations alone."""

import tempfile
import unittest
from pathlib import Path

from tools.audit.visibility import VisibilityGraph, visibility_findings


class VisibilityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        sources = {
            "src/lib.rs": 'mod facade; mod hidden; pub use facade::{PublicAlias, PublicTypeAlias}; #[cfg(feature = "tune")] pub use facade::nested as api;',
            "src/facade/mod.rs": "mod aliases; mod owner; pub mod nested; pub use aliases::PublicTypeAlias; pub use owner::{AliasOnly, Public as PublicAlias};",
            "src/facade/aliases.rs": "use super::AliasOnly; pub type PublicTypeAlias = AliasOnly;",
            "src/facade/owner.rs": "pub struct Public { value: u8 } pub struct Sealed; pub struct AliasOnly;",
            "src/facade/nested/mod.rs": "pub use super::PublicAlias as Renamed; pub struct FeatureType;",
            "src/hidden.rs": "pub struct Public;",
        }
        for relative, text in sources.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.graph = VisibilityGraph(self.root)
        self.reachable = self.graph.reachable_types()

    def test_reexports_aliases_and_feature_modules_determine_reachability(self):
        self.assertEqual(self.reachable, {"src::facade::owner::Public", "src::facade::owner::AliasOnly", "src::facade::nested::FeatureType"})
        self.assertEqual(self.graph.resolve("src::api::Renamed"), {"src::facade::owner::Public"})
        self.assertNotIn("src::hidden::Public", self.reachable)
        self.assertNotIn("src::facade::owner::Sealed", self.reachable)

    def test_only_inherent_function_headers_receive_the_exception(self):
        cases = (
            ("src/facade/owner.rs", "impl Public { pub(crate) const fn new() -> Self { Self { value: 0 } } }", 0),
            ("src/facade/owner.rs", "impl AliasOnly { pub(crate) fn new() -> Self { Self } }", 0),
            ("src/facade/owner.rs", "impl Public where Public: for<'a> Trait<'a> { pub(crate) fn update() {} }", 0),
            ("src/facade/owner.rs", "impl Sealed { pub(crate) fn new() -> Self { Self } }", 1),
            ("src/hidden.rs", "impl Public { pub(crate) fn new() -> Self { Self } }", 1),
            ("src/facade/nested/mod.rs", "impl FeatureType { pub(crate) fn update(&mut self) {} }", 0),
            ("src/facade/owner.rs", "pub(crate) fn constructor() {}", 1),
            ("src/facade/owner.rs", "pub(crate) struct Hidden;", 1),
            ("src/facade/owner.rs", "pub struct Public { pub(crate) value: u8 }", 1),
            ("src/facade/owner.rs", "impl Trait for Public { pub(crate) fn update() {} }", 1),
            ("src/facade/owner.rs", "impl Public { pub(crate) const WIDTH: usize = 4; }", 1),
            ("src/facade/owner.rs", 'const TEXT: &str = "pub(crate) struct Hidden;"; // pub(crate) fn fake() {}', 0),
            ("src/facade/owner.rs", "impl Public { pub(crate) fn f() { pub(crate) struct Nested; } }", 1),
        )
        for path, text, expected in cases:
            with self.subTest(path=path, text=text):
                self.assertEqual(len(visibility_findings(text, path, self.graph, self.reachable)), expected)


if __name__ == "__main__":
    unittest.main()
