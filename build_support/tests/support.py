"""Isolated Cargo-style environments for the build-script process tests."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class BuildScriptCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory(prefix="mp-build-compiler-")
        cls.addClassCleanup(directory.cleanup)
        cls.binary = Path(directory.name) / "build-script"
        subprocess.run(
            ["rustc", "--edition=2024", str(ROOT / "build.rs"), "-o", str(cls.binary)],
            check=True, capture_output=True, text=True, timeout=30,
        )

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="mp-build-environment-")
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.project = self.directory / "project"
        self.output = self.directory / "output"
        (self.project / "src/int").mkdir(parents=True)
        self.output.mkdir()
        self.generated = self.output / "thresholds.rs"
        self.bitmap = self.output / "prime_bitmap.bin"
        self.installed = self.project / "src/int/tuned_thresholds.rs"
        self.environment = {
            name: value for name, value in os.environ.items()
            if not name.startswith("CARGO_CFG_")
            and name not in ("MP_TUNING_PROFILE", "MP_ANAFIS_FLINT_LIB_DIR")
        }
        self.environment.update(
            CARGO_MANIFEST_DIR=str(self.project), OUT_DIR=str(self.output),
            CARGO_CFG_TARGET_ARCH="x86_64", CARGO_CFG_TARGET_POINTER_WIDTH="64",
            CARGO_CFG_TARGET_OS="linux", CARGO_CFG_TARGET_FAMILY="unix",
            CARGO_CFG_TARGET_FEATURE="",
        )

    def run_build(self, *, succeeds=True, **environment):
        result = subprocess.run(
            [str(self.binary)], cwd=self.directory,
            env={**self.environment, **environment},
            capture_output=True, text=True, timeout=30,
        )
        if succeeds:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def default_source(self):
        self.run_build()
        return self.generated.read_text()

    def replace_constant(self, source, name, value):
        prefix = f"pub const {name}: usize = "
        declarations = [line for line in source.splitlines() if line.startswith(prefix)]
        self.assertEqual(len(declarations), 1, name)
        return source.replace(declarations[0], f"{prefix}{value};")
