"""Run the Rust profile contracts independently of the library and tuner."""

from pathlib import Path
import subprocess
import tempfile
import unittest

from support import ROOT


class ProfileContractTests(unittest.TestCase):
    def test_rust_profile_contracts(self):
        with tempfile.TemporaryDirectory(prefix="mp-profile-contracts-") as directory:
            binary = Path(directory) / "profile-tests"
            compilation = subprocess.run(
                ["rustc", "--edition=2024", "--test", str(ROOT / "build_support/mod.rs"),
                 "-o", str(binary)],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(compilation.returncode, 0, compilation.stderr)
            execution = subprocess.run(
                [str(binary)], capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(execution.returncode, 0, execution.stdout + execution.stderr)
