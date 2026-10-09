"""Architecture-matrix exits distinguish crate failures from identified rust-src blocks."""

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.audit.common import ROOT


class ArchitectureMatrixTests(unittest.TestCase):
    def test_probe_failures_and_required_profile_installation_have_distinct_exit_states(self):
        script = ROOT / "tools/check_all_archs.sh"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            targets = root / "targets.txt"
            targets.write_text("\n".join(sorted(set(re.findall(r"\b[a-z][a-z0-9_]*-(?:[a-z0-9_]+-)*[a-z0-9_]+\b", script.read_text())))) + "\n")
            commands = {
                "rustc": 'cat "$ARCH_TEST_TARGETS"\n',
                "rustup": '''case "$*" in
  "target list --installed")
    if [ "$ARCH_TEST_MODE" = arm ]; then
      grep -v '^thumbv7neon-unknown-linux-gnueabihf$' "$ARCH_TEST_TARGETS"
    else
      cat "$ARCH_TEST_TARGETS"
    fi ;;
  "target add thumbv7neon-unknown-linux-gnueabihf")
    if [ "$ARCH_TEST_MODE" = arm ]; then exit 1; fi ;;
esac
''',
                "cargo": '''case "$*" in
  *"--target xtensa-esp32-none-elf"*)
    case "$ARCH_TEST_MODE" in
      crate) printf 'error: could not compile `mp_anafis` (lib)\n' >&2 ;;
      alias) printf 'error: could not compile `mp-anafis` (lib)\n' >&2 ;;
      core) printf 'error: could not compile `core` (lib)\n' >&2 ;;
      unknown) printf 'error: unexpected execution failure\n' >&2 ;;
      *) exit 0 ;;
    esac
    exit 101 ;;
esac
''',
            }
            for name, body in commands.items():
                executable = root / name
                executable.write_text("#!/bin/sh\nset -eu\n" + body)
                executable.chmod(0o755)
            for mode, expected, message in (
                ("crate", 1, "mp_anafis failed"),
                ("alias", 1, "mp_anafis failed"),
                ("core", 0, "rust-src library compilation failed"),
                ("unknown", 1, "unclassified toolchain probe failure"),
                ("arm", 1, "ARM profile target installation"),
                ("success", 0, "All required architecture checks passed"),
            ):
                env = os.environ | {"PATH": str(root) + os.pathsep + os.environ["PATH"],
                                    "ARCH_TEST_TARGETS": str(targets), "ARCH_TEST_MODE": mode}
                with self.subTest(mode=mode):
                    result = subprocess.run(["bash", str(script)], cwd=ROOT, env=env,
                                            capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                    self.assertIn(message, result.stdout)


if __name__ == "__main__":
    unittest.main()
