"""Profile resolution, transactional output, and Cargo rerun directives."""

import os

from support import BuildScriptCase


class ProfileProcessTests(BuildScriptCase):
    def test_profile_precedence_and_rerun_tracking_from_another_directory(self):
        result = self.run_build()
        source = self.generated.read_text()
        self.assertIn("cargo:rerun-if-changed=build.rs\n", result.stdout)
        self.assertIn("cargo:rerun-if-changed=build_support\n", result.stdout)
        self.assertIn("cargo:rerun-if-changed=src/int\n", result.stdout)
        self.assertIn("cargo:rerun-if-env-changed=MP_TUNING_PROFILE\n", result.stdout)
        self.assertNotIn("cargo:rerun-if-changed=src/int/tuned_thresholds.rs\n", result.stdout)

        self.installed.write_text(self.replace_constant(source, "KARATSUBA_THRESHOLD", 19))
        installed = self.run_build()
        self.assertIn("KARATSUBA_THRESHOLD: usize = 19;", self.generated.read_text())
        self.assertIn("cargo:rerun-if-changed=src/int/tuned_thresholds.rs\n", installed.stdout)

        candidate = self.project / "candidate.rs"
        candidate.write_text(self.replace_constant(source, "KARATSUBA_THRESHOLD", 21))
        self.installed.write_text("invalid installed profile")
        for path in ("candidate.rs", str(candidate)):
            with self.subTest(path=path):
                result = self.run_build(MP_TUNING_PROFILE=path)
                self.assertIn("KARATSUBA_THRESHOLD: usize = 21;", self.generated.read_text())
                self.assertIn(f"cargo:rerun-if-changed={candidate}\n", result.stdout)

    def test_invalid_explicit_profiles_preserve_both_generated_outputs(self):
        source = self.default_source()
        initial = {path: path.read_bytes() for path in (self.generated, self.bitmap)}
        timestamp = 60_000_000_000
        for path in initial:
            os.utime(path, ns=(timestamp, timestamp))
        candidate = self.project / "candidate.rs"
        invalid_sources = (
            source.replace("pub const KARATSUBA_THRESHOLD:", "// pub const KARATSUBA_THRESHOLD:"),
            source + "\npub const KARATSUBA_THRESHOLD: usize = 1;",
            source + "\nfn injected() {}",
            self.replace_constant(source, "KARATSUBA_THRESHOLD", 0),
        )
        for destination, environment in (
            (self.installed, {}),
            (candidate, {"MP_TUNING_PROFILE": str(candidate)}),
        ):
            for invalid in invalid_sources:
                with self.subTest(destination=destination, invalid=invalid[-80:]):
                    self.installed.write_text(source)
                    destination.write_text(invalid)
                    result = self.run_build(succeeds=False, **environment)
                    self.assertIn("tuning profile", result.stderr)
                    for path, contents in initial.items():
                        self.assertEqual(path.read_bytes(), contents)
                        self.assertEqual(path.stat().st_mtime_ns, timestamp)
        self.run_build(succeeds=False, MP_TUNING_PROFILE="missing.rs")
        for path, contents in initial.items():
            self.assertEqual(path.read_bytes(), contents)
            self.assertEqual(path.stat().st_mtime_ns, timestamp)

    def test_only_changed_artifacts_are_rewritten(self):
        source = self.default_source()
        timestamp = 60_000_000_000
        for path in (self.generated, self.bitmap):
            os.utime(path, ns=(timestamp, timestamp))
        self.run_build()
        for path in (self.generated, self.bitmap):
            self.assertEqual(path.stat().st_mtime_ns, timestamp)
        self.installed.write_text(self.replace_constant(source, "KARATSUBA_THRESHOLD", 19))
        self.run_build()
        self.assertIn("KARATSUBA_THRESHOLD: usize = 19;", self.generated.read_text())
        self.assertNotEqual(self.generated.stat().st_mtime_ns, timestamp)
        self.assertEqual(self.bitmap.stat().st_mtime_ns, timestamp)
        os.utime(self.generated, ns=(timestamp, timestamp))
        self.run_build(CARGO_CFG_TARGET_POINTER_WIDTH="16")
        self.assertEqual(self.generated.stat().st_mtime_ns, timestamp)
        self.assertNotEqual(self.bitmap.stat().st_mtime_ns, timestamp)
