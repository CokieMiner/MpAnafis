"""Cargo directives selected by target capabilities and benchmark linkage."""

from support import BuildScriptCase


class TargetConfigurationTests(BuildScriptCase):
    def test_thread_local_configuration_uses_exact_capabilities(self):
        cases = (
            ({}, False),
            ({"CARGO_CFG_TARGET_THREAD_LOCAL": ""}, True),
            ({"CARGO_CFG_TARGET_FAMILY": "wasm", "CARGO_CFG_TARGET_OS": "unknown"}, True),
            ({"CARGO_CFG_TARGET_FAMILY": "wasm", "CARGO_CFG_TARGET_FEATURE": "atomics"}, False),
            ({"CARGO_CFG_TARGET_FAMILY": "unix,wasm", "CARGO_CFG_TARGET_FEATURE": "simd128,atomics"}, False),
            ({"CARGO_CFG_TARGET_FAMILY": "wasm", "CARGO_CFG_TARGET_FEATURE": "nonatomics"}, True),
            ({"CARGO_CFG_TARGET_FAMILY": "wasm", "CARGO_CFG_TARGET_FEATURE": "atomics", "CARGO_CFG_TARGET_THREAD_LOCAL": ""}, True),
            *[({"CARGO_CFG_TARGET_OS": target}, True) for target in ("uefi", "zkvm", "trusty", "vexos")],
        )
        for environment, enabled in cases:
            with self.subTest(environment=environment):
                result = self.run_build(**environment)
                self.assertIn("cargo:rustc-check-cfg=cfg(mp_eager_thread_local)\n", result.stdout)
                self.assertEqual("cargo:rustc-cfg=mp_eager_thread_local\n" in result.stdout, enabled)

    def test_flint_paths_are_emitted_only_for_supported_benchmark_targets(self):
        for architecture in ("x86_64", "aarch64", "x86"):
            for width in ("16", "32", "64"):
                for target_os in ("linux", "windows", "macos", "unknown"):
                    for configured in (False, True):
                        with self.subTest(architecture=architecture, width=width, target_os=target_os, configured=configured):
                            environment = dict(
                                CARGO_CFG_TARGET_ARCH=architecture,
                                CARGO_CFG_TARGET_POINTER_WIDTH=width,
                                CARGO_CFG_TARGET_OS=target_os,
                            )
                            if configured:
                                environment["MP_ANAFIS_FLINT_LIB_DIR"] = "/opt/flint/lib"
                            result = self.run_build(**environment)
                            expected = configured and (architecture, width, target_os) == ("x86_64", "64", "linux")
                            self.assertIn("cargo:rerun-if-env-changed=MP_ANAFIS_FLINT_LIB_DIR\n", result.stdout)
                            self.assertEqual("cargo:rustc-link-search=native=/opt/flint/lib\n" in result.stdout, expected)
                            self.assertEqual("cargo:rustc-link-arg-benches=-Wl,-rpath,/opt/flint/lib\n" in result.stdout, expected)
                            self.assertNotIn("cargo:rustc-link-arg=", result.stdout)
