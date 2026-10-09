"""Backend construction preserves explicit selection and support filtering."""

import unittest
from unittest.mock import Mock, patch

from asm_analyzer.backends import make_backends, supported_backends
from asm_analyzer.models import CpuSpec


class BackendRegistryTests(unittest.TestCase):
    def test_construction_and_availability_matrix(self):
        constructors = {name: Mock() for name in ("llvm-mca", "osaca", "uica", "nanobench", "perf")}
        with patch("asm_analyzer.backends.registry._BACKEND_CLASSES", constructors):
            selected = make_backends(wsl=True)
            self.assertEqual(set(selected), set(constructors))
            for name, constructor in constructors.items():
                if name in ("llvm-mca", "osaca", "uica"):
                    constructor.assert_called_once_with(wsl=True)
                else:
                    constructor.assert_called_once_with()
            cpu = CpuSpec("fixture", "fixture")
            for available, supports in ((True, True), (True, False), (False, True), (False, False)):
                with self.subTest(available=available, supports=supports):
                    for backend in selected.values():
                        backend.available.return_value = available
                        backend.supports.return_value = supports
                    expected = list(constructors) if available and supports else []
                    self.assertEqual(supported_backends(cpu, selected), expected)
            self.assertEqual(make_backends([]), {})
            with self.assertRaisesRegex(ValueError, "unknown analysis backend"):
                make_backends(["unknown"])


if __name__ == "__main__":
    unittest.main()
