"""Unit tests for consensus error models, calibration loop, and confidence scoring."""

from __future__ import annotations

import unittest
from asm_analyzer.consensus.confidence import expected_for_variant
from asm_analyzer.consensus.dataset import Dataset, FeatureSet, MeasurementRow
from asm_analyzer.consensus.error_model import Correction, ErrorModel
from asm_analyzer.consensus.score import Cell, score_cpu


class TestConsensusCalibration(unittest.TestCase):
    def test_error_model_fit_from_dataset(self):
        feat = FeatureSet(instruction_count=10)
        # Create dataset where llvm-mca on znver3 is systematically underpredicting by +2.0 cycles
        # (measured = 12.0, predicted = 10.0)
        rows = [
            MeasurementRow(
                timestamp=100.0,
                cpu="znver3",
                kernel_id="add_mul_1",
                variant="x86_64",
                features=feat,
                models={"llvm-mca": 10.0, "osaca": 11.0},
                measured={"cycles": 12.0},
            ),
            MeasurementRow(
                timestamp=101.0,
                cpu="znver3",
                kernel_id="add_mul_2",
                variant="x86_64_adx",
                features=feat,
                models={"llvm-mca": 8.0, "osaca": 9.5},
                measured={"cycles": 10.0},
            ),
        ]
        ds = Dataset(rows)
        model = ErrorModel.fit_from_dataset(ds)

        corr_mca = model.get_correction("llvm-mca", "znver3")
        self.assertIsNotNone(corr_mca)
        self.assertAlmostEqual(corr_mca.mean_bias, 2.0)
        self.assertEqual(corr_mca.sample_count, 2)

        corr_osaca = model.get_correction("osaca", "znver3")
        self.assertIsNotNone(corr_osaca)
        # Row 1 error = 12 - 11 = 1.0; Row 2 error = 10 - 9.5 = 0.5; mean = 0.75
        self.assertAlmostEqual(corr_osaca.mean_bias, 0.75)
        self.assertEqual(corr_osaca.sample_count, 2)

    def test_confidence_and_correction_computation(self):
        feat = FeatureSet(instruction_count=10)
        cell_mca = Cell(backend="llvm-mca", cpu="znver3", costs={"orig": 10.0})
        cell_osaca = Cell(backend="osaca", cpu="znver3", costs={"orig": 10.5})
        consensus = score_cpu([cell_mca, cell_osaca], ["orig"])

        # Uncalibrated model
        empty_model = ErrorModel({})
        reports = {("orig", "llvm-mca", "znver3"): 10.0, ("orig", "osaca", "znver3"): 10.5}
        uncalibrated = expected_for_variant("orig", "znver3", reports, feat, consensus, empty_model)
        self.assertFalse(uncalibrated.covered)
        self.assertEqual(uncalibrated.n_samples, 0)
        self.assertAlmostEqual(uncalibrated.correction, 0.0)
        self.assertAlmostEqual(uncalibrated.expected, 10.0)

        # Calibrated model with correction
        calibrated_model = ErrorModel({
            "llvm-mca:znver3": Correction(mean_bias=0.2, variance=0.01, sample_count=15),
            "osaca:znver3": Correction(mean_bias=0.8, variance=0.02, sample_count=10),
        })
        calibrated = expected_for_variant("orig", "znver3", reports, feat, consensus, calibrated_model)
        self.assertTrue(calibrated.covered)
        self.assertGreater(calibrated.n_samples, 0)
        self.assertGreater(calibrated.confidence, 0.7)


if __name__ == "__main__":
    unittest.main()
