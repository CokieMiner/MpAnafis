"""Confidence scoring and variance models for multi-backend predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from .dataset import FeatureSet
from .error_model import Correction, ErrorModel
from .score import ConsensusResult


@dataclass(frozen=True)
class ExpectedResult:
    """Corrected expectation and confidence for one variant on one CPU."""
    variant: str
    expected: Optional[float] = None
    confidence: Optional[float] = None
    backend_used: Optional[str] = None
    correction: float = 0.0
    accuracy: float = 0.5
    agreement: float = 0.0
    coverage: float = 0.0
    n_samples: int = 0
    covered: bool = False

    def to_dict(self) -> Dict[str, object]:
        """Convert confidence score to structured dictionary."""
        return {
            "variant": self.variant,
            "expected_cycles": self.expected,
            "confidence": self.confidence,
            "backend_used": self.backend_used,
            "correction": self.correction,
            "accuracy": self.accuracy,
            "agreement": self.agreement,
            "coverage": self.coverage,
            "n_samples": self.n_samples,
            "covered": self.covered,
        }


def _calculate_agreement(costs: List[float]) -> float:
    """Calculate normalized simulator agreement score in [0.0, 1.0]."""
    if len(costs) <= 1:
        return 0.5 if len(costs) == 1 else 0.0
    spread = max(costs) - min(costs)
    avg_cost = sum(costs) / len(costs)
    rel_spread = spread / max(1.0, avg_cost)
    return max(0.0, min(1.0, 1.0 - rel_spread))


def expected_for_variant(
    variant: str,
    cpu: str,
    reports: Dict[Tuple[str, str, str], Optional[float]],
    features: FeatureSet,
    result: ConsensusResult,
    model: ErrorModel,
) -> ExpectedResult:
    """Compute empirically corrected cycle expectation and confidence for one variant on a CPU."""
    valid_costs: List[float] = []
    best: Optional[Tuple[str, float, float]] = None  # (backend, raw_cost, expected)

    for backend in ("llvm-mca", "osaca", "uica"):
        cost = reports.get((variant, backend, cpu))
        if cost is None:
            continue
        valid_costs.append(cost)
        corr_val = model.predict_correction(backend, cpu)
        expected = cost + corr_val
        if best is None or expected < best[2]:
            best = (backend, cost, expected)

    if best is None:
        return ExpectedResult(variant=variant)

    backend, raw_cost, expected = best
    agreement = _calculate_agreement(valid_costs)

    corr = model.get_correction(backend, cpu)
    if corr is not None and corr.sample_count > 0:
        n_samples = corr.sample_count
        correction = corr.mean_bias
        covered = True
        coverage = min(1.0, n_samples / 10.0)
        std_dev = (corr.variance) ** 0.5
        rel_error = std_dev / max(1.0, abs(expected))
        accuracy = max(0.1, min(0.99, 1.0 / (1.0 + rel_error)))
        confidence = max(0.1, min(0.99, 0.4 * agreement + 0.3 * coverage + 0.3 * accuracy))
    else:
        n_samples = 0
        correction = 0.0
        covered = False
        coverage = 0.0
        accuracy = 0.5
        confidence = max(0.1, min(0.85, 0.5 * agreement + 0.25))

    return ExpectedResult(
        variant=variant,
        expected=round(expected, 4),
        confidence=round(confidence, 4),
        backend_used=backend,
        correction=round(correction, 4),
        accuracy=round(accuracy, 4),
        agreement=round(agreement, 4),
        coverage=round(coverage, 4),
        n_samples=n_samples,
        covered=covered,
    )
