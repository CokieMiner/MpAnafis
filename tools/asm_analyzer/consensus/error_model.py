#!/usr/bin/env python3
"""Error regression models for learning backend prediction biases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .dataset import Dataset


@dataclass(frozen=True)
class Correction:
    """Additive correction learned for a specific CPU/backend cell."""
    mean_bias: float = 0.0
    variance: float = 0.0
    sample_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize correction parameters to dictionary."""
        return {
            "mean_bias": self.mean_bias,
            "variance": self.variance,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Correction:
        """Construct Correction from dictionary."""
        return cls(
            mean_bias=float(data.get("mean_bias", 0.0)),
            variance=float(data.get("variance", 0.0)),
            sample_count=int(data.get("sample_count", 0)),
        )


class ErrorModel:
    """Statistical model tracking prediction deviations against real measurements."""

    def __init__(self, corrections: Optional[Dict[str, Correction]] = None) -> None:
        self.corrections: Dict[str, Correction] = corrections or {}

    def get_correction(self, backend: str, cpu: str) -> Optional[Correction]:
        """Retrieve learned statistical correction for a (backend, cpu) pair."""
        key = f"{backend}:{cpu}"
        return self.corrections.get(key)

    def predict_correction(self, backend: str, cpu: str) -> float:
        """Predict additive cycle correction (positive means simulator underestimated)."""
        corr = self.get_correction(backend, cpu)
        return corr.mean_bias if corr else 0.0

    @classmethod
    def fit_from_dataset(cls, dataset: Dataset) -> ErrorModel:
        """Fit empirical error corrections and variances from a measurement dataset."""
        grouped_errors: Dict[str, List[float]] = {}

        for row in dataset.rows:
            for backend in row.models:
                err = row.error(backend)
                if err is not None:
                    key = f"{backend}:{row.cpu}"
                    if key not in grouped_errors:
                        grouped_errors[key] = []
                    grouped_errors[key].append(err)

        corrections: Dict[str, Correction] = {}
        for key, errors in grouped_errors.items():
            n = len(errors)
            if n == 0:
                continue
            mean_b = sum(errors) / n
            var_b = sum((err - mean_b) ** 2 for err in errors) / n if n > 1 else 0.0
            corrections[key] = Correction(
                mean_bias=mean_b,
                variance=var_b,
                sample_count=n,
            )

        return cls(corrections)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize error model to dictionary."""
        return {k: v.to_dict() for k, v in self.corrections.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ErrorModel:
        """Construct ErrorModel from serialized dictionary."""
        corrections = {
            k: Correction.from_dict(v) for k, v in data.items() if isinstance(v, dict)
        }
        return cls(corrections)
