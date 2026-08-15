#!/usr/bin/env python3
"""Append-only empirical measurement dataset for the assembly analyzer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..backends.mca_driver import REPO_ROOT

DATA_DIR = REPO_ROOT / "tools" / "asm_analyzer" / "data"
DEFAULT_DATASET = DATA_DIR / "dataset.jsonl"
SCHEMA_VERSION = 1


@dataclass
class FeatureSet:
    """Static features of one kernel variant on one CPU."""
    instruction_count: int
    uops: Optional[float] = None
    dependency_depth: Optional[float] = None
    flag_chain_length: Optional[int] = None
    code_size: Optional[int] = None
    gpr_count: Optional[int] = None
    mem_loads: Optional[int] = None
    mem_stores: Optional[int] = None
    rmw_count: Optional[int] = None
    unroll_factor: Optional[int] = None
    mul_latency_slack: Optional[int] = None
    cache_straddles: Optional[int] = None


@dataclass
class MeasurementRow:
    """One empirical measurement record."""
    timestamp: float
    cpu: str
    kernel_id: str
    variant: str
    features: FeatureSet
    models: Dict[str, Optional[float]]
    measured: Dict[str, Any]
    schema_version: int = SCHEMA_VERSION
    provenance: Dict[str, Any] = field(default_factory=dict)

    def error(self, backend: str) -> Optional[float]:
        """Compute the simulator prediction error (measured - predicted) for a backend."""
        pred = self.models.get(backend)
        meas = self.measured.get("cycles")
        if pred is None or meas is None:
            return None
        return float(meas) - float(pred)


class Dataset:
    """In-memory collection of measurement rows loaded from a JSONL file."""

    def __init__(self, rows: Optional[List[MeasurementRow]] = None) -> None:
        self.rows: List[MeasurementRow] = list(rows or [])

    @classmethod
    def load(cls, path: Optional[Path] = None) -> Dataset:
        """Load calibration dataset from JSONL path or default location."""
        p = path or DEFAULT_DATASET
        if not p.exists():
            return cls([])
        rows: List[MeasurementRow] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                f_data = data.get("features", {})
                feat = FeatureSet(**f_data)
                row = MeasurementRow(
                    timestamp=data.get("timestamp", 0.0),
                    cpu=data.get("cpu", ""),
                    kernel_id=data.get("kernel_id", ""),
                    variant=data.get("variant", ""),
                    features=feat,
                    models=data.get("models", {}),
                    measured=data.get("measured", {}),
                    schema_version=data.get("schema_version", 1),
                    provenance=data.get("provenance", {}),
                )
                rows.append(row)
            except (json.JSONDecodeError, TypeError, KeyError):
                pass
        return cls(rows)
