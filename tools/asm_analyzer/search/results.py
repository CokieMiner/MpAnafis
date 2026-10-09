"""Result models shared by static and hardware schedule searches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CandidateResult:
    """One validated schedule candidate and its modeled or measured evidence."""

    idx: int
    body: str
    is_valid: bool
    cycles: Dict[str, Dict[str, Optional[float]]]
    score: Optional[float] = None
    coverage: int = 0
    missing: int = 0
    samples: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    comparisons: Dict[str, Dict[str, object]] = field(default_factory=dict)
    provenance: Dict[str, object] = field(default_factory=dict)


__all__ = ["CandidateResult"]
