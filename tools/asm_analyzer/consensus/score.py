#!/usr/bin/env python3
"""Cross-model consensus scoring and ranking for assembly variants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

TIE_EPS = 1e-6


@dataclass
class Cell:
    """All reports for one (backend, cpu) cell across the variants."""
    backend: str
    cpu: str
    costs: Dict[str, Optional[float]] = field(default_factory=dict)
    best: Optional[float] = None
    winners: List[str] = field(default_factory=list)

    def finalize(self) -> None:
        """Calculate best cost and identify winning variants within tolerance epsilon."""
        vals = {v: c for v, c in self.costs.items() if c is not None}
        if not vals:
            self.best = None
            self.winners = []
            return
        self.best = min(vals.values())
        self.winners = sorted(
            v for v, c in vals.items()
            if abs(c - self.best) <= TIE_EPS * max(1.0, abs(self.best))
        )


@dataclass
class ConsensusResult:
    """Scoring output for one CPU across all backends/variants."""
    cpu: str
    cells: List[Cell]
    per_variant_regret: Dict[str, float]
    per_variant_mean_regret: Dict[str, float]
    per_variant_wins: Dict[str, int]
    ranking: List[str]
    consensus_winners: List[str]
    disagreement: bool
    note: str = ""


def _rank(variants: List[str], max_r: Dict[str, float],
          mean_r: Dict[str, float], wins: Dict[str, int]) -> List[str]:
    return sorted(variants, key=lambda v: (-wins.get(v, 0), max_r.get(v, float("inf")), mean_r.get(v, float("inf")), v))


def score_cpu(cells: List[Cell], variants: List[str]) -> ConsensusResult:
    """Score one CPU across available simulator cells."""
    for c in cells:
        c.finalize()
    valid_cells = [c for c in cells if c.best is not None]

    regrets: Dict[str, List[float]] = {v: [] for v in variants}
    wins: Dict[str, int] = {v: 0 for v in variants}
    for c in valid_cells:
        for v in variants:
            cost = c.costs.get(v)
            if cost is not None:
                regrets[v].append(cost / c.best)
        for v in c.winners:
            wins[v] += 1

    per_variant_regret = {v: (max(rs) if rs else float("inf")) for v, rs in regrets.items()}
    per_variant_mean = {v: (sum(rs) / len(rs) if rs else float("inf")) for v, rs in regrets.items()}

    backend_winners: Dict[str, Set[str]] = {}
    for c in valid_cells:
        backend_winners.setdefault(c.backend, set()).update(c.winners)
    present = [b for b, w in backend_winners.items() if w]
    consensus_winners: List[str] = []

    if len(present) >= 2:
        counts: Dict[str, int] = {}
        for w in backend_winners.values():
            for v in w:
                counts[v] = counts.get(v, 0) + 1
        max_count = max(counts.values()) if counts else 0
        consensus_winners = sorted(v for v, count in counts.items() if count >= 2 and count == max_count)
        if not consensus_winners:
            ranked = _rank(variants, per_variant_regret, per_variant_mean, wins)
            consensus_winners = [ranked[0]] if ranked else []

    all_winners = sorted({v for w in backend_winners.values() for v in w})
    disagreement = len(all_winners) > 1 if len(present) >= 2 else True
    ranking = _rank(variants, per_variant_regret, per_variant_mean, wins)

    return ConsensusResult(
        cpu=cells[0].cpu if cells else "",
        cells=cells,
        per_variant_regret=per_variant_regret,
        per_variant_mean_regret=per_variant_mean,
        per_variant_wins=wins,
        ranking=ranking,
        consensus_winners=consensus_winners,
        disagreement=disagreement,
    )


def build_cells(reports: Dict[tuple[str, str, str], Optional[float]],
                backends: List[str], cpus: List[str], variants: List[str]) -> List[Cell]:
    """Build and populate Cell objects from report dictionaries."""
    out: List[Cell] = []
    for cpu in cpus:
        for b in backends:
            c = Cell(backend=b, cpu=cpu)
            for v in variants:
                c.costs[v] = reports.get((v, b, cpu))
            c.finalize()
            out.append(c)
    return out
