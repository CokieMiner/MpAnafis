"""Exact statistical gates for paired hardware schedule comparisons."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass(frozen=True)
class PairedEvidence:
    """Robust effect and exact sign-test evidence for paired B/A ratios."""

    ratios: tuple[float, ...]
    effective_rounds: int
    wins: int
    losses: int
    ties: int
    median_ratio: float | None
    improvement_percent: float | None
    sign_p_value: float | None


def paired_evidence(ratios: Iterable[float]) -> PairedEvidence:
    """Summarize finite positive paired ratios without distribution assumptions."""
    kept = tuple(ratio for ratio in ratios if math.isfinite(ratio) and ratio > 0)
    wins = sum(ratio < 1.0 for ratio in kept)
    losses = sum(ratio > 1.0 for ratio in kept)
    ties = len(kept) - wins - losses
    effective_rounds = wins + losses
    median_ratio = statistics.median(kept) if kept else None
    return PairedEvidence(
        ratios=kept,
        effective_rounds=effective_rounds,
        wins=wins,
        losses=losses,
        ties=ties,
        median_ratio=median_ratio,
        improvement_percent=(
            (1.0 - median_ratio) * 100.0
            if median_ratio is not None
            else None
        ),
        sign_p_value=(
            one_sided_sign_p_value(wins, effective_rounds)
            if effective_rounds > 0
            else None
        ),
    )


def one_sided_sign_p_value(wins: int, rounds: int) -> float:
    """Return exact P(X >= wins) for X ~ Binomial(rounds, 0.5)."""
    if rounds < 0 or wins < 0 or wins > rounds:
        raise ValueError("sign-test counts require 0 <= wins <= rounds")
    return sum(math.comb(rounds, count) for count in range(wins, rounds + 1)) / (2**rounds)


def holm_bonferroni(
    p_values: Dict[int, float],
    alpha: float = 0.05,
) -> Dict[int, bool]:
    """Apply Holm's family-wise error correction to candidate P-values."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    adjusted = holm_adjusted_p_values(p_values)
    return {candidate: p_value <= alpha for candidate, p_value in adjusted.items()}


def holm_adjusted_p_values(p_values: Dict[int, float]) -> Dict[int, float]:
    """Return monotone Holm-adjusted P-values for a hypothesis family."""
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    hypotheses = len(ordered)
    adjusted: Dict[int, float] = {}
    running_max = 0.0
    for rank, (candidate, p_value) in enumerate(ordered):
        if not 0.0 <= p_value <= 1.0:
            raise ValueError("P-values must be between zero and one")
        running_max = max(running_max, (hypotheses - rank) * p_value)
        adjusted[candidate] = min(1.0, running_max)
    return adjusted


__all__ = [
    "PairedEvidence",
    "holm_adjusted_p_values",
    "holm_bonferroni",
    "one_sided_sign_p_value",
    "paired_evidence",
]
