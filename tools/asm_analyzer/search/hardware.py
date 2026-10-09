"""Paired screening and independent holdout for native hardware schedules."""

from __future__ import annotations

import math
import os
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..analyzer import Analyzer
from ..hardware import capture_hardware_context
from ..models import CpuSpec
from .results import CandidateResult
from .statistics import (
    PairedEvidence,
    holm_adjusted_p_values,
    holm_bonferroni,
    paired_evidence,
)

SCREENING_ROUNDS = 11
HOLDOUT_ROUNDS = 11
MIN_VALID_ROUNDS = 9
MIN_IMPROVEMENT = 0.02


@dataclass
class _PairRun:
    baseline_raw: List[float] = field(default_factory=list)
    candidate_raw: List[float] = field(default_factory=list)
    baseline_aggregates: List[float] = field(default_factory=list)
    candidate_aggregates: List[float] = field(default_factory=list)
    ratios: List[float] = field(default_factory=list)
    rejected_baseline: List[float] = field(default_factory=list)
    rejected_candidate: List[float] = field(default_factory=list)
    notes: set[str] = field(default_factory=set)
    failures: set[str] = field(default_factory=set)


def evaluate_hardware_candidates(
    bodies: List[str],
    valid: List[bool],
    cpus: List[CpuSpec],
    backend: Analyzer,
    screening_rounds: int = SCREENING_ROUNDS,
    holdout_rounds: int = HOLDOUT_ROUNDS,
) -> Tuple[List[CandidateResult], set[str]]:
    """Screen all schedules, then independently confirm one corrected winner."""
    if not bodies or len(valid) != len(bodies) or not valid[0]:
        return [], {"hardware comparison requires a validated original and one validity result per body"}
    if screening_rounds < 1 or holdout_rounds < 1 or not cpus:
        return [], {"hardware comparison requires positive round counts and a CPU"}
    if getattr(backend, "name", "nanobench") != "nanobench":
        return [], {"paired schedule confirmation requires nanoBench; perf totals include harness overhead"}
    valid_indices = [index for index, is_valid in enumerate(valid) if is_valid]
    screening: Dict[Tuple[int, str], _PairRun] = {}
    holdout: Dict[Tuple[int, str], _PairRun] = {}
    screening_passes: Dict[Tuple[int, str], bool] = {}
    screening_adjusted: Dict[Tuple[int, str], float] = {}
    holdout_passes: Dict[Tuple[int, str], bool] = {}
    failures: set[str] = set()

    for cpu in cpus:
        if not backend.available():
            failures.add(
                f"{cpu.name}/nanobench: executable was not found in the "
                "privileged process PATH",
            )
            continue
        if not backend.supports(cpu):
            failures.add(
                f"{cpu.name}/nanobench: requested CPU does not exactly match "
                "the detected host",
            )
            continue

        evidence_by_index: Dict[int, PairedEvidence] = {}
        for candidate_index in valid_indices:
            if candidate_index == 0:
                continue
            pair_run = _measure_pair(
                bodies[0],
                bodies[candidate_index],
                cpu,
                backend,
                screening_rounds,
            )
            screening[(candidate_index, cpu.name)] = pair_run
            evidence_by_index[candidate_index] = paired_evidence(pair_run.ratios)
            _require_enough_pairs(
                failures,
                cpu.name,
                candidate_index,
                "screening",
                pair_run,
                screening_rounds,
            )

        corrected = holm_bonferroni(
            {
                index: evidence.sign_p_value or 1.0
                for index, evidence in evidence_by_index.items()
            },
        )
        adjusted = holm_adjusted_p_values(
            {
                index: evidence.sign_p_value or 1.0
                for index, evidence in evidence_by_index.items()
            },
        )
        for index, evidence in evidence_by_index.items():
            screening_passes[(index, cpu.name)] = _passes_effect_gate(evidence) and corrected[index]
            screening_adjusted[(index, cpu.name)] = adjusted[index]

        qualified = [
            index
            for index in evidence_by_index
            if screening_passes[(index, cpu.name)]
        ]
        if not qualified:
            continue
        winner = min(
            qualified,
            key=lambda index: (
                evidence_by_index[index].median_ratio or math.inf,
                index,
            ),
        )
        confirmation = _measure_pair(
            bodies[0],
            bodies[winner],
            cpu,
            backend,
            holdout_rounds,
        )
        holdout[(winner, cpu.name)] = confirmation
        _require_enough_pairs(
            failures,
            cpu.name,
            winner,
            "holdout",
            confirmation,
            holdout_rounds,
        )
        holdout_passes[(winner, cpu.name)] = _passes_effect_gate(
            paired_evidence(confirmation.ratios),
        )

    provenance = _hardware_provenance(screening_rounds, holdout_rounds, screening)
    results = _hardware_results(
        bodies,
        valid_indices,
        cpus,
        screening,
        holdout,
        screening_passes,
        screening_adjusted,
        holdout_passes,
        provenance,
    )
    _rank_confirmed_results(results, cpus)
    return results, failures


def _measure_pair(
    original: str,
    candidate: str,
    cpu: CpuSpec,
    backend: Analyzer,
    rounds: int,
) -> _PairRun:
    pair_run = _PairRun()
    for _ in range(rounds):
        values: List[Optional[float]] = []
        for is_candidate, body in (
            (False, original),
            (True, candidate),
            (True, candidate),
            (False, original),
        ):
            try:
                report = backend.analyze_report(body, cpu)
            except Exception as error:
                pair_run.failures.add(f"{cpu.name}/nanobench: {error}")
                values.append(None)
                continue
            cycles = report.cycles
            raw = pair_run.candidate_raw if is_candidate else pair_run.baseline_raw
            rejected = (
                pair_run.rejected_candidate
                if is_candidate
                else pair_run.rejected_baseline
            )
            if cycles is not None and math.isfinite(cycles):
                raw.append(cycles)
            if report.ok and cycles is not None and math.isfinite(cycles) and cycles > 0:
                values.append(cycles)
                if report.note:
                    pair_run.notes.add(report.note)
            elif cycles is not None and math.isfinite(cycles):
                rejected.append(cycles)
                values.append(None)
            else:
                pair_run.failures.add(
                    f"{cpu.name}/nanobench: "
                    f"{report.note or 'no positive cycle estimate'}",
                )
                values.append(None)
        if all(value is not None for value in values):
            baseline = statistics.fmean((values[0], values[3]))
            measured = statistics.fmean((values[1], values[2]))
            pair_run.baseline_aggregates.append(baseline)
            pair_run.candidate_aggregates.append(measured)
            pair_run.ratios.append(measured / baseline)
    return pair_run


def _require_enough_pairs(
    failures: set[str],
    cpu_name: str,
    candidate_index: int,
    stage: str,
    pair_run: _PairRun,
    requested_rounds: int,
) -> None:
    required = min(MIN_VALID_ROUNDS, requested_rounds)
    if len(pair_run.ratios) < required:
        reasons = "; ".join(sorted(pair_run.failures))
        failures.add(
            f"{cpu_name}/nanobench candidate {candidate_index}: only "
            f"{len(pair_run.ratios)}/{requested_rounds} valid {stage} pairs; "
            f"{required} required"
            + (f" ({reasons})" if reasons else ""),
        )


def _passes_effect_gate(evidence: PairedEvidence) -> bool:
    return (
        evidence.effective_rounds >= MIN_VALID_ROUNDS
        and evidence.median_ratio is not None
        and evidence.median_ratio <= 1.0 - MIN_IMPROVEMENT
        and evidence.sign_p_value is not None
        and evidence.sign_p_value <= 0.05
    )


def _hardware_results(
    bodies: List[str],
    valid_indices: List[int],
    cpus: List[CpuSpec],
    screening: Dict[Tuple[int, str], _PairRun],
    holdout: Dict[Tuple[int, str], _PairRun],
    screening_passes: Dict[Tuple[int, str], bool],
    screening_adjusted: Dict[Tuple[int, str], float],
    holdout_passes: Dict[Tuple[int, str], bool],
    provenance: Dict[str, object],
) -> List[CandidateResult]:
    results: List[CandidateResult] = []
    for index in valid_indices:
        samples: Dict[str, Dict[str, List[float]]] = {}
        comparisons: Dict[str, Dict[str, object]] = {}
        cycles: Dict[str, Dict[str, Optional[float]]] = {}
        for cpu in cpus:
            if index == 0:
                baseline = [
                    value
                    for (candidate, cpu_name), run in screening.items()
                    if candidate != 0 and cpu_name == cpu.name
                    for value in run.baseline_raw
                ]
                samples[cpu.name] = {"nanobench": baseline}
                cycles[cpu.name] = {"nanobench": _positive_median(baseline)}
                continue
            key = (index, cpu.name)
            screen = screening.get(key, _PairRun())
            confirmation = holdout.get(key)
            samples[cpu.name] = {"nanobench": screen.candidate_raw}
            cycles[cpu.name] = {
                "nanobench": _positive_median(screen.candidate_raw),
            }
            comparisons[cpu.name] = {
                "screening": _pair_report(
                    screen,
                    screening_passes.get(key, False),
                    screening_adjusted.get(key),
                ),
                "holdout": (
                    _pair_report(
                        confirmation,
                        holdout_passes.get(key, False),
                    )
                    if confirmation is not None
                    else None
                ),
                "confirmed": holdout_passes.get(key, False),
            }
        results.append(
            CandidateResult(
                idx=index,
                body=bodies[index],
                is_valid=True,
                cycles=cycles,
                samples=samples,
                comparisons=comparisons,
                provenance=provenance,
            ),
        )
    return results


def _pair_report(
    pair_run: _PairRun,
    passed: bool,
    adjusted_p_value: Optional[float] = None,
) -> Dict[str, object]:
    evidence = paired_evidence(pair_run.ratios)
    return {
        "baseline_samples": pair_run.baseline_raw,
        "candidate_samples": pair_run.candidate_raw,
        "baseline_aggregates": pair_run.baseline_aggregates,
        "candidate_aggregates": pair_run.candidate_aggregates,
        "ratios": pair_run.ratios,
        "effective_rounds": evidence.effective_rounds,
        "paired_wins": evidence.wins,
        "paired_losses": evidence.losses,
        "ties": evidence.ties,
        "median_ratio": evidence.median_ratio,
        "median_improvement_percent": evidence.improvement_percent,
        "sign_p_value": evidence.sign_p_value,
        "holm_adjusted_p_value": adjusted_p_value,
        "rejected_baseline": pair_run.rejected_baseline,
        "rejected_candidate": pair_run.rejected_candidate,
        "measurement_errors": sorted(pair_run.failures),
        "passed": passed,
    }


def _positive_median(values: List[float]) -> Optional[float]:
    positive = [value for value in values if math.isfinite(value) and value > 0]
    return statistics.median(positive) if positive else None


def _hardware_provenance(
    screening_rounds: int,
    holdout_rounds: int,
    screening: Dict[Tuple[int, str], _PairRun],
) -> Dict[str, object]:
    try:
        affinity = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        affinity = set()
    provenance = capture_hardware_context(next(iter(affinity))) if len(affinity) == 1 else {
        "logical_cpu": None,
        "affinity": sorted(affinity),
    }
    provenance.update(
        {
            "measurement_order": "A/B/B/A",
            "screening_rounds_per_candidate": screening_rounds,
            "holdout_rounds_for_corrected_winner": holdout_rounds,
            "per_call_aggregate": "median",
            "screening_test": "exact one-sided paired sign test with Holm correction",
            "holdout_test": "fresh exact one-sided paired sign test",
            "minimum_valid_rounds": MIN_VALID_ROUNDS,
            "minimum_median_improvement_percent": MIN_IMPROVEMENT * 100.0,
            "measurement_metrics": sorted(
                note for pair_run in screening.values() for note in pair_run.notes
            ),
        },
    )
    return provenance


def _rank_confirmed_results(
    results: List[CandidateResult],
    cpus: List[CpuSpec],
) -> None:
    """Rank only schedules independently confirmed on every requested CPU."""
    for result in results:
        if result.idx == 0:
            result.coverage = sum(
                result.cycles.get(cpu.name, {}).get("nanobench") is not None
                for cpu in cpus
            )
            result.missing = len(cpus) - result.coverage
            result.score = 1.0 if result.missing == 0 else None
            continue
        confirmed_ratios = [
            comparison["holdout"]["median_ratio"]
            for comparison in result.comparisons.values()
            if comparison.get("confirmed")
            and comparison.get("holdout") is not None
            and comparison["holdout"].get("median_ratio") is not None
        ]
        result.coverage = len(confirmed_ratios)
        result.missing = len(cpus) - result.coverage
        result.score = (
            statistics.fmean(confirmed_ratios)
            if result.missing == 0
            else None
        )
    results.sort(
        key=lambda result: (
            result.missing,
            result.score if result.score is not None else math.inf,
            result.idx,
        ),
    )


__all__ = [
    "HOLDOUT_ROUNDS",
    "MIN_IMPROVEMENT",
    "MIN_VALID_ROUNDS",
    "SCREENING_ROUNDS",
    "evaluate_hardware_candidates",
]
