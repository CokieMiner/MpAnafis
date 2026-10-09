"""Repository-wide schedule search with optional native hardware confirmation."""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..asm_util import host_cpu_name
from ..backends import make_backends
from ..diff_test import supports_native_differential
from ..kernel_source import discover_kernels, extract_kernel_variants
from ..models import ANALYTICAL_BACKENDS, CPUS, CpuSpec
from ..regions import select_analysis_region
from ..search.engine import search_kernel
from ..search.hardware import evaluate_hardware_candidates
from ..search.results import CandidateResult
from ..search.source_apply import rewrite_confirmed_schedule
from ..targets import (
    architecture_for_path,
    compatible_cpus,
    default_cpus_for_architecture,
    host_architecture,
)
from ..types import ArchitectureFamily


def run_optimize(
    target_path: Optional[str] = None,
    cpus: Optional[List[CpuSpec]] = None,
    backend_names: Optional[List[str]] = None,
    candidates: int = 50,
    hardware_shortlist: int = 4,
    seed: int = 42,
    use_wsl: bool = False,
    hardware: bool = False,
    no_alias: bool = False,
    disjoint_pointers: Optional[str] = None,
    apply_confirmed: bool = False,
    as_json: bool = False,
) -> int:
    """Search every eligible local kernel and retain evidence for each stage."""
    if apply_confirmed and not hardware:
        print("Error: --apply-confirmed requires --hardware", file=sys.stderr)
        return 1
    kernel_files = _kernel_files(target_path)
    records: List[Dict[str, object]] = []
    fatal_failures = 0
    analytical_backends = [
        name
        for name in (backend_names or list(ANALYTICAL_BACKENDS))
        if name in ANALYTICAL_BACKENDS
    ]
    if not analytical_backends:
        print("Error: optimize requires at least one analytical backend", file=sys.stderr)
        return 1
    native_architecture = host_architecture()

    disjoint_bases = None
    if disjoint_pointers:
        disjoint_bases = {ptr.strip().lstrip("%") for ptr in disjoint_pointers.split(",") if ptr.strip()}
    elif no_alias:
        disjoint_bases = {"rdi", "rsi", "rdx", "rcx", "r8", "r9", "r10", "r11", "x0", "x1", "x2", "x3", "x4", "x5", "a0", "a1", "a2", "a3"}

    for path in kernel_files:
        architecture = architecture_for_path(path)
        original_source = (
            path.read_text(encoding="utf-8")
            if apply_confirmed and path.suffix == ".rs"
            else None
        )
        variants, extraction_error = extract_kernel_variants(path, use_wsl=use_wsl)
        if not variants:
            fatal_failures += 1
            records.append(
                {
                    "kernel": str(path),
                    "architecture": architecture.value,
                    "status": "extraction_failed",
                    "error": extraction_error,
                },
            )
            continue
        pending_source = original_source
        path_records: List[Dict[str, object]] = []
        source_update_failed = False
        cpu_specs = compatible_cpus(
            cpus or default_cpus_for_architecture(architecture),
            architecture,
        )
        for variant in variants:
            region = select_analysis_region(variant.asm, architecture)
            record, confirmed_body = _search_variant(
                path,
                variant.name,
                architecture,
                region.schedulable_asm(),
                cpu_specs,
                analytical_backends,
                candidates,
                hardware_shortlist,
                seed,
                use_wsl,
                hardware,
                native_architecture == architecture
                and supports_native_differential(architecture),
                disjoint_bases=disjoint_bases,
            )
            if apply_confirmed and confirmed_body is not None:
                if pending_source is None or variant.source_line is None:
                    record["source_update"] = {
                        "status": "not_applicable",
                        "reason": (
                            "confirmed schedule came from a macro-expanded or non-Rust source; "
                            "manual source mapping is required"
                        ),
                    }
                else:
                    try:
                        pending_source = rewrite_confirmed_schedule(
                            pending_source,
                            variant.source_line,
                            variant.asm,
                            region.schedulable_asm(),
                            confirmed_body,
                        )
                        record["source_update"] = {"status": "planned"}
                    except ValueError as error:
                        source_update_failed = True
                        record["source_update"] = {
                            "status": "failed",
                            "reason": str(error),
                        }
            elif apply_confirmed:
                record["source_update"] = {"status": "not_needed"}
            path_records.append(record)
            if record["status"] in ("search_failed", "hardware_failed"):
                fatal_failures += 1
        if (
            apply_confirmed
            and pending_source is not None
            and pending_source != original_source
        ):
            if source_update_failed:
                for record in path_records:
                    update = record.get("source_update")
                    if isinstance(update, dict) and update.get("status") == "planned":
                        update["status"] = "cancelled"
                        update["reason"] = (
                            "another confirmed schedule in the same file failed mapping"
                        )
            elif path.read_text(encoding="utf-8") != original_source:
                source_update_failed = True
                for record in path_records:
                    update = record.get("source_update")
                    if isinstance(update, dict) and update.get("status") == "planned":
                        update["status"] = "cancelled"
                        update["reason"] = "source changed while optimization was running"
            else:
                _atomic_write(path, pending_source)
                for record in path_records:
                    update = record.get("source_update")
                    if isinstance(update, dict) and update.get("status") == "planned":
                        update["status"] = "applied"
        if source_update_failed:
            fatal_failures += 1
        records.extend(path_records)

    if as_json:
        print(json.dumps(records, indent=2))
    else:
        _print_summary(records)
    if not records:
        print("Error: optimize discovered no inline-assembly kernels", file=sys.stderr)
        return 1
    return 1 if fatal_failures else 0


def _kernel_files(target_path: Optional[str]) -> List[Path]:
    if target_path is None:
        return discover_kernels()
    path = Path(target_path)
    return [path] if path.is_file() else discover_kernels(path)


def _search_variant(
    path: Path,
    variant_name: str,
    architecture: ArchitectureFamily,
    body: str,
    cpus: List[CpuSpec],
    backends: List[str],
    candidates: int,
    hardware_shortlist: int,
    seed: int,
    use_wsl: bool,
    hardware: bool,
    verify_natively: bool,
    disjoint_bases: Optional[Set[str] | Set[Tuple[str, str]]] = None,
) -> Tuple[Dict[str, object], Optional[str]]:
    results, diagnostics = search_kernel(
        body,
        cpus,
        backend_names=backends,
        candidates_count=candidates,
        seed=seed,
        use_wsl=use_wsl,
        run_diff_test=verify_natively,
        architecture=architecture,
        allow_unmodeled=True,
        disjoint_bases=disjoint_bases,
    )
    if not results:
        return {
            "kernel": variant_name,
            "source": str(path),
            "architecture": architecture.value,
            "status": "search_failed",
            "error": diagnostics,
        }, None
    original = next((result for result in results if result.idx == 0), None)
    if original is None:
        return {
            "kernel": variant_name,
            "source": str(path),
            "architecture": architecture.value,
            "status": "search_failed",
            "error": "candidate generation did not retain the original body",
        }, None
    unique = _unique_ranked_candidates(results, original, hardware_shortlist)
    record: Dict[str, object] = {
        "kernel": variant_name,
        "source": str(path),
        "architecture": architecture.value,
        "status": (
            "model_unavailable"
            if results[0].score is None
            else "screened" if verify_natively else "static_only"
        ),
        "generated_candidates": len(results),
        "static_original": _result_summary(original),
        "static_winner": _result_summary(results[0]),
        "static_tie_count": sum(
            result.missing == results[0].missing
            and result.score == results[0].score
            for result in results
        ),
        "static_diagnostics": diagnostics,
        "hardware": None,
    }
    if not hardware or not verify_natively:
        return record, None
    if results[0].score is None:
        record["hardware"] = {
            "reason": "hardware shortlist requires at least one analytical ranking cell",
        }
        return record, None

    hw_name = "nanobench"
    backend = make_backends([hw_name], wsl=use_wsl).get(hw_name)
    if backend is None or not backend.available():
        record["status"] = "screened_hardware_unavailable"
        record["hardware"] = {
            "reason": f"Hardware measurement backend '{hw_name}' is not available on this host",
        }
        return record, None

    host_name = host_cpu_name()
    host_cpu = CPUS.get(host_name)
    if host_cpu is None:
        record["status"] = "hardware_failed"
        record["hardware"] = {"reason": "exact host CPU model is unknown"}
        return record, None

    shortlist_bodies = [candidate.body for candidate in unique]
    hardware_results, hardware_failures = evaluate_hardware_candidates(
        shortlist_bodies,
        [True] * len(shortlist_bodies),
        [host_cpu],
        backend,
    )
    if not hardware_results:
        record["status"] = "hardware_failed"
        record["hardware"] = {"failures": sorted(hardware_failures)}
        return record, None
    source_indices = [candidate.idx for candidate in unique]
    confirmed = hardware_results[0] if hardware_results[0].idx != 0 else None
    record["status"] = "confirmed" if confirmed is not None else "original_retained"
    record["hardware"] = {
        "host_cpu": host_name,
        "shortlist_source_indices": source_indices,
        "confirmed_source_index": (
            source_indices[confirmed.idx]
            if confirmed is not None
            else None
        ),
        "failures": sorted(hardware_failures),
        "results": [_result_summary(result, include_evidence=True) for result in hardware_results],
    }
    if hardware_failures:
        record["status"] = "hardware_failed"
        return record, None
    return record, confirmed.body if confirmed is not None else None


def _atomic_write(path: Path, source: str) -> None:
    """Replace a source file only after a complete rewrite has been rendered."""
    metadata = path.stat()
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(source)
            temporary.flush()
            os.fchmod(temporary.fileno(), stat.S_IMODE(metadata.st_mode))
            if os.geteuid() == 0:
                os.fchown(temporary.fileno(), metadata.st_uid, metadata.st_gid)
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _unique_ranked_candidates(
    results: List[CandidateResult],
    original: CandidateResult,
    limit: int,
) -> List[CandidateResult]:
    selected = [original]
    seen = {original.body}
    for result in results:
        if result.body in seen or result.idx == 0:
            continue
        selected.append(result)
        seen.add(result.body)
        if len(selected) >= max(2, limit + 1):
            break
    return selected


def _result_summary(
    result: CandidateResult,
    include_evidence: bool = False,
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "index": result.idx,
        "score": result.score,
        "coverage": result.coverage,
        "missing": result.missing,
        "cycles": result.cycles,
        "body": result.body,
        "provenance": result.provenance,
    }
    if include_evidence:
        summary["samples"] = result.samples
        summary["comparisons"] = result.comparisons
    return summary


def _print_summary(records: List[Dict[str, object]]) -> None:
    counts: Dict[str, int] = {}
    for record in records:
        status = str(record["status"])
        counts[status] = counts.get(status, 0) + 1
        print(f"{status:29} {record['kernel']}")
    print("\nSummary:")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")


__all__ = ["run_optimize"]
