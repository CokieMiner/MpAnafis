#!/usr/bin/env python3
"""
Publication-Quality Benchmark Plotter & Performance Suite for MpAnafis.

Parses Divan benchmark outputs from stdout or file, computes statistics,
speedup ratios, multi-threading scaling efficiency, and generates high-res
figures and structured JSON/CSV data for reports and documentation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# ── Global Matplotlib Typography & Aesthetic Config ──────────────────────────

plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
plt.rcParams.update(
    {
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "figure.dpi": 300,
        "lines.antialiased": True,
        "patch.antialiased": True,
    }
)

PALETTE = {
    "mp": {"label": "MpAnafis Production Tower (Ambient Executor)", "color": "#008080", "ls": "-", "marker": "o", "lw": 2.6, "alpha": 0.22},
    "flint_parallel": {"label": "FLINT 3 (Configured Workers)", "color": "#E65100", "ls": "--", "marker": "s", "lw": 2.2, "alpha": 0.18},
    "flint_serial": {"label": "FLINT 3 (1 Thread)", "color": "#F57C00", "ls": ":", "marker": "v", "lw": 1.6, "alpha": 0.15},
    "gmp_serial": {"label": "GMP 6.3 (1 Thread)", "color": "#7B1FA2", "ls": ":", "marker": "d", "lw": 1.9, "alpha": 0.15},
}


def to_us(t_str: str) -> Optional[float]:
    """Converts a formatted time string (e.g., '219.6 µs', '1.499 ms', '2.117 s') to microseconds."""
    if not t_str:
        return None
    t_str = t_str.strip().replace(" ", "")
    try:
        if t_str.endswith("ns"):
            return float(t_str[:-2]) / 1000.0
        if t_str.endswith("µs") or t_str.endswith("us"):
            return float(t_str[:-2])
        if t_str.endswith("ms"):
            return float(t_str[:-2]) * 1000.0
        if t_str.endswith("s"):
            return float(t_str[:-1]) * 1_000_000.0
        return float(t_str)
    except ValueError:
        return None


def parse_divan_file(filepath: Path | str) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Parses Divan benchmark table text into structured numeric data."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    engines: Dict[str, Dict[int, Dict[str, float]]] = {}
    current_engine: Optional[str] = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        m_eng = re.search(r"[├╰]─\s+([a-zA-Z0-9_]+)\s*│", line)
        if m_eng:
            current_engine = m_eng.group(1)
            if current_engine not in engines:
                engines[current_engine] = {}
            continue

        # Format: limbs | fastest | slowest | median | mean | samples | iters
        m_row = re.search(
            r"[├╰]─\s+([0-9]+)(?:-limbs/[0-9]+-workers)?\s+"
            r"([0-9.]+\s*[µmns]+)\s*│\s*([0-9.]+\s*[µmns]+)\s*│\s*"
            r"([0-9.]+\s*[µmns]+)\s*│\s*([0-9.]+\s*[µmns]+)\s*│\s*"
            r"([0-9]+)\s*│\s*([0-9]+)",
            line,
        )
        if m_row and current_engine:
            size = int(m_row.group(1))
            fastest = to_us(m_row.group(2))
            slowest = to_us(m_row.group(3))
            median = to_us(m_row.group(4))
            mean = to_us(m_row.group(5))
            if median is not None:
                engines[current_engine][size] = {
                    "fastest": fastest if fastest is not None else median,
                    "slowest": slowest if slowest is not None else median,
                    "median": median,
                    "mean": mean if mean is not None else median,
                    "samples": int(m_row.group(6)),
                    "iterations": int(m_row.group(7)),
                }

    # Normalize huge and normal series into consolidated engine records
    merged: Dict[str, Dict[int, Dict[str, float]]] = {
        "flint_parallel": {},
        "flint_serial": {},
        "gmp_serial": {},
        "mp": {},
    }

    for eng, data in engines.items():
        base = eng.replace("_huge", "")
        if base in merged:
            merged[base].update(data)
        else:
            merged[base] = data

    return merged


def save_plot(fig: plt.Figure, filename: str, output_dirs: List[Path]) -> None:
    """Saves a figure to all configured output directories."""
    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / filename
        fig.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {filename} in {len(output_dirs)} location(s)")


def export_json_and_csv(
    data: Dict[str, Dict[int, Dict[str, float]]], output_dirs: List[Path]
) -> None:
    """Exports structured benchmark JSON and CSV data."""
    export_obj: Dict[str, Any] = {}
    csv_rows = [
        "engine,limb_count,bit_width,fastest_us,slowest_us,median_us,mean_us,samples,iterations,mlimbs_per_sec"
    ]

    for eng, series in sorted(data.items()):
        export_obj[eng] = {}
        for size, stats in sorted(series.items()):
            bits = size * 64
            med_us = stats["median"]
            mlimbs_sec = (size / (med_us / 1_000_000.0)) / 1_000_000.0 if med_us > 0 else 0.0
            row_dict = {
                "limbs": size,
                "bits": bits,
                "fastest_us": stats["fastest"],
                "slowest_us": stats["slowest"],
                "median_us": med_us,
                "mean_us": stats["mean"],
                "samples": stats["samples"],
                "iterations": stats["iterations"],
                "mlimbs_per_sec": round(mlimbs_sec, 3),
            }
            export_obj[eng][size] = row_dict
            csv_rows.append(
                f"{eng},{size},{bits},{stats['fastest']},{stats['slowest']},{med_us},"
                f"{stats['mean']},{stats['samples']},{stats['iterations']},{round(mlimbs_sec, 3)}"
            )

    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump(export_obj, f, indent=2)
        with open(out_dir / "benchmark_results.csv", "w", encoding="utf-8") as f:
            f.write("\n".join(csv_rows) + "\n")
    print("Exported benchmark_results.json and benchmark_results.csv")


def export_metadata(metadata: Dict[str, Any], output_dirs: List[Path]) -> None:
    """Exports the provenance supplied for this benchmark capture."""
    for out_dir in output_dirs:
        with open(out_dir / "benchmark_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            f.write("\n")
    print("Exported benchmark_metadata.json")


def plot_dense_scaling(
    data: Dict[str, Dict[int, Dict[str, float]]], output_dirs: List[Path]
) -> None:
    """Figure 1: Full Range Log-Log Performance Comparison."""
    fig, ax = plt.subplots(figsize=(12, 7.2))

    for eng, style in PALETTE.items():
        if eng not in data or not data[eng]:
            continue
        sizes = sorted(data[eng].keys())
        bits = np.array([s * 64 for s in sizes])
        medians = np.array([data[eng][s]["median"] for s in sizes])
        ax.plot(
            bits,
            medians,
            label=style["label"],
            color=style["color"],
            linestyle=style["ls"],
            marker=style["marker"],
            markersize=5.2,
            linewidth=style["lw"],
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("Operand Bit-Width ($n$ bits)", fontweight="bold")
    ax.set_ylabel("Execution Time (Wall-Clock)", fontweight="bold")
    ax.set_title(
        "Full-Tower Scaling: MpAnafis vs FLINT 3 vs GMP (Log-Log)\n"
        "[Median estimates | Mp runs the complete production dispatcher]",
        fontweight="bold",
        pad=12,
    )

    xticks = [
        16 * 1024,
        64 * 1024,
        256 * 1024,
        1024 * 1024,
        4 * 1024 * 1024,
        16 * 1024 * 1024,
        64 * 1024 * 1024,
        256 * 1024 * 1024,
        1024 * 1024 * 1024,
        2048 * 1024 * 1024,
    ]
    xtick_labels = ["16 Kib", "64 Kib", "256 Kib", "1 Mib", "4 Mib", "16 Mib", "64 Mib", "256 Mib", "1 Gib", "2 Gib"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

    yticks = [10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    ytick_labels = ["10 µs", "100 µs", "1 ms", "10 ms", "100 ms", "1 s", "10 s"]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.65)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#ddd", shadow=True)
    plt.tight_layout()
    save_plot(fig, "dense_scaling_all_engines.png", output_dirs)


def plot_speedup_ratios(
    data: Dict[str, Dict[int, Dict[str, float]]], output_dirs: List[Path]
) -> None:
    """Figure 2: Relative Speedup Ratios vs Baselines."""
    fig, ax = plt.subplots(figsize=(11, 6))

    mp = data.get("mp", {})
    flint_par = data.get("flint_parallel", {})
    gmp_ser = data.get("gmp_serial", {})

    ax.axhline(1.0, color="#555", linestyle="--", linewidth=1.2, label="Parity (1.0×)")

    if mp and flint_par:
        common_flint = sorted(list(set(mp.keys()) & set(flint_par.keys())))
        if common_flint:
            bits = np.array([s * 64 for s in common_flint])
            speedup = np.array([flint_par[s]["median"] / mp[s]["median"] for s in common_flint])
            ax.plot(
                bits,
                speedup,
                label="Mp Production Tower vs FLINT 3 (Configured Workers)",
                color="#008080",
                lw=2.5,
                marker="o",
                markersize=5,
            )

    if mp and gmp_ser:
        common_gmp = sorted(list(set(mp.keys()) & set(gmp_ser.keys())))
        if common_gmp:
            bits = np.array([s * 64 for s in common_gmp])
            speedup = np.array([gmp_ser[s]["median"] / mp[s]["median"] for s in common_gmp])
            ax.plot(
                bits,
                speedup,
                label="Mp Production Tower vs GMP 6.3 Serial",
                color="#7B1FA2",
                lw=2.5,
                marker="d",
                markersize=5,
            )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Operand Bit-Width ($n$ bits)", fontweight="bold")
    ax.set_ylabel("Speedup Multiplier ($T_{\\mathrm{baseline}} / T_{\\mathrm{MpAnafis}}$)", fontweight="bold")
    ax.set_title(
        "MpAnafis Production-Tower Performance vs FLINT 3 & GMP\n"
        "[Median ratios; values > 1.0× indicate MpAnafis is faster]",
        fontweight="bold",
        pad=12,
    )

    xticks = [
        16 * 1024,
        64 * 1024,
        256 * 1024,
        1024 * 1024,
        4 * 1024 * 1024,
        16 * 1024 * 1024,
        64 * 1024 * 1024,
        256 * 1024 * 1024,
        1024 * 1024 * 1024,
        2048 * 1024 * 1024,
    ]
    xtick_labels = ["16 Kib", "64 Kib", "256 Kib", "1 Mib", "4 Mib", "16 Mib", "64 Mib", "256 Mib", "1 Gib", "2 Gib"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.65)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#ddd", shadow=True)
    plt.tight_layout()
    save_plot(fig, "speedup_comparison.png", output_dirs)


def plot_single_thread_efficiency(
    data: Dict[str, Dict[int, Dict[str, float]]], output_dirs: List[Path]
) -> None:
    """Figure 3: Single-Thread Architecture Efficiency (1 Core vs 1 Core).

    Meaningful only for captures run with a one-worker ambient pool
    (`RAYON_NUM_THREADS=1` or no `rayon` feature), where the Mp arm is the
    sequential tower.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    mp = data.get("mp", {})
    flint_seq = data.get("flint_serial", {})
    gmp_ser = data.get("gmp_serial", {})

    common = sorted(list(set(mp.keys()) & set(gmp_ser.keys()) & set(flint_seq.keys())))
    if common:
        bits = np.array([s * 64 for s in common])
        ratio_gmp = np.array([gmp_ser[s]["median"] / mp[s]["median"] for s in common])
        ratio_flint = np.array([flint_seq[s]["median"] / mp[s]["median"] for s in common])

        ax.axhline(1.0, color="#555", linestyle="--", linewidth=1.2, label="Parity (1.0×)")
        ax.plot(
            bits,
            ratio_gmp,
            label="Mp Production Tower vs GMP 6.3",
            color="#7B1FA2",
            lw=2.3,
            marker="d",
            markersize=5,
        )
        ax.plot(
            bits,
            ratio_flint,
            label="Mp Production Tower vs FLINT 3",
            color="#E65100",
            lw=2.3,
            marker="s",
            markersize=5,
        )

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Operand Bit-Width ($n$ bits)", fontweight="bold")
        ax.set_ylabel("Single-Thread Speedup ($T_{\\mathrm{baseline}} / T_{\\mathrm{MpAnafis}}$)", fontweight="bold")
        ax.set_title(
            "Single-Thread Tower Efficiency\n"
            "[One-worker capture; values > 1.0× indicate Mp is faster]",
            fontweight="bold",
            pad=12,
        )

        xticks = [
            16 * 1024,
            64 * 1024,
            256 * 1024,
            1024 * 1024,
            4 * 1024 * 1024,
            16 * 1024 * 1024,
            64 * 1024 * 1024,
            256 * 1024 * 1024,
            1024 * 1024 * 1024,
            2048 * 1024 * 1024,
        ]
        xtick_labels = ["16 Kib", "64 Kib", "256 Kib", "1 Mib", "4 Mib", "16 Mib", "64 Mib", "256 Mib", "1 Gib", "2 Gib"]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.65)
        ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#ddd", shadow=True)
        plt.tight_layout()
        save_plot(fig, "single_thread_efficiency.png", output_dirs)


def plot_sample_spread(
    data: Dict[str, Dict[int, Dict[str, float]]], output_dirs: List[Path]
) -> None:
    """Figure 4: descriptive slowest/fastest estimate spread and sample depth."""
    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = False

    for eng, style in PALETTE.items():
        if eng not in data or not data[eng]:
            continue
        sizes = sorted(data[eng].keys())
        bits = np.array([size * 64 for size in sizes])
        spreads = np.array(
            [data[eng][size]["slowest"] / data[eng][size]["fastest"] for size in sizes]
        )
        ax.plot(
            bits,
            spreads,
            label=style["label"],
            color=style["color"],
            linestyle=style["ls"],
            marker=style["marker"],
            markersize=5,
            linewidth=style["lw"],
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.axhline(1.0, color="#555", linestyle="--", linewidth=1.2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Operand Bit-Width ($n$ bits)", fontweight="bold")
    ax.set_ylabel("Slowest / Fastest Estimate", fontweight="bold")
    ax.set_title(
        "Observed Benchmark Estimate Spread\n"
        "[Descriptive diagnostic, not a confidence interval; inspect exported sample counts]",
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.65)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#ddd")
    plt.tight_layout()
    save_plot(fig, "sample_estimate_spread.png", output_dirs)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publication-grade benchmark visualizer and analyzer for MpAnafis."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="ola.txt",
        help="Path to Divan benchmark stdout capture (default: ola.txt)",
    )
    parser.add_argument("--command", default="unknown", help="Exact benchmark command")
    parser.add_argument("--revision", default="unknown", help="Git revision or worktree identifier")
    parser.add_argument("--cpu", default="unknown", help="CPU model")
    parser.add_argument("--affinity", default="unpinned", help="CPU-affinity policy")
    parser.add_argument("--features", default="unknown", help="Cargo feature set")
    parser.add_argument("--rayon-workers", type=int, default=0, help="Rayon worker count")
    parser.add_argument("--flint-version", default="unknown", help="FLINT version")
    parser.add_argument("--gmp-version", default="unknown", help="GMP version")
    parser.add_argument("--notes", default="", help="Free-form run notes")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="docs/int/graphs",
        help="Primary output directory for generated graphs (default: docs/int/graphs)",
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: input file '{input_path}' not found.", file=sys.stderr)
        return 1

    primary_out = Path(args.output_dir)
    primary_out.mkdir(parents=True, exist_ok=True)

    output_dirs = [primary_out]

    print(f"Parsing benchmark data from: {input_path}")
    data = parse_divan_file(input_path)

    non_empty = {k: len(v) for k, v in data.items() if v}
    print(f"Parsed series: {non_empty}")

    if not any(non_empty.values()):
        print("Warning: No benchmark series found in input file.", file=sys.stderr)
        return 0

    export_json_and_csv(data, output_dirs)
    export_metadata(
        {
            "input": str(input_path),
            "command": args.command,
            "revision": args.revision,
            "cpu": args.cpu,
            "affinity": args.affinity,
            "features": args.features,
            "rayon_workers": args.rayon_workers,
            "flint_version": args.flint_version,
            "gmp_version": args.gmp_version,
            "notes": args.notes,
            "series": "Mp production tower with reusable state versus raw FLINT/GMP mpn",
            "band": None,
            "variability": (
                "sample_estimate_spread.png reports slowest/fastest Divan estimates; "
                "it is descriptive and not a confidence interval"
            ),
            "raw_samples": "not available in the captured Divan table; sample counts are exported",
        },
        output_dirs,
    )
    plot_dense_scaling(data, output_dirs)
    plot_speedup_ratios(data, output_dirs)
    plot_single_thread_efficiency(data, output_dirs)
    plot_sample_spread(data, output_dirs)

    print("All benchmark plots and structured data generated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
