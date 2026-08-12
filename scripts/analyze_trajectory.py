#!/usr/bin/env python3
"""Analyze cluster statistics from a saved OpenMM trajectory."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from huang_md.clusters import analyze_cluster_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze periodic cluster statistics from "
            "trajectory_positions.npz."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--surface-cutoff-reduced",
        type=float,
        default=0.1,
        help=(
            "Bonded surface-surface distance in units of "
            "particle diameter. Huang uses 0.1."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_dir = args.input_dir.resolve()

    if args.output_dir is None:
        output_dir = input_dir / "cluster_analysis"
    else:
        output_dir = args.output_dir.resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trajectory_file = (
        input_dir / "trajectory_positions.npz"
    )
    metadata_file = input_dir / "metadata.json"

    if not trajectory_file.exists():
        raise FileNotFoundError(
            f"Trajectory not found: {trajectory_file}"
        )

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata not found: {metadata_file}"
        )

    with metadata_file.open(
        "r",
        encoding="utf-8",
    ) as handle:
        metadata = json.load(handle)

    diameter_nm = float(
        metadata["diameter_nm"]
    )
    timestep_fs = float(
        metadata["timestep_fs"]
    )

    if args.surface_cutoff_reduced < 0:
        raise ValueError(
            "surface-cutoff-reduced cannot be negative."
        )

    bond_cutoff_nm = (
        diameter_nm
        * (
            1.0
            + args.surface_cutoff_reduced
        )
    )

    trajectory = np.load(
        trajectory_file,
        allow_pickle=False,
    )

    positions_nm = np.asarray(
        trajectory["positions_nm"],
        dtype=np.float64,
    )

    steps = np.asarray(
        trajectory["steps"],
        dtype=np.int64,
    )

    box_length_nm = float(
        np.asarray(
            trajectory["box_length_nm"]
        )
    )

    if positions_nm.ndim != 3:
        raise ValueError(
            "Trajectory positions must have shape "
            "(frames, particles, 3)."
        )

    if positions_nm.shape[0] != steps.size:
        raise ValueError(
            "Number of trajectory frames and saved steps differ."
        )

    print("=" * 76)
    print("Huang/refCBA trajectory cluster analysis")
    print("=" * 76)
    print(f"Input directory        : {input_dir}")
    print(f"Trajectory             : {trajectory_file}")
    print(f"Frames                 : {positions_nm.shape[0]}")
    print(f"Particles              : {positions_nm.shape[1]}")
    print(f"Box length             : {box_length_nm:.6f} nm")
    print(f"Particle diameter      : {diameter_nm:.6f} nm")
    print(
        f"Surface cutoff         : "
        f"{args.surface_cutoff_reduced:.6f} d"
    )
    print(f"Bond cutoff            : {bond_cutoff_nm:.6f} nm")

    records: list[dict[str, float | int]] = []
    cluster_histogram: Counter[int] = Counter()

    for frame_index, frame_positions in enumerate(
        positions_nm
    ):
        result = analyze_cluster_frame(
            positions_nm=frame_positions,
            box_length_nm=box_length_nm,
            bond_cutoff_nm=bond_cutoff_nm,
        )

        for cluster_size in result.cluster_sizes:
            cluster_histogram[int(cluster_size)] += 1

        step = int(steps[frame_index])

        records.append(
            {
                "frame": frame_index,
                "step": step,
                "time_ps": (
                    step
                    * timestep_fs
                    / 1000.0
                ),
                "n_bonds": result.n_bonds,
                "n_clusters_total": (
                    result.n_clusters_total
                ),
                "n_nontrivial_clusters": (
                    result.n_nontrivial_clusters
                ),
                "monomer_count": (
                    result.monomer_count
                ),
                "monomer_fraction": (
                    result.monomer_fraction
                ),
                "clustered_particle_count": (
                    result.clustered_particle_count
                ),
                "clustered_fraction": (
                    result.clustered_fraction
                ),
                "largest_cluster_size": (
                    result.largest_cluster_size
                ),
                "largest_cluster_fraction": (
                    result.largest_cluster_fraction
                ),
                "number_average_cluster_size": (
                    result.number_average_cluster_size
                ),
                "mean_nontrivial_cluster_size": (
                    result.mean_nontrivial_cluster_size
                ),
                "weight_average_nontrivial_cluster_size": (
                    result.weight_average_nontrivial_cluster_size
                ),
            }
        )

    frame_table = pd.DataFrame.from_records(
        records
    )

    frame_csv = (
        output_dir / "cluster_statistics_by_frame.csv"
    )
    frame_table.to_csv(
        frame_csv,
        index=False,
    )

    histogram_table = pd.DataFrame(
        {
            "cluster_size": sorted(
                cluster_histogram
            ),
            "observed_cluster_count": [
                cluster_histogram[size]
                for size in sorted(
                    cluster_histogram
                )
            ],
        }
    )

    histogram_csv = (
        output_dir / "cluster_size_histogram.csv"
    )
    histogram_table.to_csv(
        histogram_csv,
        index=False,
    )

    summary = {
        "input_directory": str(input_dir),
        "trajectory_frames": int(
            positions_nm.shape[0]
        ),
        "n_particles": int(
            positions_nm.shape[1]
        ),
        "box_length_nm": box_length_nm,
        "diameter_nm": diameter_nm,
        "surface_cutoff_reduced": (
            args.surface_cutoff_reduced
        ),
        "bond_cutoff_nm": bond_cutoff_nm,
        "mean_monomer_fraction": float(
            frame_table[
                "monomer_fraction"
            ].mean()
        ),
        "mean_clustered_fraction": float(
            frame_table[
                "clustered_fraction"
            ].mean()
        ),
        "mean_largest_cluster_size": float(
            frame_table[
                "largest_cluster_size"
            ].mean()
        ),
        "mean_largest_cluster_fraction": float(
            frame_table[
                "largest_cluster_fraction"
            ].mean()
        ),
        "maximum_largest_cluster_size": int(
            frame_table[
                "largest_cluster_size"
            ].max()
        ),
        "maximum_largest_cluster_fraction": float(
            frame_table[
                "largest_cluster_fraction"
            ].max()
        ),
        "final_monomer_fraction": float(
            frame_table.iloc[-1][
                "monomer_fraction"
            ]
        ),
        "final_clustered_fraction": float(
            frame_table.iloc[-1][
                "clustered_fraction"
            ]
        ),
        "final_largest_cluster_size": int(
            frame_table.iloc[-1][
                "largest_cluster_size"
            ]
        ),
        "final_largest_cluster_fraction": float(
            frame_table.iloc[-1][
                "largest_cluster_fraction"
            ]
        ),
    }

    summary_file = (
        output_dir / "cluster_summary.json"
    )

    with summary_file.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    figure, axis = plt.subplots(
        figsize=(8.0, 5.2)
    )

    axis.plot(
        frame_table["time_ps"],
        frame_table["clustered_fraction"],
        marker="o",
        label="Clustered particle fraction",
    )

    axis.plot(
        frame_table["time_ps"],
        frame_table["largest_cluster_fraction"],
        marker="o",
        label="Largest cluster fraction",
    )

    axis.set_xlabel("Simulation time (ps)")
    axis.set_ylabel("Particle fraction")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_dir / "cluster_fractions_vs_time.png",
        dpi=240,
    )
    plt.close(figure)

    print("\nSummary:")
    print(
        f"  Mean monomer fraction       : "
        f"{summary['mean_monomer_fraction']:.6f}"
    )
    print(
        f"  Mean clustered fraction     : "
        f"{summary['mean_clustered_fraction']:.6f}"
    )
    print(
        f"  Mean largest cluster size   : "
        f"{summary['mean_largest_cluster_size']:.3f}"
    )
    print(
        f"  Maximum largest cluster size: "
        f"{summary['maximum_largest_cluster_size']}"
    )
    print(
        f"  Final largest cluster size  : "
        f"{summary['final_largest_cluster_size']}"
    )

    print("\nGenerated files:")

    for output in sorted(output_dir.iterdir()):
        print(f"  {output}")

    print("\nBasic cluster analysis completed.")


if __name__ == "__main__":
    main()
