#!/usr/bin/env python3
"""Analyze percolation, bond survival and particle mobility."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from huang_md.clusters import (
    analyze_cluster_frame,
    find_periodic_bond_pairs,
)
from huang_md.percolation import (
    analyze_percolation_frame,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--surface-cutoff-reduced",
        type=float,
        default=0.1,
    )

    return parser


def pair_set(
    pairs: np.ndarray,
) -> set[tuple[int, int]]:
    return {
        (int(first), int(second))
        for first, second in pairs
    }


def unwrap_trajectory(
    positions_nm: np.ndarray,
    box_length_nm: float,
) -> np.ndarray:
    unwrapped = np.empty_like(
        positions_nm,
        dtype=np.float64,
    )

    unwrapped[0] = positions_nm[0]

    for frame_index in range(
        1,
        positions_nm.shape[0],
    ):
        displacement = (
            positions_nm[frame_index]
            - positions_nm[frame_index - 1]
        )

        displacement -= (
            box_length_nm
            * np.rint(
                displacement / box_length_nm
            )
        )

        unwrapped[frame_index] = (
            unwrapped[frame_index - 1]
            + displacement
        )

    return unwrapped


def main() -> None:
    args = build_parser().parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = input_dir / "dynamics_analysis"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        input_dir / "metadata.json"
    ).open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    trajectory = np.load(
        input_dir / "trajectory_positions.npz",
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

    diameter_nm = float(
        metadata["diameter_nm"]
    )

    timestep_fs = float(
        metadata["timestep_fs"]
    )

    bond_cutoff_nm = (
        diameter_nm
        * (
            1.0
            + args.surface_cutoff_reduced
        )
    )

    unwrapped = unwrap_trajectory(
        positions_nm=positions_nm,
        box_length_nm=box_length_nm,
    )

    displacement = (
        unwrapped
        - unwrapped[0][None, :, :]
    )

    # Remove whole-system center-of-mass drift.
    displacement -= np.mean(
        displacement,
        axis=1,
        keepdims=True,
    )

    msd_nm2 = np.mean(
        np.sum(
            displacement**2,
            axis=2,
        ),
        axis=1,
    )

    initial_pairs: set[tuple[int, int]] | None = None
    previous_pairs: set[tuple[int, int]] | None = None

    records: list[dict[str, float | int | bool]] = []

    for frame_index, frame in enumerate(
        positions_nm
    ):
        pairs_array = find_periodic_bond_pairs(
            positions_nm=frame,
            box_length_nm=box_length_nm,
            bond_cutoff_nm=bond_cutoff_nm,
        )

        current_pairs = pair_set(
            pairs_array
        )

        if initial_pairs is None:
            initial_pairs = current_pairs.copy()

        if previous_pairs is None:
            consecutive_retention = 1.0
        elif len(previous_pairs) == 0:
            consecutive_retention = (
                1.0
                if len(current_pairs) == 0
                else 0.0
            )
        else:
            consecutive_retention = (
                len(
                    previous_pairs
                    & current_pairs
                )
                / len(previous_pairs)
            )

        if len(initial_pairs) == 0:
            initial_bond_survival = (
                1.0
                if len(current_pairs) == 0
                else 0.0
            )
        else:
            initial_bond_survival = (
                len(
                    initial_pairs
                    & current_pairs
                )
                / len(initial_pairs)
            )

        cluster_result = analyze_cluster_frame(
            positions_nm=frame,
            box_length_nm=box_length_nm,
            bond_cutoff_nm=bond_cutoff_nm,
        )

        percolation_result = (
            analyze_percolation_frame(
                positions_nm=frame,
                box_length_nm=box_length_nm,
                bond_cutoff_nm=bond_cutoff_nm,
            )
        )

        records.append(
            {
                "frame": frame_index,
                "step": int(steps[frame_index]),
                "elapsed_time_ps": (
                    (
                        steps[frame_index]
                        - steps[0]
                    )
                    * timestep_fs
                    / 1000.0
                ),
                "n_bonds": len(current_pairs),
                "clustered_fraction": (
                    cluster_result.clustered_fraction
                ),
                "largest_cluster_size": (
                    cluster_result.largest_cluster_size
                ),
                "largest_cluster_fraction": (
                    cluster_result.largest_cluster_fraction
                ),
                "wraps_x": (
                    percolation_result.wraps_x
                ),
                "wraps_y": (
                    percolation_result.wraps_y
                ),
                "wraps_z": (
                    percolation_result.wraps_z
                ),
                "wraps_any": (
                    percolation_result.wraps_any
                ),
                "initial_bond_survival": (
                    initial_bond_survival
                ),
                "consecutive_bond_retention": (
                    consecutive_retention
                ),
                "msd_nm2": float(
                    msd_nm2[frame_index]
                ),
            }
        )

        previous_pairs = current_pairs

    table = pd.DataFrame.from_records(
        records
    )

    table.to_csv(
        output_dir / "dynamics_by_frame.csv",
        index=False,
    )

    percolation_fraction = float(
        table["wraps_any"].mean()
    )

    mean_clustered_fraction = float(
        table["clustered_fraction"].mean()
    )

    final_bond_survival = float(
        table.iloc[-1][
            "initial_bond_survival"
        ]
    )

    mean_consecutive_retention = float(
        table[
            "consecutive_bond_retention"
        ].iloc[1:].mean()
    )

    final_msd_nm2 = float(
        table.iloc[-1]["msd_nm2"]
    )

    late = table.iloc[max(0, len(table) // 2):]
    late_time_ns = late["elapsed_time_ps"].to_numpy(dtype=float) / 1000.0
    late_largest_fraction = late["largest_cluster_fraction"].to_numpy(dtype=float)
    late_largest_cluster_fraction_slope_per_ns = (
        float(np.polyfit(late_time_ns, late_largest_fraction, 1)[0])
        if len(late) >= 3 and np.ptp(late_time_ns) > 0.0
        else 0.0
    )

    frozen_finite_clusters = bool(
        percolation_fraction < 0.5
        and mean_clustered_fraction >= 0.8
        and final_bond_survival >= 0.9
        and mean_consecutive_retention >= 0.98
    )

    if percolation_fraction >= 0.5:
        interpretation = (
            "persistent_percolated_aggregate"
        )
    elif frozen_finite_clusters:
        interpretation = (
            "kinetically_arrested_finite_aggregates"
        )
    elif mean_clustered_fraction >= 0.8:
        interpretation = (
            "mobile_or_coarsening_clustered_state"
        )
    else:
        interpretation = "mostly_dispersed"

    summary = {
        "input_directory": str(input_dir),
        "frames": int(len(table)),
        "bond_cutoff_nm": bond_cutoff_nm,
        "mean_clustered_fraction": (
            mean_clustered_fraction
        ),
        "mean_largest_cluster_fraction": float(
            table[
                "largest_cluster_fraction"
            ].mean()
        ),
        "percolation_fraction": (
            percolation_fraction
        ),
        "wraps_x_fraction": float(
            table["wraps_x"].mean()
        ),
        "wraps_y_fraction": float(
            table["wraps_y"].mean()
        ),
        "wraps_z_fraction": float(
            table["wraps_z"].mean()
        ),
        "final_initial_bond_survival": (
            final_bond_survival
        ),
        "mean_consecutive_bond_retention": (
            mean_consecutive_retention
        ),
        "final_msd_nm2": final_msd_nm2,
        "late_largest_cluster_fraction_slope_per_ns": (
            late_largest_cluster_fraction_slope_per_ns
        ),
        "frozen_finite_clusters": (
            frozen_finite_clusters
        ),
        "provisional_interpretation": (
            interpretation
        ),
    }

    with (
        output_dir / "dynamics_summary.json"
    ).open("w", encoding="utf-8") as handle:
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
        table["elapsed_time_ps"],
        table["initial_bond_survival"],
    )
    axis.set_xlabel("Elapsed production time (ps)")
    axis.set_ylabel("Initial-bond survival fraction")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        output_dir / "bond_survival.png",
        dpi=240,
    )
    plt.close(figure)

    figure, axis = plt.subplots(
        figsize=(8.0, 5.2)
    )
    axis.plot(
        table["elapsed_time_ps"],
        table["msd_nm2"],
    )
    axis.set_xlabel("Elapsed production time (ps)")
    axis.set_ylabel("Center-of-mass-corrected MSD (nm²)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        output_dir / "particle_msd.png",
        dpi=240,
    )
    plt.close(figure)

    figure, axis = plt.subplots(
        figsize=(8.0, 5.2)
    )
    axis.step(
        table["elapsed_time_ps"],
        table["wraps_any"].astype(int),
        where="post",
    )
    axis.set_xlabel("Elapsed production time (ps)")
    axis.set_ylabel("Periodic wrapping state")
    axis.set_yticks([0, 1])
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        output_dir / "percolation_vs_time.png",
        dpi=240,
    )
    plt.close(figure)

    print("=" * 76)
    print("State dynamics and percolation analysis")
    print("=" * 76)
    print(
        f"Mean clustered fraction       : "
        f"{mean_clustered_fraction:.6f}"
    )
    print(
        f"Percolation fraction          : "
        f"{percolation_fraction:.6f}"
    )
    print(
        f"Final initial-bond survival   : "
        f"{final_bond_survival:.6f}"
    )
    print(
        f"Mean consecutive retention    : "
        f"{mean_consecutive_retention:.6f}"
    )
    print(
        f"Final corrected MSD           : "
        f"{final_msd_nm2:.6f} nm^2"
    )
    print(
        f"Provisional interpretation    : "
        f"{interpretation}"
    )

    print("\nGenerated files:")
    for output in sorted(
        output_dir.iterdir()
    ):
        print(f"  {output}")


if __name__ == "__main__":
    main()
