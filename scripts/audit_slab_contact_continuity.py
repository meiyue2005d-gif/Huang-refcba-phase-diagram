#!/usr/bin/env python3
"""Audit continuous and intermittent contact survival in slab trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--analysis-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--persistent-fraction",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--contact-cutoff-nm",
        type=float,
        default=5.35,
    )
    return parser.parse_args()


def minimum_image(
    displacement: np.ndarray,
    box_lengths: np.ndarray,
) -> np.ndarray:
    return (
        displacement
        - box_lengths
        * np.rint(displacement / box_lengths)
    )


def circular_distance_z(
    z: np.ndarray,
    center_z: float,
    box_z: float,
) -> np.ndarray:
    return (
        (z - center_z + box_z / 2.0)
        % box_z
        - box_z / 2.0
    )


def contacts_for_frame(
    positions: np.ndarray,
    box_lengths: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    cutoff_nm: float,
) -> np.ndarray:
    displacement = (
        positions[pair_i]
        - positions[pair_j]
    )

    displacement = minimum_image(
        displacement,
        box_lengths,
    )

    distance_squared = np.sum(
        displacement * displacement,
        axis=1,
    )

    return distance_squared <= cutoff_nm**2


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()

    trajectory_file = (
        input_dir / "trajectory_positions.npz"
    )
    metrics_file = (
        input_dir
        / "direct_coexistence_frame_metrics.csv"
    )
    coexistence_summary_file = (
        input_dir
        / "direct_coexistence_summary.json"
    )

    for path in (
        trajectory_file,
        metrics_file,
        coexistence_summary_file,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    trajectory = np.load(trajectory_file)

    positions = np.asarray(
        trajectory["positions_nm"],
        dtype=float,
    )
    box_lengths = np.asarray(
        trajectory["box_lengths_nm"],
        dtype=float,
    )

    metrics = pd.read_csv(metrics_file)

    coexistence_summary = json.loads(
        coexistence_summary_file.read_text()
    )

    frame_count = len(positions)
    particle_count = positions.shape[1]

    if len(metrics) != frame_count:
        raise ValueError(
            "Trajectory and metrics have different frame counts."
        )

    analysis_start = int(
        np.floor(
            frame_count
            * (1.0 - args.analysis_fraction)
        )
    )
    analysis_start = min(
        max(analysis_start, 0),
        frame_count - 2,
    )

    positions = positions[analysis_start:]

    times_ps = metrics[
        "time_ps"
    ].to_numpy(float)[analysis_start:]

    centers_z = metrics[
        "circular_center_z_nm"
    ].to_numpy(float)[analysis_start:]

    slab_thickness = float(
        coexistence_summary[
            "initial_slab_thickness_nm"
        ]
    )

    membership = np.zeros(
        (len(positions), particle_count),
        dtype=bool,
    )

    for frame_index in range(len(positions)):
        dz = circular_distance_z(
            positions[frame_index, :, 2],
            centers_z[frame_index],
            box_lengths[2],
        )

        membership[frame_index] = (
            np.abs(dz)
            <= slab_thickness / 2.0
        )

    membership_fraction = membership.mean(axis=0)

    core_indices = np.flatnonzero(
        membership_fraction
        >= args.persistent_fraction
    )

    if len(core_indices) < 10:
        raise RuntimeError(
            f"Only {len(core_indices)} persistent core particles."
        )

    core_positions = positions[:, core_indices, :]

    pair_i, pair_j = np.triu_indices(
        len(core_indices),
        k=1,
    )

    contact_history = np.asarray(
        [
            contacts_for_frame(
                frame_positions,
                box_lengths,
                pair_i,
                pair_j,
                args.contact_cutoff_nm,
            )
            for frame_positions in core_positions
        ],
        dtype=bool,
    )

    initial_contact_mask = contact_history[0]
    initial_contact_count = int(
        initial_contact_mask.sum()
    )

    if initial_contact_count == 0:
        lag_time_ns = (
            times_ps - times_ps[0]
        ) / 1000.0

        frame_output = pd.DataFrame(
            {
                "frame_index": np.arange(
                    len(contact_history)
                ),
                "lag_time_ns": lag_time_ns,
                "initial_contact_retention": np.zeros(
                    len(contact_history),
                    dtype=float,
                ),
                "continuous_contact_survival": np.zeros(
                    len(contact_history),
                    dtype=float,
                ),
                "total_contact_count": contact_history.sum(
                    axis=1
                ),
            }
        )

        summary = {
            "input_dir": str(input_dir),
            "analysis_start_frame": int(
                analysis_start
            ),
            "analysis_frames": int(
                len(contact_history)
            ),
            "analysis_duration_ns": float(
                lag_time_ns[-1]
            ),
            "persistent_core_particles": int(
                len(core_indices)
            ),
            "persistent_core_fraction": float(
                len(core_indices) / particle_count
            ),
            "contact_cutoff_nm": float(
                args.contact_cutoff_nm
            ),
            "initial_contact_count": 0,
            "final_initial_contact_retention": 0.0,
            "mean_late_initial_contact_retention": 0.0,
            "continuous_contact_survival_fraction": 0.0,
            "initial_contacts_ever_broken_fraction": 0.0,
            "initial_contacts_reformed_at_final_fraction": 0.0,
            "mean_first_break_time_ns": None,
            "median_first_break_time_ns": None,
            "zero_initial_contacts": True,
            "interpretation": (
                "No persistent-core particle pairs were within "
                "the contact cutoff at the first analyzed frame. "
                "Contact survival is therefore undefined and is "
                "stored as zero. This state must not be classified "
                "as an arrested contact aggregate."
            ),
        }

        frame_file = (
            input_dir
            / "slab_contact_continuity_frames.csv"
        )

        summary_file = (
            input_dir
            / "slab_contact_continuity_summary.json"
        )

        frame_output.to_csv(
            frame_file,
            index=False,
        )

        summary_file.write_text(
            json.dumps(summary, indent=2)
        )

        print("=" * 76)
        print("Strict slab-contact continuity audit")
        print("=" * 76)
        print(f"Input                    : {input_dir}")
        print(
            f"Persistent core          : "
            f"{len(core_indices)}/{particle_count}"
        )
        print("Initial contacts         : 0")
        print(
            "Result                   : "
            "valid zero-contact state"
        )
        print()
        print("Generated files:")
        print(f"  {frame_file}")
        print(f"  {summary_file}")
        return

    initial_contact_history = contact_history[
        :, initial_contact_mask
    ]

    intermittent_retention = (
        initial_contact_history.mean(axis=1)
    )

    continuously_alive = np.logical_and.accumulate(
        initial_contact_history,
        axis=0,
    )

    continuous_survival = (
        continuously_alive.mean(axis=1)
    )

    never_broken_mask = continuously_alive[-1]
    ever_broken_mask = ~never_broken_mask

    reformed_at_final_mask = (
        ever_broken_mask
        & initial_contact_history[-1]
    )

    lag_time_ns = (
        times_ps - times_ps[0]
    ) / 1000.0

    first_break_times = []

    for pair_index in range(initial_contact_count):
        broken_frames = np.flatnonzero(
            ~initial_contact_history[:, pair_index]
        )

        if len(broken_frames) > 0:
            first_break_times.append(
                lag_time_ns[broken_frames[0]]
            )

    if first_break_times:
        mean_first_break_ns = float(
            np.mean(first_break_times)
        )
        median_first_break_ns = float(
            np.median(first_break_times)
        )
    else:
        mean_first_break_ns = None
        median_first_break_ns = None

    late_window = max(
        1,
        len(contact_history) // 5,
    )

    frame_output = pd.DataFrame(
        {
            "frame_index": np.arange(
                len(contact_history)
            ),
            "lag_time_ns": lag_time_ns,
            "initial_contact_retention": (
                intermittent_retention
            ),
            "continuous_contact_survival": (
                continuous_survival
            ),
            "total_contact_count": (
                contact_history.sum(axis=1)
            ),
        }
    )

    summary = {
        "input_dir": str(input_dir),
        "analysis_start_frame": int(analysis_start),
        "analysis_frames": int(len(contact_history)),
        "analysis_duration_ns": float(
            lag_time_ns[-1]
        ),
        "persistent_core_particles": int(
            len(core_indices)
        ),
        "persistent_core_fraction": float(
            len(core_indices) / particle_count
        ),
        "contact_cutoff_nm": float(
            args.contact_cutoff_nm
        ),
        "initial_contact_count": int(
            initial_contact_count
        ),
        "final_initial_contact_retention": float(
            intermittent_retention[-1]
        ),
        "mean_late_initial_contact_retention": float(
            np.mean(
                intermittent_retention[-late_window:]
            )
        ),
        "continuous_contact_survival_fraction": float(
            continuous_survival[-1]
        ),
        "initial_contacts_ever_broken_fraction": float(
            ever_broken_mask.mean()
        ),
        "initial_contacts_reformed_at_final_fraction": float(
            reformed_at_final_mask.mean()
        ),
        "mean_first_break_time_ns": (
            mean_first_break_ns
        ),
        "median_first_break_time_ns": (
            median_first_break_ns
        ),
        "interpretation": (
            "Continuous survival counts only initial "
            "contacts that never broke during the full "
            "analysis interval. Intermittent retention "
            "allows breaking and reformation."
        ),
    }

    frame_file = (
        input_dir
        / "slab_contact_continuity_frames.csv"
    )
    summary_file = (
        input_dir
        / "slab_contact_continuity_summary.json"
    )

    frame_output.to_csv(
        frame_file,
        index=False,
    )

    summary_file.write_text(
        json.dumps(summary, indent=2)
    )

    print("=" * 76)
    print("Strict slab-contact continuity audit")
    print("=" * 76)
    print(f"Input                    : {input_dir}")
    print(
        f"Persistent core          : "
        f"{len(core_indices)}/{particle_count}"
    )
    print(
        f"Initial contacts         : "
        f"{initial_contact_count}"
    )
    print(
        f"Final retention          : "
        f"{intermittent_retention[-1]:.6f}"
    )
    print(
        f"Never-broken fraction    : "
        f"{continuous_survival[-1]:.6f}"
    )
    print(
        f"Ever-broken fraction     : "
        f"{ever_broken_mask.mean():.6f}"
    )
    print(
        f"Reformed at final        : "
        f"{reformed_at_final_mask.mean():.6f}"
    )
    print()
    print("Generated files:")
    print(f"  {frame_file}")
    print(f"  {summary_file}")


if __name__ == "__main__":
    main()
