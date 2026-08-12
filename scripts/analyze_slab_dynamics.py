#!/usr/bin/env python3
"""Analyze internal dynamics of an orthorhombic direct-coexistence slab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze MSD, contact survival, neighbor retention, and "
            "contact turnover inside a direct-coexistence slab."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--analysis-fraction",
        type=float,
        default=0.5,
        help="Fraction of the final trajectory used for analysis.",
    )
    parser.add_argument(
        "--persistent-fraction",
        type=float,
        default=0.8,
        help=(
            "Minimum fraction of analyzed frames for a particle to "
            "belong to the persistent slab core."
        ),
    )
    parser.add_argument(
        "--contact-cutoff-nm",
        type=float,
        default=5.35,
        help="Center-to-center contact cutoff in nm.",
    )
    return parser.parse_args()


def minimum_image(
    displacement_nm: np.ndarray,
    box_lengths_nm: np.ndarray,
) -> np.ndarray:
    return (
        displacement_nm
        - box_lengths_nm
        * np.rint(displacement_nm / box_lengths_nm)
    )


def unwrap_orthorhombic(
    positions_nm: np.ndarray,
    box_lengths_nm: np.ndarray,
) -> np.ndarray:
    unwrapped = np.empty_like(positions_nm, dtype=float)
    unwrapped[0] = positions_nm[0]

    for frame in range(1, len(positions_nm)):
        displacement = (
            positions_nm[frame]
            - positions_nm[frame - 1]
        )

        displacement = minimum_image(
            displacement,
            box_lengths_nm,
        )

        unwrapped[frame] = (
            unwrapped[frame - 1]
            + displacement
        )

    return unwrapped


def circular_distance_z(
    z_nm: np.ndarray,
    center_z_nm: float,
    box_z_nm: float,
) -> np.ndarray:
    return (
        (z_nm - center_z_nm + box_z_nm / 2.0)
        % box_z_nm
        - box_z_nm / 2.0
    )


def contact_vector(
    positions_nm: np.ndarray,
    box_lengths_nm: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    cutoff_nm: float,
) -> np.ndarray:
    displacement = (
        positions_nm[pair_i]
        - positions_nm[pair_j]
    )

    displacement = minimum_image(
        displacement,
        box_lengths_nm,
    )

    distance_squared = np.sum(
        displacement * displacement,
        axis=1,
    )

    return distance_squared <= cutoff_nm**2


def neighbor_jaccard(
    initial_contacts: np.ndarray,
    final_contacts: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    particle_count: int,
) -> float:
    initial_neighbors = [
        set() for _ in range(particle_count)
    ]
    final_neighbors = [
        set() for _ in range(particle_count)
    ]

    for pair_index, is_contact in enumerate(initial_contacts):
        if is_contact:
            i = int(pair_i[pair_index])
            j = int(pair_j[pair_index])
            initial_neighbors[i].add(j)
            initial_neighbors[j].add(i)

    for pair_index, is_contact in enumerate(final_contacts):
        if is_contact:
            i = int(pair_i[pair_index])
            j = int(pair_j[pair_index])
            final_neighbors[i].add(j)
            final_neighbors[j].add(i)

    values = []

    for initial_set, final_set in zip(
        initial_neighbors,
        final_neighbors,
    ):
        union = initial_set | final_set

        if not union:
            continue

        intersection = initial_set & final_set
        values.append(len(intersection) / len(union))

    if not values:
        return 0.0

    return float(np.mean(values))


def main() -> None:
    args = parse_args()

    if not 0.0 < args.analysis_fraction <= 1.0:
        raise ValueError(
            "--analysis-fraction must be in (0, 1]."
        )

    if not 0.0 < args.persistent_fraction <= 1.0:
        raise ValueError(
            "--persistent-fraction must be in (0, 1]."
        )

    input_dir = args.input_dir.resolve()

    trajectory_file = (
        input_dir / "trajectory_positions.npz"
    )
    metrics_file = (
        input_dir
        / "direct_coexistence_frame_metrics.csv"
    )
    summary_file = (
        input_dir
        / "direct_coexistence_summary.json"
    )

    for filename in (
        trajectory_file,
        metrics_file,
        summary_file,
    ):
        if not filename.exists():
            raise FileNotFoundError(filename)

    trajectory = np.load(trajectory_file)
    positions_nm = np.asarray(
        trajectory["positions_nm"],
        dtype=float,
    )
    steps = np.asarray(
        trajectory["steps"],
        dtype=np.int64,
    )
    box_lengths_nm = np.asarray(
        trajectory["box_lengths_nm"],
        dtype=float,
    )

    metrics = pd.read_csv(metrics_file)

    with summary_file.open() as handle:
        direct_summary = json.load(handle)

    if positions_nm.ndim != 3:
        raise ValueError(
            "positions_nm must have shape "
            "(frames, particles, 3)."
        )

    if box_lengths_nm.shape != (3,):
        raise ValueError(
            "box_lengths_nm must have shape (3,)."
        )

    if len(positions_nm) != len(metrics):
        raise ValueError(
            "Trajectory and frame-metric lengths differ: "
            f"{len(positions_nm)} versus {len(metrics)}."
        )

    frame_count = len(positions_nm)
    particle_count = positions_nm.shape[1]

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

    late_positions = positions_nm[analysis_start:]
    late_steps = steps[analysis_start:]

    late_times_ps = metrics[
        "time_ps"
    ].to_numpy(dtype=float)[analysis_start:]

    late_centers_z = metrics[
        "circular_center_z_nm"
    ].to_numpy(dtype=float)[analysis_start:]

    slab_thickness_nm = float(
        direct_summary[
            "initial_slab_thickness_nm"
        ]
    )

    membership = np.zeros(
        (
            len(late_positions),
            particle_count,
        ),
        dtype=bool,
    )

    for frame_index in range(len(late_positions)):
        dz = circular_distance_z(
            late_positions[frame_index, :, 2],
            late_centers_z[frame_index],
            box_lengths_nm[2],
        )

        membership[frame_index] = (
            np.abs(dz)
            <= slab_thickness_nm / 2.0
        )

    membership_fraction = membership.mean(axis=0)

    core_indices = np.flatnonzero(
        membership_fraction
        >= args.persistent_fraction
    )

    if len(core_indices) < 10:
        raise RuntimeError(
            "Too few persistent slab-core particles: "
            f"{len(core_indices)}"
        )

    unwrapped = unwrap_orthorhombic(
        late_positions,
        box_lengths_nm,
    )

    core_displacement = (
        unwrapped[:, core_indices, :]
        - unwrapped[0, core_indices, :][None, :, :]
    )

    collective_displacement = (
        core_displacement.mean(axis=1)
    )

    relative_displacement = (
        core_displacement
        - collective_displacement[:, None, :]
    )

    squared_displacement = (
        relative_displacement
        * relative_displacement
    )

    msd_total = squared_displacement.sum(
        axis=2
    ).mean(axis=1)

    msd_xy = squared_displacement[
        :, :, :2
    ].sum(axis=2).mean(axis=1)

    msd_z = squared_displacement[
        :, :, 2
    ].mean(axis=1)

    lag_time_ns = (
        late_times_ps - late_times_ps[0]
    ) / 1000.0

    fit_start = len(lag_time_ns) // 2

    msd_slope_nm2_per_ns = float(
        np.polyfit(
            lag_time_ns[fit_start:],
            msd_total[fit_start:],
            1,
        )[0]
    )

    apparent_diffusion_nm2_per_ns = max(
        0.0,
        msd_slope_nm2_per_ns / 6.0,
    )

    core_particle_count = len(core_indices)

    pair_i, pair_j = np.triu_indices(
        core_particle_count,
        k=1,
    )

    contact_history = []
    contact_counts = []
    contact_turnover = [0.0]

    for frame_index, frame_positions in enumerate(
        late_positions[:, core_indices, :]
    ):
        contacts = contact_vector(
            positions_nm=frame_positions,
            box_lengths_nm=box_lengths_nm,
            pair_i=pair_i,
            pair_j=pair_j,
            cutoff_nm=args.contact_cutoff_nm,
        )

        contact_history.append(contacts)
        contact_counts.append(int(contacts.sum()))

        if frame_index > 0:
            previous = contact_history[frame_index - 1]

            union_count = int(
                np.count_nonzero(previous | contacts)
            )
            changed_count = int(
                np.count_nonzero(previous ^ contacts)
            )

            contact_turnover.append(
                changed_count / max(union_count, 1)
            )

    contact_history_array = np.asarray(
        contact_history,
        dtype=bool,
    )

    initial_contacts = contact_history_array[0]
    initial_contact_count = int(
        initial_contacts.sum()
    )

    if initial_contact_count > 0:
        contact_survival = (
            contact_history_array[:, initial_contacts]
            .mean(axis=1)
        )
    else:
        contact_survival = np.zeros(
            len(contact_history_array),
            dtype=float,
        )

    final_neighbor_jaccard = neighbor_jaccard(
        initial_contacts=initial_contacts,
        final_contacts=contact_history_array[-1],
        pair_i=pair_i,
        pair_j=pair_j,
        particle_count=core_particle_count,
    )

    output_frames = pd.DataFrame(
        {
            "frame_index": np.arange(
                len(late_positions)
            ),
            "step": late_steps,
            "lag_time_ns": lag_time_ns,
            "core_relative_msd_nm2": msd_total,
            "core_relative_msd_xy_nm2": msd_xy,
            "core_relative_msd_z_nm2": msd_z,
            "contact_count": contact_counts,
            "initial_contact_survival": (
                contact_survival
            ),
            "contact_turnover_fraction": (
                contact_turnover
            ),
            "slab_membership_count": (
                membership.sum(axis=1)
            ),
        }
    )

    late_window = max(
        1,
        len(output_frames) // 5,
    )

    summary = {
        "input_dir": str(input_dir),
        "frames_total": int(frame_count),
        "analysis_start_frame": int(analysis_start),
        "analysis_frames": int(len(late_positions)),
        "analysis_duration_ns": float(
            lag_time_ns[-1]
        ),
        "particles_total": int(particle_count),
        "box_lengths_nm": box_lengths_nm.tolist(),
        "slab_thickness_nm": slab_thickness_nm,
        "persistent_fraction_threshold": float(
            args.persistent_fraction
        ),
        "persistent_core_particles": int(
            core_particle_count
        ),
        "persistent_core_fraction": float(
            core_particle_count / particle_count
        ),
        "contact_cutoff_nm": float(
            args.contact_cutoff_nm
        ),
        "initial_contact_count": int(
            initial_contact_count
        ),
        "mean_late_contact_count": float(
            np.mean(contact_counts[-late_window:])
        ),
        "final_core_relative_msd_nm2": float(
            msd_total[-1]
        ),
        "mean_late_core_relative_msd_nm2": float(
            np.mean(msd_total[-late_window:])
        ),
        "msd_late_slope_nm2_per_ns": float(
            msd_slope_nm2_per_ns
        ),
        "apparent_internal_diffusion_nm2_per_ns": float(
            apparent_diffusion_nm2_per_ns
        ),
        "final_initial_contact_survival": float(
            contact_survival[-1]
        ),
        "mean_late_initial_contact_survival": float(
            np.mean(
                contact_survival[-late_window:]
            )
        ),
        "mean_contact_turnover_fraction": float(
            np.mean(contact_turnover[1:])
        ),
        "mean_late_contact_turnover_fraction": float(
            np.mean(
                contact_turnover[-late_window:]
            )
        ),
        "final_neighbor_jaccard": float(
            final_neighbor_jaccard
        ),
        "diagnostic": (
            "Internal slab dynamics only. Compare mobile "
            "soluble-like controls with the arrested pI "
            "control before assigning liquid or aggregate."
        ),
    }

    frame_output = (
        input_dir / "slab_dynamics_frames.csv"
    )
    summary_output = (
        input_dir / "slab_dynamics_summary.json"
    )

    output_frames.to_csv(
        frame_output,
        index=False,
    )

    with summary_output.open("w") as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    print("=" * 76)
    print("Orthorhombic slab internal-dynamics analysis")
    print("=" * 76)
    print(f"Input directory        : {input_dir}")
    print(f"Analysis frames        : {len(late_positions)}")
    print(
        f"Analysis duration      : "
        f"{lag_time_ns[-1]:.6f} ns"
    )
    print(
        f"Persistent core        : "
        f"{core_particle_count}/{particle_count}"
    )
    print(
        f"Contact cutoff         : "
        f"{args.contact_cutoff_nm:.6f} nm"
    )
    print(
        f"Final relative MSD     : "
        f"{msd_total[-1]:.6f} nm^2"
    )
    print(
        f"MSD late slope         : "
        f"{msd_slope_nm2_per_ns:.6f} nm^2/ns"
    )
    print(
        f"Apparent internal D    : "
        f"{apparent_diffusion_nm2_per_ns:.6f} "
        f"nm^2/ns"
    )
    print(
        f"Final contact survival : "
        f"{contact_survival[-1]:.6f}"
    )
    print(
        f"Final neighbor Jaccard : "
        f"{final_neighbor_jaccard:.6f}"
    )
    print(
        f"Late contact turnover  : "
        f"{summary['mean_late_contact_turnover_fraction']:.6f}"
    )
    print()
    print("Generated files:")
    print(f"  {frame_output}")
    print(f"  {summary_output}")


if __name__ == "__main__":
    main()
