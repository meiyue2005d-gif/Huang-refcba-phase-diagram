#!/usr/bin/env python3
"""Analyze axial density persistence and exchange proxies in slab MD."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np


AVOGADRO = 6.02214076e23


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze direct-coexistence trajectories without declaring "
            "LLPS from a short or homogeneous simulation."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--analysis-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--smoothing-bins",
        type=int,
        default=3,
    )
    return parser


def concentration_mg_ml_from_count(
    count: np.ndarray,
    molecular_weight_kDa: float,
    volume_nm3: float,
) -> np.ndarray:
    if molecular_weight_kDa <= 0.0:
        raise ValueError("molecular_weight_kDa must be positive.")
    if volume_nm3 <= 0.0:
        raise ValueError("volume_nm3 must be positive.")

    particle_mass_mg = (
        molecular_weight_kDa
        * 1000.0
        / AVOGADRO
        * 1000.0
    )
    volume_ml = volume_nm3 * 1.0e-21

    return np.asarray(
        count,
        dtype=np.float64,
    ) * particle_mass_mg / volume_ml


def periodic_smooth(
    values: np.ndarray,
    width: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)

    if width <= 1:
        return array.copy()
    if width % 2 == 0:
        raise ValueError("smoothing-bins must be odd.")

    half_width = width // 2
    smoothed = np.zeros_like(array)

    for offset in range(-half_width, half_width + 1):
        smoothed += np.roll(array, offset)

    return smoothed / float(width)


def circular_center_nm(
    coordinates_nm: np.ndarray,
    box_length_nm: float,
) -> float:
    angles = (
        2.0
        * np.pi
        * np.asarray(coordinates_nm, dtype=np.float64)
        / box_length_nm
    )

    vector = np.mean(np.exp(1j * angles))

    if abs(vector) < 1.0e-12:
        return box_length_nm / 2.0

    angle = math.atan2(vector.imag, vector.real)

    if angle < 0.0:
        angle += 2.0 * np.pi

    return float(
        box_length_nm
        * angle
        / (2.0 * np.pi)
    )


def recenter_z_nm(
    coordinates_nm: np.ndarray,
    box_length_nm: float,
) -> tuple[np.ndarray, float]:
    center = circular_center_nm(
        coordinates_nm,
        box_length_nm,
    )

    shifted = (
        np.asarray(coordinates_nm, dtype=np.float64)
        - center
        + box_length_nm / 2.0
    ) % box_length_nm

    return shifted, center


def summarize_profile(
    density_mg_ml: np.ndarray,
) -> tuple[float, float, float, float]:
    profile = np.asarray(density_mg_ml, dtype=np.float64)

    if profile.ndim != 1 or profile.size < 6:
        raise ValueError(
            "Density profile must contain at least six bins."
        )

    sorted_profile = np.sort(profile)

    dense_bin_count = max(3, profile.size // 10)
    dilute_bin_count = max(3, profile.size // 3)

    dense = float(
        np.mean(sorted_profile[-dense_bin_count:])
    )
    dilute = float(
        np.mean(sorted_profile[:dilute_bin_count])
    )

    denominator = dense + dilute
    contrast = (
        float((dense - dilute) / denominator)
        if denominator > 0.0
        else 0.0
    )

    mean_density = float(np.mean(profile))
    axial_cv = (
        float(np.std(profile) / mean_density)
        if mean_density > 0.0
        else 0.0
    )

    return dense, dilute, contrast, axial_cv


def classify_profile_state(
    mean_contrast: float,
    mean_axial_cv: float,
) -> str:
    if (
        mean_contrast >= 0.50
        and mean_axial_cv >= 0.75
    ):
        return "persistent_axial_inhomogeneity"

    if (
        mean_contrast <= 0.20
        and mean_axial_cv <= 0.35
    ):
        return "weak_or_dissolved_profile"

    return "intermediate_profile"


def main() -> None:
    args = build_argument_parser().parse_args()

    input_dir = args.input_dir.resolve()
    metadata_file = input_dir / "metadata.json"
    trajectory_file = input_dir / "trajectory_positions.npz"

    if args.bins < 12:
        raise ValueError("bins must be at least 12.")
    if not 0.0 < args.analysis_fraction <= 1.0:
        raise ValueError(
            "analysis-fraction must lie in (0, 1]."
        )
    if (
        args.smoothing_bins <= 0
        or args.smoothing_bins % 2 == 0
    ):
        raise ValueError(
            "smoothing-bins must be a positive odd integer."
        )

    metadata: dict[str, Any] = json.loads(
        metadata_file.read_text(encoding="utf-8")
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
    box_lengths_nm = np.asarray(
        trajectory["box_lengths_nm"],
        dtype=np.float64,
    )

    if (
        positions_nm.ndim != 3
        or positions_nm.shape[2] != 3
    ):
        raise ValueError(
            "positions_nm must have shape "
            "(frames, particles, 3)."
        )
    if positions_nm.shape[0] != steps.size:
        raise ValueError(
            "Number of frames and saved steps differ."
        )
    if box_lengths_nm.shape != (3,):
        raise ValueError(
            "box_lengths_nm must have shape (3,)."
        )
    if not np.isfinite(positions_nm).all():
        raise ValueError(
            "Trajectory contains nonfinite coordinates."
        )

    n_frames, n_particles, _ = positions_nm.shape

    if n_frames < 2:
        raise ValueError(
            "At least two saved frames are required."
        )

    molecular_weight_kDa = float(
        metadata["molecular_weight_kDa"]
    )
    timestep_fs = float(
        metadata["timestep_fs"]
    )
    initial_slab_thickness_nm = float(
        metadata["initial_slab_thickness_nm"]
    )

    lx_nm, ly_nm, lz_nm = (
        float(value)
        for value in box_lengths_nm
    )

    bin_edges_nm = np.linspace(
        0.0,
        lz_nm,
        args.bins + 1,
    )
    bin_centers_nm = (
        bin_edges_nm[:-1]
        + bin_edges_nm[1:]
    ) / 2.0

    bin_volume_nm3 = (
        lx_nm
        * ly_nm
        * lz_nm
        / args.bins
    )

    lower_slab_nm = (
        lz_nm - initial_slab_thickness_nm
    ) / 2.0
    upper_slab_nm = (
        lz_nm + initial_slab_thickness_nm
    ) / 2.0

    density_profiles = np.empty(
        (n_frames, args.bins),
        dtype=np.float64,
    )
    recentered_z = np.empty(
        (n_frames, n_particles),
        dtype=np.float64,
    )
    memberships = np.empty(
        (n_frames, n_particles),
        dtype=np.bool_,
    )

    records: list[dict[str, float | int]] = []

    previous_membership: np.ndarray | None = None

    for frame_index in range(n_frames):
        z_nm, center_nm = recenter_z_nm(
            positions_nm[frame_index, :, 2],
            lz_nm,
        )

        counts, _ = np.histogram(
            z_nm,
            bins=bin_edges_nm,
        )

        raw_density = concentration_mg_ml_from_count(
            counts,
            molecular_weight_kDa,
            bin_volume_nm3,
        )
        density = periodic_smooth(
            raw_density,
            args.smoothing_bins,
        )

        dense, dilute, contrast, axial_cv = (
            summarize_profile(density)
        )

        membership = (
            (z_nm >= lower_slab_nm)
            & (z_nm <= upper_slab_nm)
        )

        if previous_membership is None:
            changed_fraction = 0.0
            entry_fraction = 0.0
            exit_fraction = 0.0
        else:
            changed_fraction = float(
                np.mean(
                    membership != previous_membership
                )
            )
            entry_fraction = float(
                np.mean(
                    membership
                    & ~previous_membership
                )
            )
            exit_fraction = float(
                np.mean(
                    ~membership
                    & previous_membership
                )
            )

        density_profiles[frame_index] = density
        recentered_z[frame_index] = z_nm
        memberships[frame_index] = membership

        records.append(
            {
                "frame_index": frame_index,
                "step": int(steps[frame_index]),
                "time_ps": (
                    float(steps[frame_index])
                    * timestep_fs
                    / 1000.0
                ),
                "circular_center_z_nm": center_nm,
                "dense_concentration_mg_ml": dense,
                "dilute_concentration_mg_ml": dilute,
                "density_contrast": contrast,
                "axial_cv": axial_cv,
                "central_slab_occupancy_fraction": float(
                    np.mean(membership)
                ),
                "membership_change_fraction": changed_fraction,
                "entry_fraction": entry_fraction,
                "exit_fraction": exit_fraction,
            }
        )

        previous_membership = membership

    analysis_start = max(
        0,
        n_frames
        - max(
            2,
            int(math.ceil(
                args.analysis_fraction
                * n_frames
            )),
        ),
    )

    late_density = density_profiles[analysis_start:]
    late_records = records[analysis_start:]

    mean_profile = np.mean(
        late_density,
        axis=0,
    )
    mean_dense, mean_dilute, mean_contrast, mean_cv = (
        summarize_profile(mean_profile)
    )

    mean_occupancy = float(
        np.mean([
            row["central_slab_occupancy_fraction"]
            for row in late_records
        ])
    )

    exchange_rows = late_records[1:]
    mean_membership_change = (
        float(np.mean([
            row["membership_change_fraction"]
            for row in exchange_rows
        ]))
        if exchange_rows
        else 0.0
    )
    mean_entry = (
        float(np.mean([
            row["entry_fraction"]
            for row in exchange_rows
        ]))
        if exchange_rows
        else 0.0
    )
    mean_exit = (
        float(np.mean([
            row["exit_fraction"]
            for row in exchange_rows
        ]))
        if exchange_rows
        else 0.0
    )

    profile_state = classify_profile_state(
        mean_contrast,
        mean_cv,
    )

    exchange_proxy_detected = bool(
        mean_entry > 0.0
        and mean_exit > 0.0
        and mean_membership_change >= 0.002
    )

    summary = {
        "input_dir": str(input_dir),
        "frames": n_frames,
        "particles": n_particles,
        "box_lengths_nm": box_lengths_nm.tolist(),
        "bins": args.bins,
        "analysis_start_frame": analysis_start,
        "analysis_fraction": args.analysis_fraction,
        "initial_slab_thickness_nm": (
            initial_slab_thickness_nm
        ),
        "late_mean_dense_concentration_mg_ml": (
            mean_dense
        ),
        "late_mean_dilute_concentration_mg_ml": (
            mean_dilute
        ),
        "late_mean_density_contrast": mean_contrast,
        "late_mean_axial_cv": mean_cv,
        "late_mean_central_slab_occupancy_fraction": (
            mean_occupancy
        ),
        "late_mean_membership_change_fraction": (
            mean_membership_change
        ),
        "late_mean_entry_fraction": mean_entry,
        "late_mean_exit_fraction": mean_exit,
        "profile_state": profile_state,
        "exchange_proxy_detected": (
            exchange_proxy_detected
        ),
        "interpretation": (
            "Diagnostic only. A persistent axial profile and "
            "bidirectional membership turnover are required before "
            "a longer replicated run can support LLPS."
        ),
    }

    frame_csv = (
        input_dir
        / "direct_coexistence_frame_metrics.csv"
    )
    with frame_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(records[0]),
        )
        writer.writeheader()
        writer.writerows(records)

    np.savez_compressed(
        input_dir
        / "direct_coexistence_density_profiles.npz",
        density_mg_ml=density_profiles,
        mean_late_density_mg_ml=mean_profile,
        bin_edges_nm=bin_edges_nm,
        bin_centers_nm=bin_centers_nm,
        steps=steps,
        box_lengths_nm=box_lengths_nm,
        recentered_z_nm=recentered_z,
        central_slab_membership=memberships,
    )

    summary_file = (
        input_dir
        / "direct_coexistence_summary.json"
    )
    summary_file.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 76)
    print("Direct-coexistence axial analysis")
    print("=" * 76)
    print(f"Input directory        : {input_dir}")
    print(f"Frames                 : {n_frames}")
    print(f"Particles              : {n_particles}")
    print(f"Box lengths            : {box_lengths_nm}")
    print(f"Analysis starts frame  : {analysis_start}")
    print(f"Late dense estimate    : {mean_dense:.6f} mg/mL")
    print(f"Late dilute estimate   : {mean_dilute:.6f} mg/mL")
    print(f"Late density contrast  : {mean_contrast:.6f}")
    print(f"Late axial CV          : {mean_cv:.6f}")
    print(f"Late slab occupancy    : {mean_occupancy:.6f}")
    print(
        "Late membership change: "
        f"{mean_membership_change:.6f}"
    )
    print(f"Late entry fraction    : {mean_entry:.6f}")
    print(f"Late exit fraction     : {mean_exit:.6f}")
    print(f"Profile state          : {profile_state}")
    print(
        "Exchange proxy         : "
        f"{exchange_proxy_detected}"
    )
    print("\nGenerated files:")
    print(f"  {frame_csv}")
    print(
        "  "
        + str(
            input_dir
            / "direct_coexistence_density_profiles.npz"
        )
    )
    print(f"  {summary_file}")
    print(
        "\nDiagnostic analysis completed. "
        "No final LLPS label was assigned."
    )


if __name__ == "__main__":
    main()
