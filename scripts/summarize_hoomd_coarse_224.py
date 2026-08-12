#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOT = ROOT / "results" / "hoomd_coarse_224_0p5ns"
MANIFEST = SCAN_ROOT / "manifest.tsv"
OUTPUT = SCAN_ROOT / "hoomd_coarse_224_metrics.csv"


def slope_per_ns(frame: pd.DataFrame, column: str) -> float:
    if len(frame) < 2:
        return float("nan")

    x = frame["time_ps"].to_numpy(dtype=float)
    y = frame[column].to_numpy(dtype=float)

    if np.ptp(x) == 0:
        return float("nan")

    return float(np.polyfit(x, y, 1)[0] * 1000.0)


manifest = pd.read_csv(MANIFEST, sep="\t")
rows = []
missing = []

for _, state in manifest.iterrows():
    state_id = str(state["state_id"])
    directory = SCAN_ROOT / state_id

    summary_path = (
        directory
        / "cluster_analysis"
        / "cluster_summary.json"
    )
    frame_path = (
        directory
        / "cluster_analysis"
        / "cluster_statistics_by_frame.csv"
    )
    thermo_path = directory / "production_thermo.csv"

    required = [summary_path, frame_path, thermo_path]

    if any(not path.exists() for path in required):
        missing.append(state_id)
        continue

    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )
    frames = pd.read_csv(frame_path)
    thermo = pd.read_csv(thermo_path)

    n_frames = len(frames)
    late = frames.iloc[int(0.8 * n_frames):].copy()
    early = frames.iloc[:max(1, int(0.2 * n_frames))].copy()

    largest = frames["largest_cluster_size"]
    late_largest = late["largest_cluster_size"]

    late_clustered = late["clustered_fraction"]
    early_clustered = early["clustered_fraction"]

    first_largest = int(largest.iloc[0])
    final_largest = int(largest.iloc[-1])
    maximum_largest = int(largest.max())

    late_mean_largest = float(late_largest.mean())
    late_mean_clustered = float(late_clustered.mean())

    n_particles = 500

    rows.append(
        {
            "state_id": state_id,
            "pH": float(state["ph"]),
            "nacl_mM": float(state["nacl_mM"]),
            "concentration_mg_ml": float(
                state["concentration_mg_ml"]
            ),
            "seed": int(state["seed"]),
            "frames": n_frames,

            "mean_monomer_fraction": float(
                summary["mean_monomer_fraction"]
            ),
            "mean_clustered_fraction": float(
                summary["mean_clustered_fraction"]
            ),
            "mean_largest_cluster": float(
                summary["mean_largest_cluster_size"]
            ),
            "maximum_largest_cluster": maximum_largest,
            "final_largest_cluster": final_largest,

            "first_largest_cluster": first_largest,
            "late_mean_largest_cluster": late_mean_largest,
            "late_min_largest_cluster": int(
                late_largest.min()
            ),
            "late_max_largest_cluster": int(
                late_largest.max()
            ),
            "late_std_largest_cluster": float(
                late_largest.std()
            ),

            "late_mean_clustered_fraction": (
                late_mean_clustered
            ),
            "late_std_clustered_fraction": float(
                late_clustered.std()
            ),

            "largest_cluster_fraction": (
                final_largest / n_particles
            ),
            "late_largest_cluster_fraction": (
                late_mean_largest / n_particles
            ),

            "largest_growth_particles": (
                final_largest - first_largest
            ),
            "clustered_fraction_growth": float(
                late_clustered.mean()
                - early_clustered.mean()
            ),

            "late_largest_slope_particles_per_ns": (
                slope_per_ns(
                    late,
                    "largest_cluster_size",
                )
            ),
            "late_clustered_slope_per_ns": (
                slope_per_ns(
                    late,
                    "clustered_fraction",
                )
            ),

            "mean_temperature_Tstar": float(
                thermo[
                    "kinetic_temperature_kBT"
                ].mean()
            ),
            "temperature_std_Tstar": float(
                thermo[
                    "kinetic_temperature_kBT"
                ].std()
            ),
            "potential_energy_change_kBT": float(
                thermo[
                    "potential_energy_kBT"
                ].iloc[-1]
                - thermo[
                    "potential_energy_kBT"
                ].iloc[0]
            ),
        }
    )

result = pd.DataFrame(rows)
result = result.sort_values(
    ["pH", "nacl_mM", "concentration_mg_ml"]
).reset_index(drop=True)

result.to_csv(OUTPUT, index=False)

print("=" * 80)
print("HOOMD COARSE 224 METRICS")
print("=" * 80)
print("Manifest states :", len(manifest))
print("Summarized      :", len(result))
print("Missing         :", len(missing))
print("Output          :", OUTPUT)

if missing:
    print("\nMissing states:")
    for state_id in missing:
        print(" ", state_id)

columns = [
    "late_mean_clustered_fraction",
    "late_mean_largest_cluster",
    "final_largest_cluster",
    "maximum_largest_cluster",
    "late_largest_slope_particles_per_ns",
    "clustered_fraction_growth",
    "mean_temperature_Tstar",
]

print("\nMetric quantiles:")
print(
    result[columns]
    .quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    .to_string(
        float_format=lambda value: f"{value:.6f}"
    )
)

print("\nTop 20 by late clustered fraction:")
print(
    result.nlargest(
        20,
        "late_mean_clustered_fraction",
    )[
        [
            "pH",
            "nacl_mM",
            "concentration_mg_ml",
            "late_mean_clustered_fraction",
            "late_mean_largest_cluster",
            "final_largest_cluster",
            "late_largest_slope_particles_per_ns",
        ]
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

print("\nTop 20 by final largest cluster:")
print(
    result.nlargest(
        20,
        "final_largest_cluster",
    )[
        [
            "pH",
            "nacl_mM",
            "concentration_mg_ml",
            "late_mean_clustered_fraction",
            "final_largest_cluster",
            "maximum_largest_cluster",
            "late_largest_slope_particles_per_ns",
        ]
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

print("\nSUMMARIZE_HOOMD_COARSE_224: PASS")
