#!/usr/bin/env python3
"""Collect pilot scan results and assign provisional states."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SCAN_ROOT = (
    ROOT / "results" / "pilot_scan_nacl0"
)

MANIFEST = SCAN_ROOT / "pilot_manifest.csv"
SUMMARY_CSV = SCAN_ROOT / "pilot_summary.csv"


def classify_provisional(
    dynamics: dict[str, object],
) -> str:
    percolation_fraction = float(
        dynamics["percolation_fraction"]
    )

    frozen = bool(
        dynamics["frozen_finite_clusters"]
    )

    interpretation = str(
        dynamics["provisional_interpretation"]
    )

    if percolation_fraction >= 0.5 or frozen:
        return "aggregate"

    if interpretation == "mobile_or_coarsening_clustered_state":
        return "LLPS_candidate"

    return "soluble"


def main() -> None:
    manifest = pd.read_csv(MANIFEST)

    records: list[dict[str, object]] = []

    for _, row in manifest.iterrows():
        output_dir = Path(str(row["output_dir"]))

        dynamics_file = (
            output_dir
            / "dynamics_analysis"
            / "dynamics_summary.json"
        )

        cluster_file = (
            output_dir
            / "cluster_analysis"
            / "cluster_summary.json"
        )

        metadata_file = output_dir / "metadata.json"

        if not (
            dynamics_file.exists()
            and cluster_file.exists()
            and metadata_file.exists()
        ):
            continue

        with dynamics_file.open(
            encoding="utf-8"
        ) as handle:
            dynamics = json.load(handle)

        with cluster_file.open(
            encoding="utf-8"
        ) as handle:
            clusters = json.load(handle)

        with metadata_file.open(
            encoding="utf-8"
        ) as handle:
            metadata = json.load(handle)

        phase = classify_provisional(
            dynamics
        )

        records.append(
            {
                "task_id": int(row["task_id"]),
                "pH": float(row["pH"]),
                "nacl_mM": float(row["nacl_mM"]),
                "concentration_mg_ml": float(
                    row["concentration_mg_ml"]
                ),
                "seed": int(row["seed"]),
                "K2_kBT": float(metadata["K2_kBT"]),
                "Z2": float(metadata["Z2"]),
                "mean_clustered_fraction": float(
                    dynamics["mean_clustered_fraction"]
                ),
                "mean_largest_cluster_fraction": float(
                    dynamics[
                        "mean_largest_cluster_fraction"
                    ]
                ),
                "percolation_fraction": float(
                    dynamics["percolation_fraction"]
                ),
                "final_initial_bond_survival": float(
                    dynamics[
                        "final_initial_bond_survival"
                    ]
                ),
                "mean_consecutive_bond_retention": float(
                    dynamics[
                        "mean_consecutive_bond_retention"
                    ]
                ),
                "final_msd_nm2": float(
                    dynamics["final_msd_nm2"]
                ),
                "frozen_finite_clusters": bool(
                    dynamics[
                        "frozen_finite_clusters"
                    ]
                ),
                "maximum_largest_cluster_size": int(
                    clusters[
                        "maximum_largest_cluster_size"
                    ]
                ),
                "provisional_interpretation": str(
                    dynamics[
                        "provisional_interpretation"
                    ]
                ),
                "provisional_phase": phase,
                "output_dir": str(output_dir),
            }
        )

    summary = pd.DataFrame.from_records(records)

    if summary.empty:
        raise RuntimeError(
            "No completed scan results were found."
        )

    summary = summary.sort_values(
        ["pH", "concentration_mg_ml"]
    )

    summary.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    phase_code = {
        "soluble": 0,
        "LLPS_candidate": 1,
        "aggregate": 2,
    }

    summary["phase_code"] = summary[
        "provisional_phase"
    ].map(phase_code)

    figure, axis = plt.subplots(
        figsize=(8.5, 6.0)
    )

    scatter = axis.scatter(
        summary["pH"],
        summary["concentration_mg_ml"],
        c=summary["phase_code"],
        s=120,
    )

    axis.set_yscale("log")
    axis.set_xlabel("pH")
    axis.set_ylabel("Concentration (mg/mL)")
    axis.set_title(
        "NaCl = 0 mM pilot assembly map"
    )
    axis.grid(alpha=0.25)

    colorbar = figure.colorbar(
        scatter,
        ax=axis,
        ticks=[0, 1, 2],
    )

    colorbar.ax.set_yticklabels(
        [
            "Soluble",
            "LLPS candidate",
            "Aggregate",
        ]
    )

    figure.tight_layout()
    figure.savefig(
        SCAN_ROOT / "pilot_phase_map.png",
        dpi=240,
    )
    plt.close(figure)

    count_table = (
        summary[
            "provisional_phase"
        ]
        .value_counts()
        .rename_axis("phase")
        .reset_index(name="count")
    )

    count_table.to_csv(
        SCAN_ROOT / "pilot_phase_counts.csv",
        index=False,
    )

    print("=" * 72)
    print("Pilot scan summary")
    print("=" * 72)
    print(f"Completed states : {len(summary)}")
    print(f"Summary CSV      : {SUMMARY_CSV}")

    print("\nProvisional state counts:")
    print(count_table.to_string(index=False))

    print("\nResults by pH and concentration:")
    print(
        summary[
            [
                "pH",
                "concentration_mg_ml",
                "provisional_phase",
                "mean_clustered_fraction",
                "percolation_fraction",
                "final_initial_bond_survival",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
