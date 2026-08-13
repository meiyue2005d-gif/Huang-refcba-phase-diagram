#!/usr/bin/env python3
"""Compact summary of refCBA MD controls and pilot results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


CONTROL_DIRECTORIES = [
    Path("results/control_soluble_pH3"),
    Path("results/control_soluble_pH9"),
    Path("results/diagnostic_pI_c20"),
]

PILOT_SUMMARY = Path(
    "results/pilot_scan_nacl0/pilot_summary.csv"
)

PILOT_COUNTS = Path(
    "results/pilot_scan_nacl0/pilot_phase_counts.csv"
)


def read_json(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def summarize_control(root: Path) -> list[dict]:
    records = []

    for dynamics_path in sorted(
        root.rglob("dynamics_summary.json")
    ):
        task_directory = (
            dynamics_path.parents[1]
        )

        cluster_path = (
            task_directory
            / "cluster_analysis"
            / "cluster_summary.json"
        )

        metadata_path = (
            task_directory
            / "metadata.json"
        )

        dynamics = read_json(
            dynamics_path
        )

        cluster = (
            read_json(cluster_path)
            if cluster_path.exists()
            else {}
        )

        metadata = (
            read_json(metadata_path)
            if metadata_path.exists()
            else {}
        )

        records.append(
            {
                "control_group": root.name,
                "task_directory": str(
                    task_directory
                ),
                "metadata_preview": json.dumps(
                    metadata,
                    ensure_ascii=False,
                )[:300],
                "mean_clustered_fraction": (
                    dynamics.get(
                        "mean_clustered_fraction"
                    )
                ),
                "mean_largest_cluster_fraction": (
                    dynamics.get(
                        "mean_largest_cluster_fraction"
                    )
                ),
                "maximum_largest_cluster_fraction": (
                    cluster.get(
                        "maximum_largest_cluster_fraction"
                    )
                ),
                "percolation_fraction": (
                    dynamics.get(
                        "percolation_fraction"
                    )
                ),
                "final_initial_bond_survival": (
                    dynamics.get(
                        "final_initial_bond_survival"
                    )
                ),
                "mean_consecutive_bond_retention": (
                    dynamics.get(
                        "mean_consecutive_bond_retention"
                    )
                ),
                "final_msd_nm2": (
                    dynamics.get(
                        "final_msd_nm2"
                    )
                ),
                "frozen_finite_clusters": (
                    dynamics.get(
                        "frozen_finite_clusters"
                    )
                ),
                "provisional_interpretation": (
                    dynamics.get(
                        "provisional_interpretation"
                    )
                ),
            }
        )

    return records


def main() -> None:
    print("=" * 112)
    print("MD control-state sanity inputs")
    print("=" * 112)

    all_control_records = []

    for directory in CONTROL_DIRECTORIES:
        print(
            f"\n{directory}: "
            f"{'FOUND' if directory.exists() else 'MISSING'}"
        )

        if directory.exists():
            records = summarize_control(
                directory
            )

            print(
                "dynamics summaries found = "
                f"{len(records)}"
            )

            all_control_records.extend(
                records
            )

    if all_control_records:
        control_table = pd.DataFrame.from_records(
            all_control_records
        )

        display_columns = [
            "control_group",
            "mean_clustered_fraction",
            "mean_largest_cluster_fraction",
            "maximum_largest_cluster_fraction",
            "percolation_fraction",
            "final_initial_bond_survival",
            "mean_consecutive_bond_retention",
            "final_msd_nm2",
            "frozen_finite_clusters",
            "provisional_interpretation",
        ]

        print("\nControl metrics:")
        print(
            control_table[
                display_columns
            ].to_string(index=False)
        )

        output_path = Path(
            "results/analysis/"
            "md_control_sanity_inputs.csv"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        control_table.to_csv(
            output_path,
            index=False,
        )

        print(
            "\nSaved control table:"
        )
        print(f"  {output_path}")
    else:
        print(
            "\nNo control dynamics summaries found."
        )

    print("\n" + "=" * 112)
    print("Pilot summary")
    print("=" * 112)

    if not PILOT_SUMMARY.exists():
        print(
            f"MISSING {PILOT_SUMMARY}"
        )
        return

    pilot = pd.read_csv(
        PILOT_SUMMARY
    )

    print(
        f"pilot rows = {len(pilot)}"
    )

    print("\npilot_summary columns:")
    for column in pilot.columns:
        print(f"  {column}")

    candidate_columns = [
        "task_id",
        "pH",
        "added_NaCl_mM",
        "added_nacl_mM",
        "concentration_mg_ml",
        "seed",
        "mean_clustered_fraction",
        "mean_largest_cluster_fraction",
        "maximum_largest_cluster_fraction",
        "percolation_fraction",
        "final_initial_bond_survival",
        "mean_consecutive_bond_retention",
        "final_msd_nm2",
        "frozen_finite_clusters",
        "provisional_interpretation",
        "classification",
        "phase",
        "final_phase",
    ]

    selected_columns = [
        column
        for column in candidate_columns
        if column in pilot.columns
    ]

    print("\nCompact pilot table:")

    if selected_columns:
        print(
            pilot[
                selected_columns
            ].to_string(
                index=False
            )
        )
    else:
        print(
            pilot.head(10).to_string(
                index=False
            )
        )

    print("\nPilot phase counts:")

    if PILOT_COUNTS.exists():
        counts = pd.read_csv(
            PILOT_COUNTS
        )

        print(
            counts.to_string(
                index=False
            )
        )
    else:
        label_column = next(
            (
                column
                for column in [
                    "final_phase",
                    "phase",
                    "classification",
                    "provisional_interpretation",
                ]
                if column in pilot.columns
            ),
            None,
        )

        if label_column is None:
            print(
                "No phase-label column detected."
            )
        else:
            print(
                pilot[
                    label_column
                ].value_counts(
                    dropna=False
                ).to_string()
            )

    print(
        "\nMD sanity-input summary completed."
    )


if __name__ == "__main__":
    main()
