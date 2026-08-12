#!/usr/bin/env python3
"""Validate and visualize the NIST LJ coexistence reference data."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_FILE = (
    ROOT
    / "data"
    / "lj_reference"
    / "nist_lj_cut5_coexistence.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "lj_nist_reference_validation"
)

CRITICAL_TEMPERATURE = 1.284
CRITICAL_DENSITY = 0.318
CRITICAL_PRESSURE = 0.118


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    table = pd.read_csv(INPUT_FILE)

    required_columns = {
        "temperature_reduced",
        "vapor_density_reduced",
        "vapor_density_uncertainty",
        "liquid_density_reduced",
        "liquid_density_uncertainty",
        "saturation_pressure_reduced",
        "saturation_pressure_uncertainty",
    }

    missing = required_columns - set(table.columns)

    if missing:
        raise RuntimeError(
            "Missing columns: "
            + ", ".join(sorted(missing))
        )

    numerical = table[
        sorted(required_columns)
    ].to_numpy(dtype=np.float64)

    checks = {
        "15 reference temperatures": (
            len(table) == 15
        ),
        "all entries finite": bool(
            np.isfinite(numerical).all()
        ),
        "temperatures strictly increasing": bool(
            (
                np.diff(
                    table["temperature_reduced"]
                )
                > 0.0
            ).all()
        ),
        "vapor density positive": bool(
            (
                table["vapor_density_reduced"]
                > 0.0
            ).all()
        ),
        "liquid density exceeds vapor": bool(
            (
                table["liquid_density_reduced"]
                >
                table["vapor_density_reduced"]
            ).all()
        ),
        "vapor density increases with T": bool(
            (
                np.diff(
                    table["vapor_density_reduced"]
                )
                > 0.0
            ).all()
        ),
        "liquid density decreases with T": bool(
            (
                np.diff(
                    table["liquid_density_reduced"]
                )
                < 0.0
            ).all()
        ),
        "pressure increases with T": bool(
            (
                np.diff(
                    table[
                        "saturation_pressure_reduced"
                    ]
                )
                > 0.0
            ).all()
        ),
    }

    table["diameter_density"] = (
        0.5
        * (
            table["liquid_density_reduced"]
            + table["vapor_density_reduced"]
        )
    )

    table["order_parameter_density"] = (
        table["liquid_density_reduced"]
        - table["vapor_density_reduced"]
    )

    augmented_file = (
        OUTPUT_DIR
        / "nist_lj_cut5_reference_augmented.csv"
    )

    table.to_csv(
        augmented_file,
        index=False,
    )

    figure, axis = plt.subplots(
        figsize=(7.8, 5.8)
    )

    axis.errorbar(
        table["vapor_density_reduced"],
        table["temperature_reduced"],
        xerr=table[
            "vapor_density_uncertainty"
        ],
        marker="o",
        linestyle="-",
        label="NIST vapor",
    )

    axis.errorbar(
        table["liquid_density_reduced"],
        table["temperature_reduced"],
        xerr=table[
            "liquid_density_uncertainty"
        ],
        marker="o",
        linestyle="-",
        label="NIST liquid",
    )

    axis.scatter(
        [CRITICAL_DENSITY],
        [CRITICAL_TEMPERATURE],
        marker="x",
        s=90,
        label="NIST critical estimate",
    )

    axis.set_xlabel(
        r"Reduced density $\rho^*$"
    )
    axis.set_ylabel(
        r"Reduced temperature $T^*$"
    )
    axis.set_title(
        r"NIST LJ coexistence, $r_c=5\sigma$"
    )
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()

    phase_file = (
        OUTPUT_DIR
        / "nist_lj_cut5_phase_diagram.png"
    )

    figure.savefig(
        phase_file,
        dpi=240,
    )

    plt.close(figure)

    figure, axis = plt.subplots(
        figsize=(7.8, 5.4)
    )

    axis.errorbar(
        table["temperature_reduced"],
        table["saturation_pressure_reduced"],
        yerr=table[
            "saturation_pressure_uncertainty"
        ],
        marker="o",
    )

    axis.scatter(
        [CRITICAL_TEMPERATURE],
        [CRITICAL_PRESSURE],
        marker="x",
        s=90,
    )

    axis.set_xlabel(
        r"Reduced temperature $T^*$"
    )
    axis.set_ylabel(
        r"Reduced saturation pressure $P^*$"
    )
    axis.set_title(
        r"NIST LJ saturation pressure, $r_c=5\sigma$"
    )
    axis.grid(alpha=0.25)

    figure.tight_layout()

    pressure_file = (
        OUTPUT_DIR
        / "nist_lj_cut5_saturation_pressure.png"
    )

    figure.savefig(
        pressure_file,
        dpi=240,
    )

    plt.close(figure)

    failed = []

    print("=" * 76)
    print("NIST Lennard-Jones reference validation")
    print("=" * 76)
    print(f"Input file       : {INPUT_FILE}")
    print(f"Temperature range: "
          f"{table['temperature_reduced'].min():.2f}"
          f"–{table['temperature_reduced'].max():.2f}")
    print(f"Reference points : {len(table)}")
    print(
        "Critical estimate: "
        f"T*={CRITICAL_TEMPERATURE:.3f}, "
        f"rho*={CRITICAL_DENSITY:.3f}, "
        f"P*={CRITICAL_PRESSURE:.3f}"
    )

    print("\nValidation checks:")

    for name, passed in checks.items():
        label = "PASS" if passed else "FAIL"
        print(f"  {label:<4s} {name}")

        if not passed:
            failed.append(name)

    print("\nSelected reference points:")
    print(
        table.loc[
            table["temperature_reduced"].isin(
                [0.60, 0.80, 1.00, 1.20, 1.25]
            ),
            [
                "temperature_reduced",
                "vapor_density_reduced",
                "liquid_density_reduced",
                "saturation_pressure_reduced",
            ],
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.8g}"
            ),
        )
    )

    print("\nGenerated files:")
    print(f"  {augmented_file}")
    print(f"  {phase_file}")
    print(f"  {pressure_file}")

    if failed:
        raise RuntimeError(
            "NIST LJ reference validation failed: "
            + ", ".join(failed)
        )

    print(
        "\nAll NIST LJ reference checks passed."
    )


if __name__ == "__main__":
    main()
