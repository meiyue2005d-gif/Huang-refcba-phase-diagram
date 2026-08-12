#!/usr/bin/env python3
"""Scan the Huang reference state over 0.1--20 mg/mL."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from huang_md.refcba_thermodynamics import (
    calculate_refcba_free_energy_point,
    load_refcba_configuration,
)


OUTPUT_DIR = (
    ROOT
    / "results"
    / "refcba_reference_state_thermodynamics"
)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    configuration = load_refcba_configuration(
        ROOT
    )

    concentrations = np.array(
        [
            0.1,
            0.2,
            0.436,
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
            15.0,
            20.0,
        ],
        dtype=np.float64,
    )

    records = []

    print("=" * 106)
    print(
        "refCBA reference-state thermodynamic scan"
    )
    print("=" * 106)
    print("pH                  = 4.5")
    print("added NaCl          = 0 mM")
    print(
        "background ionic I = "
        f"{configuration.state_model.background_ionic_strength_mM:g} mM"
    )
    print(
        "molecular weight    = "
        f"{configuration.molecular_weight_kDa:.6f} kDa"
    )

    for concentration in concentrations:
        point = (
            calculate_refcba_free_energy_point(
                configuration=configuration,
                pH=4.5,
                added_nacl_mM=0.0,
                concentration_mg_ml=float(
                    concentration
                ),
            )
        )

        records.append(
            point.__dict__
        )

        print(
            f"c={concentration:7.3f} mg/mL  "
            f"rho*={point.reduced_density_rho_sigma3:10.7f}  "
            f"a1={point.beta_a1_per_particle:12.6g}  "
            f"a2={point.beta_a2_per_particle:12.6g}  "
            f"|a2/a1|={point.second_to_first_abs_ratio:10.6g}  "
            f"{point.perturbation_status}"
        )

    table = pd.DataFrame.from_records(
        records
    )

    output_file = (
        OUTPUT_DIR
        / "reference_state_free_energy.csv"
    )

    table.to_csv(
        output_file,
        index=False,
    )

    numeric = table.select_dtypes(
        include=[np.number]
    )

    checks = {
        "all numerical values finite": bool(
            np.isfinite(
                numeric.to_numpy()
            ).all()
        ),
        "density increases monotonically": bool(
            (
                np.diff(
                    table[
                        "reduced_density_rho_sigma3"
                    ]
                )
                > 0.0
            ).all()
        ),
        "all states below rho*=0.2": bool(
            (
                table[
                    "reduced_density_rho_sigma3"
                ]
                < 0.2
            ).all()
        ),
        "second-order term negative": bool(
            (
                table[
                    "beta_a2_per_particle"
                ]
                < 0.0
            ).all()
        ),
        "total perturbation term negative": bool(
            (
                table[
                    "beta_perturbation_per_particle"
                ]
                < 0.0
            ).all()
        ),
    }

    first_order_values = table[
        "beta_a1_per_particle"
    ]

    if (first_order_values > 0.0).all():
        first_order_interpretation = (
            "net_repulsive"
        )
    elif (first_order_values < 0.0).all():
        first_order_interpretation = (
            "net_attractive"
        )
    else:
        first_order_interpretation = (
            "sign_changes_with_concentration"
        )

    print(
        "\nFirst-order interpretation:"
    )
    print(
        "  beta a1 sign = "
        f"{first_order_interpretation}"
    )
    print(
        "  In this reference state, the positive "
        "first-order term means that the weighted "
        "repulsive contribution dominates the "
        "short-range attraction."
    )

    print("\nPerturbation-status counts:")
    print(
        table[
            "perturbation_status"
        ].value_counts().to_string()
    )

    print("\nNumerical checks:")

    failed = []

    for name, passed in checks.items():
        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"  {status:<4s} {name}"
        )

        if not passed:
            failed.append(name)

    print(
        "\nGenerated file:\n"
        f"  {output_file}"
    )

    if failed:
        raise RuntimeError(
            "Reference-state thermodynamic scan failed: "
            + ", ".join(failed)
        )

    print(
        "\nReference-state numerical scan completed."
    )

    if (
        table["perturbation_status"]
        == "uncontrolled"
    ).any():
        print(
            "WARNING: One or more states have "
            "|a2/a1| > 1. These states must not be "
            "treated as quantitative binodal predictions."
        )


if __name__ == "__main__":
    main()
