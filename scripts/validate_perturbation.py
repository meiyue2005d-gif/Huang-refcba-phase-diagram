#!/usr/bin/env python3
"""Validate perturbation free energies over representative states."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from huang_md.parameters import (
    HuangPotentialParameters,
)
from huang_md.perturbation import (
    calculate_perturbation_free_energy,
)
from huang_md.state_model import (
    RefCBAStateModel,
    parameters_for_state,
)


OUTPUT_DIR = (
    ROOT
    / "results"
    / "perturbation_validation"
)

AVOGADRO = 6.02214076e23
MOLECULAR_WEIGHT_G_MOL = 8018.862
HARD_SPHERE_DIAMETER_NM = 4.278

PH_VALUES = [
    3.0,
    4.5,
    4.7,
    4.8,
    4.8852,
    5.0,
    5.2,
    6.0,
    9.0,
]

CONCENTRATIONS_MG_ML = [
    0.1,
    0.436,
    0.5,
    2.0,
    5.0,
    10.0,
    20.0,
]


def concentration_to_density(
    concentration_mg_ml: float,
) -> float:
    return (
        concentration_mg_ml
        / MOLECULAR_WEIGHT_G_MOL
        * AVOGADRO
        / 1.0e24
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline = (
        HuangPotentialParameters.from_yaml(
            ROOT
            / "configs"
            / "huang_baseline.yaml"
        )
    )

    state_model = (
        RefCBAStateModel.from_yaml(
            ROOT
            / "configs"
            / "refcba_state_model.yaml"
        )
    )

    records: list[dict[str, object]] = []

    for pH in PH_VALUES:
        params = parameters_for_state(
            baseline=baseline,
            model=state_model,
            pH=pH,
            added_nacl_mM=0.0,
        )

        for concentration in CONCENTRATIONS_MG_ML:
            density = concentration_to_density(
                concentration
            )

            result = (
                calculate_perturbation_free_energy(
                    params=params,
                    number_density_nm3=density,
                    hard_sphere_diameter_nm=(
                        HARD_SPHERE_DIAMETER_NM
                    ),
                )
            )

            records.append(
                {
                    "pH": pH,
                    "concentration_mg_ml": concentration,
                    "number_density_nm3": density,
                    "K2_kBT": params.K2_kBT,
                    "Z2": params.Z2,
                    "reduced_density": (
                        result.reduced_density_rho_sigma3
                    ),
                    "packing_fraction": (
                        result.packing_fraction
                    ),
                    "reference_compressibility": (
                        result.reference_compressibility
                    ),
                    "first_integral": (
                        result.first_moment_integral
                    ),
                    "second_integral": (
                        result.second_moment_integral
                    ),
                    "beta_a1": (
                        result.beta_a1_per_particle
                    ),
                    "beta_a2": (
                        result.beta_a2_per_particle
                    ),
                    "beta_perturbation": (
                        result.beta_perturbation_per_particle
                    ),
                    "beta_reference": (
                        result.beta_reference_free_energy_per_particle
                    ),
                    "beta_total": (
                        result.beta_total_free_energy_per_particle
                    ),
                    "abs_a2_over_abs_a1": (
                        result.second_to_first_abs_ratio
                    ),
                    "second_order_dominates": (
                        abs(result.beta_a2_per_particle)
                        >
                        abs(result.beta_a1_per_particle)
                    ),
                    "first_integral_error": (
                        result.first_integral_error
                    ),
                    "second_integral_error": (
                        result.second_integral_error
                    ),
                }
            )

    table = pd.DataFrame.from_records(
        records
    )

    csv_file = (
        OUTPUT_DIR
        / "refcba_perturbation_validation.csv"
    )

    table.to_csv(
        csv_file,
        index=False,
    )

    figure, axis = plt.subplots(
        figsize=(8.4, 5.7)
    )

    for pH in [
        3.0,
        4.5,
        4.8852,
        6.0,
        9.0,
    ]:
        subset = table.loc[
            np.isclose(
                table["pH"],
                pH,
            )
        ]

        axis.plot(
            subset["concentration_mg_ml"],
            subset["beta_perturbation"],
            marker="o",
            label=f"pH {pH:g}",
        )

    axis.set_xscale("log")
    axis.set_xlabel("Concentration (mg/mL)")
    axis.set_ylabel(
        r"Perturbation free energy "
        r"$\beta(a_1+a_2)$"
    )
    axis.set_title(
        "Second-order perturbation contribution"
    )
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()

    figure_file = (
        OUTPUT_DIR
        / "perturbation_free_energy_vs_concentration.png"
    )

    figure.savefig(
        figure_file,
        dpi=240,
    )

    plt.close(figure)

    checks = {
        "all numerical values finite": bool(
            np.isfinite(
                table.select_dtypes(
                    include=[np.number]
                ).to_numpy()
            ).all()
        ),
        "second moments nonnegative": bool(
            (
                table["second_integral"]
                >= 0.0
            ).all()
        ),
        "second-order terms nonpositive": bool(
            (
                table["beta_a2"]
                <= 1.0e-12
            ).all()
        ),
        "packing fraction inside RDF range": bool(
            (
                table["packing_fraction"]
                <= 0.061572831
            ).all()
        ),
    }

    print("=" * 80)
    print("Second-order perturbation validation")
    print("=" * 80)

    selected = table.loc[
        table["concentration_mg_ml"].isin(
            [0.436, 5.0, 20.0]
        ),
        [
            "pH",
            "concentration_mg_ml",
            "beta_a1",
            "beta_a2",
            "beta_perturbation",
            "beta_total",
            "abs_a2_over_abs_a1",
            "second_order_dominates",
        ],
    ]

    print(
        selected.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.8g}"
            ),
        )
    )

    print("\nValidation checks:")

    failed = []

    for name, passed in checks.items():
        label = "PASS" if passed else "FAIL"
        print(f"  {label:<4s} {name}")

        if not passed:
            failed.append(name)

    dominated = int(
        table[
            "second_order_dominates"
        ].sum()
    )

    print(
        "\nStates with |a2| > |a1|: "
        f"{dominated}/{len(table)}"
    )

    print("\nGenerated files:")
    print(f"  {csv_file}")
    print(f"  {figure_file}")

    if failed:
        raise RuntimeError(
            "Perturbation validation failed: "
            + ", ".join(failed)
        )

    print(
        "\nAll perturbation numerical checks passed."
    )


if __name__ == "__main__":
    main()
