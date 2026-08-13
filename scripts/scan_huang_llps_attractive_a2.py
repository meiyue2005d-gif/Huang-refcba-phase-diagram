#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import scan_huang_llps_baseline as base

from huang_md.virial import total_potential_reduced
from huang_md.perturbation import (
    calculate_perturbation_free_energy,
    perturbation_moments_reduced,
)
from huang_md.lj_perturbation import (
    hybrid_hard_sphere_rdf_reduced,
)


OUTPUT_DIR = (
    ROOT
    / "results"
    / "huang_a1_llps_attractive_a2"
)

PH_VALUES = np.array(
    [
        4.5,
        5.0,
        5.3,
        5.4,
        5.5,
        5.6,
        5.7,
        5.8,
        6.0,
    ],
    dtype=float,
)

TARGET_CONCENTRATION = 0.436


def build_attractive_a2_free_energy_table(
    params,
    sigma_hs_nm: float,
    pH: float,
) -> pd.DataFrame:
    """Use full potential in a1 and negative potential only in a2."""
    density_grid = base.build_density_grid()

    nominal_diameter_nm = float(
        params.diameter_nm
    )

    diameter_ratio = (
        sigma_hs_nm
        / nominal_diameter_nm
    )

    potential_contact_hs = (
        1.0 / diameter_ratio
    )

    additional_breakpoints = []

    if potential_contact_hs > 1.0:
        additional_breakpoints.append(
            potential_contact_hs
        )

    def attractive_potential_hs(
        distance_hs_reduced,
    ):
        distance_potential_reduced = (
            np.asarray(
                distance_hs_reduced,
                dtype=float,
            )
            * diameter_ratio
        )

        full_potential = np.asarray(
            total_potential_reduced(
                distance_potential_reduced,
                params,
            ),
            dtype=float,
        )

        return np.minimum(
            full_potential,
            0.0,
        )

    rows = []

    print()
    print("=" * 82)
    print(
        "Building attractive-only a2 table: "
        f"pH={pH:g}"
    )
    print("=" * 82)
    print(f"K2={params.K2_kBT:.9f}")
    print(f"states={density_grid.size}")

    for index, rho_star in enumerate(
        density_grid,
        start=1,
    ):
        number_density_nm3 = (
            rho_star
            / sigma_hs_nm**3
        )

        full_result = (
            calculate_perturbation_free_energy(
                params=params,
                number_density_nm3=(
                    number_density_nm3
                ),
                hard_sphere_diameter_nm=(
                    sigma_hs_nm
                ),
                thermal_wavelength_nm=1.0,
                rdf_function=(
                    hybrid_hard_sphere_rdf_reduced
                ),
            )
        )

        (
            _,
            attractive_second_integral,
            _,
            attractive_second_error,
        ) = perturbation_moments_reduced(
            potential_function=(
                attractive_potential_hs
            ),
            packing_fraction_value=(
                full_result.packing_fraction
            ),
            rdf_function=(
                hybrid_hard_sphere_rdf_reduced
            ),
            additional_breakpoints=tuple(
                additional_breakpoints
            ),
        )

        beta_a2_attractive = (
            -0.25
            * full_result.reduced_density_rho_sigma3
            * full_result.reference_compressibility
            * attractive_second_integral
        )

        beta_total_attractive_a2 = (
            full_result
            .beta_reference_free_energy_per_particle
            + full_result.beta_a1_per_particle
            + beta_a2_attractive
        )

        rows.append(
            {
                "pH": pH,
                "reduced_density": rho_star,
                "concentration_mg_ml": (
                    base.reduced_density_to_concentration(
                        rho_star,
                        sigma_hs_nm,
                    )
                ),
                "packing_fraction": (
                    full_result.packing_fraction
                ),
                "beta_reference": (
                    full_result
                    .beta_reference_free_energy_per_particle
                ),
                "beta_a1": (
                    full_result.beta_a1_per_particle
                ),
                "beta_a2_full": (
                    full_result.beta_a2_per_particle
                ),
                "beta_a2_attractive": (
                    beta_a2_attractive
                ),
                # Baseline scan reads the beta_total column.
                "beta_total": (
                    beta_total_attractive_a2
                ),
                "attractive_second_integral": (
                    attractive_second_integral
                ),
                "attractive_second_error": (
                    attractive_second_error
                ),
            }
        )

        if (
            index == 1
            or index % 50 == 0
            or index == density_grid.size
        ):
            print(
                f"  {index:4d}/{density_grid.size:4d}"
                f" rho*={rho_star:.8f}"
                f" a1={full_result.beta_a1_per_particle:.7g}"
                f" a2_attr={beta_a2_attractive:.7g}"
                f" total={beta_total_attractive_a2:.7g}"
            )

    table = pd.DataFrame(rows)

    output_file = (
        OUTPUT_DIR
        / f"free_energy_pH{str(pH).replace('.', 'p')}.csv"
    )

    table.to_csv(
        output_file,
        index=False,
    )

    return table


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() == "true"


def accept_results(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in raw.iterrows():
        output = row.to_dict()

        valid = False

        if row.get("status") == "coexistence_solved":
            try:
                vapor_density = float(
                    row["vapor_density_reduced"]
                )

                liquid_density = float(
                    row["liquid_density_reduced"]
                )

                pressure_residual = float(
                    row["pressure_residual"]
                )

                chemical_residual = float(
                    row[
                        "chemical_potential_residual"
                    ]
                )

                maxwell_residual = float(
                    row["maxwell_area_residual"]
                )

                number_of_roots = int(
                    row["number_of_spinodal_roots"]
                )

                stitch_confined = parse_bool(
                    row["stitch_roots_confined"]
                )

                optimizer_success = parse_bool(
                    row["optimizer_success"]
                )

                residuals_valid = (
                    np.isfinite(pressure_residual)
                    and np.isfinite(chemical_residual)
                    and np.isfinite(maxwell_residual)
                    and abs(pressure_residual) <= 1.0e-6
                    and abs(chemical_residual) <= 1.0e-6
                    and abs(maxwell_residual) <= 1.0e-5
                )

                root_structure_valid = (
                    number_of_roots == 2
                    or (
                        number_of_roots == 4
                        and stitch_confined
                    )
                )

                boundary_valid = (
                    vapor_density
                    > base.SEARCH_MINIMUM_DENSITY
                    * 1.001
                    and liquid_density
                    < base.SEARCH_MAXIMUM_DENSITY
                    * 0.999
                )

                valid = (
                    optimizer_success
                    and residuals_valid
                    and root_structure_valid
                    and boundary_valid
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                valid = False

        if valid:
            status = "valid_coexistence"

            vapor_concentration = float(
                row["vapor_concentration_mg_ml"]
            )

            liquid_concentration = float(
                row["liquid_concentration_mg_ml"]
            )

            target_inside = (
                vapor_concentration
                <= TARGET_CONCENTRATION
                <= liquid_concentration
            )
        elif row.get("status") == "coexistence_solved":
            status = "invalid_coexistence"
            target_inside = False
        else:
            status = row.get("status")
            target_inside = False

        output["accepted_status"] = status
        output[
            "target_0p436_inside_accepted_two_phase"
        ] = target_inside

        rows.append(output)

    return pd.DataFrame(rows)


def make_accepted_plot(
    accepted: pd.DataFrame,
) -> None:
    valid = accepted[
        accepted["accepted_status"]
        == "valid_coexistence"
    ].copy()

    if valid.empty:
        return

    valid = valid.sort_values("pH")

    figure, axis = plt.subplots(
        figsize=(7.2, 5.2)
    )

    axis.plot(
        valid["pH"],
        valid["vapor_concentration_mg_ml"],
        marker="o",
        label="Dilute branch",
    )

    axis.plot(
        valid["pH"],
        valid["liquid_concentration_mg_ml"],
        marker="o",
        label="Dense branch",
    )

    axis.axhline(
        TARGET_CONCENTRATION,
        linestyle="--",
        label="0.436 mg/mL",
    )

    axis.set_yscale("log")
    axis.set_xlabel("pH")
    axis.set_ylabel(
        "Protein concentration (mg/mL)"
    )
    axis.set_title(
        "Huang A1 attractive-only second-order sensitivity"
    )
    axis.grid(
        True,
        which="both",
        alpha=0.25,
    )
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIR
        / "huang_a1_attractive_a2_binodal.png",
        dpi=240,
    )

    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Replace baseline-scan globals and table builder.
    base.OUTPUT_DIR = OUTPUT_DIR
    base.PH_VALUES = PH_VALUES
    base.build_free_energy_table = (
        build_attractive_a2_free_energy_table
    )

    base.main()

    raw_path = (
        OUTPUT_DIR
        / "huang_a1_llps_coarse_results.csv"
    )

    raw = pd.read_csv(raw_path)
    accepted = accept_results(raw)

    accepted_path = (
        OUTPUT_DIR
        / "huang_a1_llps_attractive_a2_results.csv"
    )

    accepted.to_csv(
        accepted_path,
        index=False,
    )

    make_accepted_plot(accepted)

    columns = [
        column
        for column in [
            "pH",
            "K2_kBT",
            "accepted_status",
            "vapor_concentration_mg_ml",
            "liquid_concentration_mg_ml",
            "target_0p436_inside_accepted_two_phase",
            "number_of_spinodal_roots",
            "stitch_roots_confined",
            "pressure_residual",
            "chemical_potential_residual",
            "maxwell_area_residual",
        ]
        if column in accepted.columns
    ]

    print()
    print("=" * 88)
    print("ATTRACTIVE-ONLY SECOND-ORDER ACCEPTED RESULTS")
    print("=" * 88)
    print(
        accepted[columns].to_string(
            index=False
        )
    )

    valid = accepted[
        accepted["accepted_status"]
        == "valid_coexistence"
    ]

    inside = valid[
        valid[
            "target_0p436_inside_accepted_two_phase"
        ]
        == True
    ]

    first_inside_pH = (
        float(inside["pH"].min())
        if not inside.empty
        else None
    )

    pH4p5 = accepted[
        np.isclose(
            accepted["pH"].to_numpy(dtype=float),
            4.5,
        )
    ]

    pH4p5_inside = (
        parse_bool(
            pH4p5[
                "target_0p436_inside_accepted_two_phase"
            ].iloc[0]
        )
        if not pH4p5.empty
        else False
    )

    print()
    print(
        "first accepted pH containing "
        "0.436 mg/mL:",
        first_inside_pH,
    )

    if (
        not pH4p5_inside
        and first_inside_pH is not None
        and 5.4 <= first_inside_pH <= 5.8
    ):
        print(
            "SENSITIVITY PASS: phase ordering is "
            "consistent with the Huang boundary region."
        )
    else:
        print(
            "SENSITIVITY REVIEW: phase ordering still "
            "does not match the Huang boundary region."
        )

    print()
    print("Generated:")
    print(" ", accepted_path)
    print(
        " ",
        OUTPUT_DIR
        / "huang_a1_attractive_a2_binodal.png",
    )
    print()
    print(
        "SCAN_HUANG_LLPS_ATTRACTIVE_A2: COMPLETE"
    )


if __name__ == "__main__":
    main()
