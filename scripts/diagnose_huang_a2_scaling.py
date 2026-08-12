#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from huang_md.coexistence import (
    find_spinodal_densities,
    solve_fluid_coexistence,
)

INPUT_DIR = (
    ROOT
    / "results"
    / "huang_a1_llps_baseline_coarse"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "liquid_theory_validation"
)

TABLE_MINIMUM = 5.0e-6
TABLE_MAXIMUM = 0.89
SEARCH_MINIMUM = 1.0e-5
SEARCH_MAXIMUM = 0.885

TARGET_CONCENTRATION = 0.436

LAMBDA_VALUES = np.round(
    np.arange(0.0, 1.0001, 0.05),
    10,
)


def evaluate_table(
    table: pd.DataFrame,
    lambda_a2: float,
) -> dict[str, object]:
    pH = float(table["pH"].iloc[0])

    density = table[
        "reduced_density"
    ].to_numpy(dtype=float)

    beta_a1 = table[
        "beta_a1"
    ].to_numpy(dtype=float)

    beta_a2 = table[
        "beta_a2"
    ].to_numpy(dtype=float)

    beta_total_original = table[
        "beta_total"
    ].to_numpy(dtype=float)

    beta_reference = (
        beta_total_original
        - beta_a1
        - beta_a2
    )

    beta_total = (
        beta_reference
        + beta_a1
        + lambda_a2 * beta_a2
    )

    spline = CubicSpline(
        density,
        beta_total,
    )

    lower = float(density[0])
    upper = float(density[-1])

    def free_energy(rho: float) -> float:
        value = float(rho)

        if not lower <= value <= upper:
            raise ValueError(
                f"Density outside table: {value:.12g}"
            )

        return float(spline(value))

    concentration_factor = float(
        np.median(
            table[
                "concentration_mg_ml"
            ].to_numpy(dtype=float)
            / density
        )
    )

    try:
        roots = find_spinodal_densities(
            beta_free_energy_per_particle=free_energy,
            minimum_density=SEARCH_MINIMUM,
            maximum_density=SEARCH_MAXIMUM,
            grid_points=1800,
        )

        roots = [
            float(value)
            for value in roots
        ]
    except Exception as exc:
        return {
            "pH": pH,
            "lambda_a2": lambda_a2,
            "status": "no_spinodal",
            "error": repr(exc),
        }

    if len(roots) < 2:
        return {
            "pH": pH,
            "lambda_a2": lambda_a2,
            "status": "fewer_than_two_spinodals",
            "number_of_spinodal_roots": len(roots),
            "all_spinodal_roots": repr(roots),
        }

    inner_roots = roots[1:-1]

    stitch_roots_confined = all(
        0.195 <= value <= 0.255
        for value in inner_roots
    )

    root_structure_valid = (
        len(roots) == 2
        or (
            len(roots) == 4
            and stitch_roots_confined
        )
    )

    try:
        coexistence = solve_fluid_coexistence(
            beta_free_energy_per_particle=free_energy,
            minimum_density=SEARCH_MINIMUM,
            maximum_density=SEARCH_MAXIMUM,
            grid_points=1800,
        )
    except Exception as exc:
        return {
            "pH": pH,
            "lambda_a2": lambda_a2,
            "status": "solver_failed",
            "number_of_spinodal_roots": len(roots),
            "all_spinodal_roots": repr(roots),
            "error": repr(exc),
        }

    vapor_density = float(
        coexistence.vapor_density
    )

    liquid_density = float(
        coexistence.liquid_density
    )

    vapor_concentration = (
        vapor_density
        * concentration_factor
    )

    liquid_concentration = (
        liquid_density
        * concentration_factor
    )

    residuals_valid = (
        np.isfinite(
            coexistence.pressure_residual
        )
        and np.isfinite(
            coexistence.chemical_potential_residual
        )
        and np.isfinite(
            coexistence.maxwell_area_residual
        )
        and abs(
            coexistence.pressure_residual
        ) <= 1.0e-6
        and abs(
            coexistence.chemical_potential_residual
        ) <= 1.0e-6
        and abs(
            coexistence.maxwell_area_residual
        ) <= 1.0e-5
    )

    boundary_valid = (
        vapor_density
        > SEARCH_MINIMUM * 1.001
        and liquid_density
        < SEARCH_MAXIMUM * 0.999
    )

    valid = (
        bool(coexistence.optimizer_success)
        and residuals_valid
        and root_structure_valid
        and boundary_valid
    )

    status = (
        "valid_coexistence"
        if valid
        else "invalid_coexistence"
    )

    return {
        "pH": pH,
        "lambda_a2": lambda_a2,
        "status": status,
        "vapor_density_reduced": vapor_density,
        "liquid_density_reduced": liquid_density,
        "vapor_concentration_mg_ml": (
            vapor_concentration
        ),
        "liquid_concentration_mg_ml": (
            liquid_concentration
        ),
        "target_inside_two_phase": (
            valid
            and vapor_concentration
            <= TARGET_CONCENTRATION
            <= liquid_concentration
        ),
        "number_of_spinodal_roots": len(roots),
        "stitch_roots_confined": (
            stitch_roots_confined
        ),
        "optimizer_success": bool(
            coexistence.optimizer_success
        ),
        "pressure_residual": float(
            coexistence.pressure_residual
        ),
        "chemical_potential_residual": float(
            coexistence.chemical_potential_residual
        ),
        "maxwell_area_residual": float(
            coexistence.maxwell_area_residual
        ),
        "all_spinodal_roots": repr(roots),
    }


def main() -> None:
    files = sorted(
        INPUT_DIR.glob("free_energy_pH*.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No free-energy tables found in {INPUT_DIR}"
        )

    rows = []

    for path in files:
        table = pd.read_csv(path)

        for lambda_a2 in LAMBDA_VALUES:
            rows.append(
                evaluate_table(
                    table=table,
                    lambda_a2=float(lambda_a2),
                )
            )

    result = pd.DataFrame(rows).sort_values(
        ["lambda_a2", "pH"]
    )

    output = (
        OUTPUT_DIR
        / "huang_a2_scaling_diagnostic.csv"
    )

    result.to_csv(output, index=False)

    summary_rows = []

    for lambda_a2, group in result.groupby(
        "lambda_a2"
    ):
        valid = group[
            group["status"]
            == "valid_coexistence"
        ].copy()

        inside = valid[
            valid["target_inside_two_phase"]
            == True
        ]

        summary_rows.append(
            {
                "lambda_a2": lambda_a2,
                "valid_pH_count": len(valid),
                "minimum_valid_pH": (
                    valid["pH"].min()
                    if not valid.empty
                    else np.nan
                ),
                "first_pH_0p436_inside": (
                    inside["pH"].min()
                    if not inside.empty
                    else np.nan
                ),
                "pH4p5_vapor_mg_ml": (
                    valid.loc[
                        np.isclose(
                            valid["pH"],
                            4.5,
                        ),
                        "vapor_concentration_mg_ml",
                    ].iloc[0]
                    if np.isclose(
                        valid["pH"],
                        4.5,
                    ).any()
                    else np.nan
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    summary_output = (
        OUTPUT_DIR
        / "huang_a2_scaling_summary.csv"
    )

    summary.to_csv(
        summary_output,
        index=False,
    )

    print("=" * 82)
    print("HUANG SECOND-ORDER SCALING SUMMARY")
    print("=" * 82)
    print(summary.to_string(index=False))
    print()
    print("Generated:")
    print(" ", output)
    print(" ", summary_output)
    print()
    print("DIAGNOSE_HUANG_A2_SCALING: COMPLETE")


if __name__ == "__main__":
    main()
