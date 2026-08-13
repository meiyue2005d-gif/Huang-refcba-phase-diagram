#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    ROOT
    / "results"
    / "huang_a1_llps_attractive_a2"
)

OUTPUT_FILE = (
    ROOT
    / "results"
    / "liquid_theory_validation"
    / "huang_bridge_excluded_coexistence.csv"
)

LOW_SPLINE_MAXIMUM = 0.199
LOW_SEARCH_MINIMUM = 1.0e-5
LOW_SEARCH_MAXIMUM = 0.195

HIGH_SPLINE_MINIMUM = 0.251
HIGH_SEARCH_MINIMUM = 0.255
HIGH_SEARCH_MAXIMUM = 0.885

TARGET_CONCENTRATION = 0.436

RESIDUAL_TOLERANCE = 1.0e-8
TANGENT_SUPPORT_TOLERANCE = 1.0e-6


def state(
    spline: CubicSpline,
    derivative: CubicSpline,
    density: float,
) -> tuple[float, float, float, float]:
    rho = float(density)
    a = float(spline(rho))
    da = float(derivative(rho))

    free_energy_density = rho * a
    chemical_potential = a + rho * da
    pressure = rho**2 * da

    return (
        a,
        free_energy_density,
        chemical_potential,
        pressure,
    )


def analyze(path: Path) -> dict[str, object]:
    table = (
        pd.read_csv(path)
        .sort_values("reduced_density")
        .reset_index(drop=True)
    )

    pH = float(table["pH"].iloc[0])

    low = table[
        table["reduced_density"]
        <= LOW_SPLINE_MAXIMUM
    ].copy()

    high = table[
        table["reduced_density"]
        >= HIGH_SPLINE_MINIMUM
    ].copy()

    if len(low) < 10 or len(high) < 10:
        return {
            "pH": pH,
            "status": "insufficient_branch_data",
        }

    low_rho = low["reduced_density"].to_numpy(dtype=float)
    low_a = low["beta_total"].to_numpy(dtype=float)

    high_rho = high["reduced_density"].to_numpy(dtype=float)
    high_a = high["beta_total"].to_numpy(dtype=float)

    low_spline = CubicSpline(
        low_rho,
        low_a,
        extrapolate=False,
    )

    high_spline = CubicSpline(
        high_rho,
        high_a,
        extrapolate=False,
    )

    low_derivative = low_spline.derivative(1)
    high_derivative = high_spline.derivative(1)

    concentration_factor = float(
        np.median(
            table["concentration_mg_ml"].to_numpy(dtype=float)
            / table["reduced_density"].to_numpy(dtype=float)
        )
    )

    vapor_seeds = np.geomspace(
        LOW_SEARCH_MINIMUM * 1.05,
        LOW_SEARCH_MAXIMUM * 0.98,
        20,
    )

    liquid_seeds = np.linspace(
        HIGH_SEARCH_MINIMUM * 1.02,
        HIGH_SEARCH_MAXIMUM * 0.98,
        20,
    )

    candidates = []

    for vapor_seed in vapor_seeds:
        for liquid_seed in liquid_seeds:
            _, _, mu_v0, p_v0 = state(
                low_spline,
                low_derivative,
                vapor_seed,
            )

            _, _, mu_l0, p_l0 = state(
                high_spline,
                high_derivative,
                liquid_seed,
            )

            mu_scale = max(
                1.0,
                abs(mu_v0),
                abs(mu_l0),
            )

            pressure_scale = max(
                1.0e-6,
                abs(p_v0),
                abs(p_l0),
            )

            def residuals(log_densities):
                rho_v = float(np.exp(log_densities[0]))
                rho_l = float(np.exp(log_densities[1]))

                _, _, mu_v, pressure_v = state(
                    low_spline,
                    low_derivative,
                    rho_v,
                )

                _, _, mu_l, pressure_l = state(
                    high_spline,
                    high_derivative,
                    rho_l,
                )

                return np.array(
                    [
                        (mu_v - mu_l) / mu_scale,
                        (
                            pressure_v
                            - pressure_l
                        ) / pressure_scale,
                    ],
                    dtype=float,
                )

            solution = least_squares(
                residuals,
                x0=np.log(
                    [vapor_seed, liquid_seed]
                ),
                bounds=(
                    np.log(
                        [
                            LOW_SEARCH_MINIMUM,
                            HIGH_SEARCH_MINIMUM,
                        ]
                    ),
                    np.log(
                        [
                            LOW_SEARCH_MAXIMUM,
                            HIGH_SEARCH_MAXIMUM,
                        ]
                    ),
                ),
                xtol=1.0e-13,
                ftol=1.0e-13,
                gtol=1.0e-13,
                max_nfev=4000,
            )

            rho_v = float(np.exp(solution.x[0]))
            rho_l = float(np.exp(solution.x[1]))

            _, f_v, mu_v, pressure_v = state(
                low_spline,
                low_derivative,
                rho_v,
            )

            _, f_l, mu_l, pressure_l = state(
                high_spline,
                high_derivative,
                rho_l,
            )

            mu_residual = mu_v - mu_l
            pressure_residual = pressure_v - pressure_l

            mu_coexistence = 0.5 * (mu_v + mu_l)
            pressure_coexistence = 0.5 * (
                pressure_v + pressure_l
            )

            endpoint_v_residual = (
                f_v
                - (
                    mu_coexistence * rho_v
                    - pressure_coexistence
                )
            )

            endpoint_l_residual = (
                f_l
                - (
                    mu_coexistence * rho_l
                    - pressure_coexistence
                )
            )

            low_grid = np.geomspace(
                LOW_SEARCH_MINIMUM,
                LOW_SEARCH_MAXIMUM,
                1200,
            )

            high_grid = np.linspace(
                HIGH_SEARCH_MINIMUM,
                HIGH_SEARCH_MAXIMUM,
                1600,
            )

            low_gap = (
                low_grid * low_spline(low_grid)
                - (
                    mu_coexistence * low_grid
                    - pressure_coexistence
                )
            )

            high_gap = (
                high_grid * high_spline(high_grid)
                - (
                    mu_coexistence * high_grid
                    - pressure_coexistence
                )
            )

            minimum_tangent_gap = float(
                min(
                    np.min(low_gap),
                    np.min(high_gap),
                )
            )

            interior = (
                rho_v
                > LOW_SEARCH_MINIMUM * 1.001
                and rho_v
                < LOW_SEARCH_MAXIMUM * 0.999
                and rho_l
                > HIGH_SEARCH_MINIMUM * 1.001
                and rho_l
                < HIGH_SEARCH_MAXIMUM * 0.999
            )

            equations_valid = (
                abs(mu_residual)
                <= RESIDUAL_TOLERANCE
                and abs(pressure_residual)
                <= RESIDUAL_TOLERANCE
                and abs(endpoint_v_residual)
                <= RESIDUAL_TOLERANCE
                and abs(endpoint_l_residual)
                <= RESIDUAL_TOLERANCE
            )

            supporting_tangent = (
                minimum_tangent_gap
                >= -TANGENT_SUPPORT_TOLERANCE
            )

            valid = (
                bool(solution.success)
                and interior
                and equations_valid
                and supporting_tangent
            )

            candidates.append(
                {
                    "valid": valid,
                    "optimizer_success": bool(solution.success),
                    "vapor_density_reduced": rho_v,
                    "liquid_density_reduced": rho_l,
                    "chemical_potential_residual": mu_residual,
                    "pressure_residual": pressure_residual,
                    "vapor_tangent_residual": endpoint_v_residual,
                    "liquid_tangent_residual": endpoint_l_residual,
                    "minimum_tangent_gap": minimum_tangent_gap,
                    "raw_residual_norm": float(
                        np.hypot(
                            mu_residual,
                            pressure_residual,
                        )
                    ),
                }
            )

    valid_candidates = [
        item
        for item in candidates
        if item["valid"]
    ]

    if valid_candidates:
        best = min(
            valid_candidates,
            key=lambda item: item["raw_residual_norm"],
        )
        status = "valid_bridge_excluded_coexistence"
    else:
        best = min(
            candidates,
            key=lambda item: item["raw_residual_norm"],
        )
        status = "no_valid_bridge_excluded_solution"

    vapor_concentration = (
        best["vapor_density_reduced"]
        * concentration_factor
    )

    liquid_concentration = (
        best["liquid_density_reduced"]
        * concentration_factor
    )

    return {
        "pH": pH,
        "status": status,
        "vapor_density_reduced": (
            best["vapor_density_reduced"]
        ),
        "liquid_density_reduced": (
            best["liquid_density_reduced"]
        ),
        "vapor_concentration_mg_ml": (
            vapor_concentration
        ),
        "liquid_concentration_mg_ml": (
            liquid_concentration
        ),
        "target_0p436_inside": (
            status
            == "valid_bridge_excluded_coexistence"
            and vapor_concentration
            <= TARGET_CONCENTRATION
            <= liquid_concentration
        ),
        "optimizer_success": (
            best["optimizer_success"]
        ),
        "pressure_residual": (
            best["pressure_residual"]
        ),
        "chemical_potential_residual": (
            best["chemical_potential_residual"]
        ),
        "vapor_tangent_residual": (
            best["vapor_tangent_residual"]
        ),
        "liquid_tangent_residual": (
            best["liquid_tangent_residual"]
        ),
        "minimum_tangent_gap": (
            best["minimum_tangent_gap"]
        ),
        "number_of_initial_guesses": len(candidates),
        "number_of_valid_candidates": (
            len(valid_candidates)
        ),
    }


def main():
    files = sorted(
        INPUT_DIR.glob("free_energy_pH*.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No free-energy files found: {INPUT_DIR}"
        )

    result = pd.DataFrame(
        [
            analyze(path)
            for path in files
        ]
    ).sort_values("pH")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("=" * 118)
    print("HUANG BRIDGE-EXCLUDED BRANCHWISE COEXISTENCE")
    print("=" * 118)

    print(result.to_string(index=False))

    valid = result[
        result["status"]
        == "valid_bridge_excluded_coexistence"
    ]

    inside = valid[
        valid["target_0p436_inside"] == True
    ]

    first_inside = (
        float(inside["pH"].min())
        if not inside.empty
        else None
    )

    print()
    print(
        "first valid pH containing 0.436 mg/mL:",
        first_inside,
    )

    print()
    print("saved:", OUTPUT_FILE)
    print("HUANG_BRIDGE_EXCLUDED_SCAN: COMPLETE")


if __name__ == "__main__":
    main()
