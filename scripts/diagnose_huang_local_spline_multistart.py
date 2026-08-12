#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, least_squares


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
    / "huang_local_spline_multistart.csv"
)

SEARCH_MINIMUM = 1.0e-5
SEARCH_MAXIMUM = 0.199

TARGET_CONCENTRATION_MG_ML = 0.436

MU_TOLERANCE = 1.0e-8
PRESSURE_TOLERANCE = 1.0e-8
MAXWELL_TOLERANCE = 1.0e-7


def build_search_grid() -> np.ndarray:
    return np.unique(
        np.concatenate(
            [
                np.geomspace(
                    SEARCH_MINIMUM,
                    0.02,
                    900,
                    endpoint=False,
                ),
                np.linspace(
                    0.02,
                    SEARCH_MAXIMUM,
                    1400,
                ),
            ]
        )
    )


def deduplicate_roots(
    roots: list[float],
    tolerance: float = 1.0e-7,
) -> list[float]:
    output: list[float] = []

    for value in sorted(roots):
        if (
            not output
            or abs(value - output[-1]) > tolerance
        ):
            output.append(float(value))

    return output


def find_analytic_spinodals(
    spline: CubicSpline,
) -> list[float]:
    first = spline.derivative(1)
    second = spline.derivative(2)

    def pressure_derivative(
        density: float,
    ) -> float:
        rho = float(density)

        return float(
            2.0 * rho * first(rho)
            + rho**2 * second(rho)
        )

    grid = build_search_grid()

    values = np.array(
        [
            pressure_derivative(rho)
            for rho in grid
        ],
        dtype=float,
    )

    roots: list[float] = []

    for left, right, f_left, f_right in zip(
        grid[:-1],
        grid[1:],
        values[:-1],
        values[1:],
    ):
        if not (
            np.isfinite(f_left)
            and np.isfinite(f_right)
        ):
            continue

        if f_left == 0.0:
            roots.append(float(left))
            continue

        if f_left * f_right < 0.0:
            root = brentq(
                pressure_derivative,
                float(left),
                float(right),
                xtol=1.0e-13,
                rtol=1.0e-13,
                maxiter=300,
            )

            roots.append(float(root))

    return deduplicate_roots(roots)


def analyze_table(
    path: Path,
) -> dict[str, object]:
    table = pd.read_csv(path)

    pH = float(table["pH"].iloc[0])

    density = table[
        "reduced_density"
    ].to_numpy(dtype=float)

    free_energy = table[
        "beta_total"
    ].to_numpy(dtype=float)

    spline = CubicSpline(
        density,
        free_energy,
    )

    first = spline.derivative(1)

    concentration_factor = float(
        np.median(
            table[
                "concentration_mg_ml"
            ].to_numpy(dtype=float)
            / density
        )
    )

    def state(
        rho: float,
    ) -> tuple[float, float, float]:
        value = float(rho)

        a = float(spline(value))
        derivative = float(first(value))

        chemical_potential = (
            a + value * derivative
        )

        pressure = (
            value**2 * derivative
        )

        return (
            a,
            chemical_potential,
            pressure,
        )

    roots = find_analytic_spinodals(
        spline
    )

    base_result: dict[str, object] = {
        "pH": pH,
        "number_of_spinodal_roots": len(roots),
        "all_spinodal_roots": repr(roots),
    }

    if len(roots) != 2:
        base_result.update(
            {
                "status": "not_one_local_loop",
            }
        )

        return base_result

    lower_spinodal = roots[0]
    upper_spinodal = roots[1]

    vapor_lower = (
        SEARCH_MINIMUM * 1.001
    )

    vapor_upper = (
        lower_spinodal * (1.0 - 1.0e-5)
    )

    liquid_lower = (
        upper_spinodal * (1.0 + 1.0e-5)
    )

    liquid_upper = (
        SEARCH_MAXIMUM * (1.0 - 1.0e-5)
    )

    if not (
        vapor_lower
        < vapor_upper
        < liquid_lower
        < liquid_upper
    ):
        base_result.update(
            {
                "status": "invalid_bounds",
            }
        )

        return base_result

    vapor_seeds = np.geomspace(
        vapor_lower * 1.01,
        vapor_upper * 0.99,
        16,
    )

    liquid_seeds = np.linspace(
        liquid_lower * 1.01,
        liquid_upper * 0.99,
        16,
    )

    candidates: list[dict[str, object]] = []

    for vapor_seed in vapor_seeds:
        for liquid_seed in liquid_seeds:
            _, mu_v0, pressure_v0 = state(
                float(vapor_seed)
            )

            _, mu_l0, pressure_l0 = state(
                float(liquid_seed)
            )

            mu_scale = max(
                1.0,
                abs(mu_v0),
                abs(mu_l0),
            )

            pressure_scale = max(
                1.0e-6,
                abs(pressure_v0),
                abs(pressure_l0),
            )

            def scaled_residuals(
                log_densities: np.ndarray,
            ) -> np.ndarray:
                rho_v = float(
                    np.exp(log_densities[0])
                )

                rho_l = float(
                    np.exp(log_densities[1])
                )

                _, mu_v, pressure_v = state(
                    rho_v
                )

                _, mu_l, pressure_l = state(
                    rho_l
                )

                return np.array(
                    [
                        (
                            mu_v - mu_l
                        ) / mu_scale,
                        (
                            pressure_v
                            - pressure_l
                        ) / pressure_scale,
                    ],
                    dtype=float,
                )

            solution = least_squares(
                scaled_residuals,
                x0=np.log(
                    [
                        vapor_seed,
                        liquid_seed,
                    ]
                ),
                bounds=(
                    np.log(
                        [
                            vapor_lower,
                            liquid_lower,
                        ]
                    ),
                    np.log(
                        [
                            vapor_upper,
                            liquid_upper,
                        ]
                    ),
                ),
                xtol=1.0e-13,
                ftol=1.0e-13,
                gtol=1.0e-13,
                max_nfev=3000,
            )

            rho_v = float(
                np.exp(solution.x[0])
            )

            rho_l = float(
                np.exp(solution.x[1])
            )

            a_v, mu_v, pressure_v = state(
                rho_v
            )

            a_l, mu_l, pressure_l = state(
                rho_l
            )

            mu_residual = (
                mu_v - mu_l
            )

            pressure_residual = (
                pressure_v - pressure_l
            )

            coexistence_pressure = (
                0.5
                * (
                    pressure_v
                    + pressure_l
                )
            )

            # Exact density-coordinate Maxwell residual:
            # integral[(P-Pcoex)/rho^2]drho
            maxwell_residual = (
                a_l
                - a_v
                + coexistence_pressure
                * (
                    1.0 / rho_l
                    - 1.0 / rho_v
                )
            )

            raw_norm = float(
                np.sqrt(
                    mu_residual**2
                    + pressure_residual**2
                )
            )

            interior = (
                rho_v
                > vapor_lower * 1.0001
                and rho_v
                < vapor_upper * 0.9999
                and rho_l
                > liquid_lower * 1.0001
                and rho_l
                < liquid_upper * 0.9999
            )

            valid = (
                bool(solution.success)
                and interior
                and abs(mu_residual)
                <= MU_TOLERANCE
                and abs(pressure_residual)
                <= PRESSURE_TOLERANCE
                and abs(maxwell_residual)
                <= MAXWELL_TOLERANCE
            )

            candidates.append(
                {
                    "valid": valid,
                    "optimizer_success": bool(
                        solution.success
                    ),
                    "vapor_density_reduced": rho_v,
                    "liquid_density_reduced": rho_l,
                    "chemical_potential_residual": (
                        mu_residual
                    ),
                    "pressure_residual": (
                        pressure_residual
                    ),
                    "maxwell_area_residual": (
                        maxwell_residual
                    ),
                    "raw_residual_norm": raw_norm,
                    "optimizer_message": str(
                        solution.message
                    ),
                }
            )

    valid_candidates = [
        candidate
        for candidate in candidates
        if bool(candidate["valid"])
    ]

    if valid_candidates:
        best = min(
            valid_candidates,
            key=lambda item: float(
                item["raw_residual_norm"]
            ),
        )

        status = (
            "valid_local_coexistence"
        )
    else:
        best = min(
            candidates,
            key=lambda item: float(
                item["raw_residual_norm"]
            ),
        )

        status = (
            "no_valid_multistart_solution"
        )

    vapor_density = float(
        best["vapor_density_reduced"]
    )

    liquid_density = float(
        best["liquid_density_reduced"]
    )

    vapor_concentration = (
        vapor_density
        * concentration_factor
    )

    liquid_concentration = (
        liquid_density
        * concentration_factor
    )

    base_result.update(
        {
            "status": status,
            "vapor_spinodal": (
                lower_spinodal
            ),
            "liquid_spinodal": (
                upper_spinodal
            ),
            "vapor_density_reduced": (
                vapor_density
            ),
            "liquid_density_reduced": (
                liquid_density
            ),
            "vapor_concentration_mg_ml": (
                vapor_concentration
            ),
            "liquid_concentration_mg_ml": (
                liquid_concentration
            ),
            "target_0p436_inside": (
                status
                == "valid_local_coexistence"
                and vapor_concentration
                <= TARGET_CONCENTRATION_MG_ML
                <= liquid_concentration
            ),
            "optimizer_success": (
                best["optimizer_success"]
            ),
            "pressure_residual": (
                best["pressure_residual"]
            ),
            "chemical_potential_residual": (
                best[
                    "chemical_potential_residual"
                ]
            ),
            "maxwell_area_residual": (
                best["maxwell_area_residual"]
            ),
            "raw_residual_norm": (
                best["raw_residual_norm"]
            ),
            "number_of_initial_guesses": (
                len(candidates)
            ),
        }
    )

    return base_result


def main() -> None:
    files = sorted(
        INPUT_DIR.glob(
            "free_energy_pH*.csv"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No free-energy tables found: "
            f"{INPUT_DIR}"
        )

    rows = [
        analyze_table(path)
        for path in files
    ]

    result = (
        pd.DataFrame(rows)
        .sort_values("pH")
        .reset_index(drop=True)
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    columns = [
        column
        for column in [
            "pH",
            "status",
            "vapor_spinodal",
            "liquid_spinodal",
            "vapor_concentration_mg_ml",
            "liquid_concentration_mg_ml",
            "target_0p436_inside",
            "pressure_residual",
            "chemical_potential_residual",
            "maxwell_area_residual",
            "raw_residual_norm",
        ]
        if column in result.columns
    ]

    print("=" * 108)
    print(
        "HUANG LOCAL SPLINE-DERIVATIVE "
        "MULTISTART COEXISTENCE"
    )
    print("=" * 108)

    print(
        result[columns].to_string(
            index=False
        )
    )

    valid = result[
        result["status"]
        == "valid_local_coexistence"
    ]

    inside = valid[
        valid["target_0p436_inside"]
        == True
    ]

    first_inside = (
        float(inside["pH"].min())
        if not inside.empty
        else None
    )

    print()
    print(
        "first valid pH containing "
        "0.436 mg/mL:",
        first_inside,
    )

    print()
    print("saved:", OUTPUT_FILE)
    print(
        "HUANG_LOCAL_SPLINE_MULTISTART: COMPLETE"
    )


if __name__ == "__main__":
    main()
