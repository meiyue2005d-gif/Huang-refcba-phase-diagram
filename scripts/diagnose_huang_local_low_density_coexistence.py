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
    / "huang_a1_llps_attractive_a2"
)

OUTPUT_FILE = (
    ROOT
    / "results"
    / "liquid_theory_validation"
    / "huang_local_low_density_coexistence.csv"
)

SEARCH_MINIMUM = 1.0e-5

# Stay strictly below the project RDF bridge beginning at rho*=0.2.
SEARCH_MAXIMUM = 0.199

TARGET_CONCENTRATION_MG_ML = 0.436


def analyze_table(path: Path) -> dict[str, object]:
    table = pd.read_csv(path)

    pH = float(table["pH"].iloc[0])

    density = table[
        "reduced_density"
    ].to_numpy(dtype=float)

    free_energy_values = table[
        "beta_total"
    ].to_numpy(dtype=float)

    spline = CubicSpline(
        density,
        free_energy_values,
    )

    table_minimum = float(density.min())
    table_maximum = float(density.max())

    def free_energy(rho: float) -> float:
        value = float(rho)

        if not table_minimum <= value <= table_maximum:
            raise ValueError(
                f"Density outside table: {value:.12g}"
            )

        return float(spline(value))

    concentration_factor = float(
        np.median(
            table["concentration_mg_ml"].to_numpy(
                dtype=float
            )
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

        roots = [float(root) for root in roots]

    except Exception as exc:
        return {
            "pH": pH,
            "status": "spinodal_search_failed",
            "error": repr(exc),
        }

    if len(roots) != 2:
        return {
            "pH": pH,
            "status": "not_one_local_loop",
            "number_of_spinodal_roots": len(roots),
            "all_spinodal_roots": repr(roots),
        }

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
            "status": "coexistence_failed",
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
        vapor_density * concentration_factor
    )

    liquid_concentration = (
        liquid_density * concentration_factor
    )

    residuals_valid = (
        np.isfinite(coexistence.pressure_residual)
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
        and boundary_valid
    )

    return {
        "pH": pH,
        "status": (
            "valid_local_coexistence"
            if valid
            else "invalid_local_coexistence"
        ),
        "number_of_spinodal_roots": len(roots),
        "vapor_spinodal": roots[0],
        "liquid_spinodal": roots[1],
        "vapor_density_reduced": vapor_density,
        "liquid_density_reduced": liquid_density,
        "vapor_concentration_mg_ml": (
            vapor_concentration
        ),
        "liquid_concentration_mg_ml": (
            liquid_concentration
        ),
        "target_0p436_inside": (
            valid
            and vapor_concentration
            <= TARGET_CONCENTRATION_MG_ML
            <= liquid_concentration
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

    rows = [
        analyze_table(path)
        for path in files
    ]

    results = (
        pd.DataFrame(rows)
        .sort_values("pH")
        .reset_index(drop=True)
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
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
        ]
        if column in results.columns
    ]

    print("=" * 100)
    print("HUANG LOCAL LOW-DENSITY COEXISTENCE")
    print("=" * 100)
    print(results[columns].to_string(index=False))

    valid = results[
        results["status"]
        == "valid_local_coexistence"
    ]

    inside = valid[
        valid["target_0p436_inside"] == True
    ]

    print()
    print(
        "first valid pH containing 0.436 mg/mL:",
        (
            float(inside["pH"].min())
            if not inside.empty
            else None
        ),
    )

    print()
    print("saved:", OUTPUT_FILE)
    print(
        "DIAGNOSE_LOCAL_LOW_DENSITY_COEXISTENCE: COMPLETE"
    )


if __name__ == "__main__":
    main()
