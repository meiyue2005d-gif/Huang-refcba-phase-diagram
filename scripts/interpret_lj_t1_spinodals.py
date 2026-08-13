#!/usr/bin/env python3
"""Interpret physical and RDF-stitch spinodals at LJ T*=1."""

from __future__ import annotations

import json
import sys
from math import log
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from huang_md.coexistence import (
    find_spinodal_densities,
    solve_fluid_coexistence,
)


TEMPERATURE = 1.0
SEARCH_MINIMUM = 1.0e-4
SEARCH_MAXIMUM = 0.885

BLEND_START = 0.20
BLEND_END = 0.25
BLEND_MARGIN = 0.005

OUTPUT_DIR = (
    ROOT
    / "results"
    / "lj_coexistence_t1_diagnostic"
)


def main() -> None:
    table_path = (
        OUTPUT_DIR
        / "free_energy_table.csv"
    )

    if not table_path.exists():
        raise FileNotFoundError(table_path)

    table = pd.read_csv(table_path)

    densities = table[
        "reduced_density"
    ].to_numpy(dtype=np.float64)

    residual = table[
        "beta_residual_free_energy"
    ].to_numpy(dtype=np.float64)

    spline = CubicSpline(
        densities,
        residual,
        bc_type="natural",
        extrapolate=False,
    )

    def free_energy(rho: float) -> float:
        value = float(rho)

        if not (
            densities[0]
            <= value
            <= densities[-1]
        ):
            raise ValueError(
                f"Density outside spline table: {value}"
            )

        return float(
            log(value)
            - 1.0
            + spline(value)
        )

    all_spinodals = find_spinodal_densities(
        beta_free_energy_per_particle=free_energy,
        minimum_density=SEARCH_MINIMUM,
        maximum_density=SEARCH_MAXIMUM,
        grid_points=1800,
    )

    if len(all_spinodals) < 2:
        raise RuntimeError(
            "Fewer than two spinodal roots were found."
        )

    physical_spinodals = (
        float(all_spinodals[0]),
        float(all_spinodals[-1]),
    )

    stitch_spinodals = [
        float(root)
        for root in all_spinodals[1:-1]
    ]

    stitch_roots_confined = all(
        (
            BLEND_START - BLEND_MARGIN
            <= root
            <= BLEND_END + BLEND_MARGIN
        )
        for root in stitch_spinodals
    )

    coexistence = solve_fluid_coexistence(
        beta_free_energy_per_particle=free_energy,
        minimum_density=SEARCH_MINIMUM,
        maximum_density=SEARCH_MAXIMUM,
        grid_points=1800,
    )

    reference_path = (
        ROOT
        / "results"
        / "lj_nist_reference_validation"
        / "nist_lj_cut5_reference_augmented.csv"
    )

    reference = pd.read_csv(reference_path)

    row = reference.loc[
        np.isclose(
            reference["temperature_reduced"],
            TEMPERATURE,
            atol=1.0e-12,
            rtol=0.0,
        )
    ]

    if len(row) != 1:
        raise RuntimeError(
            f"Expected one NIST row, found {len(row)}."
        )

    row = row.iloc[0]

    reference_vapor = float(
        row["vapor_density_reduced"]
    )
    reference_liquid = float(
        row["liquid_density_reduced"]
    )
    reference_pressure = float(
        row["saturation_pressure_reduced"]
    )

    theory_pressure = float(
        TEMPERATURE
        * coexistence.beta_pressure
    )

    result = {
        "temperature_reduced": TEMPERATURE,
        "all_spinodal_roots": [
            float(root)
            for root in all_spinodals
        ],
        "physical_spinodal_roots": [
            physical_spinodals[0],
            physical_spinodals[1],
        ],
        "rdf_stitch_spinodal_roots": (
            stitch_spinodals
        ),
        "stitch_roots_confined_to_blend_region": (
            stitch_roots_confined
        ),
        "theory_vapor_density": float(
            coexistence.vapor_density
        ),
        "theory_liquid_density": float(
            coexistence.liquid_density
        ),
        "theory_pressure_reduced": (
            theory_pressure
        ),
        "reference_vapor_density": (
            reference_vapor
        ),
        "reference_liquid_density": (
            reference_liquid
        ),
        "reference_pressure_reduced": (
            reference_pressure
        ),
        "vapor_relative_error": abs(
            coexistence.vapor_density
            - reference_vapor
        ) / reference_vapor,
        "liquid_relative_error": abs(
            coexistence.liquid_density
            - reference_liquid
        ) / reference_liquid,
        "pressure_relative_error": abs(
            theory_pressure
            - reference_pressure
        ) / reference_pressure,
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
    }

    output_file = (
        OUTPUT_DIR
        / "spinodal_interpretation_t1.json"
    )

    output_file.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 78)
    print("LJ T*=1 spinodal interpretation")
    print("=" * 78)

    print("\nAll roots:")
    for root in all_spinodals:
        print(f"  rho*={root:.10f}")

    print("\nPhysical outer roots:")
    print(
        f"  vapor-side spinodal = "
        f"{physical_spinodals[0]:.10f}"
    )
    print(
        f"  liquid-side spinodal = "
        f"{physical_spinodals[1]:.10f}"
    )

    print("\nRDF-stitch roots:")
    for root in stitch_spinodals:
        print(f"  rho*={root:.10f}")

    print("\nBinodal:")
    print(
        f"  vapor density  = "
        f"{coexistence.vapor_density:.10f}"
    )
    print(
        f"  liquid density = "
        f"{coexistence.liquid_density:.10f}"
    )
    print(
        f"  pressure P*    = "
        f"{theory_pressure:.10f}"
    )

    print("\nRelative NIST errors:")
    print(
        f"  vapor density  = "
        f"{result['vapor_relative_error']:.3%}"
    )
    print(
        f"  liquid density = "
        f"{result['liquid_relative_error']:.3%}"
    )
    print(
        f"  pressure       = "
        f"{result['pressure_relative_error']:.3%}"
    )

    checks = {
        "at least two spinodal roots":
            len(all_spinodals) >= 2,
        "ordered physical spinodals":
            (
                physical_spinodals[0]
                < physical_spinodals[1]
            ),
        "extra roots confined to RDF blend":
            stitch_roots_confined,
        "ordered binodal densities":
            (
                coexistence.vapor_density
                < coexistence.liquid_density
            ),
        "optimizer success":
            bool(coexistence.optimizer_success),
        "pressure residual":
            abs(
                coexistence.pressure_residual
            ) < 1.0e-6,
        "chemical-potential residual":
            abs(
                coexistence.chemical_potential_residual
            ) < 1.0e-6,
    }

    failed = []

    print("\nChecks:")

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status:<4s} {name}")

        if not passed:
            failed.append(name)

    print(f"\nGenerated file:\n  {output_file}")

    if failed:
        raise RuntimeError(
            "Spinodal interpretation failed: "
            + ", ".join(failed)
        )

    print(
        "\nLJ T*=1 diagnostic accepted with "
        "documented RDF-stitch artifacts."
    )


if __name__ == "__main__":
    main()
