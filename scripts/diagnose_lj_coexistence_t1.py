#!/usr/bin/env python3
"""Diagnose LJ vapor-liquid coexistence at T*=1.0.

A dense table of the residual Helmholtz free energy is first
constructed using the full perturbation integrals. A C2 cubic
spline is then used by the numerical coexistence solver.

Interpolating only the residual free energy avoids attempting
to spline the logarithmic ideal-gas divergence at rho -> 0.
"""

from __future__ import annotations

import sys
from math import log
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from huang_md.coexistence import (
    find_spinodal_densities,
    solve_fluid_coexistence,
    thermodynamic_state,
)
from huang_md.lj_perturbation import (
    make_lj_free_energy_function,
)


TEMPERATURE = 1.0
MINIMUM_DENSITY = 1.0e-5
MAXIMUM_DENSITY = 0.89
SPINODAL_SEARCH_MAXIMUM_DENSITY = 0.885

OUTPUT_DIR = (
    ROOT
    / "results"
    / "lj_coexistence_t1_diagnostic"
)


def ideal_free_energy(
    reduced_density: float,
) -> float:
    rho = float(reduced_density)

    if rho <= 0.0:
        raise ValueError(
            "Reduced density must be positive."
        )

    return log(rho) - 1.0


def build_density_grid() -> np.ndarray:
    """Return a nonuniform grid with extra points near RDF blending."""
    sections = [
        np.geomspace(
            MINIMUM_DENSITY,
            0.03,
            65,
            endpoint=False,
        ),
        np.linspace(
            0.03,
            0.19,
            75,
            endpoint=False,
        ),
        np.linspace(
            0.19,
            0.26,
            101,
            endpoint=False,
        ),
        np.linspace(
            0.26,
            MAXIMUM_DENSITY,
            230,
        ),
    ]

    return np.unique(
        np.concatenate(sections)
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    direct_free_energy = (
        make_lj_free_energy_function(
            temperature_reduced=TEMPERATURE
        )
    )

    densities = build_density_grid()

    print("=" * 84)
    print("Building LJ residual-free-energy table")
    print("=" * 84)
    print(f"T*                 = {TEMPERATURE}")
    print(f"number of states   = {len(densities)}")
    print(
        "density range      = "
        f"{densities[0]:.6g} to {densities[-1]:.6g}"
    )

    total_free_energies = []
    residual_free_energies = []

    for index, density in enumerate(
        densities,
        start=1,
    ):
        total = float(
            direct_free_energy(
                float(density)
            )
        )

        residual = (
            total
            - ideal_free_energy(
                float(density)
            )
        )

        total_free_energies.append(total)
        residual_free_energies.append(
            residual
        )

        if (
            index == 1
            or index % 50 == 0
            or index == len(densities)
        ):
            print(
                f"  completed {index:4d}/"
                f"{len(densities):4d}  "
                f"rho*={density:.7f}  "
                f"beta a_res={residual:.8f}"
            )

    total_free_energies = np.asarray(
        total_free_energies,
        dtype=np.float64,
    )

    residual_free_energies = np.asarray(
        residual_free_energies,
        dtype=np.float64,
    )

    residual_spline = CubicSpline(
        densities,
        residual_free_energies,
        bc_type="natural",
        extrapolate=False,
    )

    def spline_free_energy(
        reduced_density: float,
    ) -> float:
        rho = float(reduced_density)

        if not (
            densities[0]
            <= rho
            <= densities[-1]
        ):
            raise ValueError(
                "Density outside interpolation table: "
                f"{rho}"
            )

        return float(
            ideal_free_energy(rho)
            + residual_spline(rho)
        )

    table = pd.DataFrame(
        {
            "reduced_density": densities,
            "beta_total_free_energy": (
                total_free_energies
            ),
            "beta_ideal_free_energy": [
                ideal_free_energy(float(rho))
                for rho in densities
            ],
            "beta_residual_free_energy": (
                residual_free_energies
            ),
        }
    )

    table_file = (
        OUTPUT_DIR
        / "free_energy_table.csv"
    )

    table.to_csv(
        table_file,
        index=False,
    )

    # Validate the interpolation at points that were not knots.
    validation_densities = np.array(
        [
            0.004,
            0.025,
            0.075,
            0.15,
            0.197,
            0.215,
            0.242,
            0.31,
            0.55,
            0.81,
        ],
        dtype=np.float64,
    )

    interpolation_records = []

    for density in validation_densities:
        direct_value = float(
            direct_free_energy(
                float(density)
            )
        )

        spline_value = float(
            spline_free_energy(
                float(density)
            )
        )

        interpolation_records.append(
            {
                "reduced_density": density,
                "direct_beta_free_energy": (
                    direct_value
                ),
                "spline_beta_free_energy": (
                    spline_value
                ),
                "absolute_error": abs(
                    direct_value
                    - spline_value
                ),
            }
        )

    interpolation_table = pd.DataFrame(
        interpolation_records
    )

    interpolation_file = (
        OUTPUT_DIR
        / "interpolation_validation.csv"
    )

    interpolation_table.to_csv(
        interpolation_file,
        index=False,
    )

    maximum_interpolation_error = float(
        interpolation_table[
            "absolute_error"
        ].max()
    )

    print("\nInterpolation validation:")
    print(
        interpolation_table.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.10g}"
            ),
        )
    )

    print(
        "\nmaximum interpolation error = "
        f"{maximum_interpolation_error:.6e}"
    )

    spinodals = find_spinodal_densities(
        beta_free_energy_per_particle=(
            spline_free_energy
        ),
        minimum_density=1.0e-4,
        maximum_density=SPINODAL_SEARCH_MAXIMUM_DENSITY,
        grid_points=1800,
    )

    print("\nSpinodal roots:")
    for index, density in enumerate(
        spinodals,
        start=1,
    ):
        print(
            f"  root {index}: rho*={density:.10f}"
        )

    artificial_boundary_roots = [
        root
        for root in spinodals
        if (
            abs(root - 0.20) < 5.0e-3
            or abs(root - 0.25) < 5.0e-3
        )
    ]

    coexistence = solve_fluid_coexistence(
        beta_free_energy_per_particle=(
            spline_free_energy
        ),
        minimum_density=1.0e-4,
        maximum_density=SPINODAL_SEARCH_MAXIMUM_DENSITY,
        grid_points=1800,
    )

    print("\nSpline coexistence result:")
    print(
        "  vapor density             = "
        f"{coexistence.vapor_density:.10f}"
    )
    print(
        "  liquid density            = "
        f"{coexistence.liquid_density:.10f}"
    )
    print(
        "  beta pressure             = "
        f"{coexistence.beta_pressure:.10f}"
    )
    print(
        "  reduced pressure P*       = "
        f"{TEMPERATURE * coexistence.beta_pressure:.10f}"
    )
    print(
        "  beta chemical potential   = "
        f"{coexistence.beta_chemical_potential:.10f}"
    )
    print(
        "  pressure residual         = "
        f"{coexistence.pressure_residual:.6e}"
    )
    print(
        "  chemical-potential resid. = "
        f"{coexistence.chemical_potential_residual:.6e}"
    )
    print(
        "  Maxwell residual          = "
        f"{coexistence.maxwell_area_residual:.6e}"
    )
    print(
        "  optimizer success         = "
        f"{coexistence.optimizer_success}"
    )
    print(
        "  optimizer message         = "
        f"{coexistence.optimizer_message}"
    )

    # Recheck the final coexistence densities with direct
    # perturbation integrations rather than the spline.
    direct_vapor_state = thermodynamic_state(
        beta_free_energy_per_particle=(
            direct_free_energy
        ),
        number_density=(
            coexistence.vapor_density
        ),
    )

    direct_liquid_state = thermodynamic_state(
        beta_free_energy_per_particle=(
            direct_free_energy
        ),
        number_density=(
            coexistence.liquid_density
        ),
    )

    direct_pressure_residual = (
        direct_vapor_state.beta_pressure
        - direct_liquid_state.beta_pressure
    )

    direct_mu_residual = (
        direct_vapor_state.beta_chemical_potential
        - direct_liquid_state.beta_chemical_potential
    )

    print("\nDirect-integration recheck:")
    print(
        "  pressure residual         = "
        f"{direct_pressure_residual:.6e}"
    )
    print(
        "  chemical-potential resid. = "
        f"{direct_mu_residual:.6e}"
    )

    reference_path = (
        ROOT
        / "results"
        / "lj_nist_reference_validation"
        / "nist_lj_cut5_reference_augmented.csv"
    )

    reference = pd.read_csv(
        reference_path
    )

    reference_row = reference.loc[
        np.isclose(
            reference[
                "temperature_reduced"
            ],
            TEMPERATURE,
            rtol=0.0,
            atol=1.0e-12,
        )
    ]

    if len(reference_row) != 1:
        raise RuntimeError(
            "Expected exactly one NIST row at T*=1.0; "
            f"found {len(reference_row)}."
        )

    reference_row = reference_row.iloc[0]

    comparison = {
        "temperature_reduced": TEMPERATURE,
        "theory_vapor_density": (
            coexistence.vapor_density
        ),
        "reference_vapor_density": float(
            reference_row[
                "vapor_density_reduced"
            ]
        ),
        "vapor_density_absolute_error": abs(
            coexistence.vapor_density
            - float(
                reference_row[
                    "vapor_density_reduced"
                ]
            )
        ),
        "theory_liquid_density": (
            coexistence.liquid_density
        ),
        "reference_liquid_density": float(
            reference_row[
                "liquid_density_reduced"
            ]
        ),
        "liquid_density_absolute_error": abs(
            coexistence.liquid_density
            - float(
                reference_row[
                    "liquid_density_reduced"
                ]
            )
        ),
        "theory_pressure_reduced": (
            TEMPERATURE
            * coexistence.beta_pressure
        ),
        "reference_pressure_reduced": float(
            reference_row[
                "saturation_pressure_reduced"
            ]
        ),
        "pressure_absolute_error": abs(
            TEMPERATURE
            * coexistence.beta_pressure
            - float(
                reference_row[
                    "saturation_pressure_reduced"
                ]
            )
        ),
    }

    comparison_file = (
        OUTPUT_DIR
        / "nist_comparison_t1.csv"
    )

    pd.DataFrame(
        [comparison]
    ).to_csv(
        comparison_file,
        index=False,
    )

    print("\nNIST comparison:")
    for key, value in comparison.items():
        print(
            f"  {key:<32s} = {value:.10g}"
        )

    pressure_densities = np.linspace(
        1.0e-4,
        SPINODAL_SEARCH_MAXIMUM_DENSITY,
        700,
    )

    pressure_values = np.array(
        [
            thermodynamic_state(
                spline_free_energy,
                float(density),
            ).beta_pressure
            * TEMPERATURE
            for density in pressure_densities
        ],
        dtype=np.float64,
    )

    figure, axis = plt.subplots(
        figsize=(8.4, 5.7)
    )

    axis.plot(
        pressure_densities,
        pressure_values,
        label=r"Theory $P^*(\rho^*)$",
    )

    axis.axhline(
        TEMPERATURE
        * coexistence.beta_pressure,
        linestyle="--",
        label="Coexistence pressure",
    )

    axis.scatter(
        [
            coexistence.vapor_density,
            coexistence.liquid_density,
        ],
        [
            TEMPERATURE
            * coexistence.beta_pressure,
            TEMPERATURE
            * coexistence.beta_pressure,
        ],
        zorder=4,
        label="Theory binodal",
    )

    for index, spinodal in enumerate(
        spinodals
    ):
        state = thermodynamic_state(
            spline_free_energy,
            spinodal,
        )

        axis.scatter(
            [spinodal],
            [
                TEMPERATURE
                * state.beta_pressure
            ],
            marker="x",
            s=70,
            label=(
                "Spinodal"
                if index == 0
                else None
            ),
        )

    axis.set_xlabel(
        r"Reduced density $\rho^*$"
    )
    axis.set_ylabel(
        r"Reduced pressure $P^*$"
    )
    axis.set_title(
        r"LJ coexistence diagnostic at $T^*=1.0$"
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()

    plot_file = (
        OUTPUT_DIR
        / "pressure_density_t1.png"
    )

    figure.savefig(
        plot_file,
        dpi=240,
    )

    plt.close(figure)

    checks = {
        "interpolation accuracy": (
            maximum_interpolation_error
            < 2.0e-4
        ),
        "exactly two spinodal roots": (
            len(spinodals) == 2
        ),
        "no RDF-boundary spinodal": (
            len(
                artificial_boundary_roots
            )
            == 0
        ),
        "ordered spinodal roots": (
            len(spinodals) == 2
            and spinodals[0]
            < spinodals[1]
        ),
        "ordered coexistence densities": (
            coexistence.vapor_density
            < coexistence.liquid_density
        ),
        "optimizer success": bool(
            coexistence.optimizer_success
        ),
        "spline pressure residual": (
            abs(
                coexistence.pressure_residual
            )
            < 1.0e-6
        ),
        "spline chemical-potential residual": (
            abs(
                coexistence.chemical_potential_residual
            )
            < 1.0e-6
        ),
        "direct pressure residual": (
            abs(
                direct_pressure_residual
            )
            < 5.0e-3
        ),
        "direct chemical-potential residual": (
            abs(
                direct_mu_residual
            )
            < 5.0e-3
        ),
    }

    failed = []

    print("\nDiagnostic checks:")

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

    print("\nGenerated files:")
    print(f"  {table_file}")
    print(f"  {interpolation_file}")
    print(f"  {comparison_file}")
    print(f"  {plot_file}")

    if failed:
        raise RuntimeError(
            "LJ T*=1.0 diagnostic failed: "
            + ", ".join(failed)
        )

    print(
        "\nAll LJ T*=1.0 coexistence "
        "diagnostic checks passed."
    )


if __name__ == "__main__":
    main()
