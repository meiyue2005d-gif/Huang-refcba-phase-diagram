#!/usr/bin/env python3
"""Scan truncated-LJ coexistence over all NIST temperatures.

The expensive perturbation integrals are evaluated once at T*=1:

    beta*a1(T*) = beta*a1(1) / T*
    beta*a2(T*) = beta*a2(1) / T*^2

A residual-free-energy spline is then constructed separately
for each temperature.

Internal spinodal roots inside the RDF transition interval
0.20 < rho* < 0.25 are recorded as stitch artifacts. The
outermost pair is treated as the physical spinodal pair.
"""

from __future__ import annotations

import json
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
)
from huang_md.lj_perturbation import (
    calculate_lj_perturbation_free_energy,
)


SEARCH_MINIMUM = 1.0e-4
SEARCH_MAXIMUM = 0.885

BLEND_START = 0.20
BLEND_END = 0.25
BLEND_MARGIN = 0.005

OUTPUT_DIR = (
    ROOT
    / "results"
    / "lj_nist_coexistence_scan"
)

MASTER_TABLE_FILE = (
    OUTPUT_DIR
    / "lj_density_master_t1.csv"
)

REFERENCE_FILE = (
    ROOT
    / "results"
    / "lj_nist_reference_validation"
    / "nist_lj_cut5_reference_augmented.csv"
)


def build_density_grid() -> np.ndarray:
    sections = [
        np.geomspace(
            1.0e-5,
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
            0.89,
            230,
        ),
    ]

    return np.unique(
        np.concatenate(sections)
    )


def build_master_table() -> pd.DataFrame:
    """Evaluate density-dependent terms once at T*=1."""
    densities = build_density_grid()
    records = []

    print("=" * 88)
    print("Building LJ T*=1 master perturbation table")
    print("=" * 88)
    print(f"number of densities = {len(densities)}")

    for index, density in enumerate(
        densities,
        start=1,
    ):
        result = (
            calculate_lj_perturbation_free_energy(
                temperature_reduced=1.0,
                reduced_density=float(density),
            )
        )

        records.append(
            {
                "reduced_density": float(density),
                "packing_fraction": (
                    result.packing_fraction
                ),
                "beta_hard_sphere_excess": (
                    result
                    .beta_hard_sphere_excess_per_particle
                ),
                "beta_a1_t1": (
                    result.beta_a1_per_particle
                ),
                "beta_a2_t1": (
                    result.beta_a2_per_particle
                ),
                "second_to_first_abs_ratio_t1": (
                    result.second_to_first_abs_ratio
                ),
                "first_integral_error": (
                    result.first_integral_error
                ),
                "second_integral_error": (
                    result.second_integral_error
                ),
            }
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
                f"a1={result.beta_a1_per_particle:.8f}  "
                f"a2={result.beta_a2_per_particle:.8f}"
            )

    table = pd.DataFrame.from_records(records)

    table.to_csv(
        MASTER_TABLE_FILE,
        index=False,
    )

    print(
        "\nGenerated master table:\n"
        f"  {MASTER_TABLE_FILE}"
    )

    return table


def load_or_build_master_table() -> pd.DataFrame:
    if MASTER_TABLE_FILE.exists():
        print(
            "Using existing master table:\n"
            f"  {MASTER_TABLE_FILE}"
        )

        return pd.read_csv(
            MASTER_TABLE_FILE
        )

    return build_master_table()


def make_free_energy_function(
    master: pd.DataFrame,
    temperature: float,
):
    densities = master[
        "reduced_density"
    ].to_numpy(dtype=np.float64)

    hard_sphere_excess = master[
        "beta_hard_sphere_excess"
    ].to_numpy(dtype=np.float64)

    first_order = master[
        "beta_a1_t1"
    ].to_numpy(dtype=np.float64) / temperature

    second_order = master[
        "beta_a2_t1"
    ].to_numpy(dtype=np.float64) / temperature**2

    residual = (
        hard_sphere_excess
        + first_order
        + second_order
    )

    spline = CubicSpline(
        densities,
        residual,
        bc_type="natural",
        extrapolate=False,
    )

    def free_energy(
        reduced_density: float,
    ) -> float:
        rho = float(reduced_density)

        if not (
            densities[0]
            <= rho
            <= densities[-1]
        ):
            raise ValueError(
                "Density outside free-energy table: "
                f"{rho}"
            )

        return float(
            log(rho)
            - 1.0
            + spline(rho)
        )

    return free_energy


def scan_temperature(
    master: pd.DataFrame,
    reference_row: pd.Series,
) -> dict[str, object]:
    temperature = float(
        reference_row["temperature_reduced"]
    )

    free_energy = make_free_energy_function(
        master,
        temperature,
    )

    roots = find_spinodal_densities(
        beta_free_energy_per_particle=free_energy,
        minimum_density=SEARCH_MINIMUM,
        maximum_density=SEARCH_MAXIMUM,
        grid_points=1800,
    )

    if len(roots) < 2:
        raise RuntimeError(
            f"T*={temperature}: fewer than two "
            f"spinodal roots found: {roots}"
        )

    physical_roots = (
        float(roots[0]),
        float(roots[-1]),
    )

    stitch_roots = [
        float(root)
        for root in roots[1:-1]
    ]

    stitch_roots_confined = all(
        (
            BLEND_START - BLEND_MARGIN
            <= root
            <= BLEND_END + BLEND_MARGIN
        )
        for root in stitch_roots
    )

    coexistence = solve_fluid_coexistence(
        beta_free_energy_per_particle=free_energy,
        minimum_density=SEARCH_MINIMUM,
        maximum_density=SEARCH_MAXIMUM,
        grid_points=1800,
    )

    theory_vapor = float(
        coexistence.vapor_density
    )

    theory_liquid = float(
        coexistence.liquid_density
    )

    theory_pressure = float(
        temperature
        * coexistence.beta_pressure
    )

    reference_vapor = float(
        reference_row[
            "vapor_density_reduced"
        ]
    )

    reference_liquid = float(
        reference_row[
            "liquid_density_reduced"
        ]
    )

    reference_pressure = float(
        reference_row[
            "saturation_pressure_reduced"
        ]
    )

    return {
        "temperature_reduced": temperature,
        "theory_vapor_density": theory_vapor,
        "reference_vapor_density": reference_vapor,
        "vapor_absolute_error": abs(
            theory_vapor
            - reference_vapor
        ),
        "vapor_relative_error": abs(
            theory_vapor
            - reference_vapor
        ) / reference_vapor,
        "theory_liquid_density": theory_liquid,
        "reference_liquid_density": reference_liquid,
        "liquid_absolute_error": abs(
            theory_liquid
            - reference_liquid
        ),
        "liquid_relative_error": abs(
            theory_liquid
            - reference_liquid
        ) / reference_liquid,
        "theory_pressure_reduced": theory_pressure,
        "reference_pressure_reduced": (
            reference_pressure
        ),
        "pressure_absolute_error": abs(
            theory_pressure
            - reference_pressure
        ),
        "pressure_relative_error": abs(
            theory_pressure
            - reference_pressure
        ) / reference_pressure,
        "physical_vapor_spinodal": (
            physical_roots[0]
        ),
        "physical_liquid_spinodal": (
            physical_roots[1]
        ),
        "all_spinodal_roots": json.dumps(
            [float(root) for root in roots]
        ),
        "rdf_stitch_roots": json.dumps(
            stitch_roots
        ),
        "stitch_roots_confined": (
            stitch_roots_confined
        ),
        "number_of_spinodal_roots": (
            len(roots)
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
        "optimizer_message": (
            coexistence.optimizer_message
        ),
    }


def generate_plots(
    results: pd.DataFrame,
) -> None:
    figure, axis = plt.subplots(
        figsize=(8.3, 5.6)
    )

    axis.semilogy(
        results["temperature_reduced"],
        results["theory_vapor_density"],
        marker="o",
        label="Theory",
    )

    axis.scatter(
        results["temperature_reduced"],
        results["reference_vapor_density"],
        marker="x",
        s=55,
        label="NIST",
    )

    axis.set_xlabel(
        r"Reduced temperature $T^*$"
    )
    axis.set_ylabel(
        r"Vapor density $\rho_v^*$"
    )
    axis.set_title(
        "LJ vapor coexistence branch"
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIR
        / "vapor_branch_comparison.png",
        dpi=240,
    )

    plt.close(figure)

    figure, axis = plt.subplots(
        figsize=(8.3, 5.6)
    )

    axis.plot(
        results["temperature_reduced"],
        results["theory_liquid_density"],
        marker="o",
        label="Theory",
    )

    axis.scatter(
        results["temperature_reduced"],
        results["reference_liquid_density"],
        marker="x",
        s=55,
        label="NIST",
    )

    axis.set_xlabel(
        r"Reduced temperature $T^*$"
    )
    axis.set_ylabel(
        r"Liquid density $\rho_l^*$"
    )
    axis.set_title(
        "LJ liquid coexistence branch"
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIR
        / "liquid_branch_comparison.png",
        dpi=240,
    )

    plt.close(figure)

    figure, axis = plt.subplots(
        figsize=(8.3, 5.6)
    )

    axis.plot(
        results["temperature_reduced"],
        results["theory_pressure_reduced"],
        marker="o",
        label="Theory",
    )

    axis.scatter(
        results["temperature_reduced"],
        results["reference_pressure_reduced"],
        marker="x",
        s=55,
        label="NIST",
    )

    axis.set_xlabel(
        r"Reduced temperature $T^*$"
    )
    axis.set_ylabel(
        r"Saturation pressure $P^*$"
    )
    axis.set_title(
        "LJ saturation-pressure comparison"
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIR
        / "pressure_comparison.png",
        dpi=240,
    )

    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            REFERENCE_FILE
        )

    reference = pd.read_csv(
        REFERENCE_FILE
    ).sort_values(
        "temperature_reduced"
    )

    master = load_or_build_master_table()

    print("\n" + "=" * 112)
    print("Scanning all NIST LJ coexistence temperatures")
    print("=" * 112)

    records = []

    for _, reference_row in reference.iterrows():
        temperature = float(
            reference_row[
                "temperature_reduced"
            ]
        )

        try:
            result = scan_temperature(
                master,
                reference_row,
            )

            records.append(result)

            print(
                f"T*={temperature:4.2f}  "
                f"rho_v={result['theory_vapor_density']:.7f}  "
                f"rho_l={result['theory_liquid_density']:.7f}  "
                f"P*={result['theory_pressure_reduced']:.7f}  "
                f"errors="
                f"{result['vapor_relative_error']:.2%}/"
                f"{result['liquid_relative_error']:.2%}/"
                f"{result['pressure_relative_error']:.2%}  "
                f"roots={result['number_of_spinodal_roots']}"
            )

        except Exception as error:
            records.append(
                {
                    "temperature_reduced": (
                        temperature
                    ),
                    "optimizer_success": False,
                    "failure_message": repr(error),
                }
            )

            print(
                f"T*={temperature:4.2f}  FAILED  "
                f"{error!r}"
            )

    results = pd.DataFrame.from_records(
        records
    )

    results_file = (
        OUTPUT_DIR
        / "lj_nist_coexistence_results.csv"
    )

    results.to_csv(
        results_file,
        index=False,
    )

    successful = results.loc[
        results["optimizer_success"] == True
    ].copy()

    if len(successful) > 0:
        generate_plots(successful)

    print("\n" + "=" * 88)
    print("LJ coexistence scan summary")
    print("=" * 88)
    print(
        f"successful temperatures = "
        f"{len(successful)}/{len(reference)}"
    )

    if len(successful) > 0:
        print(
            "mean relative vapor error    = "
            f"{successful['vapor_relative_error'].mean():.3%}"
        )
        print(
            "mean relative liquid error   = "
            f"{successful['liquid_relative_error'].mean():.3%}"
        )
        print(
            "mean relative pressure error = "
            f"{successful['pressure_relative_error'].mean():.3%}"
        )
        print(
            "max relative vapor error     = "
            f"{successful['vapor_relative_error'].max():.3%}"
        )
        print(
            "max relative liquid error    = "
            f"{successful['liquid_relative_error'].max():.3%}"
        )
        print(
            "max relative pressure error  = "
            f"{successful['pressure_relative_error'].max():.3%}"
        )

        maximum_pressure_residual = float(
            successful[
                "pressure_residual"
            ].abs().max()
        )

        maximum_mu_residual = float(
            successful[
                "chemical_potential_residual"
            ].abs().max()
        )

        print(
            "max pressure residual         = "
            f"{maximum_pressure_residual:.6e}"
        )
        print(
            "max chemical-potential resid. = "
            f"{maximum_mu_residual:.6e}"
        )

    print("\nGenerated files:")
    print(f"  {MASTER_TABLE_FILE}")
    print(f"  {results_file}")
    print(
        f"  {OUTPUT_DIR / 'vapor_branch_comparison.png'}"
    )
    print(
        f"  {OUTPUT_DIR / 'liquid_branch_comparison.png'}"
    )
    print(
        f"  {OUTPUT_DIR / 'pressure_comparison.png'}"
    )

    checks = {
        "all temperatures solved":
            len(successful) == len(reference),

        "all coexistence densities ordered":
            (
                successful[
                    "theory_vapor_density"
                ]
                < successful[
                    "theory_liquid_density"
                ]
            ).all()
            if len(successful) > 0
            else False,

        "all optimizers successful":
            bool(
                successful[
                    "optimizer_success"
                ].all()
            )
            if len(successful) > 0
            else False,

        "pressure residuals controlled":
            bool(
                (
                    successful[
                        "pressure_residual"
                    ].abs()
                    < 1.0e-5
                ).all()
            )
            if len(successful) > 0
            else False,

        "chemical-potential residuals controlled":
            bool(
                (
                    successful[
                        "chemical_potential_residual"
                    ].abs()
                    < 1.0e-5
                ).all()
            )
            if len(successful) > 0
            else False,
    }

    failed = []

    print("\nNumerical checks:")

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status:<4s} {name}")

        if not passed:
            failed.append(name)

    if failed:
        raise RuntimeError(
            "LJ NIST coexistence scan failed: "
            + ", ".join(failed)
        )

    print(
        "\nAll 15 LJ coexistence temperatures "
        "were solved successfully."
    )


if __name__ == "__main__":
    main()
