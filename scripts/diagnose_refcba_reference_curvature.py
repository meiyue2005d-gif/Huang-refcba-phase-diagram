#!/usr/bin/env python3
"""Diagnose refCBA thermodynamic curvature at pH 4.5, NaCl 0 mM.

This script scans the requested concentration window:

    0.1 <= c <= 20 mg/mL

The residual Helmholtz free energy is interpolated as a function
of reduced density rho* = rho * sigma^3.

For beta*a(rho*):

    beta*P*sigma^3 = rho* + rho*^2 * d(a_res)/d(rho*)

and

    d(beta*P*sigma^3)/d(rho*)
        = 1 + 2*rho* a_res' + rho*^2 a_res''

Negative pressure derivative indicates a candidate spinodal
instability. Because |a2/a1| > 1 throughout this state, any
instability found here is only a thermodynamic candidate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from huang_md.refcba_thermodynamics import (
    calculate_refcba_free_energy_point,
    load_refcba_configuration,
    reduced_density_to_concentration,
)


PH_VALUE = 4.5
ADDED_NACL_MM = 0.0

MINIMUM_CONCENTRATION = 0.1
MAXIMUM_CONCENTRATION = 20.0

OUTPUT_DIR = (
    ROOT
    / "results"
    / "refcba_reference_state_curvature"
)


def build_concentration_grid() -> np.ndarray:
    """Dense grid with additional resolution at low concentration."""
    low_section = np.geomspace(
        MINIMUM_CONCENTRATION,
        2.0,
        50,
        endpoint=False,
    )

    high_section = np.linspace(
        2.0,
        MAXIMUM_CONCENTRATION,
        91,
    )

    return np.unique(
        np.concatenate(
            [low_section, high_section]
        )
    )


def find_roots(
    function,
    minimum_density: float,
    maximum_density: float,
    grid_points: int = 4000,
) -> tuple[float, ...]:
    """Locate all sign-changing roots in the requested interval."""
    grid = np.linspace(
        minimum_density,
        maximum_density,
        grid_points,
    )

    values = np.asarray(
        [function(float(value)) for value in grid],
        dtype=np.float64,
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
            root = float(left)

        elif f_left * f_right < 0.0:
            root = float(
                brentq(
                    function,
                    float(left),
                    float(right),
                    xtol=1.0e-13,
                    rtol=1.0e-12,
                )
            )

        else:
            continue

        if (
            not roots
            or abs(root - roots[-1]) > 1.0e-7
        ):
            roots.append(root)

    return tuple(roots)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    configuration = load_refcba_configuration(
        ROOT
    )

    concentrations = build_concentration_grid()

    records = []

    print("=" * 92)
    print("refCBA reference-state free-energy curvature diagnostic")
    print("=" * 92)
    print(f"pH                  = {PH_VALUE}")
    print(
        f"added NaCl          = "
        f"{ADDED_NACL_MM:g} mM"
    )
    print(
        f"concentration range = "
        f"{MINIMUM_CONCENTRATION:g} to "
        f"{MAXIMUM_CONCENTRATION:g} mg/mL"
    )
    print(
        f"number of states    = "
        f"{len(concentrations)}"
    )

    for index, concentration in enumerate(
        concentrations,
        start=1,
    ):
        point = calculate_refcba_free_energy_point(
            configuration=configuration,
            pH=PH_VALUE,
            added_nacl_mM=ADDED_NACL_MM,
            concentration_mg_ml=float(
                concentration
            ),
        )

        # calculate_perturbation_free_energy used a thermal
        # wavelength of 1 nm, so the ideal term is:
        #
        # beta*a_id = ln(rho_nm^-3) - 1
        beta_ideal = (
            np.log(point.number_density_nm3)
            - 1.0
        )

        beta_residual = (
            point.beta_total_free_energy_per_particle
            - beta_ideal
        )

        records.append(
            {
                **point.__dict__,
                "beta_ideal_free_energy_per_particle": (
                    beta_ideal
                ),
                "beta_residual_free_energy_per_particle": (
                    beta_residual
                ),
            }
        )

        if (
            index == 1
            or index % 20 == 0
            or index == len(concentrations)
        ):
            print(
                f"  completed {index:3d}/"
                f"{len(concentrations):3d}  "
                f"c={concentration:9.5f} mg/mL  "
                f"rho*="
                f"{point.reduced_density_rho_sigma3:.8f}  "
                f"a_res={beta_residual:.8g}  "
                f"|a2/a1|="
                f"{point.second_to_first_abs_ratio:.6g}"
            )

    table = pd.DataFrame.from_records(
        records
    ).sort_values(
        "reduced_density_rho_sigma3"
    )

    rho_star = table[
        "reduced_density_rho_sigma3"
    ].to_numpy(dtype=np.float64)

    residual_free_energy = table[
        "beta_residual_free_energy_per_particle"
    ].to_numpy(dtype=np.float64)

    residual_spline = CubicSpline(
        rho_star,
        residual_free_energy,
        extrapolate=False,
    )

    first_derivative = residual_spline.derivative(1)
    second_derivative = residual_spline.derivative(2)

    def beta_pressure_sigma3(
        density: float,
    ) -> float:
        value = float(density)

        return float(
            value
            + value**2
            * first_derivative(value)
        )

    def pressure_derivative(
        density: float,
    ) -> float:
        value = float(density)

        return float(
            1.0
            + 2.0
            * value
            * first_derivative(value)
            + value**2
            * second_derivative(value)
        )

    def beta_chemical_potential(
        density: float,
    ) -> float:
        value = float(density)

        number_density_nm3 = (
            value
            / configuration.baseline.diameter_nm**3
        )

        residual = float(
            residual_spline(value)
        )

        return float(
            np.log(number_density_nm3)
            + residual
            + value
            * first_derivative(value)
        )

    spinodal_roots = find_roots(
        pressure_derivative,
        float(rho_star[0]),
        float(rho_star[-1]),
    )

    evaluation_density = np.linspace(
        float(rho_star[0]),
        float(rho_star[-1]),
        1200,
    )

    evaluation_concentration = np.asarray(
        [
            reduced_density_to_concentration(
                reduced_density=float(value),
                molecular_weight_kDa=(
                    configuration.molecular_weight_kDa
                ),
                hard_sphere_diameter_nm=(
                    configuration.baseline.diameter_nm
                ),
            )
            for value in evaluation_density
        ],
        dtype=np.float64,
    )

    pressure_values = np.asarray(
        [
            beta_pressure_sigma3(float(value))
            for value in evaluation_density
        ],
        dtype=np.float64,
    )

    pressure_derivative_values = np.asarray(
        [
            pressure_derivative(float(value))
            for value in evaluation_density
        ],
        dtype=np.float64,
    )

    chemical_potential_values = np.asarray(
        [
            beta_chemical_potential(float(value))
            for value in evaluation_density
        ],
        dtype=np.float64,
    )

    diagnostic_table = pd.DataFrame(
        {
            "concentration_mg_ml": (
                evaluation_concentration
            ),
            "reduced_density_rho_sigma3": (
                evaluation_density
            ),
            "beta_pressure_sigma3": (
                pressure_values
            ),
            "pressure_derivative": (
                pressure_derivative_values
            ),
            "beta_chemical_potential": (
                chemical_potential_values
            ),
        }
    )

    free_energy_file = (
        OUTPUT_DIR
        / "reference_state_free_energy_dense.csv"
    )

    curvature_file = (
        OUTPUT_DIR
        / "reference_state_curvature.csv"
    )

    summary_file = (
        OUTPUT_DIR
        / "reference_state_curvature_summary.json"
    )

    table.to_csv(
        free_energy_file,
        index=False,
    )

    diagnostic_table.to_csv(
        curvature_file,
        index=False,
    )

    spinodal_concentrations = [
        reduced_density_to_concentration(
            reduced_density=float(root),
            molecular_weight_kDa=(
                configuration.molecular_weight_kDa
            ),
            hard_sphere_diameter_nm=(
                configuration.baseline.diameter_nm
            ),
        )
        for root in spinodal_roots
    ]

    minimum_derivative_index = int(
        np.argmin(
            pressure_derivative_values
        )
    )

    minimum_pressure_derivative = float(
        pressure_derivative_values[
            minimum_derivative_index
        ]
    )

    minimum_derivative_concentration = float(
        evaluation_concentration[
            minimum_derivative_index
        ]
    )

    has_negative_compressibility = bool(
        minimum_pressure_derivative < 0.0
    )

    if has_negative_compressibility:
        thermodynamic_signal = (
            "candidate_instability_in_requested_window"
        )
    else:
        thermodynamic_signal = (
            "no_spinodal_instability_in_requested_window"
        )

    maximum_ratio = float(
        table[
            "second_to_first_abs_ratio"
        ].max()
    )

    minimum_ratio = float(
        table[
            "second_to_first_abs_ratio"
        ].min()
    )

    summary = {
        "pH": PH_VALUE,
        "added_nacl_mM": ADDED_NACL_MM,
        "minimum_concentration_mg_ml": (
            MINIMUM_CONCENTRATION
        ),
        "maximum_concentration_mg_ml": (
            MAXIMUM_CONCENTRATION
        ),
        "spinodal_roots_reduced_density": [
            float(root)
            for root in spinodal_roots
        ],
        "spinodal_roots_concentration_mg_ml": [
            float(value)
            for value in spinodal_concentrations
        ],
        "minimum_pressure_derivative": (
            minimum_pressure_derivative
        ),
        "minimum_derivative_concentration_mg_ml": (
            minimum_derivative_concentration
        ),
        "has_negative_compressibility": (
            has_negative_compressibility
        ),
        "thermodynamic_signal": (
            thermodynamic_signal
        ),
        "minimum_second_to_first_abs_ratio": (
            minimum_ratio
        ),
        "maximum_second_to_first_abs_ratio": (
            maximum_ratio
        ),
        "perturbation_control": "uncontrolled",
        "interpretation": (
            "Candidate signal only; requires MD validation."
        ),
    }

    summary_file.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    figure, axis = plt.subplots(
        figsize=(8.4, 5.6)
    )

    axis.plot(
        evaluation_concentration,
        pressure_values,
    )

    for concentration in spinodal_concentrations:
        axis.axvline(
            concentration,
            linestyle="--",
        )

    axis.set_xlabel(
        "Concentration (mg/mL)"
    )
    axis.set_ylabel(
        r"$\beta P\sigma^3$"
    )
    axis.set_title(
        "refCBA reference-state pressure curve"
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()

    pressure_plot = (
        OUTPUT_DIR
        / "reference_state_pressure.png"
    )

    figure.savefig(
        pressure_plot,
        dpi=240,
    )

    plt.close(figure)

    figure, axis = plt.subplots(
        figsize=(8.4, 5.6)
    )

    axis.plot(
        evaluation_concentration,
        pressure_derivative_values,
    )

    axis.axhline(
        0.0,
        linestyle="--",
    )

    for concentration in spinodal_concentrations:
        axis.axvline(
            concentration,
            linestyle=":",
        )

    axis.set_xlabel(
        "Concentration (mg/mL)"
    )
    axis.set_ylabel(
        r"$d(\beta P\sigma^3)/d\rho^*$"
    )
    axis.set_title(
        "refCBA reference-state thermodynamic curvature"
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()

    curvature_plot = (
        OUTPUT_DIR
        / "reference_state_pressure_derivative.png"
    )

    figure.savefig(
        curvature_plot,
        dpi=240,
    )

    plt.close(figure)

    print("\n" + "=" * 92)
    print("Curvature diagnostic summary")
    print("=" * 92)

    print(
        "minimum pressure derivative = "
        f"{minimum_pressure_derivative:.10g}"
    )
    print(
        "minimum occurs at           = "
        f"{minimum_derivative_concentration:.10g} mg/mL"
    )

    print("\nSpinodal roots inside 0.1--20 mg/mL:")

    if spinodal_roots:
        for index, (
            root,
            concentration,
        ) in enumerate(
            zip(
                spinodal_roots,
                spinodal_concentrations,
            ),
            start=1,
        ):
            print(
                f"  root {index}: "
                f"rho*={root:.10g}, "
                f"c={concentration:.10g} mg/mL"
            )
    else:
        print("  none")

    print(
        "\nThermodynamic signal:"
    )
    print(
        f"  {thermodynamic_signal}"
    )

    print(
        "\nPerturbation ratio range:"
    )
    print(
        f"  minimum |a2/a1| = {minimum_ratio:.8g}"
    )
    print(
        f"  maximum |a2/a1| = {maximum_ratio:.8g}"
    )
    print(
        "  status          = uncontrolled"
    )

    checks = {
        "all spline inputs finite": bool(
            np.isfinite(
                residual_free_energy
            ).all()
        ),
        "all diagnostic outputs finite": bool(
            np.isfinite(
                diagnostic_table.select_dtypes(
                    include=[np.number]
                ).to_numpy()
            ).all()
        ),
        "ordered concentration grid": bool(
            (
                np.diff(concentrations)
                > 0.0
            ).all()
        ),
        "all states below rho*=0.2": bool(
            (
                rho_star < 0.2
            ).all()
        ),
        "all spinodal roots in domain": bool(
            all(
                rho_star[0]
                <= root
                <= rho_star[-1]
                for root in spinodal_roots
            )
        ),
    }

    failed = []

    print("\nNumerical checks:")

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status:<4s} {name}")

        if not passed:
            failed.append(name)

    print("\nGenerated files:")
    print(f"  {free_energy_file}")
    print(f"  {curvature_file}")
    print(f"  {summary_file}")
    print(f"  {pressure_plot}")
    print(f"  {curvature_plot}")

    if failed:
        raise RuntimeError(
            "Reference curvature diagnostic failed: "
            + ", ".join(failed)
        )

    print(
        "\nReference-state curvature diagnostic completed."
    )
    print(
        "WARNING: The perturbation expansion is uncontrolled; "
        "the result is a candidate thermodynamic signal only."
    )


if __name__ == "__main__":
    main()
