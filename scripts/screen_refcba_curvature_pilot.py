#!/usr/bin/env python3
"""Pilot pH--NaCl thermodynamic-curvature screen for refCBA.

This script scans:

    pH = 3.0, 4.5, 4.8852, 6.0, 9.0
    added NaCl = 0, 100, 500 mM
    concentration = 0.1--20 mg/mL

The output is deliberately labeled as a candidate
thermodynamic signal because the second-order perturbation
expansion may be uncontrolled.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from huang_md.refcba_thermodynamics import (
    calculate_refcba_free_energy_point,
    load_refcba_configuration,
    reduced_density_to_concentration,
    state_parameters,
)


PH_VALUES = [
    3.0,
    4.5,
    4.8852,
    6.0,
    9.0,
]

NACL_VALUES_MM = [
    0.0,
    100.0,
    500.0,
]

MINIMUM_CONCENTRATION = 0.1
MAXIMUM_CONCENTRATION = 20.0

OUTPUT_DIR = (
    ROOT
    / "results"
    / "refcba_curvature_pilot"
)


def build_concentration_grid() -> np.ndarray:
    """Use extra resolution in the low-concentration region."""
    low = np.geomspace(
        MINIMUM_CONCENTRATION,
        2.0,
        25,
        endpoint=False,
    )

    high = np.linspace(
        2.0,
        MAXIMUM_CONCENTRATION,
        37,
    )

    return np.unique(
        np.concatenate([low, high])
    )


def find_roots(
    function,
    minimum_density: float,
    maximum_density: float,
    grid_points: int = 2400,
) -> tuple[float, ...]:
    grid = np.linspace(
        minimum_density,
        maximum_density,
        grid_points,
    )

    values = np.asarray(
        [
            function(float(value))
            for value in grid
        ],
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
                    xtol=1.0e-12,
                    rtol=1.0e-11,
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


def analyze_state(
    configuration,
    pH: float,
    added_nacl_mM: float,
    concentrations: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    params = state_parameters(
        configuration,
        pH=pH,
        added_nacl_mM=added_nacl_mM,
    )

    point_records: list[dict[str, object]] = []

    for concentration in concentrations:
        point = calculate_refcba_free_energy_point(
            configuration=configuration,
            pH=pH,
            added_nacl_mM=added_nacl_mM,
            concentration_mg_ml=float(
                concentration
            ),
        )

        beta_ideal = (
            np.log(point.number_density_nm3)
            - 1.0
        )

        beta_residual = (
            point.beta_total_free_energy_per_particle
            - beta_ideal
        )

        point_records.append(
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

    table = pd.DataFrame.from_records(
        point_records
    ).sort_values(
        "reduced_density_rho_sigma3"
    )

    rho_star = table[
        "reduced_density_rho_sigma3"
    ].to_numpy(dtype=np.float64)

    residual = table[
        "beta_residual_free_energy_per_particle"
    ].to_numpy(dtype=np.float64)

    if not (
        np.isfinite(rho_star).all()
        and np.isfinite(residual).all()
    ):
        raise FloatingPointError(
            "Non-finite spline input generated."
        )

    spline = CubicSpline(
        rho_star,
        residual,
        extrapolate=False,
    )

    first_derivative = spline.derivative(1)
    second_derivative = spline.derivative(2)

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

    roots = find_roots(
        pressure_derivative,
        float(rho_star[0]),
        float(rho_star[-1]),
    )

    root_concentrations = [
        reduced_density_to_concentration(
            reduced_density=float(root),
            molecular_weight_kDa=(
                configuration.molecular_weight_kDa
            ),
            hard_sphere_diameter_nm=(
                params.diameter_nm
            ),
        )
        for root in roots
    ]

    evaluation_density = np.linspace(
        float(rho_star[0]),
        float(rho_star[-1]),
        1000,
    )

    derivative_values = np.asarray(
        [
            pressure_derivative(
                float(value)
            )
            for value in evaluation_density
        ],
        dtype=np.float64,
    )

    minimum_index = int(
        np.argmin(derivative_values)
    )

    minimum_derivative = float(
        derivative_values[minimum_index]
    )

    minimum_derivative_concentration = (
        reduced_density_to_concentration(
            reduced_density=float(
                evaluation_density[
                    minimum_index
                ]
            ),
            molecular_weight_kDa=(
                configuration.molecular_weight_kDa
            ),
            hard_sphere_diameter_nm=(
                params.diameter_nm
            ),
        )
    )

    minimum_ratio = float(
        table[
            "second_to_first_abs_ratio"
        ].min()
    )

    maximum_ratio = float(
        table[
            "second_to_first_abs_ratio"
        ].max()
    )

    all_uncontrolled = bool(
        (
            table["perturbation_status"]
            == "uncontrolled"
        ).all()
    )

    if minimum_derivative >= 0.0:
        signal = (
            "no_spinodal_in_requested_window"
        )

    elif len(roots) >= 2:
        signal = (
            "candidate_internal_instability"
        )

    elif len(roots) == 1:
        signal = (
            "candidate_boundary_limited_instability"
        )

    else:
        signal = (
            "candidate_full_window_instability"
        )

    if all_uncontrolled:
        interpretation = (
            signal + "_uncontrolled"
        )
    else:
        interpretation = signal

    summary = {
        "pH": float(pH),
        "added_nacl_mM": float(
            added_nacl_mM
        ),
        "total_ionic_strength_mM": float(
            configuration
            .state_model
            .background_ionic_strength_mM
            + added_nacl_mM
        ),
        "K1_kBT": float(params.K1_kBT),
        "Z1": float(params.Z1),
        "K2_kBT": float(params.K2_kBT),
        "Z2": float(params.Z2),
        "minimum_pressure_derivative": (
            minimum_derivative
        ),
        "minimum_derivative_concentration_mg_ml": (
            minimum_derivative_concentration
        ),
        "number_of_spinodal_roots": (
            len(roots)
        ),
        "spinodal_roots_rho_star": (
            json.dumps(
                [
                    float(root)
                    for root in roots
                ]
            )
        ),
        "spinodal_roots_concentration_mg_ml": (
            json.dumps(
                [
                    float(value)
                    for value
                    in root_concentrations
                ]
            )
        ),
        "lower_spinodal_concentration_mg_ml": (
            float(root_concentrations[0])
            if len(root_concentrations) >= 1
            else np.nan
        ),
        "upper_spinodal_concentration_mg_ml": (
            float(root_concentrations[-1])
            if len(root_concentrations) >= 2
            else np.nan
        ),
        "minimum_second_to_first_abs_ratio": (
            minimum_ratio
        ),
        "maximum_second_to_first_abs_ratio": (
            maximum_ratio
        ),
        "all_points_uncontrolled": (
            all_uncontrolled
        ),
        "thermodynamic_signal": signal,
        "interpretation": interpretation,
        "status": "success",
        "failure_message": "",
    }

    return summary, point_records


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    configuration = load_refcba_configuration(
        ROOT
    )

    concentrations = (
        build_concentration_grid()
    )

    summaries: list[dict[str, object]] = []
    all_points: list[dict[str, object]] = []

    total_states = (
        len(PH_VALUES)
        * len(NACL_VALUES_MM)
    )

    print("=" * 126)
    print(
        "refCBA pilot pH--NaCl "
        "thermodynamic-curvature screen"
    )
    print("=" * 126)
    print(
        f"states              = {total_states}"
    )
    print(
        f"concentrations/state = "
        f"{len(concentrations)}"
    )
    print(
        f"range               = "
        f"{MINIMUM_CONCENTRATION:g}--"
        f"{MAXIMUM_CONCENTRATION:g} mg/mL"
    )
    print(
        "NOTE: Every instability is a candidate "
        "until validated by MD."
    )

    state_index = 0

    for added_nacl_mM in NACL_VALUES_MM:
        for pH in PH_VALUES:
            state_index += 1

            print(
                f"\n[{state_index:02d}/{total_states:02d}] "
                f"pH={pH:.4g}, "
                f"NaCl={added_nacl_mM:.0f} mM"
            )

            try:
                summary, point_records = (
                    analyze_state(
                        configuration=configuration,
                        pH=float(pH),
                        added_nacl_mM=float(
                            added_nacl_mM
                        ),
                        concentrations=(
                            concentrations
                        ),
                    )
                )

                summaries.append(summary)
                all_points.extend(
                    point_records
                )

                roots_text = summary[
                    "spinodal_roots_concentration_mg_ml"
                ]

                print(
                    "  "
                    f"K2={summary['K2_kBT']:.6g}, "
                    f"Z2={summary['Z2']:.6g}, "
                    f"min dP/drho="
                    f"{summary['minimum_pressure_derivative']:.6g}, "
                    f"roots={roots_text}, "
                    f"ratio="
                    f"{summary['minimum_second_to_first_abs_ratio']:.4g}"
                    f"--"
                    f"{summary['maximum_second_to_first_abs_ratio']:.4g}"
                )

                print(
                    "  signal="
                    f"{summary['interpretation']}"
                )

            except Exception as error:
                summaries.append(
                    {
                        "pH": float(pH),
                        "added_nacl_mM": float(
                            added_nacl_mM
                        ),
                        "status": "failed",
                        "failure_message": repr(
                            error
                        ),
                    }
                )

                print(
                    f"  FAILED: {error!r}"
                )

    summary_table = pd.DataFrame.from_records(
        summaries
    )

    point_table = pd.DataFrame.from_records(
        all_points
    )

    summary_file = (
        OUTPUT_DIR
        / "pilot_curvature_summary.csv"
    )

    point_file = (
        OUTPUT_DIR
        / "pilot_free_energy_points.csv"
    )

    summary_table.to_csv(
        summary_file,
        index=False,
    )

    point_table.to_csv(
        point_file,
        index=False,
    )

    successful = summary_table.loc[
        summary_table["status"]
        == "success"
    ].copy()

    failed = summary_table.loc[
        summary_table["status"]
        != "success"
    ].copy()

    print("\n" + "=" * 126)
    print("Pilot screen summary")
    print("=" * 126)

    display_columns = [
        "pH",
        "added_nacl_mM",
        "K2_kBT",
        "Z2",
        "minimum_pressure_derivative",
        "lower_spinodal_concentration_mg_ml",
        "upper_spinodal_concentration_mg_ml",
        "minimum_second_to_first_abs_ratio",
        "maximum_second_to_first_abs_ratio",
        "interpretation",
    ]

    if len(successful) > 0:
        print(
            successful[
                display_columns
            ].to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.7g}"
                ),
            )
        )

    print(
        "\nSignal counts:"
    )

    if len(successful) > 0:
        print(
            successful[
                "interpretation"
            ].value_counts().to_string()
        )
    else:
        print("  none")

    print(
        f"\nSuccessful states: "
        f"{len(successful)}/{total_states}"
    )

    print(
        f"Failed states:     "
        f"{len(failed)}/{total_states}"
    )

    print("\nGenerated files:")
    print(f"  {summary_file}")
    print(f"  {point_file}")

    if len(failed) > 0:
        print("\nFailures:")
        print(
            failed[
                [
                    "pH",
                    "added_nacl_mM",
                    "failure_message",
                ]
            ].to_string(index=False)
        )

        raise RuntimeError(
            "One or more pilot states failed."
        )

    if len(successful) != total_states:
        raise RuntimeError(
            "Pilot output contains an unexpected "
            "number of successful states."
        )

    print(
        "\nAll refCBA pilot curvature states completed."
    )
    print(
        "WARNING: Candidate spinodals remain "
        "non-quantitative wherever the perturbation "
        "expansion is uncontrolled."
    )


if __name__ == "__main__":
    main()
