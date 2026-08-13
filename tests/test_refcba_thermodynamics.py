"""Tests for refCBA concentration and free-energy utilities."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from huang_md.refcba_thermodynamics import (
    LOW_DENSITY_RDF_LIMIT,
    calculate_refcba_free_energy_point,
    concentration_to_number_density_nm3,
    concentration_to_reduced_density,
    load_refcba_configuration,
    number_density_nm3_to_concentration,
    state_parameters,
)


def assert_close(
    actual: float,
    expected: float,
    tolerance: float,
    name: str,
) -> None:
    error = abs(actual - expected)

    if error > tolerance:
        raise AssertionError(
            f"{name}: expected={expected:.12g}, "
            f"actual={actual:.12g}, "
            f"error={error:.3e}"
        )

    print(
        f"PASS  {name:<47s} "
        f"value={actual:.10g}"
    )


def main() -> None:
    configuration = load_refcba_configuration(
        ROOT
    )

    for concentration in [
        0.1,
        0.436,
        1.0,
        10.0,
        20.0,
    ]:
        number_density = (
            concentration_to_number_density_nm3(
                concentration,
                configuration.molecular_weight_kDa,
            )
        )

        recovered = (
            number_density_nm3_to_concentration(
                number_density,
                configuration.molecular_weight_kDa,
            )
        )

        assert_close(
            recovered,
            concentration,
            1.0e-12,
            (
                "concentration round trip "
                f"{concentration:g} mg/mL"
            ),
        )

    params = state_parameters(
        configuration,
        pH=4.5,
        added_nacl_mM=0.0,
    )

    assert_close(
        params.K2_kBT,
        53.056,
        1.0e-12,
        "reference-state K2",
    )

    assert_close(
        params.Z2,
        1.483,
        1.0e-12,
        "reference-state Z2",
    )

    maximum_rho_star = (
        concentration_to_reduced_density(
            concentration_mg_ml=20.0,
            molecular_weight_kDa=(
                configuration.molecular_weight_kDa
            ),
            hard_sphere_diameter_nm=(
                params.diameter_nm
            ),
        )
    )

    print(
        "PASS  rho* at 20 mg/mL"
        f"{'':<26s} "
        f"value={maximum_rho_star:.10g}"
    )

    if not (
        0.117
        < maximum_rho_star
        < 0.118
    ):
        raise AssertionError(
            "Unexpected reduced density at 20 mg/mL: "
            f"{maximum_rho_star}"
        )

    if maximum_rho_star >= LOW_DENSITY_RDF_LIMIT:
        raise AssertionError(
            "Target concentration range exceeds "
            "the low-density RDF domain."
        )

    print(
        "PASS  target domain remains below rho*=0.2"
    )

    point = calculate_refcba_free_energy_point(
        configuration=configuration,
        pH=4.5,
        added_nacl_mM=0.0,
        concentration_mg_ml=1.0,
    )

    values = np.asarray(
        [
            point.number_density_nm3,
            point.reduced_density_rho_sigma3,
            point.beta_a1_per_particle,
            point.beta_a2_per_particle,
            point.beta_total_free_energy_per_particle,
            point.second_to_first_abs_ratio,
        ],
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise AssertionError(
            "The reference free-energy point is non-finite."
        )

    print(
        "PASS  finite reference free-energy point"
    )

    print(
        "PASS  perturbation diagnostic status"
        f"{'':<17s} "
        f"value={point.perturbation_status}"
    )

    print(
        "\nAll refCBA thermodynamic utility tests passed."
    )


if __name__ == "__main__":
    main()
