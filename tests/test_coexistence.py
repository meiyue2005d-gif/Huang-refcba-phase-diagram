"""Tests for thermodynamic and coexistence solvers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from huang_md.coexistence import (
    find_spinodal_densities,
    solve_fluid_coexistence,
    thermodynamic_state,
)


ATTRACTION = 4.0
EXCLUDED_VOLUME = 1.0


def vdw_beta_free_energy_per_particle(
    density: float,
) -> float:
    rho = float(density)
    b = EXCLUDED_VOLUME
    attraction = ATTRACTION

    if rho <= 0.0 or b * rho >= 1.0:
        raise ValueError(
            "Density outside the van der Waals fluid domain."
        )

    return (
        np.log(rho)
        - 1.0
        - np.log(1.0 - b * rho)
        - attraction * rho
    )


def vdw_beta_pressure_analytic(
    density: float,
) -> float:
    rho = float(density)

    return (
        rho
        / (
            1.0
            - EXCLUDED_VOLUME * rho
        )
        - ATTRACTION * rho**2
    )


def vdw_beta_mu_analytic(
    density: float,
) -> float:
    rho = float(density)
    b = EXCLUDED_VOLUME

    return (
        np.log(
            rho / (1.0 - b * rho)
        )
        + (
            b * rho
            / (1.0 - b * rho)
        )
        - 2.0 * ATTRACTION * rho
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
        f"PASS  {name:<42s} "
        f"value={actual:.10g}"
    )


def main() -> None:
    test_density = 0.2

    state = thermodynamic_state(
        vdw_beta_free_energy_per_particle,
        test_density,
    )

    assert_close(
        state.beta_pressure,
        vdw_beta_pressure_analytic(
            test_density
        ),
        1.0e-9,
        "van der Waals pressure",
    )

    assert_close(
        state.beta_chemical_potential,
        vdw_beta_mu_analytic(
            test_density
        ),
        1.0e-9,
        "van der Waals chemical potential",
    )

    spinodals = find_spinodal_densities(
        vdw_beta_free_energy_per_particle,
        minimum_density=1.0e-5,
        maximum_density=0.95,
        grid_points=1200,
    )

    if len(spinodals) != 2:
        raise AssertionError(
            f"Expected two spinodals, found {spinodals}."
        )

    assert_close(
        spinodals[0],
        0.1909830056,
        2.0e-6,
        "lower spinodal density",
    )

    assert_close(
        spinodals[1],
        0.5,
        2.0e-6,
        "upper spinodal density",
    )

    coexistence = solve_fluid_coexistence(
        vdw_beta_free_energy_per_particle,
        minimum_density=1.0e-5,
        maximum_density=0.95,
        grid_points=1200,
    )

    if not coexistence.optimizer_success:
        raise AssertionError(
            coexistence.optimizer_message
        )

    assert_close(
        coexistence.vapor_density,
        0.1028455,
        2.0e-5,
        "coexisting vapor density",
    )

    assert_close(
        coexistence.liquid_density,
        0.6079769,
        2.0e-5,
        "coexisting liquid density",
    )

    assert_close(
        coexistence.beta_pressure,
        0.07232643,
        2.0e-7,
        "coexistence pressure",
    )

    assert_close(
        coexistence.pressure_residual,
        0.0,
        1.0e-8,
        "equal-pressure residual",
    )

    assert_close(
        coexistence.chemical_potential_residual,
        0.0,
        1.0e-8,
        "equal-chemical-potential residual",
    )

    assert_close(
        coexistence.maxwell_area_residual,
        0.0,
        1.0e-7,
        "Maxwell equal-area residual",
    )

    print("\nAll coexistence solver tests passed.")


if __name__ == "__main__":
    main()
