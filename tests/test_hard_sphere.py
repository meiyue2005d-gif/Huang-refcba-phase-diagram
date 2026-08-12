"""Tests for Carnahan-Starling hard-sphere thermodynamics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from huang_md.hard_sphere import (
    cs_beta_chemical_potential,
    cs_beta_helmholtz_free_energy_per_particle,
    cs_beta_pressure,
    cs_compressibility_factor,
    cs_d_beta_pressure_d_density,
    cs_excess_chemical_potential,
    cs_excess_free_energy_per_particle,
    cs_reduced_isothermal_compressibility,
    number_density_from_packing_fraction,
    packing_fraction,
)


def assert_close(
    actual: float,
    expected: float,
    tolerance: float,
    name: str,
) -> None:
    difference = abs(actual - expected)

    if difference > tolerance:
        raise AssertionError(
            f"{name}: expected {expected:.12g}, "
            f"obtained {actual:.12g}, "
            f"difference={difference:.3e}"
        )

    print(
        f"PASS  {name:<42s} "
        f"value={actual:.10g}"
    )


def scalar(value) -> float:
    return float(
        np.asarray(value).reshape(-1)[0]
    )


def main() -> None:
    diameter_nm = 4.278
    eta = 0.1

    density = scalar(
        number_density_from_packing_fraction(
            eta,
            diameter_nm,
        )
    )

    recovered_eta = scalar(
        packing_fraction(
            density,
            diameter_nm,
        )
    )

    assert_close(
        recovered_eta,
        eta,
        1.0e-14,
        "packing-fraction round trip",
    )

    z_value = scalar(
        cs_compressibility_factor(eta)
    )

    assert_close(
        z_value,
        1.5212620027434842,
        1.0e-12,
        "Carnahan-Starling Z at eta=0.1",
    )

    excess_free_energy = scalar(
        cs_excess_free_energy_per_particle(
            eta
        )
    )

    assert_close(
        excess_free_energy,
        0.4567901234567901,
        1.0e-12,
        "CS excess free energy at eta=0.1",
    )

    excess_mu = scalar(
        cs_excess_chemical_potential(
            eta
        )
    )

    assert_close(
        excess_mu,
        0.9780521262002742,
        1.0e-12,
        "CS excess chemical potential at eta=0.1",
    )

    # Thermodynamic identity:
    # mu_ex = a_ex + Z - 1.
    assert_close(
        excess_free_energy + z_value - 1.0,
        excess_mu,
        1.0e-12,
        "mu_ex = a_ex + Z - 1",
    )

    pressure = scalar(
        cs_beta_pressure(
            density,
            diameter_nm,
        )
    )

    assert_close(
        pressure / density,
        z_value,
        1.0e-12,
        "beta P / rho = Z",
    )

    analytic_derivative = scalar(
        cs_d_beta_pressure_d_density(
            eta
        )
    )

    density_step = density * 1.0e-5

    pressure_plus = scalar(
        cs_beta_pressure(
            density + density_step,
            diameter_nm,
        )
    )

    pressure_minus = scalar(
        cs_beta_pressure(
            density - density_step,
            diameter_nm,
        )
    )

    numerical_derivative = (
        pressure_plus
        - pressure_minus
    ) / (2.0 * density_step)

    assert_close(
        numerical_derivative,
        analytic_derivative,
        2.0e-9,
        "pressure derivative",
    )

    reduced_compressibility = scalar(
        cs_reduced_isothermal_compressibility(
            eta
        )
    )

    assert_close(
        reduced_compressibility
        * analytic_derivative,
        1.0,
        1.0e-12,
        "compressibility inverse derivative",
    )

    free_energy = scalar(
        cs_beta_helmholtz_free_energy_per_particle(
            density,
            diameter_nm,
        )
    )

    chemical_potential = scalar(
        cs_beta_chemical_potential(
            density,
            diameter_nm,
        )
    )

    # mu = a + P/rho.
    assert_close(
        free_energy
        + pressure / density,
        chemical_potential,
        1.0e-12,
        "mu = a + P/rho",
    )

    dilute_eta = 1.0e-10

    assert_close(
        scalar(
            cs_compressibility_factor(
                dilute_eta
            )
        ),
        1.0,
        1.0e-9,
        "ideal-gas limit of Z",
    )

    assert_close(
        scalar(
            cs_reduced_isothermal_compressibility(
                dilute_eta
            )
        ),
        1.0,
        1.0e-9,
        "ideal-gas compressibility limit",
    )

    print(
        "\nAll hard-sphere thermodynamic tests passed."
    )


if __name__ == "__main__":
    main()
