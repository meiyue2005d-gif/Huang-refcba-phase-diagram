"""Carnahan-Starling hard-sphere reference thermodynamics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _as_array(values: ArrayLike) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


def _validate_packing_fraction(
    packing_fraction: FloatArray,
) -> None:
    if np.any(~np.isfinite(packing_fraction)):
        raise ValueError(
            "Packing fraction must contain only finite values."
        )

    if np.any(packing_fraction < 0.0):
        raise ValueError(
            "Packing fraction cannot be negative."
        )

    if np.any(packing_fraction >= 1.0):
        raise ValueError(
            "Packing fraction must be smaller than 1."
        )


def packing_fraction(
    number_density_nm3: ArrayLike,
    hard_sphere_diameter_nm: float,
) -> FloatArray:
    """Return eta = pi*rho*sigma^3/6."""
    density = _as_array(number_density_nm3)
    diameter = float(hard_sphere_diameter_nm)

    if diameter <= 0.0:
        raise ValueError(
            "hard_sphere_diameter_nm must be positive."
        )

    if np.any(~np.isfinite(density)):
        raise ValueError(
            "Number density must contain finite values."
        )

    if np.any(density < 0.0):
        raise ValueError(
            "Number density cannot be negative."
        )

    eta = (
        np.pi
        * density
        * diameter**3
        / 6.0
    )

    _validate_packing_fraction(eta)
    return eta


def number_density_from_packing_fraction(
    packing_fraction_value: ArrayLike,
    hard_sphere_diameter_nm: float,
) -> FloatArray:
    """Return number density in particles/nm^3."""
    eta = _as_array(packing_fraction_value)
    diameter = float(hard_sphere_diameter_nm)

    if diameter <= 0.0:
        raise ValueError(
            "hard_sphere_diameter_nm must be positive."
        )

    _validate_packing_fraction(eta)

    return (
        6.0
        * eta
        / (
            np.pi
            * diameter**3
        )
    )


def cs_compressibility_factor(
    packing_fraction_value: ArrayLike,
) -> FloatArray:
    """Carnahan-Starling Z = beta*P/rho."""
    eta = _as_array(
        packing_fraction_value
    )
    _validate_packing_fraction(eta)

    return (
        1.0
        + eta
        + eta**2
        - eta**3
    ) / (1.0 - eta) ** 3


def cs_excess_free_energy_per_particle(
    packing_fraction_value: ArrayLike,
) -> FloatArray:
    """Return beta*A_ex/N."""
    eta = _as_array(
        packing_fraction_value
    )
    _validate_packing_fraction(eta)

    return (
        4.0 * eta
        - 3.0 * eta**2
    ) / (1.0 - eta) ** 2


def cs_excess_chemical_potential(
    packing_fraction_value: ArrayLike,
) -> FloatArray:
    """Return beta*mu_ex."""
    eta = _as_array(
        packing_fraction_value
    )
    _validate_packing_fraction(eta)

    return (
        8.0 * eta
        - 9.0 * eta**2
        + 3.0 * eta**3
    ) / (1.0 - eta) ** 3


def cs_d_beta_pressure_d_density(
    packing_fraction_value: ArrayLike,
) -> FloatArray:
    """Return d(beta P)/d rho for the CS reference fluid."""
    eta = _as_array(
        packing_fraction_value
    )
    _validate_packing_fraction(eta)

    return (
        1.0
        + 4.0 * eta
        + 4.0 * eta**2
        - 4.0 * eta**3
        + eta**4
    ) / (1.0 - eta) ** 4


def cs_reduced_isothermal_compressibility(
    packing_fraction_value: ArrayLike,
) -> FloatArray:
    """Return rho*kBT*kappa_T.

    This is equal to:

        [d(beta P)/d rho]^{-1}

    and is the hard-sphere compressibility factor needed by the
    second-order macroscopic-compressibility approximation.
    """
    derivative = (
        cs_d_beta_pressure_d_density(
            packing_fraction_value
        )
    )

    return 1.0 / derivative


def cs_beta_pressure(
    number_density_nm3: ArrayLike,
    hard_sphere_diameter_nm: float,
) -> FloatArray:
    """Return beta*P in nm^-3."""
    density = _as_array(
        number_density_nm3
    )

    eta = packing_fraction(
        density,
        hard_sphere_diameter_nm,
    )

    return (
        density
        * cs_compressibility_factor(eta)
    )


def cs_beta_helmholtz_free_energy_per_particle(
    number_density_nm3: ArrayLike,
    hard_sphere_diameter_nm: float,
    thermal_wavelength_nm: float = 1.0,
) -> FloatArray:
    """Return beta*A/N, including the ideal contribution.

    The thermal wavelength only adds a density-independent reference
    to the chemical potential and does not change phase coexistence.
    """
    density = _as_array(
        number_density_nm3
    )
    wavelength = float(
        thermal_wavelength_nm
    )

    if wavelength <= 0.0:
        raise ValueError(
            "thermal_wavelength_nm must be positive."
        )

    if np.any(density <= 0.0):
        raise ValueError(
            "Number density must be positive for free energy."
        )

    eta = packing_fraction(
        density,
        hard_sphere_diameter_nm,
    )

    ideal = (
        np.log(
            density
            * wavelength**3
        )
        - 1.0
    )

    return (
        ideal
        + cs_excess_free_energy_per_particle(
            eta
        )
    )


def cs_beta_chemical_potential(
    number_density_nm3: ArrayLike,
    hard_sphere_diameter_nm: float,
    thermal_wavelength_nm: float = 1.0,
) -> FloatArray:
    """Return beta*mu."""
    density = _as_array(
        number_density_nm3
    )
    wavelength = float(
        thermal_wavelength_nm
    )

    if wavelength <= 0.0:
        raise ValueError(
            "thermal_wavelength_nm must be positive."
        )

    if np.any(density <= 0.0):
        raise ValueError(
            "Number density must be positive for chemical potential."
        )

    eta = packing_fraction(
        density,
        hard_sphere_diameter_nm,
    )

    return (
        np.log(
            density
            * wavelength**3
        )
        + cs_excess_chemical_potential(
            eta
        )
    )
