"""Second-order liquid perturbation theory for the Huang potential.

The implementation follows the macroscopic-compressibility
approximation described in Eq. S5 of the Huang supporting material.

All interaction energies are expressed in kBT and all returned
free energies are beta*A/N.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad

from huang_md.hard_sphere import (
    cs_beta_helmholtz_free_energy_per_particle,
    cs_reduced_isothermal_compressibility,
    packing_fraction,
)
from huang_md.hard_sphere_rdf_low_density import (
    hard_sphere_rdf_reduced as low_density_hard_sphere_rdf_reduced,
)
from huang_md.parameters import HuangPotentialParameters
from huang_md.potential import total_potential_reduced


FloatArray = NDArray[np.float64]
PotentialFunction = Callable[[FloatArray], ArrayLike]
RDFFunction = Callable[[FloatArray, float], ArrayLike]


@dataclass(frozen=True)
class PerturbationResult:
    number_density_nm3: float
    hard_sphere_diameter_nm: float
    reduced_density_rho_sigma3: float
    packing_fraction: float
    reference_compressibility: float

    first_moment_integral: float
    second_moment_integral: float

    beta_a1_per_particle: float
    beta_a2_per_particle: float
    beta_perturbation_per_particle: float

    beta_reference_free_energy_per_particle: float
    beta_total_free_energy_per_particle: float

    second_to_first_abs_ratio: float

    first_integral_error: float
    second_integral_error: float


def _scalar_value(values: ArrayLike) -> float:
    array = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    if array.size != 1:
        raise ValueError(
            "Potential callable must return one value "
            "for one input distance."
        )

    value = float(array[0])

    if not np.isfinite(value):
        raise FloatingPointError(
            f"Non-finite potential value: {value}"
        )

    return value


def _piecewise_quad(
    integrand: Callable[[float], float],
    breakpoints: list[float],
    epsabs: float,
    epsrel: float,
) -> tuple[float, float]:
    points = sorted(
        {
            float(point)
            for point in breakpoints
            if float(point) > 0.0
        }
    )

    if not points:
        raise ValueError(
            "At least one finite lower breakpoint is required."
        )

    edges = [
        *points,
        float("inf"),
    ]

    total_value = 0.0
    total_error = 0.0

    for left, right in zip(
        edges[:-1],
        edges[1:],
    ):
        if right <= left:
            continue

        value, error = quad(
            integrand,
            left,
            right,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=500,
        )

        total_value += float(value)
        total_error += float(error)

    return total_value, total_error


def perturbation_moments_reduced(
    potential_function: PotentialFunction,
    packing_fraction_value: float,
    rdf_function: RDFFunction = (
        low_density_hard_sphere_rdf_reduced
    ),
    additional_breakpoints: tuple[float, ...] = (),
    epsabs: float = 1.0e-8,
    epsrel: float = 1.0e-8,
) -> tuple[float, float, float, float]:
    """Calculate the two integrals appearing in Eq. S5.

    The integration variable is y=r/sigma_HS.

    I1 = integral u*(y) g0(y) 4*pi*y^2 dy
    I2 = integral [u*(y)]^2 g0(y) 4*pi*y^2 dy
    """
    eta = float(packing_fraction_value)

    if not np.isfinite(eta):
        raise ValueError(
            "Packing fraction must be finite."
        )

    if eta < 0.0:
        raise ValueError(
            "Packing fraction cannot be negative."
        )

    def first_integrand(
        distance_reduced: float,
    ) -> float:
        distance_array = np.array(
            [distance_reduced],
            dtype=np.float64,
        )

        potential = _scalar_value(
            potential_function(
                distance_array
            )
        )

        rdf = float(
            np.asarray(
                rdf_function(
                    distance_array,
                    eta,
                ),
                dtype=np.float64,
            ).reshape(-1)[0]
        )

        return (
            4.0
            * pi
            * potential
            * rdf
            * distance_reduced**2
        )

    def second_integrand(
        distance_reduced: float,
    ) -> float:
        distance_array = np.array(
            [distance_reduced],
            dtype=np.float64,
        )

        potential = _scalar_value(
            potential_function(
                distance_array
            )
        )

        rdf = float(
            np.asarray(
                rdf_function(
                    distance_array,
                    eta,
                ),
                dtype=np.float64,
            ).reshape(-1)[0]
        )

        return (
            4.0
            * pi
            * potential**2
            * rdf
            * distance_reduced**2
        )

    breakpoints = [
        1.0,
        2.0,
        *additional_breakpoints,
    ]

    first_value, first_error = (
        _piecewise_quad(
            first_integrand,
            breakpoints,
            epsabs,
            epsrel,
        )
    )

    second_value, second_error = (
        _piecewise_quad(
            second_integrand,
            breakpoints,
            epsabs,
            epsrel,
        )
    )

    return (
        float(first_value),
        float(second_value),
        float(first_error),
        float(second_error),
    )


def calculate_perturbation_free_energy(
    params: HuangPotentialParameters,
    number_density_nm3: float,
    hard_sphere_diameter_nm: float,
    thermal_wavelength_nm: float = 1.0,
    rdf_function: RDFFunction = (
        low_density_hard_sphere_rdf_reduced
    ),
) -> PerturbationResult:
    """Calculate reference, first- and second-order free energies."""
    density = float(number_density_nm3)
    sigma_hs = float(
        hard_sphere_diameter_nm
    )
    nominal_diameter = float(
        params.diameter_nm
    )

    if not np.isfinite(density) or density <= 0.0:
        raise ValueError(
            "number_density_nm3 must be positive and finite."
        )

    if not np.isfinite(sigma_hs) or sigma_hs <= 0.0:
        raise ValueError(
            "hard_sphere_diameter_nm must be positive."
        )

    if nominal_diameter <= 0.0:
        raise ValueError(
            "Potential diameter must be positive."
        )

    eta = float(
        np.asarray(
            packing_fraction(
                density,
                sigma_hs,
            )
        ).reshape(-1)[0]
    )

    reduced_density = (
        density
        * sigma_hs**3
    )

    reference_compressibility = float(
        np.asarray(
            cs_reduced_isothermal_compressibility(
                eta
            )
        ).reshape(-1)[0]
    )

    # The Huang potential uses x=r/d, while the hard-sphere RDF uses
    # y=r/sigma_HS. Therefore x = y*sigma_HS/d.
    diameter_ratio = (
        sigma_hs
        / nominal_diameter
    )

    def potential_in_hs_coordinates(
        distance_hs_reduced: FloatArray,
    ) -> FloatArray:
        potential_distance = (
            distance_hs_reduced
            * diameter_ratio
        )

        return np.asarray(
            total_potential_reduced(
                potential_distance,
                params,
            ),
            dtype=np.float64,
        )

    potential_contact_in_hs_units = (
        1.0 / diameter_ratio
    )

    additional_breakpoints: list[float] = []

    if potential_contact_in_hs_units > 1.0:
        additional_breakpoints.append(
            potential_contact_in_hs_units
        )

    (
        first_integral,
        second_integral,
        first_error,
        second_error,
    ) = perturbation_moments_reduced(
        potential_function=(
            potential_in_hs_coordinates
        ),
        packing_fraction_value=eta,
        rdf_function=rdf_function,
        additional_breakpoints=tuple(
            additional_breakpoints
        ),
    )

    # Eq. S5 in reduced coordinates.
    beta_a1 = (
        0.5
        * reduced_density
        * first_integral
    )

    beta_a2 = (
        -0.25
        * reduced_density
        * reference_compressibility
        * second_integral
    )

    beta_reference = float(
        np.asarray(
            cs_beta_helmholtz_free_energy_per_particle(
                density,
                sigma_hs,
                thermal_wavelength_nm=(
                    thermal_wavelength_nm
                ),
            )
        ).reshape(-1)[0]
    )

    beta_perturbation = (
        beta_a1
        + beta_a2
    )

    beta_total = (
        beta_reference
        + beta_perturbation
    )

    ratio = (
        abs(beta_a2)
        / max(
            abs(beta_a1),
            1.0e-15,
        )
    )

    return PerturbationResult(
        number_density_nm3=density,
        hard_sphere_diameter_nm=sigma_hs,
        reduced_density_rho_sigma3=float(
            reduced_density
        ),
        packing_fraction=eta,
        reference_compressibility=(
            reference_compressibility
        ),
        first_moment_integral=(
            first_integral
        ),
        second_moment_integral=(
            second_integral
        ),
        beta_a1_per_particle=float(
            beta_a1
        ),
        beta_a2_per_particle=float(
            beta_a2
        ),
        beta_perturbation_per_particle=float(
            beta_perturbation
        ),
        beta_reference_free_energy_per_particle=float(
            beta_reference
        ),
        beta_total_free_energy_per_particle=float(
            beta_total
        ),
        second_to_first_abs_ratio=float(
            ratio
        ),
        first_integral_error=float(
            first_error
        ),
        second_integral_error=float(
            second_error
        ),
    )
