"""Second-order liquid perturbation theory for a truncated LJ fluid.

Reduced units
-------------
x       = r / sigma
rho*    = rho * sigma**3
T*      = k_B T / epsilon
beta u  = u / (k_B T)

The Lennard-Jones interaction is cut directly at x = 5,
without shifting and without a long-range tail correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import log, pi
from typing import Callable

import numpy as np

from huang_md.hard_sphere import (
    cs_excess_free_energy_per_particle,
    cs_reduced_isothermal_compressibility,
)
from huang_md.hard_sphere_rdf_low_density import (
    hard_sphere_rdf_reduced as low_density_rdf_reduced,
)
from huang_md.perturbation import (
    perturbation_moments_reduced,
)
from huang_md.trokhymchuk_rdf import (
    trokhymchuk_rdf_reduced,
)


ArrayLike = float | list[float] | tuple[float, ...] | np.ndarray
FreeEnergyFunction = Callable[[float], float]

DEFAULT_CUTOFF_REDUCED = 5.0
TROKHYMCHUK_MIN_DENSITY = 0.2
TROKHYMCHUK_MAX_DENSITY = 0.9

# A short C2-smooth transition prevents artificial derivative
# discontinuity where the two RDF representations meet.
RDF_BLEND_START_DENSITY = 0.2
RDF_BLEND_END_DENSITY = 0.25


@dataclass(frozen=True)
class LJPerturbationResult:
    """Thermodynamic result at one reduced LJ state."""

    temperature_reduced: float
    reduced_density: float
    packing_fraction: float
    cutoff_reduced: float

    reference_compressibility: float

    first_moment_integral: float
    second_moment_integral: float

    beta_a1_per_particle: float
    beta_a2_per_particle: float
    beta_perturbation_per_particle: float

    beta_ideal_free_energy_per_particle: float
    beta_hard_sphere_excess_per_particle: float
    beta_reference_free_energy_per_particle: float
    beta_total_free_energy_per_particle: float

    second_to_first_abs_ratio: float

    first_integral_error: float
    second_integral_error: float


def _as_scalar(value: object) -> float:
    array = np.asarray(value, dtype=np.float64).reshape(-1)

    if array.size != 1:
        raise ValueError(
            "Expected one scalar value, "
            f"received {array.size} values."
        )

    result = float(array[0])

    if not np.isfinite(result):
        raise FloatingPointError(
            "Non-finite scalar generated."
        )

    return result


def _validate_temperature(
    temperature_reduced: float,
) -> float:
    value = float(temperature_reduced)

    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            "Reduced temperature must be positive and finite."
        )

    return value


def _validate_density(
    reduced_density: float,
) -> float:
    value = float(reduced_density)

    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            "Reduced density must be positive and finite."
        )

    if value > TROKHYMCHUK_MAX_DENSITY + 1.0e-12:
        raise ValueError(
            "Current LJ perturbation model is restricted to "
            f"rho*sigma^3 <= {TROKHYMCHUK_MAX_DENSITY}; "
            f"received {value}."
        )

    return min(
        value,
        TROKHYMCHUK_MAX_DENSITY,
    )


def packing_fraction_from_reduced_density(
    reduced_density: float,
) -> float:
    """Return eta = pi*rho*/6."""
    density = _validate_density(
        reduced_density
    )

    return pi * density / 6.0


def reduced_density_from_packing_fraction(
    packing_fraction: float,
) -> float:
    eta = float(packing_fraction)

    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError(
            "Packing fraction must be positive and finite."
        )

    density = 6.0 * eta / pi

    return _validate_density(density)


def lj_beta_potential_reduced(
    distance_reduced: ArrayLike,
    temperature_reduced: float,
    cutoff_reduced: float = DEFAULT_CUTOFF_REDUCED,
) -> float | np.ndarray:
    """Return beta*u_LJ for the directly cut LJ potential.

    The hard-sphere reference excludes x < 1. The perturbation
    potential is therefore explicitly set to zero inside the core
    to avoid evaluating the divergent LJ expression there.
    """
    temperature = _validate_temperature(
        temperature_reduced
    )

    cutoff = float(cutoff_reduced)

    if not np.isfinite(cutoff) or cutoff <= 1.0:
        raise ValueError(
            "Reduced cutoff must be finite and greater than one."
        )

    original = np.asarray(
        distance_reduced,
        dtype=np.float64,
    )

    scalar_input = original.ndim == 0
    shape = original.shape
    x = original.reshape(-1)

    if not np.isfinite(x).all():
        raise ValueError(
            "Reduced distances must all be finite."
        )

    if np.any(x < 0.0):
        raise ValueError(
            "Reduced distances must be non-negative."
        )

    result = np.zeros_like(
        x,
        dtype=np.float64,
    )

    mask = (
        (x >= 1.0)
        & (x < cutoff)
    )

    if np.any(mask):
        inverse_six = x[mask] ** -6

        result[mask] = (
            4.0
            / temperature
            * (
                inverse_six**2
                - inverse_six
            )
        )

    if scalar_input:
        return float(result[0])

    return result.reshape(shape)


def _smoothstep(value: float) -> float:
    """C2-smooth quintic interpolation weight on [0, 1].

    The quintic smootherstep has zero first and second
    derivatives at both endpoints. This is important because
    spinodal calculations use second derivatives of the
    Helmholtz free energy with respect to density.
    """
    clipped = min(
        max(float(value), 0.0),
        1.0,
    )

    return (
        clipped**3
        * (
            clipped
            * (
                6.0 * clipped
                - 15.0
            )
            + 10.0
        )
    )


def hybrid_hard_sphere_rdf_reduced(
    distance_reduced: ArrayLike,
    packing_fraction: float,
) -> float | np.ndarray:
    """Hard-sphere RDF over vapor and liquid densities.

    rho* <= 0.2
        Existing low-density, compressibility-consistent RDF.

    0.2 < rho* < 0.25
        C2-smooth numerical bridge between the two models.

    rho* >= 0.25
        Trokhymchuk high-density RDF.

    The narrow bridge is a project-level numerical device,
    not an equation stated in Trokhymchuk et al.
    """
    density = reduced_density_from_packing_fraction(
        packing_fraction
    )

    if density <= RDF_BLEND_START_DENSITY:
        return low_density_rdf_reduced(
            distance_reduced,
            packing_fraction,
            enforce_low_density_range=True,
        )

    if density >= RDF_BLEND_END_DENSITY:
        return trokhymchuk_rdf_reduced(
            distance_reduced,
            packing_fraction,
        )

    transition_coordinate = (
        (
            density
            - RDF_BLEND_START_DENSITY
        )
        / (
            RDF_BLEND_END_DENSITY
            - RDF_BLEND_START_DENSITY
        )
    )

    weight = _smoothstep(
        transition_coordinate
    )

    low_value = np.asarray(
        low_density_rdf_reduced(
            distance_reduced,
            packing_fraction,
            enforce_low_density_range=False,
        ),
        dtype=np.float64,
    )

    high_value = np.asarray(
        trokhymchuk_rdf_reduced(
            distance_reduced,
            packing_fraction,
        ),
        dtype=np.float64,
    )

    blended = (
        (1.0 - weight) * low_value
        + weight * high_value
    )

    if np.asarray(distance_reduced).ndim == 0:
        return float(blended.reshape(-1)[0])

    return blended


@lru_cache(maxsize=100_000)
def calculate_lj_perturbation_free_energy(
    temperature_reduced: float,
    reduced_density: float,
    cutoff_reduced: float = DEFAULT_CUTOFF_REDUCED,
) -> LJPerturbationResult:
    """Calculate the second-order LJ Helmholtz free energy."""
    temperature = _validate_temperature(
        temperature_reduced
    )

    density = _validate_density(
        reduced_density
    )

    cutoff = float(cutoff_reduced)

    if not np.isfinite(cutoff) or cutoff <= 1.0:
        raise ValueError(
            "Reduced cutoff must be greater than one."
        )

    eta = packing_fraction_from_reduced_density(
        density
    )

    def potential_function(
        distance: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(
            lj_beta_potential_reduced(
                distance,
                temperature_reduced=temperature,
                cutoff_reduced=cutoff,
            ),
            dtype=np.float64,
        )

    (
        first_integral,
        second_integral,
        first_error,
        second_error,
    ) = perturbation_moments_reduced(
        potential_function=potential_function,
        packing_fraction_value=eta,
        rdf_function=hybrid_hard_sphere_rdf_reduced,
        additional_breakpoints=(cutoff,),
        epsabs=1.0e-9,
        epsrel=1.0e-9,
    )

    reference_compressibility = _as_scalar(
        cs_reduced_isothermal_compressibility(
            eta
        )
    )

    beta_a1 = (
        0.5
        * density
        * first_integral
    )

    beta_a2 = (
        -0.25
        * density
        * reference_compressibility
        * second_integral
    )

    beta_perturbation = (
        beta_a1
        + beta_a2
    )

    beta_ideal = (
        log(density)
        - 1.0
    )

    beta_hard_sphere_excess = _as_scalar(
        cs_excess_free_energy_per_particle(
            eta
        )
    )

    beta_reference = (
        beta_ideal
        + beta_hard_sphere_excess
    )

    beta_total = (
        beta_reference
        + beta_perturbation
    )

    if abs(beta_a1) > 1.0e-15:
        ratio = abs(
            beta_a2 / beta_a1
        )
    else:
        ratio = float("inf")

    values = np.array(
        [
            temperature,
            density,
            eta,
            cutoff,
            reference_compressibility,
            first_integral,
            second_integral,
            beta_a1,
            beta_a2,
            beta_perturbation,
            beta_ideal,
            beta_hard_sphere_excess,
            beta_reference,
            beta_total,
            first_error,
            second_error,
        ],
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise FloatingPointError(
            "Non-finite LJ perturbation result generated."
        )

    return LJPerturbationResult(
        temperature_reduced=temperature,
        reduced_density=density,
        packing_fraction=eta,
        cutoff_reduced=cutoff,
        reference_compressibility=(
            reference_compressibility
        ),
        first_moment_integral=float(
            first_integral
        ),
        second_moment_integral=float(
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
        beta_ideal_free_energy_per_particle=float(
            beta_ideal
        ),
        beta_hard_sphere_excess_per_particle=float(
            beta_hard_sphere_excess
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


def beta_lj_free_energy_per_particle(
    reduced_density: float,
    temperature_reduced: float,
    cutoff_reduced: float = DEFAULT_CUTOFF_REDUCED,
) -> float:
    """Return beta*A/N for coexistence calculations."""
    result = calculate_lj_perturbation_free_energy(
        temperature_reduced=temperature_reduced,
        reduced_density=reduced_density,
        cutoff_reduced=cutoff_reduced,
    )

    return result.beta_total_free_energy_per_particle


def make_lj_free_energy_function(
    temperature_reduced: float,
    cutoff_reduced: float = DEFAULT_CUTOFF_REDUCED,
) -> FreeEnergyFunction:
    """Return a one-variable free-energy function f(rho*)."""
    temperature = _validate_temperature(
        temperature_reduced
    )

    cutoff = float(cutoff_reduced)

    def free_energy(
        reduced_density: float,
    ) -> float:
        return beta_lj_free_energy_per_particle(
            reduced_density=reduced_density,
            temperature_reduced=temperature,
            cutoff_reduced=cutoff,
        )

    return free_energy
