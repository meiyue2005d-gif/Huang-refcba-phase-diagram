"""Trokhymchuk hard-sphere radial distribution function.

Implements the piecewise analytical RDF from:

Trokhymchuk et al.,
J. Chem. Phys. 123, 024501 (2005).

The corrected 2006 erratum expressions for mu and gamma
are supplied by ``trokhymchuk_parameters.py``.

Dimensionless conventions
-------------------------
x = r / sigma
rho_star = rho * sigma**3
eta = pi * rho_star / 6

Validity range
--------------
0.2 <= rho_star <= 0.9
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Union

import numpy as np

from huang_md.trokhymchuk_parameters import (
    TrokhymchukParameters,
    calculate_trokhymchuk_parameters,
)


ArrayInput = Union[
    float,
    list[float],
    tuple[float, ...],
    np.ndarray,
]


@dataclass(frozen=True)
class TrokhymchukRDFCoefficients:
    """All coefficients required to evaluate the full RDF."""

    base: TrokhymchukParameters

    coefficient_a: float
    coefficient_b: float
    coefficient_c: float

    structural_phase: float

    depletion_derivative_at_merge: float
    structural_derivative_at_merge: float


def _validate_finite_nonnegative_distance(
    distance_reduced: ArrayInput,
) -> tuple[np.ndarray, bool, tuple[int, ...]]:
    original = np.asarray(
        distance_reduced,
        dtype=np.float64,
    )

    scalar_input = original.ndim == 0
    original_shape = original.shape
    flat = original.reshape(-1)

    if not np.isfinite(flat).all():
        raise ValueError(
            "Reduced distances must all be finite."
        )

    if np.any(flat < 0.0):
        raise ValueError(
            "Reduced distances must be non-negative."
        )

    return flat, scalar_input, original_shape


def _restore_output_shape(
    values: np.ndarray,
    scalar_input: bool,
    original_shape: tuple[int, ...],
) -> float | np.ndarray:
    if scalar_input:
        return float(values[0])

    return values.reshape(original_shape)


def reduced_density_from_packing_fraction(
    packing_fraction: float,
) -> float:
    """Convert eta to rho*sigma^3."""
    eta = float(packing_fraction)

    if not np.isfinite(eta):
        raise ValueError(
            "Packing fraction must be finite."
        )

    if eta <= 0.0:
        raise ValueError(
            "Packing fraction must be positive."
        )

    return 6.0 * eta / pi


def depletion_rdf_value(
    distance_reduced: float,
    coefficients: TrokhymchukRDFCoefficients,
) -> float:
    """Evaluate the short-range depletion branch."""
    x = float(distance_reduced)
    p = coefficients.base

    if x <= 0.0:
        raise ValueError(
            "Distance must be positive for the depletion branch."
        )

    separation = x - 1.0

    exponential_mu = np.exp(
        p.mu * separation
    )

    exponential_alpha = np.exp(
        p.alpha * separation
    )

    oscillatory_angle = (
        p.beta * separation
        + p.gamma
    )

    return float(
        coefficients.coefficient_a
        / x
        * exponential_mu
        + coefficients.coefficient_b
        / x
        * np.cos(oscillatory_angle)
        * exponential_alpha
    )


def structural_rdf_value(
    distance_reduced: float,
    coefficients: TrokhymchukRDFCoefficients,
) -> float:
    """Evaluate the long-range structural branch."""
    x = float(distance_reduced)
    p = coefficients.base

    if x <= 0.0:
        raise ValueError(
            "Distance must be positive for the structural branch."
        )

    angle = (
        p.omega * x
        + coefficients.structural_phase
    )

    return float(
        1.0
        + coefficients.coefficient_c
        / x
        * np.cos(angle)
        * np.exp(-p.kappa * x)
    )


def depletion_rdf_derivative(
    distance_reduced: float,
    coefficients: TrokhymchukRDFCoefficients,
) -> float:
    """Analytical derivative d g_dep / d x."""
    x = float(distance_reduced)
    p = coefficients.base

    if x <= 0.0:
        raise ValueError(
            "Distance must be positive."
        )

    separation = x - 1.0

    exp_mu = np.exp(
        p.mu * separation
    )

    exp_alpha = np.exp(
        p.alpha * separation
    )

    angle = (
        p.beta * separation
        + p.gamma
    )

    cosine = np.cos(angle)
    sine = np.sin(angle)

    numerator = (
        coefficients.coefficient_a
        * exp_mu
        + coefficients.coefficient_b
        * cosine
        * exp_alpha
    )

    numerator_derivative = (
        coefficients.coefficient_a
        * p.mu
        * exp_mu
        + coefficients.coefficient_b
        * exp_alpha
        * (
            p.alpha * cosine
            - p.beta * sine
        )
    )

    return float(
        (
            x * numerator_derivative
            - numerator
        )
        / x**2
    )


def structural_rdf_derivative(
    distance_reduced: float,
    coefficients: TrokhymchukRDFCoefficients,
) -> float:
    """Analytical derivative d g_str / d x."""
    x = float(distance_reduced)
    p = coefficients.base

    if x <= 0.0:
        raise ValueError(
            "Distance must be positive."
        )

    angle = (
        p.omega * x
        + coefficients.structural_phase
    )

    cosine = np.cos(angle)
    sine = np.sin(angle)

    exponential = np.exp(
        -p.kappa * x
    )

    numerator = (
        coefficients.coefficient_c
        * exponential
        * cosine
    )

    numerator_derivative = (
        coefficients.coefficient_c
        * exponential
        * (
            -p.kappa * cosine
            - p.omega * sine
        )
    )

    return float(
        (
            x * numerator_derivative
            - numerator
        )
        / x**2
    )


def calculate_trokhymchuk_rdf_coefficients(
    reduced_density: float,
) -> TrokhymchukRDFCoefficients:
    """Calculate A, B, C, and delta from Eqs. 21-24."""
    p = calculate_trokhymchuk_parameters(
        reduced_density
    )

    merge_position = p.minimum_position
    separation = merge_position - 1.0

    exp_mu = np.exp(
        p.mu * separation
    )

    exp_alpha = np.exp(
        p.alpha * separation
    )

    denominator_b = (
        np.cos(
            p.beta * separation
            + p.gamma
        )
        * exp_alpha
        - np.cos(p.gamma)
        * exp_mu
    )

    if abs(denominator_b) <= 1.0e-13:
        raise FloatingPointError(
            "Coefficient-B denominator is too close to zero."
        )

    coefficient_b = (
        merge_position
        * (
            p.minimum_rdf
            - (
                p.contact_rdf
                / merge_position
            )
            * exp_mu
        )
        / denominator_b
    )

    coefficient_a = (
        p.contact_rdf
        - coefficient_b
        * np.cos(p.gamma)
    )

    structural_phase = (
        -p.omega * merge_position
        - np.arctan(
            (
                p.kappa * merge_position
                + 1.0
            )
            / (
                p.omega * merge_position
            )
        )
    )

    structural_cosine = np.cos(
        p.omega * merge_position
        + structural_phase
    )

    if abs(structural_cosine) <= 1.0e-13:
        raise FloatingPointError(
            "Coefficient-C denominator is too close to zero."
        )

    coefficient_c = (
        merge_position
        * (
            p.minimum_rdf
            - 1.0
        )
        * np.exp(
            p.kappa * merge_position
        )
        / structural_cosine
    )

    provisional = TrokhymchukRDFCoefficients(
        base=p,
        coefficient_a=float(
            coefficient_a
        ),
        coefficient_b=float(
            coefficient_b
        ),
        coefficient_c=float(
            coefficient_c
        ),
        structural_phase=float(
            structural_phase
        ),
        depletion_derivative_at_merge=0.0,
        structural_derivative_at_merge=0.0,
    )

    depletion_slope = (
        depletion_rdf_derivative(
            merge_position,
            provisional,
        )
    )

    structural_slope = (
        structural_rdf_derivative(
            merge_position,
            provisional,
        )
    )

    result = TrokhymchukRDFCoefficients(
        base=p,
        coefficient_a=float(
            coefficient_a
        ),
        coefficient_b=float(
            coefficient_b
        ),
        coefficient_c=float(
            coefficient_c
        ),
        structural_phase=float(
            structural_phase
        ),
        depletion_derivative_at_merge=float(
            depletion_slope
        ),
        structural_derivative_at_merge=float(
            structural_slope
        ),
    )

    values = np.array(
        [
            result.coefficient_a,
            result.coefficient_b,
            result.coefficient_c,
            result.structural_phase,
            result.depletion_derivative_at_merge,
            result.structural_derivative_at_merge,
        ],
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise FloatingPointError(
            "Non-finite full-RDF coefficient generated."
        )

    return result


def trokhymchuk_rdf_from_reduced_density(
    distance_reduced: ArrayInput,
    reduced_density: float,
) -> float | np.ndarray:
    """Evaluate g(x) using rho*sigma^3 as density input."""
    x, scalar_input, original_shape = (
        _validate_finite_nonnegative_distance(
            distance_reduced
        )
    )

    coefficients = (
        calculate_trokhymchuk_rdf_coefficients(
            reduced_density
        )
    )

    p = coefficients.base
    result = np.zeros_like(
        x,
        dtype=np.float64,
    )

    depletion_mask = (
        (x >= 1.0)
        & (
            x
            <= p.minimum_position
        )
    )

    structural_mask = (
        x
        > p.minimum_position
    )

    if np.any(depletion_mask):
        xd = x[depletion_mask]
        separation = xd - 1.0

        result[depletion_mask] = (
            coefficients.coefficient_a
            / xd
            * np.exp(
                p.mu * separation
            )
            + coefficients.coefficient_b
            / xd
            * np.cos(
                p.beta * separation
                + p.gamma
            )
            * np.exp(
                p.alpha * separation
            )
        )

    if np.any(structural_mask):
        xs = x[structural_mask]

        result[structural_mask] = (
            1.0
            + coefficients.coefficient_c
            / xs
            * np.cos(
                p.omega * xs
                + coefficients.structural_phase
            )
            * np.exp(
                -p.kappa * xs
            )
        )

    return _restore_output_shape(
        result,
        scalar_input,
        original_shape,
    )


def trokhymchuk_rdf_reduced(
    distance_reduced: ArrayInput,
    packing_fraction: float,
) -> float | np.ndarray:
    """Perturbation-module-compatible RDF interface.

    Parameters
    ----------
    distance_reduced
        Distance x = r/sigma.
    packing_fraction
        Hard-sphere packing fraction eta.
    """
    reduced_density = (
        reduced_density_from_packing_fraction(
            packing_fraction
        )
    )

    return trokhymchuk_rdf_from_reduced_density(
        distance_reduced,
        reduced_density,
    )
