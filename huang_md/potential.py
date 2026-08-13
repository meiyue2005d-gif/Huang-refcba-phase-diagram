"""Huang et al. two-Yukawa plus Gaussian-core pair potential."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from huang_md.parameters import HuangPotentialParameters


FloatArray = NDArray[np.float64]


def _as_array(values: ArrayLike) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


def attractive_yukawa_reduced(
    x: ArrayLike,
    params: HuangPotentialParameters,
) -> FloatArray:
    """Short-range attractive contribution in units of kBT.

    x = r/d is dimensionless center-to-center separation.
    """
    x_array = _as_array(x)

    if np.any(x_array <= 0):
        raise ValueError("All reduced distances must be positive.")

    return (
        -params.K1_kBT
        * np.exp(-params.Z1 * (x_array - 1.0))
        / x_array
    )


def repulsive_yukawa_reduced(
    x: ArrayLike,
    params: HuangPotentialParameters,
) -> FloatArray:
    """Long-range repulsive contribution in units of kBT."""
    x_array = _as_array(x)

    if np.any(x_array <= 0):
        raise ValueError("All reduced distances must be positive.")

    return (
        params.K2_kBT
        * np.exp(-params.Z2 * (x_array - 1.0))
        / x_array
    )


def two_yukawa_reduced(
    x: ArrayLike,
    params: HuangPotentialParameters,
) -> FloatArray:
    """Two-Yukawa SA-LR contribution in units of kBT."""
    return (
        attractive_yukawa_reduced(x, params)
        + repulsive_yukawa_reduced(x, params)
    )


def two_yukawa_derivative_reduced(
    x: ArrayLike,
    params: HuangPotentialParameters,
) -> FloatArray:
    """Derivative d(U/kBT)/d(r/d)."""
    x_array = _as_array(x)

    if np.any(x_array <= 0):
        raise ValueError("All reduced distances must be positive.")

    attraction_derivative = (
        params.K1_kBT
        * np.exp(-params.Z1 * (x_array - 1.0))
        * (
            params.Z1 / x_array
            + 1.0 / x_array**2
        )
    )

    repulsion_derivative = (
        -params.K2_kBT
        * np.exp(-params.Z2 * (x_array - 1.0))
        * (
            params.Z2 / x_array
            + 1.0 / x_array**2
        )
    )

    return attraction_derivative + repulsion_derivative


def total_potential_reduced(
    x: ArrayLike,
    params: HuangPotentialParameters,
) -> FloatArray:
    """Complete piecewise Huang potential in units of kBT.

    For x >= 1:
        ordinary two-Yukawa potential.

    For x < 1:
        Gaussian core plus a linear continuation chosen so that both
        the potential and its first derivative match the two-Yukawa
        expression at x = 1.
    """
    params.validate()

    x_array = _as_array(x)

    if np.any(x_array <= 0):
        raise ValueError("All reduced distances must be positive.")

    result = np.empty_like(x_array)

    outer_mask = x_array >= 1.0
    inner_mask = ~outer_mask

    if np.any(outer_mask):
        result[outer_mask] = two_yukawa_reduced(
            x_array[outer_mask],
            params,
        )

    if np.any(inner_mask):
        x_inner = x_array[inner_mask]

        sigma = params.gaussian_sigma_reduced
        epsilon = params.gaussian_epsilon_kBT

        u_at_one = -params.K1_kBT + params.K2_kBT

        du_at_one = (
            params.K1_kBT * (params.Z1 + 1.0)
            - params.K2_kBT * (params.Z2 + 1.0)
        )

        gaussian_at_one = np.exp(-1.0 / sigma**2)
        gaussian_derivative_at_one = (
            -2.0
            / sigma**2
            * gaussian_at_one
        )

        gaussian = np.exp(-(x_inner**2) / sigma**2)

        linear_intercept = (
            u_at_one
            - epsilon * gaussian_at_one
        )

        linear_slope = (
            du_at_one
            - epsilon * gaussian_derivative_at_one
        )

        result[inner_mask] = (
            epsilon * gaussian
            + linear_intercept
            + linear_slope * (x_inner - 1.0)
        )

    return result


def total_derivative_reduced(
    x: ArrayLike,
    params: HuangPotentialParameters,
) -> FloatArray:
    """Derivative d(U/kBT)/d(r/d) of the complete potential."""
    params.validate()

    x_array = _as_array(x)

    if np.any(x_array <= 0):
        raise ValueError("All reduced distances must be positive.")

    result = np.empty_like(x_array)

    outer_mask = x_array >= 1.0
    inner_mask = ~outer_mask

    if np.any(outer_mask):
        result[outer_mask] = two_yukawa_derivative_reduced(
            x_array[outer_mask],
            params,
        )

    if np.any(inner_mask):
        x_inner = x_array[inner_mask]

        sigma = params.gaussian_sigma_reduced
        epsilon = params.gaussian_epsilon_kBT

        du_at_one = (
            params.K1_kBT * (params.Z1 + 1.0)
            - params.K2_kBT * (params.Z2 + 1.0)
        )

        gaussian_at_one = np.exp(-1.0 / sigma**2)

        gaussian_derivative_at_one = (
            -2.0
            / sigma**2
            * gaussian_at_one
        )

        gaussian = np.exp(-(x_inner**2) / sigma**2)

        gaussian_derivative = (
            -2.0
            * x_inner
            / sigma**2
            * gaussian
        )

        linear_slope = (
            du_at_one
            - epsilon * gaussian_derivative_at_one
        )

        result[inner_mask] = (
            epsilon * gaussian_derivative
            + linear_slope
        )

    return result


def force_reduced(
    x: ArrayLike,
    params: HuangPotentialParameters,
) -> FloatArray:
    """Reduced radial force: -d(U/kBT)/d(r/d)."""
    return -total_derivative_reduced(x, params)


def distance_nm_to_reduced(
    distance_nm: ArrayLike,
    params: HuangPotentialParameters,
) -> FloatArray:
    return _as_array(distance_nm) / params.diameter_nm


def reduced_to_distance_nm(
    x: ArrayLike,
    params: HuangPotentialParameters,
) -> FloatArray:
    return _as_array(x) * params.diameter_nm
