"""Second-virial calculations for the Huang/refCBA potential."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad

from huang_md.parameters import HuangPotentialParameters
from huang_md.potential import total_potential_reduced


FloatArray = NDArray[np.float64]
PotentialFunction = Callable[[FloatArray], ArrayLike]


@dataclass(frozen=True)
class VirialResult:
    """Second-virial properties of one interaction potential."""

    full_B2_reduced: float
    full_B2_nm3: float

    core_B2_reduced: float
    core_B2_nm3: float

    effective_hs_diameter_reduced: float
    effective_hs_diameter_nm: float

    reduced_second_virial_B2star: float

    full_integration_error: float
    core_integration_error: float


def gaussian_soft_core_reduced(
    distance_reduced: ArrayLike,
    params: HuangPotentialParameters,
) -> FloatArray:
    """Repulsive Gaussian core with shifted-force matching at x=1.

    The core and its first derivative both approach zero at x=1.
    It is zero for x >= 1.
    """
    x = np.asarray(
        distance_reduced,
        dtype=np.float64,
    )

    sigma = float(params.gaussian_sigma_reduced)
    epsilon = float(params.gaussian_epsilon_kBT)

    if sigma <= 0:
        raise ValueError(
            "gaussian_sigma_reduced must be positive."
        )

    if epsilon <= 0:
        raise ValueError(
            "gaussian_epsilon_kBT must be positive."
        )

    # Huang et al. Gaussian-core reference:
    # U_core(x) = epsilon * exp[-(x/sigma)^2].
    #
    # The full interaction is piecewise: the Gaussian excluded-volume
    # contribution is used inside the nominal particle diameter, while
    # the two-Yukawa interaction is used outside.
    gaussian = epsilon * np.exp(
        -(x / sigma) ** 2
    )

    return np.where(
        x < 1.0,
        gaussian,
        0.0,
    )


def shifted_truncated_potential_reduced(
    distance_reduced: ArrayLike,
    params: HuangPotentialParameters,
) -> FloatArray:
    """Return the truncated and energy-shifted full potential.

    This matches the finite-cutoff convention used in the OpenMM
    implementation: U(x_cut)=0 and U(x>=x_cut)=0.
    """
    x = np.asarray(
        distance_reduced,
        dtype=np.float64,
    )

    cutoff = float(params.cutoff_reduced)

    if cutoff <= 1.0:
        raise ValueError(
            "cutoff_reduced must be greater than 1."
        )

    cutoff_inside = np.nextafter(
        cutoff,
        0.0,
    )

    cutoff_energy = float(
        np.asarray(
            total_potential_reduced(
                np.array(
                    [cutoff_inside],
                    dtype=np.float64,
                ),
                params,
            )
        ).reshape(-1)[0]
    )

    raw_energy = np.asarray(
        total_potential_reduced(
            x,
            params,
        ),
        dtype=np.float64,
    )

    return np.where(
        x < cutoff,
        raw_energy - cutoff_energy,
        0.0,
    )


def _scalar_value(
    values: ArrayLike,
) -> float:
    array = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    if array.size == 0:
        raise ValueError(
            "Potential function returned an empty array."
        )

    value = float(array[0])

    if not np.isfinite(value):
        raise FloatingPointError(
            f"Non-finite potential value: {value}"
        )

    return value


def _safe_mayer_function(
    potential_kBT: float,
) -> float:
    """Return exp(-U/kBT)-1 with overflow protection."""
    exponent = np.clip(
        -potential_kBT,
        -745.0,
        700.0,
    )

    return float(np.expm1(exponent))


def second_virial_reduced_from_callable(
    potential_function: PotentialFunction,
    upper_bound_reduced: float,
    breakpoints: tuple[float, ...] = (),
    lower_bound_reduced: float = 0.0,
    epsabs: float = 1.0e-9,
    epsrel: float = 1.0e-8,
) -> tuple[float, float]:
    """Calculate B2/d^3 for an arbitrary reduced potential.

    B2/d^3 = -2*pi integral [
        exp(-U(x)/kBT)-1
    ] x^2 dx
    """
    lower = float(lower_bound_reduced)
    upper = float(upper_bound_reduced)

    if lower < 0:
        raise ValueError(
            "lower_bound_reduced cannot be negative."
        )

    if upper <= lower:
        raise ValueError(
            "upper_bound_reduced must exceed the lower bound."
        )

    internal_points = sorted(
        {
            float(point)
            for point in breakpoints
            if lower < float(point) < upper
        }
    )

    interval_edges = [
        lower,
        *internal_points,
        upper,
    ]

    total_value = 0.0
    total_error = 0.0

    def integrand(
        distance_reduced: float,
    ) -> float:
        potential_value = _scalar_value(
            potential_function(
                np.array(
                    [distance_reduced],
                    dtype=np.float64,
                )
            )
        )

        mayer = _safe_mayer_function(
            potential_value
        )

        return (
            -2.0
            * pi
            * mayer
            * distance_reduced**2
        )

    for left, right in zip(
        interval_edges[:-1],
        interval_edges[1:],
    ):
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


def hard_sphere_diameter_from_B2_reduced(
    core_B2_reduced: float,
) -> float:
    """Return d_HS/d from core B2/d^3 matching."""
    value = float(core_B2_reduced)

    if not np.isfinite(value):
        raise ValueError(
            "core_B2_reduced must be finite."
        )

    if value <= 0:
        raise ValueError(
            "Repulsive-core B2 must be positive."
        )

    return (
        3.0 * value / (2.0 * pi)
    ) ** (1.0 / 3.0)


def calculate_virial_properties(
    params: HuangPotentialParameters,
) -> VirialResult:
    """Calculate full and repulsive-core second virial properties."""
    diameter_nm = float(
        params.diameter_nm
    )

    if diameter_nm <= 0:
        raise ValueError(
            "Particle diameter must be positive."
        )

    cutoff = float(
        params.cutoff_reduced
    )

    # Thermodynamic B2 is calculated from the original, unshifted
    # interaction and integrated to infinite separation. The shifted
    # finite-cutoff form is retained only for comparison with the MD
    # implementation and must not define the liquid-state theory.
    full_B2_reduced, full_error = (
        second_virial_reduced_from_callable(
            potential_function=lambda x: (
                total_potential_reduced(
                    x,
                    params,
                )
            ),
            lower_bound_reduced=1.0e-8,
            upper_bound_reduced=float("inf"),
            breakpoints=(1.0,),
        )
    )

    core_B2_reduced, core_error = (
        second_virial_reduced_from_callable(
            potential_function=lambda x: (
                gaussian_soft_core_reduced(
                    x,
                    params,
                )
            ),
            lower_bound_reduced=0.0,
            upper_bound_reduced=1.0,
        )
    )

    effective_diameter_reduced = (
        hard_sphere_diameter_from_B2_reduced(
            core_B2_reduced
        )
    )

    full_B2_nm3 = (
        full_B2_reduced
        * diameter_nm**3
    )

    core_B2_nm3 = (
        core_B2_reduced
        * diameter_nm**3
    )

    effective_diameter_nm = (
        effective_diameter_reduced
        * diameter_nm
    )

    reduced_B2star = (
        full_B2_reduced
        / core_B2_reduced
    )

    return VirialResult(
        full_B2_reduced=float(
            full_B2_reduced
        ),
        full_B2_nm3=float(
            full_B2_nm3
        ),
        core_B2_reduced=float(
            core_B2_reduced
        ),
        core_B2_nm3=float(
            core_B2_nm3
        ),
        effective_hs_diameter_reduced=float(
            effective_diameter_reduced
        ),
        effective_hs_diameter_nm=float(
            effective_diameter_nm
        ),
        reduced_second_virial_B2star=float(
            reduced_B2star
        ),
        full_integration_error=float(
            full_error
        ),
        core_integration_error=float(
            core_error
        ),
    )
