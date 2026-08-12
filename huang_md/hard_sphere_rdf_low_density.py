"""Low-density hard-sphere radial distribution function.

The model is designed for the concentration range used by the
refCBA project, where rho*sigma^3 < 0.2.

It satisfies:
1. g(r) = 0 for r < sigma;
2. Carnahan-Starling contact value;
3. Carnahan-Starling compressibility sum rule;
4. g(r) = 1 for r >= 2*sigma.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad

from huang_md.hard_sphere import (
    cs_reduced_isothermal_compressibility,
)


FloatArray = NDArray[np.float64]

MAX_REDUCED_DENSITY = 0.2
MAX_PACKING_FRACTION = (
    pi * MAX_REDUCED_DENSITY / 6.0
)

# Integrals from x=1 to x=2:
#
# integral f_contact(x)*x^2 dx = 17/30
# integral f_correction(x)*x^2 dx = 1/6
CONTACT_BASIS_INTEGRAL = 17.0 / 30.0
CORRECTION_BASIS_INTEGRAL = 1.0 / 6.0


@dataclass(frozen=True)
class LowDensityRDFParameters:
    packing_fraction: float
    reduced_density: float
    contact_value: float
    contact_amplitude: float
    correction_amplitude: float


def _validate_packing_fraction(
    packing_fraction: float,
    enforce_low_density_range: bool,
) -> float:
    eta = float(packing_fraction)

    if not np.isfinite(eta):
        raise ValueError(
            "Packing fraction must be finite."
        )

    if eta < 0.0:
        raise ValueError(
            "Packing fraction cannot be negative."
        )

    if eta >= 1.0:
        raise ValueError(
            "Packing fraction must be below 1."
        )

    if (
        enforce_low_density_range
        and eta > MAX_PACKING_FRACTION
    ):
        raise ValueError(
            "Low-density RDF requested outside its "
            f"recommended range: eta={eta:.8f}, "
            f"limit={MAX_PACKING_FRACTION:.8f}."
        )

    return eta


def cs_contact_rdf(
    packing_fraction: float,
) -> float:
    """Carnahan-Starling hard-sphere contact value."""
    eta = _validate_packing_fraction(
        packing_fraction,
        enforce_low_density_range=False,
    )

    return (
        1.0 - 0.5 * eta
    ) / (1.0 - eta) ** 3


def reduced_density_from_packing_fraction(
    packing_fraction: float,
) -> float:
    """Return rho*sigma^3 = 6*eta/pi."""
    eta = _validate_packing_fraction(
        packing_fraction,
        enforce_low_density_range=False,
    )

    return 6.0 * eta / pi


def _contact_basis(
    distance_reduced: FloatArray,
) -> FloatArray:
    """Normalized hard-sphere overlap-volume shape.

    It equals one at x=1 and zero at x=2.
    """
    x = distance_reduced

    return (
        (4.0 + x)
        * (2.0 - x) ** 2
        / 5.0
    )


def _correction_basis(
    distance_reduced: FloatArray,
) -> FloatArray:
    """Shape that vanishes at both x=1 and x=2."""
    x = distance_reduced

    return (
        (x - 1.0)
        * (2.0 - x) ** 2
    )


def low_density_rdf_parameters(
    packing_fraction: float,
    enforce_low_density_range: bool = True,
) -> LowDensityRDFParameters:
    """Determine RDF amplitudes from contact and compressibility."""
    eta = _validate_packing_fraction(
        packing_fraction,
        enforce_low_density_range,
    )

    reduced_density = (
        reduced_density_from_packing_fraction(
            eta
        )
    )

    contact_value = cs_contact_rdf(eta)
    contact_amplitude = contact_value - 1.0

    if eta <= 1.0e-14:
        correction_amplitude = 0.0
    else:
        target_compressibility = float(
            np.asarray(
                cs_reduced_isothermal_compressibility(
                    eta
                )
            ).reshape(-1)[0]
        )

        # Compressibility sum rule:
        #
        # chi = 1 + 4*pi*rho* integral[
        #     (g(x)-1)*x^2 dx
        # ]
        #
        # The hard core contributes -1/3.
        target_integral = (
            target_compressibility - 1.0
        ) / (
            4.0
            * pi
            * reduced_density
        )

        correction_amplitude = (
            target_integral
            + 1.0 / 3.0
            - (
                contact_amplitude
                * CONTACT_BASIS_INTEGRAL
            )
        ) / CORRECTION_BASIS_INTEGRAL

    return LowDensityRDFParameters(
        packing_fraction=eta,
        reduced_density=reduced_density,
        contact_value=contact_value,
        contact_amplitude=contact_amplitude,
        correction_amplitude=float(
            correction_amplitude
        ),
    )


def hard_sphere_rdf_reduced(
    distance_reduced: ArrayLike,
    packing_fraction: float,
    enforce_low_density_range: bool = True,
) -> FloatArray:
    """Return hard-sphere g0(x), where x=r/sigma."""
    x = np.asarray(
        distance_reduced,
        dtype=np.float64,
    )

    if np.any(~np.isfinite(x)):
        raise ValueError(
            "Distances must be finite."
        )

    if np.any(x < 0.0):
        raise ValueError(
            "Distances cannot be negative."
        )

    parameters = low_density_rdf_parameters(
        packing_fraction,
        enforce_low_density_range,
    )

    result = np.ones_like(
        x,
        dtype=np.float64,
    )

    result[x < 1.0] = 0.0

    middle = (
        (x >= 1.0)
        & (x < 2.0)
    )

    if np.any(middle):
        middle_x = x[middle]

        result[middle] = (
            1.0
            + parameters.contact_amplitude
            * _contact_basis(middle_x)
            + parameters.correction_amplitude
            * _correction_basis(middle_x)
        )

    return result


def compressibility_from_rdf(
    packing_fraction: float,
) -> float:
    """Numerically evaluate the compressibility sum rule."""
    eta = _validate_packing_fraction(
        packing_fraction,
        enforce_low_density_range=True,
    )

    if eta <= 1.0e-14:
        return 1.0

    reduced_density = (
        reduced_density_from_packing_fraction(
            eta
        )
    )

    # Inside the hard core, g-1 = -1 exactly.
    core_integral = -1.0 / 3.0

    def middle_integrand(
        distance_reduced: float,
    ) -> float:
        g_value = float(
            hard_sphere_rdf_reduced(
                np.array([distance_reduced]),
                eta,
            )[0]
        )

        return (
            (g_value - 1.0)
            * distance_reduced**2
        )

    middle_integral, _ = quad(
        middle_integrand,
        1.0,
        2.0,
        epsabs=1.0e-12,
        epsrel=1.0e-12,
        limit=200,
    )

    return (
        1.0
        + 4.0
        * pi
        * reduced_density
        * (
            core_integral
            + middle_integral
        )
    )
