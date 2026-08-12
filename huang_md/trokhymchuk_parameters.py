"""Parameters for the Trokhymchuk hard-sphere RDF.

References
----------
Trokhymchuk et al., J. Chem. Phys. 123, 024501 (2005).
Erratum, J. Chem. Phys. 124, 149902 (2006).

The fitted high-density representation is intended for:

    0.2 <= rho * sigma^3 <= 0.9

All inverse-length parameters are expressed in units of 1/sigma.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import numpy as np


MIN_REDUCED_DENSITY = 0.2
MAX_REDUCED_DENSITY = 0.9


@dataclass(frozen=True)
class TrokhymchukParameters:
    """Dimensionless parameters at one hard-sphere density."""

    reduced_density: float
    packing_fraction: float

    wertheim_d: float

    mu: float
    alpha_py: float
    beta_py: float
    gamma: float

    omega: float
    kappa: float

    alpha: float
    beta: float

    minimum_position: float
    minimum_rdf: float
    contact_rdf: float


def validate_reduced_density(
    reduced_density: float,
) -> float:
    value = float(reduced_density)

    if not np.isfinite(value):
        raise ValueError(
            "Reduced density must be finite."
        )

    if not (
        MIN_REDUCED_DENSITY
        <= value
        <= MAX_REDUCED_DENSITY
    ):
        raise ValueError(
            "Trokhymchuk fitted RDF requires "
            f"{MIN_REDUCED_DENSITY} <= rho*sigma^3 "
            f"<= {MAX_REDUCED_DENSITY}; "
            f"received {value}."
        )

    return value


def packing_fraction_from_reduced_density(
    reduced_density: float,
) -> float:
    """Return eta = pi*rho*sigma^3/6."""
    value = validate_reduced_density(
        reduced_density
    )

    return pi * value / 6.0


def wertheim_d(
    packing_fraction: float,
) -> float:
    """Return the auxiliary Wertheim cubic-root parameter."""
    eta = float(packing_fraction)

    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError(
            "Packing fraction must be positive and finite."
        )

    inner_square_root = sqrt(
        3.0
        * (
            eta**4
            - 2.0 * eta**3
            + eta**2
            + 6.0 * eta
            + 3.0
        )
    )

    radicand = (
        2.0
        * eta
        * (
            eta**2
            - 3.0 * eta
            - 3.0
            + inner_square_root
        )
    )

    if radicand <= 0.0:
        raise FloatingPointError(
            "Non-positive Wertheim radicand."
        )

    return radicand ** (1.0 / 3.0)


def corrected_mu(
    packing_fraction: float,
    d_value: float,
) -> float:
    """Return corrected mu*sigma from the 2006 erratum.

    Corrected Eq. 29:

        mu*sigma = 2*eta/(1-eta)
                   * [-1 - d/(2*eta) + eta/d]
    """
    eta = float(packing_fraction)
    d = float(d_value)

    if not 0.0 < eta < 1.0:
        raise ValueError(
            "Packing fraction must lie between zero and one."
        )

    if not np.isfinite(d) or d <= 0.0:
        raise ValueError(
            "Wertheim d must be positive and finite."
        )

    return (
        2.0
        * eta
        / (1.0 - eta)
        * (
            -1.0
            - d / (2.0 * eta)
            + eta / d
        )
    )


def py_alpha(
    packing_fraction: float,
    d_value: float,
) -> float:
    """Return the Percus-Yevick alpha_0*sigma."""
    eta = float(packing_fraction)
    d = float(d_value)

    return (
        2.0
        * eta
        / (1.0 - eta)
        * (
            -1.0
            + d / (4.0 * eta)
            - eta / (2.0 * d)
        )
    )


def py_beta(
    packing_fraction: float,
    d_value: float,
) -> float:
    """Return the Percus-Yevick beta_0*sigma."""
    eta = float(packing_fraction)
    d = float(d_value)

    return (
        2.0
        * eta
        / (1.0 - eta)
        * sqrt(3.0)
        * (
            -d / (4.0 * eta)
            - eta / (2.0 * d)
        )
    )


def corrected_gamma(
    packing_fraction: float,
    mu_value: float,
    alpha_value: float,
    beta_value: float,
) -> float:
    """Return corrected gamma from erratum Eq. 30.

    All inverse-length parameters are already reduced by sigma,
    so the explicit sigma factors in Eq. 30 equal one.
    """
    eta = float(packing_fraction)
    mu = float(mu_value)
    alpha = float(alpha_value)
    beta = float(beta_value)

    values = np.array(
        [eta, mu, alpha, beta],
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Gamma inputs must all be finite."
        )

    if not 0.0 < eta < 1.0:
        raise ValueError(
            "Packing fraction must lie between zero and one."
        )

    if abs(beta) <= 1.0e-14:
        raise ValueError(
            "beta_py is too close to zero for Eq. 30."
        )

    alpha_squared = alpha**2
    beta_squared = beta**2
    sum_squared = alpha_squared + beta_squared

    numerator = (
        (
            alpha * sum_squared
            - mu * (
                alpha_squared
                - beta_squared
            )
        )
        * (
            1.0
            + 0.5 * eta
        )
        + (
            sum_squared
            - mu * alpha
        )
        * (
            1.0
            + 2.0 * eta
        )
    )

    denominator = (
        (
            sum_squared
            - 2.0 * mu * alpha
        )
        * (
            1.0
            + 0.5 * eta
        )
        - mu
        * (
            1.0
            + 2.0 * eta
        )
    )

    if abs(denominator) <= 1.0e-14:
        raise FloatingPointError(
            "Corrected gamma denominator is too close to zero."
        )

    argument = (
        -numerator
        / (
            beta
            * denominator
        )
    )

    return float(np.arctan(argument))


def structural_omega(
    packing_fraction: float,
) -> float:
    """Return omega*sigma, Eq. 27."""
    eta = float(packing_fraction)

    return (
        -0.682 * np.exp(-24.697 * eta)
        + 4.720
        + 4.450 * eta
    )


def structural_kappa(
    packing_fraction: float,
) -> float:
    """Return kappa*sigma, Eq. 28."""
    eta = float(packing_fraction)

    return (
        4.674 * np.exp(-3.935 * eta)
        + 3.536 * np.exp(-56.270 * eta)
    )


def fitted_alpha(
    packing_fraction: float,
) -> float:
    """Return fitted alpha*sigma, Eq. 33."""
    eta = float(packing_fraction)

    return (
        44.554
        + 79.868 * eta
        + 116.432 * eta**2
        - 44.652 * np.exp(2.0 * eta)
    )


def fitted_beta(
    packing_fraction: float,
) -> float:
    """Return fitted beta*sigma, Eq. 34."""
    eta = float(packing_fraction)

    return (
        -5.022
        + 5.857 * eta
        + 5.089 * np.exp(-4.0 * eta)
    )


def fitted_minimum_position(
    packing_fraction: float,
) -> float:
    """Return r_min/sigma."""
    eta = float(packing_fraction)

    return (
        2.0116
        - 1.0647 * eta
        + 0.0538 * eta**2
    )


def fitted_minimum_rdf(
    packing_fraction: float,
) -> float:
    """Return the RDF value at the first minimum."""
    eta = float(packing_fraction)

    return (
        1.0286
        - 0.6095 * eta
        + 3.5781 * eta**2
        - 21.3651 * eta**3
        + 42.6344 * eta**4
        - 33.8485 * eta**5
    )


def experimental_contact_rdf(
    packing_fraction: float,
) -> float:
    """Return the Kolafa contact value used in the model."""
    eta = float(packing_fraction)

    if not 0.0 < eta < 1.0:
        raise ValueError(
            "Packing fraction must lie between zero and one."
        )

    compressibility_factor = (
        1.0
        + eta
        + eta**2
        - (2.0 / 3.0) * eta**3
        - (2.0 / 3.0) * eta**4
    ) / (1.0 - eta) ** 3

    return (
        compressibility_factor - 1.0
    ) / (4.0 * eta)


def calculate_trokhymchuk_parameters(
    reduced_density: float,
) -> TrokhymchukParameters:
    """Calculate all unambiguous high-density RDF parameters."""
    rho_star = validate_reduced_density(
        reduced_density
    )

    eta = packing_fraction_from_reduced_density(
        rho_star
    )

    d_value = wertheim_d(eta)

    result = TrokhymchukParameters(
        reduced_density=rho_star,
        packing_fraction=eta,
        wertheim_d=d_value,
        mu=corrected_mu(
            eta,
            d_value,
        ),
        alpha_py=py_alpha(
            eta,
            d_value,
        ),
        beta_py=py_beta(
            eta,
            d_value,
        ),
        gamma=corrected_gamma(
            eta,
            corrected_mu(
                eta,
                d_value,
            ),
            py_alpha(
                eta,
                d_value,
            ),
            py_beta(
                eta,
                d_value,
            ),
        ),
        omega=structural_omega(eta),
        kappa=structural_kappa(eta),
        alpha=fitted_alpha(eta),
        beta=fitted_beta(eta),
        minimum_position=(
            fitted_minimum_position(eta)
        ),
        minimum_rdf=(
            fitted_minimum_rdf(eta)
        ),
        contact_rdf=(
            experimental_contact_rdf(eta)
        ),
    )

    values = np.array(
        [
            result.packing_fraction,
            result.wertheim_d,
            result.mu,
            result.alpha_py,
            result.beta_py,
            result.gamma,
            result.omega,
            result.kappa,
            result.alpha,
            result.beta,
            result.minimum_position,
            result.minimum_rdf,
            result.contact_rdf,
        ],
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise FloatingPointError(
            "Non-finite Trokhymchuk parameter generated."
        )

    return result
