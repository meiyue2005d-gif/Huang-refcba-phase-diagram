"""Generic thermodynamic and fluid-coexistence solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, least_squares


FreeEnergyFunction = Callable[[float], float]


@dataclass(frozen=True)
class ThermodynamicState:
    number_density: float
    beta_free_energy_per_particle: float
    beta_chemical_potential: float
    beta_pressure: float


@dataclass(frozen=True)
class CoexistenceResult:
    vapor_density: float
    liquid_density: float
    beta_pressure: float
    beta_chemical_potential: float

    lower_spinodal_density: float
    upper_spinodal_density: float

    chemical_potential_residual: float
    pressure_residual: float
    maxwell_area_residual: float

    optimizer_success: bool
    optimizer_message: str


def derivative_five_point(
    function: Callable[[float], float],
    x: float,
    relative_step: float = 1.0e-4,
    absolute_step: float = 1.0e-8,
) -> float:
    """Return a five-point central numerical derivative."""
    value = float(x)

    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            "Derivative position must be positive and finite."
        )

    step = max(
        abs(value) * relative_step,
        absolute_step,
    )

    # Ensure x - 2h remains positive.
    step = min(
        step,
        0.24 * value,
    )

    if step <= 0.0:
        raise ValueError(
            "Unable to construct a positive derivative step."
        )

    samples = [
        float(function(value - 2.0 * step)),
        float(function(value - step)),
        float(function(value + step)),
        float(function(value + 2.0 * step)),
    ]

    if not np.isfinite(samples).all():
        raise FloatingPointError(
            "Non-finite value encountered during differentiation."
        )

    return (
        samples[0]
        - 8.0 * samples[1]
        + 8.0 * samples[2]
        - samples[3]
    ) / (12.0 * step)


def thermodynamic_state(
    beta_free_energy_per_particle: FreeEnergyFunction,
    number_density: float,
) -> ThermodynamicState:
    """Calculate beta*mu and beta*P from beta*a(rho).

    beta*mu = a + rho da/drho
    beta*P  = rho^2 da/drho
    """
    density = float(number_density)

    if not np.isfinite(density) or density <= 0.0:
        raise ValueError(
            "Number density must be positive and finite."
        )

    free_energy = float(
        beta_free_energy_per_particle(density)
    )

    derivative = derivative_five_point(
        beta_free_energy_per_particle,
        density,
    )

    chemical_potential = (
        free_energy
        + density * derivative
    )

    pressure = (
        density**2
        * derivative
    )

    return ThermodynamicState(
        number_density=density,
        beta_free_energy_per_particle=free_energy,
        beta_chemical_potential=float(
            chemical_potential
        ),
        beta_pressure=float(pressure),
    )


def beta_pressure_derivative(
    beta_free_energy_per_particle: FreeEnergyFunction,
    number_density: float,
) -> float:
    """Return d(beta*P)/d rho."""

    def pressure_function(
        density: float,
    ) -> float:
        return thermodynamic_state(
            beta_free_energy_per_particle,
            density,
        ).beta_pressure

    return derivative_five_point(
        pressure_function,
        number_density,
        relative_step=5.0e-4,
    )


def find_spinodal_densities(
    beta_free_energy_per_particle: FreeEnergyFunction,
    minimum_density: float,
    maximum_density: float,
    grid_points: int = 1000,
) -> tuple[float, ...]:
    """Find roots of d(beta P)/d rho = 0."""
    rho_min = float(minimum_density)
    rho_max = float(maximum_density)

    if rho_min <= 0.0 or rho_max <= rho_min:
        raise ValueError(
            "Invalid density interval."
        )

    if grid_points < 20:
        raise ValueError(
            "grid_points must be at least 20."
        )

    density_grid = np.geomspace(
        rho_min,
        rho_max,
        grid_points,
    )

    derivatives = np.array(
        [
            beta_pressure_derivative(
                beta_free_energy_per_particle,
                density,
            )
            for density in density_grid
        ],
        dtype=np.float64,
    )

    roots: list[float] = []

    for left, right, f_left, f_right in zip(
        density_grid[:-1],
        density_grid[1:],
        derivatives[:-1],
        derivatives[1:],
    ):
        if not (
            np.isfinite(f_left)
            and np.isfinite(f_right)
        ):
            continue

        if f_left == 0.0:
            roots.append(float(left))
            continue

        if f_left * f_right < 0.0:
            root = brentq(
                lambda density: (
                    beta_pressure_derivative(
                        beta_free_energy_per_particle,
                        density,
                    )
                ),
                float(left),
                float(right),
                xtol=1.0e-12,
                rtol=1.0e-10,
                maxiter=200,
            )

            if (
                not roots
                or abs(root - roots[-1])
                > 1.0e-7
            ):
                roots.append(float(root))

    return tuple(roots)


def maxwell_area_residual(
    beta_free_energy_per_particle: FreeEnergyFunction,
    vapor_density: float,
    liquid_density: float,
    coexistence_pressure: float,
) -> float:
    """Evaluate the Maxwell equal-area residual.

    In density coordinates:

        integral[(P(rho)-Pcoex)/rho^2] d rho
    """
    rho_v = float(vapor_density)
    rho_l = float(liquid_density)
    pressure_coex = float(
        coexistence_pressure
    )

    if not (
        0.0 < rho_v < rho_l
    ):
        raise ValueError(
            "Require 0 < vapor density < liquid density."
        )

    def integrand(
        density: float,
    ) -> float:
        pressure = thermodynamic_state(
            beta_free_energy_per_particle,
            density,
        ).beta_pressure

        return (
            pressure
            - pressure_coex
        ) / density**2

    value, _ = quad(
        integrand,
        rho_v,
        rho_l,
        epsabs=1.0e-9,
        epsrel=1.0e-9,
        limit=500,
    )

    return float(value)


def solve_fluid_coexistence(
    beta_free_energy_per_particle: FreeEnergyFunction,
    minimum_density: float,
    maximum_density: float,
    grid_points: int = 1000,
) -> CoexistenceResult:
    """Solve equal-pressure and equal-chemical-potential conditions."""
    rho_min = float(minimum_density)
    rho_max = float(maximum_density)

    spinodals = find_spinodal_densities(
        beta_free_energy_per_particle,
        rho_min,
        rho_max,
        grid_points=grid_points,
    )

    if len(spinodals) < 2:
        raise RuntimeError(
            "Fewer than two spinodal roots were found; "
            "no van der Waals loop was detected."
        )

    lower_spinodal = spinodals[0]
    upper_spinodal = spinodals[-1]

    vapor_upper = (
        lower_spinodal
        * (1.0 - 1.0e-5)
    )

    liquid_lower = (
        upper_spinodal
        * (1.0 + 1.0e-5)
    )

    if not (
        rho_min < vapor_upper
        < liquid_lower < rho_max
    ):
        raise RuntimeError(
            "Spinodal roots do not define valid coexistence bounds."
        )

    vapor_initial = np.sqrt(
        rho_min * vapor_upper
    )

    liquid_initial = np.sqrt(
        liquid_lower * rho_max
    )

    vapor_state_initial = thermodynamic_state(
        beta_free_energy_per_particle,
        vapor_initial,
    )

    liquid_state_initial = thermodynamic_state(
        beta_free_energy_per_particle,
        liquid_initial,
    )

    mu_scale = max(
        1.0,
        abs(
            vapor_state_initial
            .beta_chemical_potential
        ),
        abs(
            liquid_state_initial
            .beta_chemical_potential
        ),
    )

    pressure_scale = max(
        1.0e-6,
        abs(
            vapor_state_initial
            .beta_pressure
        ),
        abs(
            liquid_state_initial
            .beta_pressure
        ),
    )

    def residuals(
        log_densities: np.ndarray,
    ) -> np.ndarray:
        rho_v = float(
            np.exp(log_densities[0])
        )

        rho_l = float(
            np.exp(log_densities[1])
        )

        vapor_state = thermodynamic_state(
            beta_free_energy_per_particle,
            rho_v,
        )

        liquid_state = thermodynamic_state(
            beta_free_energy_per_particle,
            rho_l,
        )

        return np.array(
            [
                (
                    vapor_state
                    .beta_chemical_potential
                    -
                    liquid_state
                    .beta_chemical_potential
                ) / mu_scale,
                (
                    vapor_state.beta_pressure
                    -
                    liquid_state.beta_pressure
                ) / pressure_scale,
            ],
            dtype=np.float64,
        )

    solution = least_squares(
        residuals,
        x0=np.log(
            [
                vapor_initial,
                liquid_initial,
            ]
        ),
        bounds=(
            np.log(
                [
                    rho_min,
                    liquid_lower,
                ]
            ),
            np.log(
                [
                    vapor_upper,
                    rho_max,
                ]
            ),
        ),
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=1000,
    )

    vapor_density = float(
        np.exp(solution.x[0])
    )

    liquid_density = float(
        np.exp(solution.x[1])
    )

    vapor_state = thermodynamic_state(
        beta_free_energy_per_particle,
        vapor_density,
    )

    liquid_state = thermodynamic_state(
        beta_free_energy_per_particle,
        liquid_density,
    )

    pressure_residual = (
        vapor_state.beta_pressure
        - liquid_state.beta_pressure
    )

    chemical_potential_residual = (
        vapor_state.beta_chemical_potential
        -
        liquid_state.beta_chemical_potential
    )

    coexistence_pressure = 0.5 * (
        vapor_state.beta_pressure
        + liquid_state.beta_pressure
    )

    coexistence_mu = 0.5 * (
        vapor_state.beta_chemical_potential
        +
        liquid_state.beta_chemical_potential
    )

    area_residual = maxwell_area_residual(
        beta_free_energy_per_particle,
        vapor_density,
        liquid_density,
        coexistence_pressure,
    )

    return CoexistenceResult(
        vapor_density=vapor_density,
        liquid_density=liquid_density,
        beta_pressure=float(
            coexistence_pressure
        ),
        beta_chemical_potential=float(
            coexistence_mu
        ),
        lower_spinodal_density=float(
            lower_spinodal
        ),
        upper_spinodal_density=float(
            upper_spinodal
        ),
        chemical_potential_residual=float(
            chemical_potential_residual
        ),
        pressure_residual=float(
            pressure_residual
        ),
        maxwell_area_residual=float(
            area_residual
        ),
        optimizer_success=bool(
            solution.success
        ),
        optimizer_message=str(
            solution.message
        ),
    )
