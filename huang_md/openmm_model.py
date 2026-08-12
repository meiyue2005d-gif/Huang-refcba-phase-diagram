"""OpenMM implementation of the Huang/refCBA colloidal model."""

from __future__ import annotations

from math import exp

import numpy as np
from numpy.typing import NDArray

from openmm import (
    AndersenThermostat,
    CustomNonbondedForce,
    System,
    Vec3,
    unit,
)
from openmm.app import Element, Topology

from huang_md.parameters import HuangPotentialParameters


FloatArray = NDArray[np.float64]

R_KJ_MOL_K = 8.31446261815324e-3


def calculate_kbt_kj_mol(
    temperature_K: float,
) -> float:
    """Return kBT in OpenMM energy units of kJ/mol."""
    if temperature_K <= 0:
        raise ValueError("temperature_K must be positive.")

    return R_KJ_MOL_K * temperature_K


def calculate_inner_matching_constants(
    params: HuangPotentialParameters,
) -> tuple[float, float]:
    """Return linear continuation intercept and slope.

    The inner Gaussian-core potential is

        epsilon*exp(-x^2/sigma^2) + a0 + a1*(x-1)

    and is matched in value and derivative to the two-Yukawa
    potential at x=1.
    """
    sigma = params.gaussian_sigma_reduced
    epsilon = params.gaussian_epsilon_kBT

    u_at_one = -params.K1_kBT + params.K2_kBT

    derivative_at_one = (
        params.K1_kBT * (params.Z1 + 1.0)
        - params.K2_kBT * (params.Z2 + 1.0)
    )

    gaussian_at_one = exp(-1.0 / sigma**2)

    gaussian_derivative_at_one = (
        -2.0
        / sigma**2
        * gaussian_at_one
    )

    intercept = (
        u_at_one
        - epsilon * gaussian_at_one
    )

    slope = (
        derivative_at_one
        - epsilon * gaussian_derivative_at_one
    )

    return intercept, slope


def calculate_cutoff_shift_reduced(
    params: HuangPotentialParameters,
) -> float:
    """Two-Yukawa potential at the cutoff in units of kBT."""
    x_cut = params.cutoff_reduced

    attraction = (
        -params.K1_kBT
        * exp(-params.Z1 * (x_cut - 1.0))
        / x_cut
    )

    repulsion = (
        params.K2_kBT
        * exp(-params.Z2 * (x_cut - 1.0))
        / x_cut
    )

    return attraction + repulsion


def create_pair_force(
    params: HuangPotentialParameters,
    n_particles: int,
) -> CustomNonbondedForce:
    """Create the complete piecewise pair interaction."""
    params.validate()

    if n_particles <= 0:
        raise ValueError("n_particles must be positive.")

    intercept, slope = calculate_inner_matching_constants(
        params
    )

    cutoff_shift = calculate_cutoff_shift_reduced(
        params
    )

    kbt_kj_mol = calculate_kbt_kj_mol(
        params.temperature_K
    )

    expression = (
        "kBT*(raw-ucut);"
        "raw=outer*step(x-1)+inner*(1-step(x-1));"
        "outer=-K1*exp(-Z1*(x-1))/x"
        "+K2*exp(-Z2*(x-1))/x;"
        "inner=epsilon*exp(-(x*x)/(sigma*sigma))"
        "+a0+a1*(x-1);"
        "x=r/d"
    )

    force = CustomNonbondedForce(expression)

    force.addGlobalParameter(
        "kBT",
        kbt_kj_mol,
    )
    force.addGlobalParameter(
        "K1",
        params.K1_kBT,
    )
    force.addGlobalParameter(
        "Z1",
        params.Z1,
    )
    force.addGlobalParameter(
        "K2",
        params.K2_kBT,
    )
    force.addGlobalParameter(
        "Z2",
        params.Z2,
    )
    force.addGlobalParameter(
        "epsilon",
        params.gaussian_epsilon_kBT,
    )
    force.addGlobalParameter(
        "sigma",
        params.gaussian_sigma_reduced,
    )
    force.addGlobalParameter(
        "a0",
        intercept,
    )
    force.addGlobalParameter(
        "a1",
        slope,
    )
    force.addGlobalParameter(
        "ucut",
        cutoff_shift,
    )
    force.addGlobalParameter(
        "d",
        params.diameter_nm,
    )

    for _ in range(n_particles):
        force.addParticle([])

    force.setNonbondedMethod(
        CustomNonbondedForce.CutoffPeriodic
    )

    force.setCutoffDistance(
        params.cutoff_reduced
        * params.diameter_nm
        * unit.nanometer
    )

    force.setUseSwitchingFunction(False)
    force.setUseLongRangeCorrection(False)

    return force


def create_system(
    params: HuangPotentialParameters,
    n_particles: int,
    molecular_weight_kDa: float,
    box_length_nm: float,
    collision_frequency_per_ps: float,
) -> System:
    """Create an OpenMM NVT colloidal system."""
    if molecular_weight_kDa <= 0:
        raise ValueError(
            "molecular_weight_kDa must be positive."
        )

    if box_length_nm <= 0:
        raise ValueError(
            "box_length_nm must be positive."
        )

    if collision_frequency_per_ps <= 0:
        raise ValueError(
            "collision_frequency_per_ps must be positive."
        )

    cutoff_nm = (
        params.cutoff_reduced
        * params.diameter_nm
    )

    if cutoff_nm >= box_length_nm / 2.0:
        raise ValueError(
            "Pair cutoff must be smaller than half the "
            f"periodic box length. cutoff={cutoff_nm:.6f} nm, "
            f"L/2={box_length_nm / 2.0:.6f} nm."
        )

    system = System()

    particle_mass_dalton = molecular_weight_kDa * 1000.0

    for _ in range(n_particles):
        system.addParticle(
            particle_mass_dalton * unit.dalton
        )

    system.setDefaultPeriodicBoxVectors(
        Vec3(box_length_nm, 0.0, 0.0) * unit.nanometer,
        Vec3(0.0, box_length_nm, 0.0) * unit.nanometer,
        Vec3(0.0, 0.0, box_length_nm) * unit.nanometer,
    )

    pair_force = create_pair_force(
        params=params,
        n_particles=n_particles,
    )

    system.addForce(pair_force)

    thermostat = AndersenThermostat(
        params.temperature_K * unit.kelvin,
        collision_frequency_per_ps
        / unit.picosecond,
    )

    system.addForce(thermostat)

    return system


def create_colloid_topology(
    n_particles: int,
    box_length_nm: float,
) -> Topology:
    """Create a minimal one-atom-per-colloid topology."""
    topology = Topology()
    chain = topology.addChain("A")

    carbon = Element.getBySymbol("C")

    for index in range(n_particles):
        residue = topology.addResidue(
            "COL",
            chain,
            id=str(index + 1),
        )

        topology.addAtom(
            "C",
            carbon,
            residue,
            id=str(index + 1),
        )

    # For an orthogonal cubic box, setUnitCellDimensions avoids
    # nested Quantity objects in Topology.setPeriodicBoxVectors().
    topology.setUnitCellDimensions(
        Vec3(
            float(box_length_nm),
            float(box_length_nm),
            float(box_length_nm),
        ) * unit.nanometer
    )

    return topology


def generate_random_positions_nm(
    n_particles: int,
    box_length_nm: float,
    minimum_distance_nm: float,
    seed: int,
    maximum_attempts_per_particle: int = 100000,
) -> FloatArray:
    """Generate dispersed random positions with periodic separation."""
    if minimum_distance_nm <= 0:
        raise ValueError(
            "minimum_distance_nm must be positive."
        )

    rng = np.random.default_rng(seed)

    positions = np.empty(
        (n_particles, 3),
        dtype=np.float64,
    )

    for particle_index in range(n_particles):
        accepted = False

        for _ in range(maximum_attempts_per_particle):
            candidate = rng.uniform(
                0.0,
                box_length_nm,
                size=3,
            )

            if particle_index == 0:
                accepted = True
            else:
                displacement = (
                    candidate
                    - positions[:particle_index]
                )

                displacement -= (
                    box_length_nm
                    * np.round(
                        displacement / box_length_nm
                    )
                )

                squared_distance = np.sum(
                    displacement**2,
                    axis=1,
                )

                accepted = bool(
                    np.all(
                        squared_distance
                        >= minimum_distance_nm**2
                    )
                )

            if accepted:
                positions[particle_index] = candidate
                break

        if not accepted:
            raise RuntimeError(
                "Could not generate a nonoverlapping initial "
                f"position for particle {particle_index}. "
                "Try reducing the minimum initial separation."
            )

    return positions
