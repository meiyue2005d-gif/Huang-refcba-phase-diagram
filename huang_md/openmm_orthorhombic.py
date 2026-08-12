"""OpenMM builders for orthorhombic refCBA simulation boxes."""

from __future__ import annotations

from openmm import AndersenThermostat, System, Vec3, unit
from openmm.app import Element, Topology

from huang_md.openmm_model import create_pair_force
from huang_md.parameters import HuangPotentialParameters
from huang_md.slab_initialization import OrthorhombicBox


def create_orthorhombic_system(
    params: HuangPotentialParameters,
    n_particles: int,
    molecular_weight_kDa: float,
    box: OrthorhombicBox,
    collision_frequency_per_ps: float,
) -> System:
    """Create an NVT OpenMM system in an orthorhombic periodic box."""
    params.validate()

    if n_particles <= 0:
        raise ValueError("n_particles must be positive.")

    if molecular_weight_kDa <= 0.0:
        raise ValueError(
            "molecular_weight_kDa must be positive."
        )

    if collision_frequency_per_ps <= 0.0:
        raise ValueError(
            "collision_frequency_per_ps must be positive."
        )

    cutoff_nm = (
        params.cutoff_reduced
        * params.diameter_nm
    )

    minimum_box_length_nm = min(
        box.lengths_nm
    )

    if cutoff_nm >= minimum_box_length_nm / 2.0:
        raise ValueError(
            "Pair cutoff must be smaller than half the "
            "shortest periodic box length. "
            f"cutoff={cutoff_nm:.6f} nm, "
            f"min(L)/2={minimum_box_length_nm / 2.0:.6f} nm."
        )

    system = System()

    particle_mass_dalton = (
        molecular_weight_kDa
        * 1000.0
    )

    for _ in range(n_particles):
        system.addParticle(
            particle_mass_dalton
            * unit.dalton
        )

    system.setDefaultPeriodicBoxVectors(
        Vec3(
            box.length_x_nm,
            0.0,
            0.0,
        ) * unit.nanometer,
        Vec3(
            0.0,
            box.length_y_nm,
            0.0,
        ) * unit.nanometer,
        Vec3(
            0.0,
            0.0,
            box.length_z_nm,
        ) * unit.nanometer,
    )

    system.addForce(
        create_pair_force(
            params=params,
            n_particles=n_particles,
        )
    )

    system.addForce(
        AndersenThermostat(
            params.temperature_K
            * unit.kelvin,
            collision_frequency_per_ps
            / unit.picosecond,
        )
    )

    return system


def create_orthorhombic_topology(
    n_particles: int,
    box: OrthorhombicBox,
) -> Topology:
    """Create a minimal one-atom-per-colloid topology."""
    if n_particles <= 0:
        raise ValueError(
            "n_particles must be positive."
        )

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

    topology.setUnitCellDimensions(
        Vec3(
            box.length_x_nm,
            box.length_y_nm,
            box.length_z_nm,
        ) * unit.nanometer
    )

    return topology
