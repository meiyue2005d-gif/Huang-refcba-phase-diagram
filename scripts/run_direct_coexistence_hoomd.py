#!/usr/bin/env python3
"""Run one refCBA state with HOOMD-blue GPU."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import hoomd
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--equil-steps",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--prod-steps",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--report-interval",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--minimize-max-steps",
        type=int,
        default=500,
    )

    return parser


def write_thermo_header(
    path: Path,
) -> tuple[Any, csv.DictWriter]:
    handle = path.open(
        "w",
        newline="",
        encoding="utf-8",
    )

    fieldnames = [
        "phase",
        "logical_step",
        "hoomd_step",
        "time_ps",
        "potential_energy_kBT",
        "kinetic_energy_kBT",
        "total_energy_kBT",
        "kinetic_temperature_kBT",
    ]

    writer = csv.DictWriter(
        handle,
        fieldnames=fieldnames,
    )
    writer.writeheader()

    return handle, writer


def write_thermo_row(
    writer: csv.DictWriter,
    phase: str,
    logical_step: int,
    simulation: hoomd.Simulation,
    thermo: hoomd.md.compute.ThermodynamicQuantities,
    timestep_fs: float,
) -> None:
    potential = float(thermo.potential_energy)
    kinetic = float(thermo.kinetic_energy)

    writer.writerow(
        {
            "phase": phase,
            "logical_step": logical_step,
            "hoomd_step": int(simulation.timestep),
            "time_ps": (
                logical_step
                * timestep_fs
                / 1000.0
            ),
            "potential_energy_kBT": potential,
            "kinetic_energy_kBT": kinetic,
            "total_energy_kBT": (
                potential + kinetic
            ),
            "kinetic_temperature_kBT": float(
                thermo.kinetic_temperature
            ),
        }
    )


def main() -> None:
    args = build_parser().parse_args()

    for value, name in [
        (args.equil_steps, "equil-steps"),
        (args.prod_steps, "prod-steps"),
        (args.report_interval, "report-interval"),
    ]:
        if value <= 0:
            raise ValueError(f"{name} must be positive.")

    if args.minimize_max_steps < 0:
        raise ValueError(
            "minimize-max-steps cannot be negative."
        )

    input_dir = args.input_dir.resolve()

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    input_file = input_dir / "hoomd_input.npz"
    input_metadata_file = (
        input_dir / "hoomd_input_metadata.json"
    )

    if not input_file.exists():
        raise FileNotFoundError(input_file)

    if not input_metadata_file.exists():
        raise FileNotFoundError(input_metadata_file)

    data = np.load(
        input_file,
        allow_pickle=False,
    )

    with input_metadata_file.open(
        "r",
        encoding="utf-8",
    ) as handle:
        metadata = json.load(handle)

    positions_reduced = np.asarray(
        data["positions_reduced"],
        dtype=np.float64,
    )
    table_U = np.asarray(
        data["table_U"],
        dtype=np.float64,
    )
    table_F = np.asarray(
        data["table_F"],
        dtype=np.float64,
    )

    r_min = float(data["r_min"])
    r_cut = float(data["r_cut"])
    box_lengths_reduced = np.asarray(
        data["box_lengths_reduced"],
        dtype=np.float64,
    )

    if box_lengths_reduced.shape != (3,):
        raise ValueError(
            "box_lengths_reduced must have shape (3,)."
        )

    # Compatibility alias for inherited diagnostics.
    box_length_reduced = float(
        box_lengths_reduced[0]
    )

    n_particles = int(metadata["n_particles"])
    timestep_reduced = float(
        metadata["timestep_reduced"]
    )
    timestep_fs = float(metadata["timestep_fs"])
    gamma_reduced = float(
        metadata["langevin_gamma_reduced"]
    )
    diameter_nm = float(metadata["diameter_nm"])
    box_lengths_nm = np.asarray(
        metadata["box_lengths_nm"],
        dtype=np.float64,
    )

    if box_lengths_nm.shape != (3,):
        raise ValueError(
            "box_lengths_nm must have shape (3,)."
        )

    # Compatibility alias for inherited diagnostics.
    box_length_nm = float(
        box_lengths_nm[0]
    )
    hoomd_seed = int(metadata["hoomd_seed"])

    if positions_reduced.shape != (
        n_particles,
        3,
    ):
        raise ValueError(
            "Unexpected initial-position array shape."
        )

    device = hoomd.device.GPU(
        notice_level=2,
    )

    simulation = hoomd.Simulation(
        device=device,
        seed=hoomd_seed,
    )

    snapshot = hoomd.Snapshot(
        device.communicator
    )

    if snapshot.communicator.rank == 0:
        snapshot.configuration.box = [
            float(box_lengths_reduced[0]),
            float(box_lengths_reduced[1]),
            float(box_lengths_reduced[2]),
            0.0,
            0.0,
            0.0,
        ]

        snapshot.particles.N = n_particles
        snapshot.particles.types = ["A"]
        snapshot.particles.typeid[:] = 0
        snapshot.particles.mass[:] = 1.0
        snapshot.particles.diameter[:] = 1.0
        snapshot.particles.position[:] = (
            positions_reduced
        )

    simulation.create_state_from_snapshot(
        snapshot
    )

    nlist = hoomd.md.nlist.Cell(
        buffer=0.4,
    )

    pair_table = hoomd.md.pair.Table(
        nlist=nlist,
        default_r_cut=r_cut,
    )

    pair_table.params[("A", "A")] = {
        "r_min": r_min,
        "U": table_U.tolist(),
        "F": table_F.tolist(),
    }
    pair_table.r_cut[("A", "A")] = r_cut

    all_particles = hoomd.filter.All()

    # FIRE uses a larger adaptive maximum step than production MD.
    fire_dt = min(
        5.0e-4,
        max(
            100.0 * timestep_reduced,
            1.0e-5,
        ),
    )

    fire = hoomd.md.minimize.FIRE(
        dt=fire_dt,
        force_tol=1.0e-2,
        angmom_tol=1.0e-2,
        energy_tol=1.0e-7,
        forces=[pair_table],
        methods=[
            hoomd.md.methods.ConstantVolume(
                filter=all_particles,
            )
        ],
    )

    simulation.operations.integrator = fire
    simulation.run(0)

    initial_potential = float(
        pair_table.energy
    )

    minimization_steps = 0

    while (
        not fire.converged
        and minimization_steps
        < args.minimize_max_steps
    ):
        chunk = min(
            50,
            args.minimize_max_steps
            - minimization_steps,
        )

        simulation.run(chunk)
        minimization_steps += chunk

    minimized_potential = float(
        pair_table.energy
    )

    langevin = hoomd.md.methods.Langevin(
        filter=all_particles,
        kT=1.0,
        default_gamma=gamma_reduced,
    )

    integrator = hoomd.md.Integrator(
        dt=timestep_reduced,
        methods=[langevin],
        forces=[pair_table],
    )

    simulation.operations.integrator = integrator

    thermo = (
        hoomd.md.compute.ThermodynamicQuantities(
            filter=all_particles
        )
    )
    simulation.operations.computes.append(
        thermo
    )

    simulation.state.thermalize_particle_momenta(
        filter=all_particles,
        kT=1.0,
    )

    simulation.run(0)

    print("=" * 72)
    print("Huang/refCBA HOOMD-blue simulation")
    print("=" * 72)
    print("HOOMD version          :", hoomd.version.version)
    print("GPU device             :", device.device)
    print("Output directory       :", output_dir)
    print("pH                     :", metadata["pH"])
    print(
        "Added NaCl             :",
        metadata["added_nacl_mM"],
        "mM",
    )
    print(
        "Concentration          :",
        metadata["concentration_mg_ml"],
        "mg/mL",
    )
    print("Particles              :", n_particles)
    print("Box length             :", box_length_nm, "nm")
    print("Reduced box length     :", box_length_reduced)
    print("Reduced timestep       :", timestep_reduced)
    print("Time unit              :", metadata["time_unit_ps"], "ps")
    print("Langevin gamma         :", gamma_reduced)
    print("Initial potential      :", initial_potential, "kBT")
    print("Minimized potential    :", minimized_potential, "kBT")
    print("FIRE steps             :", minimization_steps)
    print("FIRE converged         :", fire.converged)
    print("Equilibration steps    :", args.equil_steps)
    print("Production steps       :", args.prod_steps)

    equil_handle, equil_writer = write_thermo_header(
        output_dir / "equilibration_thermo.csv"
    )

    logical_step = 0

    try:
        remaining = args.equil_steps

        while remaining > 0:
            chunk = min(
                args.report_interval,
                remaining,
            )
            simulation.run(chunk)

            logical_step += chunk
            remaining -= chunk

            write_thermo_row(
                writer=equil_writer,
                phase="equilibration",
                logical_step=logical_step,
                simulation=simulation,
                thermo=thermo,
                timestep_fs=timestep_fs,
            )

            print(
                "Equilibration:",
                logical_step,
                "/",
                args.equil_steps,
                "T*=",
                float(thermo.kinetic_temperature),
                "U*=",
                float(thermo.potential_energy),
            )
    finally:
        equil_handle.close()

    prod_handle, prod_writer = write_thermo_header(
        output_dir / "production_thermo.csv"
    )

    trajectory_frames: list[np.ndarray] = []
    trajectory_steps: list[int] = []

    production_elapsed = 0

    try:
        remaining = args.prod_steps

        while remaining > 0:
            chunk = min(
                args.report_interval,
                remaining,
            )
            simulation.run(chunk)

            logical_step += chunk
            production_elapsed += chunk
            remaining -= chunk

            write_thermo_row(
                writer=prod_writer,
                phase="production",
                logical_step=logical_step,
                simulation=simulation,
                thermo=thermo,
                timestep_fs=timestep_fs,
            )

            current_snapshot = (
                simulation.state.get_snapshot()
            )

            if (
                current_snapshot.communicator.rank
                == 0
            ):
                reduced_positions = np.asarray(
                    current_snapshot
                    .particles
                    .position,
                    dtype=np.float64,
                )

                # Convert centered HOOMD coordinates to [0,L),
                # matching the existing OpenMM trajectory convention.
                wrapped_reduced = np.mod(
                    (
                        reduced_positions
                    )
                    + 0.5 * box_lengths_reduced,
                    box_lengths_reduced,
                )

                positions_nm = (
                    wrapped_reduced
                    * diameter_nm
                )

                trajectory_frames.append(
                    np.asarray(
                        positions_nm,
                        dtype=np.float32,
                    )
                )
                trajectory_steps.append(
                    int(logical_step)
                )

            print(
                "Production:",
                production_elapsed,
                "/",
                args.prod_steps,
                "T*=",
                float(thermo.kinetic_temperature),
                "U*=",
                float(thermo.potential_energy),
            )
    finally:
        prod_handle.close()

    if device.communicator.rank != 0:
        return

    trajectory_file = (
        output_dir
        / "trajectory_positions.npz"
    )

    np.savez_compressed(
        trajectory_file,
        positions_nm=np.stack(
            trajectory_frames,
            axis=0,
        ),
        steps=np.asarray(
            trajectory_steps,
            dtype=np.int64,
        ),
        box_lengths_nm=np.asarray(
            box_lengths_nm,
            dtype=np.float64,
        ),
    )

    final_snapshot = (
        simulation.state.get_snapshot()
    )

    final_positions_reduced = np.asarray(
        final_snapshot.particles.position,
        dtype=np.float64,
    )
    final_velocities_reduced = np.asarray(
        final_snapshot.particles.velocity,
        dtype=np.float64,
    )

    np.savez_compressed(
        output_dir / "final_state_hoomd.npz",
        positions_reduced=(
            final_positions_reduced
        ),
        velocities_reduced=(
            final_velocities_reduced
        ),
        box_lengths_reduced=np.asarray(
            box_lengths_reduced,
            dtype=np.float64,
        ),
        hoomd_timestep=np.int64(
            simulation.timestep
        ),
    )

    final_metadata = dict(metadata)
    final_metadata.update(
        {
            "engine": "HOOMD-blue",
            "hoomd_version": (
                hoomd.version.version
            ),
            "gpu_device": str(device.device),
            "thermostat": "Langevin",
            "equil_steps": args.equil_steps,
            "prod_steps": args.prod_steps,
            "report_interval_steps": (
                args.report_interval
            ),
            "minimize_max_steps": (
                args.minimize_max_steps
            ),
            "minimization_steps": (
                minimization_steps
            ),
            "minimization_converged": bool(
                fire.converged
            ),
            "initial_potential_kBT": (
                initial_potential
            ),
            "minimized_potential_kBT": (
                minimized_potential
            ),
            "final_potential_kBT": float(
                thermo.potential_energy
            ),
            "final_kinetic_kBT": float(
                thermo.kinetic_energy
            ),
            "trajectory_frames": len(
                trajectory_frames
            ),
        }
    )

    with (
        output_dir / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            final_metadata,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("Trajectory:", trajectory_file)
    print(
        "Frames    :",
        len(trajectory_frames),
    )
    print("RUN_DIRECT_COEXISTENCE_HOOMD: PASS")


if __name__ == "__main__":
    main()
