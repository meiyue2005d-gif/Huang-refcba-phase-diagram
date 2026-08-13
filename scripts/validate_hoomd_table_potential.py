#!/usr/bin/env python3
"""Validate the HOOMD tabulated potential against OpenMM reference values."""

from __future__ import annotations

import csv
from pathlib import Path

import hoomd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "hoomd_validation"

TABLE_FILE = DATA_DIR / "baseline_hoomd_table.npz"
REFERENCE_FILE = DATA_DIR / "openmm_two_particle_reference.csv"
OUTPUT_FILE = DATA_DIR / "hoomd_two_particle_validation.csv"


def read_reference() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []

    with REFERENCE_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    key: float(value)
                    for key, value in raw.items()
                }
            )

    return rows


def evaluate_pair(
    device: hoomd.device.Device,
    x: float,
    r_min: float,
    r_cut: float,
    U: np.ndarray,
    F: np.ndarray,
) -> tuple[float, float]:
    """Evaluate one two-particle configuration with HOOMD."""
    box_length = max(12.0, 3.0 * r_cut)

    snapshot = hoomd.Snapshot(device.communicator)

    if snapshot.communicator.rank == 0:
        snapshot.configuration.box = [
            box_length,
            box_length,
            box_length,
            0.0,
            0.0,
            0.0,
        ]

        snapshot.particles.N = 2
        snapshot.particles.types = ["A"]
        snapshot.particles.typeid[:] = [0, 0]
        snapshot.particles.mass[:] = [1.0, 1.0]

        snapshot.particles.position[:] = [
            [-0.5 * x, 0.0, 0.0],
            [0.5 * x, 0.0, 0.0],
        ]

    simulation = hoomd.Simulation(
        device=device,
        seed=10097,
    )
    simulation.create_state_from_snapshot(snapshot)

    nlist = hoomd.md.nlist.Cell(buffer=0.4)

    table = hoomd.md.pair.Table(
        nlist=nlist,
        default_r_cut=r_cut,
    )

    table.params[("A", "A")] = {
        "r_min": r_min,
        "U": U.tolist(),
        "F": F.tolist(),
    }
    table.r_cut[("A", "A")] = r_cut

    method = hoomd.md.methods.ConstantVolume(
        filter=hoomd.filter.All(),
    )

    integrator = hoomd.md.Integrator(
        dt=1.0e-5,
        methods=[method],
        forces=[table],
    )

    simulation.operations.integrator = integrator

    # Attach operations and compute energy/forces without advancing time.
    simulation.run(0)

    energy = float(table.energy)
    forces = np.asarray(table.forces, dtype=np.float64)

    # Particle 1 lies on the positive x side.
    radial_force = float(forces[1, 0])

    return energy, radial_force


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not TABLE_FILE.exists():
        raise FileNotFoundError(
            f"Missing table file: {TABLE_FILE}"
        )

    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            f"Missing reference file: {REFERENCE_FILE}"
        )

    data = np.load(TABLE_FILE, allow_pickle=False)

    x_grid = np.asarray(data["x"], dtype=np.float64)
    U = np.asarray(data["U"], dtype=np.float64)
    F = np.asarray(data["F"], dtype=np.float64)

    r_min = float(data["r_min"])
    r_cut = float(data["r_cut"])

    if len(x_grid) != len(U) or len(U) != len(F):
        raise ValueError("x, U and F table lengths differ.")

    spacing = np.diff(x_grid)

    if not np.allclose(
        spacing,
        spacing[0],
        rtol=1.0e-10,
        atol=1.0e-14,
    ):
        raise ValueError("The table grid is not equally spaced.")

    print("===== HOOMD environment =====")
    print("HOOMD version :", hoomd.version.version)
    print("GPU build     :", hoomd.version.gpu_enabled)
    print("GPU available :", hoomd.device.GPU.is_available())
    print(
        "Available GPUs:",
        hoomd.device.GPU.get_available_devices(),
    )
    print("Table width   :", len(U))
    print("r_min         :", r_min)
    print("r_cut         :", r_cut)
    print("Grid spacing  :", spacing[0])

    if not hoomd.version.gpu_enabled:
        raise RuntimeError("Installed HOOMD is not a GPU build.")

    if not hoomd.device.GPU.is_available():
        raise RuntimeError("HOOMD cannot access a GPU.")

    device = hoomd.device.GPU(notice_level=2)
    references = read_reference()

    output_rows: list[dict[str, float]] = []

    for ref in references:
        x = ref["x_reduced"]

        hoomd_U, hoomd_F = evaluate_pair(
            device=device,
            x=x,
            r_min=r_min,
            r_cut=r_cut,
            U=U,
            F=F,
        )

        openmm_U = ref["openmm_U_kBT"]
        openmm_F = ref["openmm_F_kBT_per_d"]

        energy_error = abs(hoomd_U - openmm_U)
        force_error = abs(hoomd_F - openmm_F)

        energy_scaled_error = (
            energy_error / max(1.0, abs(openmm_U))
        )
        force_scaled_error = (
            force_error / max(1.0, abs(openmm_F))
        )

        output_rows.append(
            {
                "x_reduced": x,
                "openmm_U_kBT": openmm_U,
                "hoomd_U_kBT": hoomd_U,
                "energy_abs_error": energy_error,
                "energy_scaled_error": energy_scaled_error,
                "openmm_F_kBT_per_d": openmm_F,
                "hoomd_F_kBT_per_d": hoomd_F,
                "force_abs_error": force_error,
                "force_scaled_error": force_scaled_error,
            }
        )

        print(
            f"x={x:7.4f}  "
            f"U_OMM={openmm_U: .8e}  "
            f"U_HOOMD={hoomd_U: .8e}  "
            f"F_OMM={openmm_F: .8e}  "
            f"F_HOOMD={hoomd_F: .8e}"
        )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(output_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    max_energy_abs = max(
        row["energy_abs_error"]
        for row in output_rows
    )
    max_force_abs = max(
        row["force_abs_error"]
        for row in output_rows
    )
    max_energy_scaled = max(
        row["energy_scaled_error"]
        for row in output_rows
    )
    max_force_scaled = max(
        row["force_scaled_error"]
        for row in output_rows
    )

    print()
    print("===== Error summary =====")
    print(f"Max energy abs error   : {max_energy_abs:.12e}")
    print(f"Max force abs error    : {max_force_abs:.12e}")
    print(f"Max energy scaled error: {max_energy_scaled:.12e}")
    print(f"Max force scaled error : {max_force_scaled:.12e}")
    print(f"Output CSV             : {OUTPUT_FILE}")

    # HOOMD GPU uses mixed-precision arithmetic for local force
    # calculations. Use the conventional combined absolute-relative
    # tolerance rather than dividing every error by max(1, |reference|).
    energy_atol = 1.0e-2
    energy_rtol = 2.0e-4
    force_atol = 1.0e-2
    force_rtol = 2.0e-3

    failed_energy = []
    failed_force = []
    energy_error_ratios = []
    force_error_ratios = []

    for row in output_rows:
        energy_limit = (
            energy_atol
            + energy_rtol * abs(row["openmm_U_kBT"])
        )
        force_limit = (
            force_atol
            + force_rtol
            * abs(row["openmm_F_kBT_per_d"])
        )

        energy_ratio = (
            row["energy_abs_error"] / energy_limit
        )
        force_ratio = (
            row["force_abs_error"] / force_limit
        )

        energy_error_ratios.append(energy_ratio)
        force_error_ratios.append(force_ratio)

        if row["energy_abs_error"] > energy_limit:
            failed_energy.append(
                (
                    row["x_reduced"],
                    row["energy_abs_error"],
                    energy_limit,
                )
            )

        if row["force_abs_error"] > force_limit:
            failed_force.append(
                (
                    row["x_reduced"],
                    row["force_abs_error"],
                    force_limit,
                )
            )

    print()
    print("===== Mixed-precision acceptance =====")
    print(
        "Energy criterion: error <= "
        f"{energy_atol} + {energy_rtol}*|U_ref|"
    )
    print(
        "Force criterion : error <= "
        f"{force_atol} + {force_rtol}*|F_ref|"
    )
    print(
        "Max energy error/limit:",
        f"{max(energy_error_ratios):.8e}",
    )
    print(
        "Max force error/limit :",
        f"{max(force_error_ratios):.8e}",
    )

    if failed_energy:
        print("Failed energy points:", failed_energy)
        raise RuntimeError(
            "HOOMD energy error exceeds mixed-precision tolerance."
        )

    if failed_force:
        print("Failed force points:", failed_force)
        raise RuntimeError(
            "HOOMD force error exceeds mixed-precision tolerance."
        )

    print("HOOMD_TABLE_VALIDATION: PASS")


if __name__ == "__main__":
    main()
