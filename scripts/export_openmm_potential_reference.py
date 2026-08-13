#!/usr/bin/env python3
"""Export the current OpenMM Huang potential as a HOOMD table reference."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from openmm import Context, Platform, System, Vec3, VerletIntegrator, unit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from huang_md.openmm_model import (  # noqa: E402
    calculate_cutoff_shift_reduced,
    create_pair_force,
)
from huang_md.parameters import HuangPotentialParameters  # noqa: E402
from huang_md.potential import (  # noqa: E402
    force_reduced,
    total_derivative_reduced,
    total_potential_reduced,
)


CONFIG = ROOT / "configs" / "huang_baseline.yaml"
OUTPUT_DIR = ROOT / "results" / "hoomd_validation"

# A dense table is necessary because the Gaussian core has epsilon = 1e5 kBT.
TABLE_WIDTH = 65536
R_MIN_REDUCED = 1.0e-4

PROBE_X = np.array(
    [
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.95,
        0.999,
        1.000,
        1.001,
        1.05,
        1.25,
        1.50,
        2.00,
        3.00,
        3.90,
        3.999,
    ],
    dtype=np.float64,
)


def calculate_kbt_kj_mol(temperature_K: float) -> float:
    """Return kBT in kJ/mol."""
    return (
        unit.MOLAR_GAS_CONSTANT_R
        * temperature_K
        * unit.kelvin
    ).value_in_unit(unit.kilojoule_per_mole)


def build_openmm_context(
    params: HuangPotentialParameters,
) -> tuple[Context, VerletIntegrator]:
    """Build an exact two-particle OpenMM reference system."""
    system = System()

    # Mass does not affect a static energy/force evaluation.
    system.addParticle(1.0 * unit.dalton)
    system.addParticle(1.0 * unit.dalton)

    pair_force = create_pair_force(
        params=params,
        n_particles=2,
    )
    system.addForce(pair_force)

    cutoff_nm = params.cutoff_reduced * params.diameter_nm

    # CutoffPeriodic requires every box dimension to exceed 2*r_cut.
    box_nm = max(3.0 * cutoff_nm, 60.0)

    system.setDefaultPeriodicBoxVectors(
        Vec3(box_nm, 0.0, 0.0) * unit.nanometer,
        Vec3(0.0, box_nm, 0.0) * unit.nanometer,
        Vec3(0.0, 0.0, box_nm) * unit.nanometer,
    )

    integrator = VerletIntegrator(
        0.001 * unit.picoseconds
    )

    try:
        platform = Platform.getPlatformByName("Reference")
    except Exception:
        platform = Platform.getPlatformByName("CPU")

    context = Context(
        system,
        integrator,
        platform,
    )

    return context, integrator


def evaluate_openmm(
    context: Context,
    params: HuangPotentialParameters,
    x: float,
) -> tuple[float, float]:
    """Return two-particle energy and radial force in reduced units."""
    distance_nm = x * params.diameter_nm

    context.setPositions(
        [
            Vec3(0.0, 0.0, 0.0),
            Vec3(distance_nm, 0.0, 0.0),
        ]
        * unit.nanometer
    )

    state = context.getState(
        getEnergy=True,
        getForces=True,
    )

    kbt_kj_mol = calculate_kbt_kj_mol(
        params.temperature_K
    )

    energy_reduced = (
        state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )
        / kbt_kj_mol
    )

    forces = state.getForces(
        asNumpy=True
    ).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )

    # Particle 1 is located on the positive x side.
    radial_force_reduced = (
        float(forces[1, 0])
        * params.diameter_nm
        / kbt_kj_mol
    )

    return float(energy_reduced), float(radial_force_reduced)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    params = HuangPotentialParameters.from_yaml(CONFIG)
    params.validate()

    x_cut = params.cutoff_reduced
    cutoff_shift = calculate_cutoff_shift_reduced(params)

    x_grid = np.linspace(
        R_MIN_REDUCED,
        x_cut,
        TABLE_WIDTH,
        dtype=np.float64,
    )

    # OpenMM expression is kBT*(raw - ucut).
    U_table = (
        total_potential_reduced(x_grid, params)
        - cutoff_shift
    )

    # A constant energy shift does not change force.
    F_table = force_reduced(x_grid, params)

    table_path = OUTPUT_DIR / "baseline_hoomd_table.npz"

    np.savez_compressed(
        table_path,
        x=x_grid,
        U=U_table,
        F=F_table,
        r_min=np.float64(R_MIN_REDUCED),
        r_cut=np.float64(x_cut),
        table_width=np.int64(TABLE_WIDTH),
        diameter_nm=np.float64(params.diameter_nm),
        temperature_K=np.float64(params.temperature_K),
        K1=np.float64(params.K1_kBT),
        Z1=np.float64(params.Z1),
        K2=np.float64(params.K2_kBT),
        Z2=np.float64(params.Z2),
        gaussian_sigma=np.float64(
            params.gaussian_sigma_reduced
        ),
        gaussian_epsilon=np.float64(
            params.gaussian_epsilon_kBT
        ),
        cutoff_shift=np.float64(cutoff_shift),
    )

    context, integrator = build_openmm_context(params)

    csv_path = OUTPUT_DIR / "openmm_two_particle_reference.csv"

    rows: list[dict[str, float]] = []

    try:
        for x in PROBE_X:
            analytic_U = float(
                total_potential_reduced(
                    np.array([x]),
                    params,
                )[0]
                - cutoff_shift
            )

            analytic_F = float(
                force_reduced(
                    np.array([x]),
                    params,
                )[0]
            )

            openmm_U, openmm_F = evaluate_openmm(
                context=context,
                params=params,
                x=float(x),
            )

            rows.append(
                {
                    "x_reduced": float(x),
                    "analytic_U_kBT": analytic_U,
                    "openmm_U_kBT": openmm_U,
                    "energy_abs_error": abs(
                        analytic_U - openmm_U
                    ),
                    "analytic_F_kBT_per_d": analytic_F,
                    "openmm_F_kBT_per_d": openmm_F,
                    "force_abs_error": abs(
                        analytic_F - openmm_F
                    ),
                }
            )
    finally:
        del context
        del integrator

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    max_energy_error = max(
        row["energy_abs_error"] for row in rows
    )
    max_force_error = max(
        row["force_abs_error"] for row in rows
    )

    epsilon = 1.0e-8

    left_x = np.array([1.0 - epsilon])
    right_x = np.array([1.0 + epsilon])

    left_U = float(
        total_potential_reduced(left_x, params)[0]
    )
    right_U = float(
        total_potential_reduced(right_x, params)[0]
    )

    left_dU = float(
        total_derivative_reduced(left_x, params)[0]
    )
    right_dU = float(
        total_derivative_reduced(right_x, params)[0]
    )

    print("===== OpenMM reference export =====")
    print(f"Config              : {CONFIG}")
    print(f"Table               : {table_path}")
    print(f"Reference CSV       : {csv_path}")
    print(f"Table width         : {TABLE_WIDTH}")
    print(f"Reduced r_min       : {R_MIN_REDUCED:.8g}")
    print(f"Reduced r_cut       : {x_cut:.8g}")
    print(f"Cutoff shift        : {cutoff_shift:.12e} kBT")
    print(f"U(r_cut) table      : {U_table[-1]:.12e} kBT")
    print(f"F(r_cut-) table     : {F_table[-1]:.12e} kBT/d")
    print(f"x=1 U jump estimate : {right_U-left_U:.12e}")
    print(f"x=1 dU jump estimate: {right_dU-left_dU:.12e}")
    print(f"Max OpenMM U error  : {max_energy_error:.12e}")
    print(f"Max OpenMM F error  : {max_force_error:.12e}")

    tolerance = 1.0e-6

    if max_energy_error > tolerance:
        raise RuntimeError(
            "OpenMM energy does not match the analytic implementation."
        )

    if max_force_error > tolerance:
        raise RuntimeError(
            "OpenMM force does not match the analytic implementation."
        )

    print("OPENMM_REFERENCE_VALIDATION: PASS")


if __name__ == "__main__":
    main()
