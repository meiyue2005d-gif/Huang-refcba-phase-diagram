#!/usr/bin/env python3
"""Run one refCBA direct-coexistence simulation from a centered dense slab."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from openmm import Platform, VerletIntegrator, unit
from openmm.app import PDBFile, Simulation, StateDataReporter

from huang_md.concentration import (
    box_length_to_concentration_mg_ml,
    concentration_to_box_length_nm,
)
from huang_md.openmm_orthorhombic import (
    create_orthorhombic_system,
    create_orthorhombic_topology,
)
from huang_md.parameters import HuangPotentialParameters
from huang_md.slab_initialization import (
    centered_slab_bounds_nm,
    generate_centered_slab_positions_nm,
    minimum_pair_distance_nm,
    orthorhombic_box_from_cubic_length,
    slab_fraction_from_concentrations,
)
from huang_md.state_model import RefCBAStateModel, parameters_for_state


BASELINE_CONFIG = ROOT / "configs" / "huang_baseline.yaml"
STATE_CONFIG = ROOT / "configs" / "refcba_state_model.yaml"
MD_CONFIG = ROOT / "configs" / "refcba_md.yaml"


def load_md_config(path: Path = MD_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if "simulation" not in raw:
        raise ValueError("MD YAML must contain a 'simulation' section.")

    return raw["simulation"]


def positive_integer(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a refCBA direct-coexistence test from a centered dense slab."
        )
    )

    parser.add_argument("--ph", type=float, default=5.6)
    parser.add_argument("--nacl-mM", type=float, default=0.0)
    parser.add_argument(
        "--global-concentration-mg-ml",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--initial-slab-concentration-mg-ml",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--z-aspect-ratio",
        type=float,
        default=3.0,
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
        "--minimize-max-iterations",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--report-interval",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260718,
    )
    parser.add_argument(
        "--device-index",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "direct_coexistence_smoke_pH5p6",
    )
    parser.add_argument(
        "--state-config",
        type=Path,
        default=STATE_CONFIG,
        help="Versioned electrostatic state-model YAML.",
    )
    parser.add_argument(
        "--md-config",
        type=Path,
        default=MD_CONFIG,
        help="Protein/engine YAML; use huang_a1_md.yaml for strict A1.",
    )

    return parser


def energy_kj_mol(state: Any, kind: str) -> float:
    if kind == "potential":
        value = state.getPotentialEnergy()
    elif kind == "kinetic":
        value = state.getKineticEnergy()
    else:
        raise ValueError(f"Unknown energy kind: {kind}")

    return float(value.value_in_unit(unit.kilojoule_per_mole))


def write_pdb(
    path: Path,
    topology: Any,
    positions: Any,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(
            topology,
            positions,
            handle,
            keepIds=True,
        )


def main() -> None:
    args = build_argument_parser().parse_args()

    positive_integer(args.equil_steps, "equil_steps")
    positive_integer(args.prod_steps, "prod_steps")

    if args.minimize_max_iterations < 0:
        raise ValueError(
            "minimize_max_iterations cannot be negative."
        )

    if not 3.0 <= args.ph <= 9.0:
        raise ValueError("pH must be between 3 and 9.")

    if not 0.0 <= args.nacl_mM <= 500.0:
        raise ValueError("NaCl must be between 0 and 500 mM.")

    if args.global_concentration_mg_ml <= 0.0:
        raise ValueError(
            "Global concentration must be positive."
        )

    if (
        args.initial_slab_concentration_mg_ml
        <= args.global_concentration_mg_ml
    ):
        raise ValueError(
            "Initial slab concentration must exceed global concentration."
        )

    if args.z_aspect_ratio < 1.0:
        raise ValueError("z-aspect-ratio must be at least 1.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = HuangPotentialParameters.from_yaml(
        BASELINE_CONFIG
    )
    state_model = RefCBAStateModel.from_yaml(
        args.state_config
    )
    md_config = load_md_config(args.md_config)

    n_particles = int(md_config["n_particles"])
    molecular_weight_kDa = float(
        md_config["molecular_weight_kDa"]
    )
    temperature_K = float(md_config["temperature_K"])
    timestep_fs = float(md_config["timestep_fs"])
    collision_frequency = float(
        md_config["andersen_collision_frequency_per_ps"]
    )
    minimum_separation_reduced = float(
        md_config["initial_min_separation_reduced"]
    )

    report_interval = (
        int(args.report_interval)
        if args.report_interval is not None
        else int(md_config["report_interval_steps"])
    )
    positive_integer(report_interval, "report_interval")

    device_index = (
        args.device_index
        if args.device_index is not None
        else str(md_config["default_device_index"])
    )

    state_params = parameters_for_state(
        baseline=baseline,
        model=state_model,
        pH=args.ph,
        added_nacl_mM=args.nacl_mM,
    )

    if abs(state_params.temperature_K - temperature_K) > 1e-8:
        raise ValueError(
            "Temperature differs between potential and MD configuration."
        )

    cubic_equivalent_length_nm = concentration_to_box_length_nm(
        concentration_mg_ml=args.global_concentration_mg_ml,
        molecular_weight_kDa=molecular_weight_kDa,
        n_particles=n_particles,
    )

    recovered_global_concentration = (
        box_length_to_concentration_mg_ml(
            box_length_nm=cubic_equivalent_length_nm,
            molecular_weight_kDa=molecular_weight_kDa,
            n_particles=n_particles,
        )
    )

    box = orthorhombic_box_from_cubic_length(
        cubic_length_nm=cubic_equivalent_length_nm,
        z_aspect_ratio=args.z_aspect_ratio,
    )

    slab_fraction_z = slab_fraction_from_concentrations(
        global_concentration_mg_ml=(
            args.global_concentration_mg_ml
        ),
        initial_slab_concentration_mg_ml=(
            args.initial_slab_concentration_mg_ml
        ),
    )

    slab_lower_nm, slab_upper_nm = (
        centered_slab_bounds_nm(
            box=box,
            slab_fraction_z=slab_fraction_z,
        )
    )

    slab_thickness_nm = slab_upper_nm - slab_lower_nm
    cutoff_nm = (
        state_params.cutoff_reduced
        * state_params.diameter_nm
    )
    vapor_gap_each_side_nm = (
        box.length_z_nm - slab_thickness_nm
    ) / 2.0

    if slab_thickness_nm <= 2.0 * cutoff_nm:
        raise ValueError(
            "Initial slab must be thicker than twice the pair cutoff. "
            f"slab={slab_thickness_nm:.6f} nm, "
            f"2*cutoff={2.0 * cutoff_nm:.6f} nm."
        )

    if vapor_gap_each_side_nm <= cutoff_nm:
        raise ValueError(
            "Each dilute-side gap must exceed the pair cutoff."
        )

    minimum_distance_nm = (
        minimum_separation_reduced
        * state_params.diameter_nm
    )

    initial_positions_nm = (
        generate_centered_slab_positions_nm(
            n_particles=n_particles,
            box=box,
            slab_fraction_z=slab_fraction_z,
            minimum_distance_nm=minimum_distance_nm,
            seed=args.seed,
        )
    )

    initial_minimum_distance_nm = (
        minimum_pair_distance_nm(
            positions_nm=initial_positions_nm,
            box=box,
        )
    )

    print("=" * 84)
    print("Huang/refCBA direct-coexistence slab simulation")
    print("=" * 84)
    print(f"Output directory             : {output_dir}")
    print(f"pH                           : {args.ph:.6f}")
    print(f"Added NaCl                   : {args.nacl_mM:.6f} mM")
    print(
        "Global concentration         : "
        f"{args.global_concentration_mg_ml:.6f} mg/mL"
    )
    print(
        "Initial slab concentration   : "
        f"{args.initial_slab_concentration_mg_ml:.6f} mg/mL"
    )
    print(f"Particles                    : {n_particles}")
    print(f"Box lengths                  : {box.lengths_nm} nm")
    print(f"Box z aspect ratio           : {args.z_aspect_ratio:.6f}")
    print(f"Initial slab fraction z      : {slab_fraction_z:.6f}")
    print(
        "Initial slab bounds          : "
        f"({slab_lower_nm:.6f}, {slab_upper_nm:.6f}) nm"
    )
    print(f"Initial slab thickness       : {slab_thickness_nm:.6f} nm")
    print(f"Dilute gap on each side      : {vapor_gap_each_side_nm:.6f} nm")
    print(f"Initial minimum distance     : {initial_minimum_distance_nm:.6f} nm")
    print(f"Particle diameter            : {state_params.diameter_nm:.6f} nm")
    print(f"Pair cutoff                  : {cutoff_nm:.6f} nm")
    print(f"K1                           : {state_params.K1_kBT:.6f} kBT")
    print(f"Z1                           : {state_params.Z1:.6f}")
    print(f"K2                           : {state_params.K2_kBT:.6f} kBT")
    print(f"Z2                           : {state_params.Z2:.6f}")
    print(f"Strict SA-LR state           : {state_params.is_salr}")
    print(f"Temperature                  : {temperature_K:.3f} K")
    print(f"Timestep                     : {timestep_fs:.6f} fs")
    print(f"Equilibration steps          : {args.equil_steps}")
    print(f"Production steps             : {args.prod_steps}")
    print(f"Report interval              : {report_interval}")
    print(f"Random seed                  : {args.seed}")

    system = create_orthorhombic_system(
        params=state_params,
        n_particles=n_particles,
        molecular_weight_kDa=molecular_weight_kDa,
        box=box,
        collision_frequency_per_ps=collision_frequency,
    )

    topology = create_orthorhombic_topology(
        n_particles=n_particles,
        box=box,
    )

    integrator = VerletIntegrator(
        timestep_fs * unit.femtosecond
    )

    platform_name = str(md_config["platform"])
    platform = Platform.getPlatformByName(platform_name)

    platform_properties: dict[str, str] = {}
    if platform_name == "CUDA":
        platform_properties = {
            "CudaPrecision": str(md_config["cuda_precision"]),
            "DeviceIndex": device_index,
        }

    simulation = Simulation(
        topology,
        system,
        integrator,
        platform,
        platform_properties,
    )

    actual_platform = (
        simulation.context.getPlatform().getName()
    )
    print(f"OpenMM platform              : {actual_platform}")

    if actual_platform != platform_name:
        raise RuntimeError(
            f"Requested {platform_name}, but OpenMM used "
            f"{actual_platform}."
        )

    simulation.context.setPositions(
        initial_positions_nm * unit.nanometer
    )

    initial_state = simulation.context.getState(
        getEnergy=True,
        getPositions=True,
    )
    initial_potential = energy_kj_mol(
        initial_state,
        "potential",
    )

    write_pdb(
        output_dir / "initial_structure.pdb",
        topology,
        initial_state.getPositions(),
    )

    print(f"Initial potential energy     : {initial_potential:.6f} kJ/mol")
    print("\nEnergy minimization...")

    simulation.minimizeEnergy(
        maxIterations=args.minimize_max_iterations
    )

    minimized_state = simulation.context.getState(
        getEnergy=True,
        getPositions=True,
        enforcePeriodicBox=True,
    )
    minimized_potential = energy_kj_mol(
        minimized_state,
        "potential",
    )

    write_pdb(
        output_dir / "minimized_structure.pdb",
        topology,
        minimized_state.getPositions(),
    )

    print(f"Minimized potential energy   : {minimized_potential:.6f} kJ/mol")

    simulation.context.setVelocitiesToTemperature(
        temperature_K * unit.kelvin,
        args.seed,
    )

    simulation.reporters.append(
        StateDataReporter(
            str(output_dir / "equilibration_thermo.csv"),
            report_interval,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            speed=True,
            separator=",",
        )
    )

    simulation.reporters.append(
        StateDataReporter(
            sys.stdout,
            report_interval,
            step=True,
            temperature=True,
            potentialEnergy=True,
            speed=True,
        )
    )

    print("\nEquilibration...")
    simulation.step(args.equil_steps)

    simulation.reporters.clear()

    simulation.reporters.append(
        StateDataReporter(
            str(output_dir / "production_thermo.csv"),
            report_interval,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            speed=True,
            separator=",",
        )
    )

    simulation.reporters.append(
        StateDataReporter(
            sys.stdout,
            report_interval,
            step=True,
            temperature=True,
            potentialEnergy=True,
            speed=True,
        )
    )

    print("\nProduction...")
    trajectory_positions: list[np.ndarray] = []
    trajectory_steps: list[int] = []

    remaining_steps = args.prod_steps
    while remaining_steps > 0:
        chunk = min(report_interval, remaining_steps)
        simulation.step(chunk)
        remaining_steps -= chunk

        frame_state = simulation.context.getState(
            getPositions=True,
            enforcePeriodicBox=True,
        )

        frame_positions_nm = np.asarray(
            frame_state.getPositions(
                asNumpy=True
            ).value_in_unit(unit.nanometer),
            dtype=np.float64,
        )

        trajectory_positions.append(frame_positions_nm.copy())
        trajectory_steps.append(int(simulation.currentStep))

    trajectory_array = np.stack(
        trajectory_positions,
        axis=0,
    )

    np.savez_compressed(
        output_dir / "trajectory_positions.npz",
        positions_nm=trajectory_array,
        steps=np.asarray(
            trajectory_steps,
            dtype=np.int64,
        ),
        box_lengths_nm=np.asarray(
            box.lengths_nm,
            dtype=np.float64,
        ),
    )

    final_state = simulation.context.getState(
        getEnergy=True,
        getPositions=True,
        enforcePeriodicBox=True,
    )
    final_potential = energy_kj_mol(
        final_state,
        "potential",
    )
    final_kinetic = energy_kj_mol(
        final_state,
        "kinetic",
    )
    final_total = final_potential + final_kinetic

    write_pdb(
        output_dir / "final_structure.pdb",
        topology,
        final_state.getPositions(),
    )

    metadata = {
        "state_model_id": state_model.model_id,
        "charge_mapping": state_model.charge_mapping,
        "protein_id": state_model.protein_id,
        "protein_sequence_length_aa": len(state_model.protein_sequence),
        "state_config": str(args.state_config.resolve()),
        "md_config": str(args.md_config.resolve()),
        **state_model.applicability_flags(args.ph, args.nacl_mM),
        "initialization_type": "centered_slab",
        "pH": args.ph,
        "added_NaCl_mM": args.nacl_mM,
        "global_concentration_mg_ml": (
            args.global_concentration_mg_ml
        ),
        "concentration_mg_ml": (
            args.global_concentration_mg_ml
        ),
        "recovered_concentration_mg_ml": (
            recovered_global_concentration
        ),
        "initial_slab_concentration_mg_ml": (
            args.initial_slab_concentration_mg_ml
        ),
        "initial_slab_fraction_z": slab_fraction_z,
        "initial_slab_bounds_nm": [
            slab_lower_nm,
            slab_upper_nm,
        ],
        "initial_slab_thickness_nm": slab_thickness_nm,
        "dilute_gap_each_side_nm": vapor_gap_each_side_nm,
        "n_particles": n_particles,
        "molecular_weight_kDa": molecular_weight_kDa,
        "cubic_equivalent_box_length_nm": (
            cubic_equivalent_length_nm
        ),
        "box_lengths_nm": list(box.lengths_nm),
        "box_aspect_ratio_z": args.z_aspect_ratio,
        "box_volume_nm3": box.volume_nm3,
        "diameter_nm": state_params.diameter_nm,
        "cutoff_nm": cutoff_nm,
        "temperature_K": temperature_K,
        "timestep_fs": timestep_fs,
        "collision_frequency_per_ps": collision_frequency,
        "equil_steps": args.equil_steps,
        "prod_steps": args.prod_steps,
        "report_interval_steps": report_interval,
        "seed": args.seed,
        "platform": actual_platform,
        "device_index": device_index,
        "K1_kBT": state_params.K1_kBT,
        "Z1": state_params.Z1,
        "K2_kBT": state_params.K2_kBT,
        "Z2": state_params.Z2,
        "strict_salr": state_params.is_salr,
        "initial_minimum_distance_nm": (
            initial_minimum_distance_nm
        ),
        "initial_potential_kj_mol": initial_potential,
        "minimized_potential_kj_mol": minimized_potential,
        "final_potential_kj_mol": final_potential,
        "final_kinetic_kj_mol": final_kinetic,
        "final_total_kj_mol": final_total,
        "trajectory_frames": int(
            trajectory_array.shape[0]
        ),
        "trajectory_box_key": "box_lengths_nm",
    }

    with (
        output_dir / "metadata.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print("\nFinal energies:")
    print(f"  Potential               : {final_potential:.6f} kJ/mol")
    print(f"  Kinetic                 : {final_kinetic:.6f} kJ/mol")
    print(f"  Total                   : {final_total:.6f} kJ/mol")
    print(f"  Trajectory frames       : {trajectory_array.shape[0]}")

    print("\nGenerated files:")
    for path in sorted(output_dir.iterdir()):
        print(f"  {path}")

    print("\nDirect-coexistence run completed.")


if __name__ == "__main__":
    main()
