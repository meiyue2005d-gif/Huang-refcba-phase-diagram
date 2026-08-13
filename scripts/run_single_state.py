#!/usr/bin/env python3
"""Run one pH-NaCl-concentration state using OpenMM CUDA."""

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
from openmm.app import (
    PDBFile,
    Simulation,
    StateDataReporter,
)

from huang_md.concentration import (
    box_length_to_concentration_mg_ml,
    concentration_to_box_length_nm,
)
from huang_md.openmm_model import (
    create_colloid_topology,
    create_system,
    generate_random_positions_nm,
)
from huang_md.parameters import HuangPotentialParameters
from huang_md.state_model import (
    RefCBAStateModel,
    parameters_for_state,
)


BASELINE_CONFIG = ROOT / "configs" / "huang_baseline.yaml"
STATE_CONFIG = ROOT / "configs" / "refcba_state_model.yaml"
MD_CONFIG = ROOT / "configs" / "refcba_md.yaml"


def load_md_config(path: Path = MD_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if "simulation" not in raw:
        raise ValueError(
            "MD YAML must contain a 'simulation' section."
        )

    return raw["simulation"]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one Huang/refCBA coarse-grained colloidal state."
        )
    )

    parser.add_argument(
        "--ph",
        type=float,
        default=4.5,
    )

    parser.add_argument(
        "--nacl-mM",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--concentration-mg-ml",
        type=float,
        default=0.436,
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
        default=20260716,
    )

    parser.add_argument(
        "--device-index",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "single_state_smoke",
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


def ensure_positive_integer(
    value: int,
    name: str,
) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def main() -> None:
    args = build_argument_parser().parse_args()

    ensure_positive_integer(
        args.equil_steps,
        "equil_steps",
    )
    ensure_positive_integer(
        args.prod_steps,
        "prod_steps",
    )

    if args.minimize_max_iterations < 0:
        raise ValueError(
            "minimize_max_iterations cannot be negative."
        )

    if not 3.0 <= args.ph <= 9.0:
        raise ValueError("pH must be between 3 and 9.")

    if not 0.0 <= args.nacl_mM <= 500.0:
        raise ValueError(
            "NaCl must be between 0 and 500 mM."
        )

    if args.concentration_mg_ml <= 0:
        raise ValueError(
            "Concentration must be positive."
        )

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
    temperature_K = float(
        md_config["temperature_K"]
    )
    timestep_fs = float(
        md_config["timestep_fs"]
    )
    collision_frequency = float(
        md_config[
            "andersen_collision_frequency_per_ps"
        ]
    )
    minimum_separation_reduced = float(
        md_config[
            "initial_min_separation_reduced"
        ]
    )

    report_interval = (
        int(args.report_interval)
        if args.report_interval is not None
        else int(md_config["report_interval_steps"])
    )

    ensure_positive_integer(
        report_interval,
        "report_interval",
    )

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
            "Temperature differs between the potential and MD "
            "configuration files."
        )

    box_length_nm = concentration_to_box_length_nm(
        concentration_mg_ml=args.concentration_mg_ml,
        molecular_weight_kDa=molecular_weight_kDa,
        n_particles=n_particles,
    )

    recovered_concentration = (
        box_length_to_concentration_mg_ml(
            box_length_nm=box_length_nm,
            molecular_weight_kDa=molecular_weight_kDa,
            n_particles=n_particles,
        )
    )

    cutoff_nm = (
        state_params.cutoff_reduced
        * state_params.diameter_nm
    )

    minimum_distance_nm = (
        minimum_separation_reduced
        * state_params.diameter_nm
    )

    print("=" * 76)
    print("Huang/refCBA single-state OpenMM simulation")
    print("=" * 76)
    print(f"Output directory       : {output_dir}")
    print(f"pH                     : {args.ph:.6f}")
    print(f"Added NaCl             : {args.nacl_mM:.6f} mM")
    print(
        f"Concentration          : "
        f"{args.concentration_mg_ml:.6f} mg/mL"
    )
    print(f"Particles              : {n_particles}")
    print(
        f"Molecular weight       : "
        f"{molecular_weight_kDa:.6f} kDa"
    )
    print(f"Box length             : {box_length_nm:.6f} nm")
    print(f"Recovered concentration: {recovered_concentration:.6f} mg/mL")
    print(f"Particle diameter      : {state_params.diameter_nm:.6f} nm")
    print(f"Cutoff                 : {cutoff_nm:.6f} nm")
    print(f"K1                     : {state_params.K1_kBT:.6f} kBT")
    print(f"Z1                     : {state_params.Z1:.6f}")
    print(f"K2                     : {state_params.K2_kBT:.6f} kBT")
    print(f"Z2                     : {state_params.Z2:.6f}")
    print(f"Strict SA-LR state     : {state_params.is_salr}")
    print(f"Temperature            : {temperature_K:.3f} K")
    print(f"Timestep               : {timestep_fs:.6f} fs")
    print(f"Equilibration steps    : {args.equil_steps}")
    print(f"Production steps       : {args.prod_steps}")
    print(f"Report interval        : {report_interval}")
    print(f"Random seed            : {args.seed}")

    system = create_system(
        params=state_params,
        n_particles=n_particles,
        molecular_weight_kDa=molecular_weight_kDa,
        box_length_nm=box_length_nm,
        collision_frequency_per_ps=collision_frequency,
    )

    topology = create_colloid_topology(
        n_particles=n_particles,
        box_length_nm=box_length_nm,
    )

    initial_positions_nm = generate_random_positions_nm(
        n_particles=n_particles,
        box_length_nm=box_length_nm,
        minimum_distance_nm=minimum_distance_nm,
        seed=args.seed,
    )

    integrator = VerletIntegrator(
        timestep_fs * unit.femtosecond
    )

    platform_name = str(md_config["platform"])
    platform = Platform.getPlatformByName(
        platform_name
    )

    platform_properties: dict[str, str] = {}

    if platform_name == "CUDA":
        platform_properties = {
            "CudaPrecision": str(
                md_config["cuda_precision"]
            ),
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
        simulation.context
        .getPlatform()
        .getName()
    )

    print(f"OpenMM platform        : {actual_platform}")

    if actual_platform != platform_name:
        raise RuntimeError(
            f"Requested {platform_name}, but OpenMM used "
            f"{actual_platform}."
        )

    simulation.context.setPositions(
        initial_positions_nm * unit.nanometer
    )

    initial_state = simulation.context.getState(
        getEnergy=True
    )

    initial_potential = (
        initial_state
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )

    print(
        f"Initial potential energy: "
        f"{initial_potential:.6f} kJ/mol"
    )

    print("\nEnergy minimization...")

    simulation.minimizeEnergy(
        maxIterations=args.minimize_max_iterations
    )

    minimized_state = simulation.context.getState(
        getEnergy=True,
        getPositions=True,
    )

    minimized_potential = (
        minimized_state
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )

    print(
        f"Minimized potential     : "
        f"{minimized_potential:.6f} kJ/mol"
    )

    simulation.context.setVelocitiesToTemperature(
        temperature_K * unit.kelvin,
        args.seed,
    )

    equilibration_csv = (
        output_dir / "equilibration_thermo.csv"
    )

    simulation.reporters.append(
        StateDataReporter(
            str(equilibration_csv),
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

    production_csv = (
        output_dir / "production_thermo.csv"
    )

    simulation.reporters.append(
        StateDataReporter(
            str(production_csv),
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

    trajectory_frames: list[np.ndarray] = []
    trajectory_steps: list[int] = []

    remaining_steps = args.prod_steps

    while remaining_steps > 0:
        chunk_steps = min(
            report_interval,
            remaining_steps,
        )

        simulation.step(chunk_steps)

        trajectory_state = simulation.context.getState(
            getPositions=True,
            enforcePeriodicBox=True,
        )

        positions_nm = (
            trajectory_state
            .getPositions(asNumpy=True)
            .value_in_unit(unit.nanometer)
        )

        trajectory_frames.append(
            np.asarray(
                positions_nm,
                dtype=np.float32,
            )
        )

        trajectory_steps.append(
            int(simulation.currentStep)
        )

        remaining_steps -= chunk_steps

    simulation.reporters.clear()

    final_state = simulation.context.getState(
        getEnergy=True,
        getPositions=True,
        getVelocities=True,
        enforcePeriodicBox=True,
    )

    final_potential = (
        final_state
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )

    final_kinetic = (
        final_state
        .getKineticEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )

    final_total = final_potential + final_kinetic

    print("\nFinal state:")
    print(
        f"  Potential energy      : "
        f"{final_potential:.6f} kJ/mol"
    )
    print(
        f"  Kinetic energy        : "
        f"{final_kinetic:.6f} kJ/mol"
    )
    print(
        f"  Total energy          : "
        f"{final_total:.6f} kJ/mol"
    )

    trajectory_array = np.stack(
        trajectory_frames,
        axis=0,
    )

    np.savez_compressed(
        output_dir / "trajectory_positions.npz",
        positions_nm=trajectory_array,
        steps=np.asarray(
            trajectory_steps,
            dtype=np.int64,
        ),
        box_length_nm=np.asarray(
            box_length_nm,
            dtype=np.float64,
        ),
    )

    final_positions = final_state.getPositions()

    with (
        output_dir / "final_structure.pdb"
    ).open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(
            topology,
            final_positions,
            handle,
            keepIds=True,
        )

    simulation.saveCheckpoint(
        str(output_dir / "final_checkpoint.chk")
    )

    metadata = {
        "state_model_id": state_model.model_id,
        "charge_mapping": state_model.charge_mapping,
        "protein_id": state_model.protein_id,
        "protein_sequence_length_aa": len(state_model.protein_sequence),
        "state_config": str(args.state_config.resolve()),
        "md_config": str(args.md_config.resolve()),
        **state_model.applicability_flags(args.ph, args.nacl_mM),
        "strict_salr_interaction": bool(state_params.is_salr),
        "outside_strict_salr_regime": not bool(state_params.is_salr),
        "pH": args.ph,
        "added_NaCl_mM": args.nacl_mM,
        "concentration_mg_ml": args.concentration_mg_ml,
        "recovered_concentration_mg_ml": recovered_concentration,
        "n_particles": n_particles,
        "molecular_weight_kDa": molecular_weight_kDa,
        "box_length_nm": box_length_nm,
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
        "initial_potential_kj_mol": initial_potential,
        "minimized_potential_kj_mol": minimized_potential,
        "final_potential_kj_mol": final_potential,
        "final_kinetic_kj_mol": final_kinetic,
        "final_total_kj_mol": final_total,
        "trajectory_frames": int(
            trajectory_array.shape[0]
        ),
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

    print("\nGenerated files:")
    for path in sorted(output_dir.iterdir()):
        print(f"  {path}")

    print("\nSingle-state CUDA smoke test completed.")


if __name__ == "__main__":
    main()
